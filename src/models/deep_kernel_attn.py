"""Deep kernel / atención sobre el train — el kernel relacional APRENDIDO (supervisado).

Generaliza GBLUP. Recordá que la predicción de GBLUP es

    ŷ_q = Σ_j w_qj · y_j ,   w = G_PS,TS (G_TS + λI)^-1 ,   G = ZZ'/p

es decir una **suma ponderada de los fenotipos del train**, con pesos = parentesco genómico
aditivo, LINEAL y FIJO. Acá esos pesos se **aprenden**: un encoder ``φ`` (MLP) mapea los SNPs a un
embedding, la similitud coseno ``φ(x_q)·φ(x_j)`` (temperatura ``τ``) define un score, y un softmax
sobre el train da los pesos de atención. La predicción es la mezcla ``Σ_j a_qj · y_j``. El kernel se
entrena SUPERVISADO para predecir el fenotipo (no para medir parentesco), que es justo lo que a
GBLUP le falta.

Entrenamiento (leave-one-out por LÍNEA): cada fila de train actúa como *query* y atiende sobre las
demás filas de train (*keys/values*), con las keys que comparten ``line_id`` con la query
enmascaradas (incluida ella misma). Sin ese mask el modelo copiaría su propio fenotipo (o el de la
misma línea en otro año) y no generalizaría a líneas nuevas. En predicción, las líneas de test son
disjuntas del train (regla 2, garantizado por el split) ⇒ sin mask.

Reglas del proyecto: preprocesamiento train-only DENTRO de ``fit`` (regla 1: z-score de features y
del target con stats de train); split temporal lo maneja ``run_evaluation`` (regla 3); la salida es
un fenotipo continuo (la BCE de la regla 4 es para autoencoders, no acá). Persistencia torch
(``state_dict`` + arquitectura + embeddings de las keys), como todo modelo torch del proyecto.

`params` (sub-dict ``model.params`` del config):
    embed_dim      : dim del embedding φ(x) (default 64).
    hidden_dims    : capas ocultas del encoder antes del embedding (default [512]).
    dropout        : dropout tras cada capa oculta del encoder (default 0.3).
    input_dropout  : dropout sobre los SNPs de entrada (clave en p>>n; default 0.3).
    activation     : relu|gelu|leaky_relu|elu|tanh (default relu).
    batchnorm      : BatchNorm1d en el encoder (default False).
    cosine         : normaliza los embeddings (similitud coseno) antes del score (default True).
    tau            : temperatura del softmax de atención; score = sim/τ (default 0.1).
    lr             : learning rate de Adam (default 1e-3).
    weight_decay   : L2 de Adam (default 1e-4).
    max_epochs     : épocas (full-batch; default 300).
    random_state   : seed del RNG de torch (default 575).
    device         : "cpu"|"cuda"|"auto" (default auto).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.eval.metrics import pearson_r
from src.models.base import BaseModel
from src.models.mlp import _MLPNet

_DEFAULTS: dict[str, Any] = {
    "embed_dim": 64,
    "hidden_dims": [512],
    "dropout": 0.3,
    "input_dropout": 0.3,
    "activation": "relu",
    "batchnorm": False,
    "cosine": True,
    "tau": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 300,
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "auto",
}


class DeepKernelAttentionRegressor(BaseModel):
    """Regresión por kernel/atención aprendida sobre el train (generaliza GBLUP)."""

    # --- Resolución de HP ---------------------------------------------------

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["embed_dim"] = int(cfg["embed_dim"])
        cfg["hidden_dims"] = [int(h) for h in cfg["hidden_dims"]]
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["cosine"] = bool(cfg["cosine"])
        cfg["tau"] = float(cfg["tau"])
        if cfg["tau"] <= 0:
            raise ValueError("tau debe ser > 0.")
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        return cfg

    def _resolve_device(self, device: str) -> "torch.device":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return torch.device(device)

    # --- Estandarización train-only (regla 1) -------------------------------

    def _compute_scalers(self, X: np.ndarray, y: np.ndarray, cfg: dict[str, Any]):
        if cfg["standardize_inputs"]:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0)
            x_std = np.where(x_std == 0.0, 1.0, x_std)
        else:
            x_mean = np.zeros(X.shape[1])
            x_std = np.ones(X.shape[1])
        if cfg["standardize_target"]:
            y_mean = float(y.mean())
            y_std = float(y.std()) or 1.0
        else:
            y_mean, y_std = 0.0, 1.0
        return (np.asarray(x_mean, float), np.asarray(x_std, float), float(y_mean), float(y_std))

    def _apply_x(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_std_

    def _build_net(self, in_dim: int, cfg: dict[str, Any]) -> _MLPNet:
        # Encoder φ: mismo bloque _MLPNet que la MLP, con out_dim = embed_dim (embedding lineal).
        return _MLPNet(
            in_dim,
            cfg["hidden_dims"],
            dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"],
            activation=cfg["activation"],
            batchnorm=cfg["batchnorm"],
            out_dim=cfg["embed_dim"],
        )

    @staticmethod
    def _embed(net: _MLPNet, X: torch.Tensor, cosine: bool) -> torch.Tensor:
        h = net(X)                                  # (n, d)
        if cosine:
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
        return h

    # --- Entrenamiento ------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepKernelAttentionRegressor":
        cfg = self._resolve_cfg()
        self.cfg_ = cfg
        device = self._resolve_device(cfg["device"])
        torch.manual_seed(cfg["random_state"])

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("DeepKernelAttentionRegressor.fit recibió y vacío.")
        self.n_features_in_ = int(X.shape[1])

        self.x_mean_, self.x_std_, self.y_mean_, self.y_std_ = self._compute_scalers(X, y, cfg)
        Xs = self._apply_x(X)
        ys = (y - self.y_mean_) / self.y_std_

        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        yt = torch.as_tensor(ys, dtype=torch.float32, device=device)
        n = Xt.shape[0]

        # Mask leave-one-out por LÍNEA: keys que comparten line_id con la query (incl. diagonal)
        # quedan fuera. Sin line_id, cae a mask de solo la diagonal (leave-one-out por fila).
        line_id = self._fit_context.get("line_id")
        if line_id is not None and len(line_id) == n:
            gid = np.asarray(line_id)
            same = torch.as_tensor(gid[:, None] == gid[None, :], device=device)
        else:
            same = torch.eye(n, dtype=torch.bool, device=device)
        valid_q = (~same).any(dim=1)                # queries con al menos una key válida
        neg_inf = torch.finfo(torch.float32).min

        # Val externo del fold SOLO para la curva por época (sin selección => sin leakage).
        mon = self._monitor(device)

        net = self._build_net(self.n_features_in_, cfg).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        tau = cfg["tau"]
        history: list[dict[str, float]] = []

        for epoch in range(cfg["max_epochs"]):
            net.train()
            optim.zero_grad()
            h = self._embed(net, Xt, cfg["cosine"])          # (n, d)
            scores = (h @ h.t()) / tau                       # (n, n)
            scores = scores.masked_fill(same, neg_inf)       # leave-one-out por línea
            attn = torch.softmax(scores, dim=1)              # (n, n)
            pred = attn @ yt                                 # (n,)  mezcla de fenotipos del train
            loss = ((pred[valid_q] - yt[valid_q]) ** 2).mean()
            loss.backward()
            optim.step()

            rec: dict[str, float] = {"epoch": epoch, "train_loss": float(loss.item())}
            if mon is not None:
                Xv, yv_np = mon
                net.eval()
                with torch.no_grad():
                    hq = self._embed(net, Xv, cfg["cosine"])         # (m, d) queries del val
                    hk = self._embed(net, Xt, cfg["cosine"])         # (n, d) keys del train actuales
                    av = torch.softmax((hq @ hk.t()) / tau, dim=1)   # sin mask (val disjunto del train)
                    pv = (av @ yt).cpu().numpy()                     # escala estandarizada
                # Pearson es invariante a afines por variable ⇒ da igual std vs cruda.
                rec["val_pearson"] = pearson_r(yv_np, pv)
            history.append(rec)

        # Congela el encoder y precomputa los embeddings de las KEYS (train) para predecir.
        net.eval()
        with torch.no_grad():
            key_emb = self._embed(net, Xt, cfg["cosine"]).detach().cpu().numpy()
        self.net_ = net.to("cpu")
        self.key_emb_ = key_emb.astype(np.float32)           # (n_train, d), ya normalizado si cosine
        self.val_y_ = ys.astype(np.float32)                  # (n_train,) fenotipos estandarizados
        self.history_ = history
        self.device_ = str(device)
        self.n_params_ = int(sum(p.numel() for p in net.parameters()))
        return self

    def _monitor(self, device):
        """Val del fold (eval_X/eval_y) estandarizado, solo para la curva por época."""
        Xe = self._fit_context.get("eval_X")
        ye = self._fit_context.get("eval_y")
        if Xe is None or ye is None:
            return None
        Xe = np.asarray(Xe, dtype=np.float64)
        ye = np.asarray(ye, dtype=np.float64).ravel()
        if Xe.ndim != 2 or Xe.shape[0] == 0 or Xe.shape[1] != self.n_features_in_ or ye.size != Xe.shape[0]:
            return None
        Xv = torch.as_tensor(self._apply_x(Xe), dtype=torch.float32, device=device)
        return Xv, ye

    # --- Predicción ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "net_"):
            raise RuntimeError("DeepKernelAttentionRegressor.predict llamado antes de fit.")
        cfg = self.cfg_
        Xs = self._apply_x(X)
        self.net_.eval()
        with torch.no_grad():
            hq = self._embed(self.net_, torch.as_tensor(Xs, dtype=torch.float32), cfg["cosine"])  # (m, d)
            keys = torch.as_tensor(self.key_emb_, dtype=torch.float32)                              # (n, d)
            scores = (hq @ keys.t()) / cfg["tau"]            # (m, n) — sin mask (test disjunto)
            attn = torch.softmax(scores, dim=1)
            vals = torch.as_tensor(self.val_y_, dtype=torch.float32)
            pred_std = (attn @ vals).cpu().numpy()           # (m,) en escala estandarizada
        return (pred_std * self.y_std_ + self.y_mean_).astype(float)

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "net_"):
            return {}
        return {
            "deep_kernel_attn": 1,
            "embed_dim": int(self.cfg_["embed_dim"]),
            "tau": float(self.cfg_["tau"]),
            "cosine": int(self.cfg_["cosine"]),
            "n_train_keys": int(self.key_emb_.shape[0]),
            "n_params": getattr(self, "n_params_", None),
            "hidden_dims": str(self.cfg_["hidden_dims"]),   # no-numérico => metadata
        }

    # --- Persistencia (torch: state_dict + arquitectura + keys) -------------

    def save(self, path: str | Path) -> Path:
        if not hasattr(self, "net_"):
            raise RuntimeError("DeepKernelAttentionRegressor.save llamado antes de fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "cfg_": self.cfg_,
                "state_dict": self.net_.state_dict(),
                "n_features_in_": self.n_features_in_,
                "x_mean_": np.asarray(self.x_mean_),
                "x_std_": np.asarray(self.x_std_),
                "y_mean_": float(self.y_mean_),
                "y_std_": float(self.y_std_),
                "key_emb_": self.key_emb_,
                "val_y_": self.val_y_,
                "n_params_": getattr(self, "n_params_", None),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DeepKernelAttentionRegressor":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(ckpt.get("config"))
        obj.cfg_ = ckpt["cfg_"]
        obj.n_features_in_ = int(ckpt["n_features_in_"])
        obj.x_mean_ = np.asarray(ckpt["x_mean_"], dtype=np.float64)
        obj.x_std_ = np.asarray(ckpt["x_std_"], dtype=np.float64)
        obj.y_mean_ = float(ckpt["y_mean_"])
        obj.y_std_ = float(ckpt["y_std_"])
        net = obj._build_net(obj.n_features_in_, obj.cfg_)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        obj.net_ = net
        obj.key_emb_ = np.asarray(ckpt["key_emb_"], dtype=np.float32)
        obj.val_y_ = np.asarray(ckpt["val_y_"], dtype=np.float32)
        obj.n_params_ = ckpt.get("n_params_")
        obj.history_ = []
        return obj
