"""MLP multitask (PyTorch): un tronco compartido y varias cabezas de regresión.

Extiende el MLP single-task (src/models/mlp.py) al caso multitask: una sola red
comparte representación entre varios traits (GY, PH, AD) y una cabeza Linear(prev,
n_targets) los predice todos a la vez, para ver si las tareas auxiliares (PH, AD)
ayudan a la portada (GY) en p >> n. Reusa _MLPNet y pearson_loss del single-task.

La estandarización train-only va dentro de fit: z-score de features por SNP y del
target por columna, ignorando los valores enmascarados. Los mu/sigma del target se
registran como buffers del net para que viajen en el state_dict. Cada observación
puede no tener medidos todos los traits: Y trae los faltantes en 0 e Y_mask marca
los reales, y la loss usa solo las filas válidas de cada columna.

La loss es una suma ponderada de Pearson por trait (más peso a GY); si un trait tiene
<2 ejemplos válidos en el batch su término es 0. loss="pearson" (default) o
"pearson_mse". No implementa early stopping (patience debe ser null). save/load
guardan state_dict + arquitectura + escalas + nombres de traits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.eval.metrics import pearson_r
from src.models.base import BaseModel
from src.models.mlp import _MLPNet, pearson_loss

_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [256],
    "dropout": 0.5,
    "input_dropout": 0.2,
    "activation": "relu",
    "batchnorm": False,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "loss": "pearson",
    "mse_weight": 0.3,
    "patience": None,
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",
    "loss_weights": {},
}


def masked_weighted_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    mse_weight: float = 0.0,
) -> torch.Tensor:
    """Suma ponderada de pérdidas Pearson por trait, usando solo filas válidas.

    pred/target/mask tienen shape (B, T) y weights (T,). Para cada trait j toma las
    filas con mask[:, j] y suma w_j * (1 - r_j) (más w_j * mse_weight * MSE_j si
    mse_weight > 0). Si un trait tiene <2 filas válidas en el batch su término se omite
    para no propagar NaN. Reusa pearson_loss.
    """
    total = pred.new_zeros(())
    for j in range(pred.shape[1]):
        valid = mask[:, j]
        if int(valid.sum().item()) < 2:
            continue
        pj = pred[valid, j]
        tj = target[valid, j]
        lj = pearson_loss(pj, tj)
        if mse_weight > 0.0:
            lj = lj + mse_weight * torch.mean((pj - tj) ** 2)
        total = total + weights[j] * lj
    return total


class MLPMultitaskRegressor(BaseModel):
    """Perceptrón multicapa multitask (PyTorch): tronco compartido + cabeza Linear(prev, n_targets).

    params (sub-dict model.params del config) — además de los del MLP single-task:
        loss               : "pearson" (default) | "pearson_mse".
        loss_weights       : dict {trait: w} para la suma ponderada (default 1.0).
        patience           : debe ser null (early stopping no soportado en v1).

    fit(X, Y, Y_mask) recibe Y (n, T) (faltantes en 0) e Y_mask (n, T)
    bool. Los nombres de los traits llegan por set_fit_context(target_names=...) para
    alinear loss_weights con las columnas de Y. predict(X) devuelve (n, T).
    """


    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["hidden_dims"] = [int(h) for h in cfg["hidden_dims"]]
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["loss"] = str(cfg["loss"]).lower()
        cfg["mse_weight"] = float(cfg["mse_weight"])
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["patience"] = None if cfg["patience"] is None else int(cfg["patience"])
        cfg["loss_weights"] = dict(cfg["loss_weights"] or {})
        return cfg

    def _resolve_device(self, device: str) -> "torch.device":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return torch.device(device)

    @staticmethod
    def _seed(seed: int) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


    @staticmethod
    def _compute_x_scalers(X: np.ndarray, cfg: dict[str, Any]):
        """z-score de features con train. SNP constante => std 1."""
        if cfg["standardize_inputs"]:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0)
            x_std = np.where(x_std == 0.0, 1.0, x_std)
        else:
            x_mean = np.zeros(X.shape[1])
            x_std = np.ones(X.shape[1])
        return np.asarray(x_mean, dtype=np.float64), np.asarray(x_std, dtype=np.float64)

    @staticmethod
    def _compute_y_scalers(Y: np.ndarray, M: np.ndarray, cfg: dict[str, Any]):
        """z-score del target POR COLUMNA, ignorando los valores enmascarados."""
        T = Y.shape[1]
        mu = np.zeros(T, dtype=np.float64)
        sd = np.ones(T, dtype=np.float64)
        if cfg["standardize_target"]:
            for j in range(T):
                col = Y[M[:, j], j]
                if col.size > 0:
                    mu[j] = float(col.mean())
                    s = float(col.std())
                    sd[j] = s if s > 0 else 1.0
        return mu, sd


    def _build_net(self, in_dim: int, out_dim: int, cfg: dict[str, Any]) -> _MLPNet:
        return _MLPNet(
            in_dim,
            cfg["hidden_dims"],
            dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"],
            activation=cfg["activation"],
            batchnorm=cfg["batchnorm"],
            out_dim=out_dim,
        )

    def _forward2d(self, x: torch.Tensor) -> torch.Tensor:
        """Salida del net garantizada 2D (n, T) (con T=1, _MLPNet hace squeeze)."""
        out = self.net_(x)
        return out.unsqueeze(-1) if out.ndim == 1 else out

    def _monitor_val(self, device: "torch.device"):
        """Val externo del fold (de set_fit_context) para la curva por época.

        Devuelve (Xv_estandarizado, Yv_raw, Mv) o None. No selecciona nada (sin early
        stopping). Como el Pearson es invariante a afines, compara la predicción
        estandarizada contra el target crudo.
        """
        Xe = self._fit_context.get("eval_X")
        Ye = self._fit_context.get("eval_Y")
        Me = self._fit_context.get("eval_Ymask")
        if Xe is None or Ye is None:
            return None
        Xe = np.asarray(Xe, dtype=np.float64)
        Ye = np.asarray(Ye, dtype=np.float64)
        if Ye.ndim == 1:
            Ye = Ye.reshape(-1, 1)
        Me = np.ones_like(Ye, dtype=bool) if Me is None else np.asarray(Me, dtype=bool)
        if (
            Xe.ndim != 2
            or Xe.shape[0] == 0
            or Xe.shape[1] != self.n_features_in_
            or Ye.shape[0] != Xe.shape[0]
            or Ye.shape[1] != self.n_targets_
        ):
            return None
        Xs = (Xe - self.x_mean_) / self.x_std_
        return torch.as_tensor(Xs, dtype=torch.float32, device=device), Ye, Me


    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MLPMultitaskRegressor":
        cfg = self._resolve_cfg()
        self.cfg_ = cfg
        if cfg["patience"] is not None:
            raise NotImplementedError(
                "early stopping multitask no soportado en v1; usá patience: null."
            )
        device = self._resolve_device(cfg["device"])
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("MLPMultitaskRegressor.fit recibió Y vacío.")

        self.n_features_in_ = int(X.shape[1])
        self.n_targets_ = int(Y.shape[1])
        names = self._fit_context.get("target_names")
        self.target_names_ = (
            [str(n) for n in names] if names is not None else [f"t{j}" for j in range(self.n_targets_)]
        )
        if len(self.target_names_) != self.n_targets_:
            raise ValueError(
                f"target_names ({len(self.target_names_)}) no coincide con n_targets ({self.n_targets_})."
            )
        self.loss_weights_ = np.asarray(
            [float(cfg["loss_weights"].get(n, 1.0)) for n in self.target_names_], dtype=np.float64
        )

        self._seed(cfg["random_state"])
        self.x_mean_, self.x_std_ = self._compute_x_scalers(X, cfg)
        y_mean, y_std = self._compute_y_scalers(Y, Y_mask, cfg)
        Xs = (X - self.x_mean_) / self.x_std_
        Ys = (Y - y_mean) / y_std

        net = self._build_net(self.n_features_in_, self.n_targets_, cfg).to(device)
        net.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32, device=device))
        net.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32, device=device))
        self.net_ = net

        self.history_ = self._train(Xs, Ys, Y_mask, cfg=cfg, device=device)
        self._finalize(device)
        return self

    def _train(self, Xs, Ys, M, *, cfg, device) -> list[dict[str, float]]:
        net = self.net_
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Ys, dtype=torch.float32, device=device)
        Mt = torch.as_tensor(M, dtype=torch.bool, device=device)
        w = torch.as_tensor(self.loss_weights_, dtype=torch.float32, device=device)
        mse_w = cfg["mse_weight"] if cfg["loss"] == "pearson_mse" else 0.0

        n = Xt.shape[0]
        bs = cfg["batch_size"]
        full_batch = bs <= 0 or bs >= n
        mon = self._monitor_val(device)

        history: list[dict[str, float]] = []
        for epoch in range(int(cfg["max_epochs"])):
            net.train()
            if full_batch:
                batches = [torch.arange(n, device=device)]
            else:
                perm = torch.randperm(n, device=device)
                batches = list(torch.split(perm, bs))
            run_loss = 0.0
            seen = 0
            for idx in batches:
                optim.zero_grad()
                pred = self._forward2d(Xt[idx])
                loss = masked_weighted_pearson_loss(pred, Yt[idx], Mt[idx], w, mse_weight=mse_w)
                if loss.requires_grad:
                    loss.backward()
                    optim.step()
                run_loss += float(loss.item()) * idx.numel()
                seen += int(idx.numel())
            rec: dict[str, float] = {"epoch": epoch, "train_loss": run_loss / max(seen, 1)}
            if mon is not None:
                net.eval()
                with torch.no_grad():
                    Xv, Yv_np, Mv_np = mon
                    pv = self._forward2d(Xv).detach().cpu().numpy()
                for j, name in enumerate(self.target_names_):
                    colv = Mv_np[:, j]
                    if colv.sum() >= 2:
                        rec[f"val_pearson_{name}"] = pearson_r(Yv_np[colv, j], pv[colv, j])
            history.append(rec)
        return history

    def _finalize(self, device: "torch.device") -> None:
        self.net_.eval()
        self.device_ = str(device)
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "net_"):
            raise RuntimeError("MLPMultitaskRegressor.predict llamado antes de fit.")
        device = next(self.net_.parameters()).device
        Xs = (np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_std_
        self.net_.eval()
        with torch.no_grad():
            out = self._forward2d(torch.as_tensor(Xs, dtype=torch.float32, device=device))
            out = out * self.net_.y_std + self.net_.y_mean
        return out.detach().cpu().numpy().astype(float)

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "net_"):
            return {}
        return {
            "selected_epochs": int(self.cfg_["max_epochs"]),
            "n_params": getattr(self, "n_params_", None),
            "n_targets": getattr(self, "n_targets_", None),
            "hidden_dims": str(self.cfg_["hidden_dims"]),
            "loss_weights": str(
                {n: float(wv) for n, wv in zip(self.target_names_, self.loss_weights_)}
            ),
            "target_names": str(self.target_names_),
        }


    def save(self, path: str | Path) -> Path:
        if not hasattr(self, "net_"):
            raise RuntimeError("MLPMultitaskRegressor.save llamado antes de fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "config": self.config,
            "cfg_": self.cfg_,
            "state_dict": self.net_.state_dict(),
            "n_features_in_": self.n_features_in_,
            "n_targets_": self.n_targets_,
            "target_names_": list(self.target_names_),
            "loss_weights_": np.asarray(self.loss_weights_),
            "x_mean_": np.asarray(self.x_mean_),
            "x_std_": np.asarray(self.x_std_),
            "n_params_": getattr(self, "n_params_", None),
            "device_": getattr(self, "device_", "cpu"),
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLPMultitaskRegressor":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(ckpt.get("config"))
        obj.cfg_ = ckpt["cfg_"]
        obj.n_features_in_ = int(ckpt["n_features_in_"])
        obj.n_targets_ = int(ckpt["n_targets_"])
        obj.target_names_ = list(ckpt["target_names_"])
        obj.loss_weights_ = np.asarray(ckpt["loss_weights_"], dtype=np.float64)
        obj.x_mean_ = np.asarray(ckpt["x_mean_"], dtype=np.float64)
        obj.x_std_ = np.asarray(ckpt["x_std_"], dtype=np.float64)
        net = obj._build_net(obj.n_features_in_, obj.n_targets_, obj.cfg_)
        net.register_buffer("y_mean", torch.zeros(obj.n_targets_))
        net.register_buffer("y_std", torch.ones(obj.n_targets_))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        obj.net_ = net
        obj.n_params_ = ckpt.get("n_params_")
        obj.device_ = ckpt.get("device_", "cpu")
        obj.history_ = []
        return obj
