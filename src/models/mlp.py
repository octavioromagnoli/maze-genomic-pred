"""MLP — perceptrón multicapa (PyTorch) para regresión genómica.

Red neuronal *baseline* que toma los marcadores crudos como capa de entrada y
predice un fenotipo continuo (por defecto GY óptimo) con una cabeza de regresión
lineal. Respeta la interfaz común (`fit`/`predict`) y se enchufa en
`run_evaluation` como cualquier otro modelo.

Decisiones de diseño alineadas con las reglas del proyecto:

- **Sin leakage (regla 1).** El modelo hace su preprocesamiento train-only
  DENTRO de `fit`: estandariza las features (z-score por SNP) y el target con la
  media/desvío del train. En la selección interna de épocas (early stopping) los
  estadísticos se recomputan con el inner-train. Por eso este modelo **no usa el
  bloque `preprocess:`** ni `PreprocessedModel`: ese wrapper persiste con joblib
  (picklearía el `nn.Module`), lo que choca con el contrato de que los modelos
  torch serializan `state_dict` + arquitectura. Todo el preprocesamiento viaja
  dentro del checkpoint torch.
- **Validación temporal en la selección interna (regla 3).** Con `params.patience`
  se activa early stopping reservando el ÚLTIMO año del train como inner-val
  (p.ej. 2017 -> 2018) solo para elegir cuántas épocas entrenar. Luego, por
  default, se reentrena un modelo final desde cero con TODO el train del fold
  (p.ej. 2017+2018) por el número de épocas seleccionado. El val/test externo no
  se toca. Si hay un solo año en train, se entrena `max_epochs` fijo.
- **Pérdida configurable: MSE, Pearson o Pearson+MSE.** La regla 4 (BCE) aplica a
  autoencoders que reconstruyen SNPs binarios; acá la salida es un fenotipo
  continuo. Por `params.loss` se elige entre ``"mse"`` (default, convención
  segura), ``"pearson"`` (minimiza ``1 - r``, optimizando directamente la
  métrica de portada, regla 5) y ``"pearson_mse"``. El r es invariante a
  transformaciones afines, así que con la pérdida Pearson pura la salida queda sin
  calibrar en escala (RMSE/R² dejan de ser informativos; el Pearson —lo que
  importa— no se ve afectado). ``"pearson_mse"`` suma ``mse_weight·MSE`` (sobre el
  target estandarizado): mantiene el ranking que da el término Pearson pero ancla
  la escala, así RMSE/R² vuelven a ser informativos (``mse_weight`` regula el
  trade-off; default 0.3).
- **Reproducibilidad.** `random_state` siembra el RNG de torch dentro de `fit`
  (la seed global de `set_seed` no alcanza para fijar la inicialización de pesos
  de forma independiente por fold).

Persistencia: override de `save`/`load` (regla del proyecto para modelos torch).
El checkpoint guarda arquitectura + `state_dict` + estadísticos de escala +
diagnóstico, así `load_model_artifact` lo reconstruye sin reentrenar.

`get_fitted_params` expone épocas seleccionadas, mejor época / pérdida de
inner-val y tamaño de la red para que `run_evaluation`/`train.py` los logueen.
El historial por época queda en `self.history_` (`train_loss` siempre; `val_loss` y
`val_pearson` cuando hay inner-val de early stopping O cuando `run_evaluation` pasa el
val del fold como set de MONITOREO vía `set_fit_context(eval_X=, eval_y=)`). Ese
monitoreo es solo para la curva — nunca selecciona ni entrena sobre el val (sin
leakage). `train.py` lo loguea por época a wandb (`curve/train_loss`, `curve/val_*`).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import pearson_r
from src.models.base import BaseModel

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}

_PEARSON_EPS = 1e-8


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, *, eps: float = _PEARSON_EPS) -> torch.Tensor:
    """Pérdida ``1 - r`` de Pearson (diferenciable) entre predicción y target.

    Maximizar el Pearson (métrica de portada, regla 5) equivale a minimizar
    ``1 - r``. Usa medias (``ddof=0``), igual que `src.eval.metrics.pearson_r`, así
    la pérdida y la métrica de evaluación coinciden. El r es invariante a
    transformaciones afines de cada variable, de modo que da lo mismo que el target
    venga estandarizado o crudo. Necesita un batch con >=2 muestras y varianza no
    nula en ``pred``; con un batch degenerado (1 muestra o predicción constante) el
    denominador colapsa y el ``eps`` evita el NaN devolviendo ~1.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    cov = (pred_c * target_c).mean()
    denom = torch.sqrt((pred_c ** 2).mean() + eps) * torch.sqrt((target_c ** 2).mean() + eps)
    return 1.0 - cov / (denom + eps)

# Hiperparámetros por defecto: una red chica y regularizada, punto de partida
# razonable para p >> n (una capa oculta angosta, dropout fuerte en la entrada y
# en la capa oculta, weight decay). Se overridean por `model.params` en el YAML.
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
    "loss": "mse",            # "mse" | "pearson" (1 - r) | "pearson_mse" (1 - r + w·MSE)
    "mse_weight": 0.3,        # peso del término MSE cuando loss="pearson_mse"
    "patience": None,         # None => sin early stopping
    "eval_scheme": "temporal",
    "refit_full_train": True,
    "val_metric": "loss",     # "loss" (pérdida inner-val) | "pearson"
    "min_delta": 0.0,
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",          # "cpu" | "cuda" | "auto"
}


def temporal_inner_split(years: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Inner-split temporal dentro del train: años previos vs. último año.

    Devuelve ``(idx_inner_train, idx_inner_val)`` con el último año como inner-val,
    o ``None`` si no hay al menos 2 años distintos (no se puede partir por año).
    """
    if years is None:
        return None
    years = np.asarray(years)
    uniq = sorted(int(y) for y in np.unique(years))
    if len(uniq) < 2:
        return None
    vy = uniq[-1]
    return np.flatnonzero(years < vy), np.flatnonzero(years == vy)


class _MLPNet(nn.Module):
    """Red feed-forward: [Dropout entrada] -> bloques (Linear[, BN], act, Dropout) -> Linear(1)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        *,
        dropout: float,
        input_dropout: float,
        activation: str,
        batchnorm: bool,
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"activation desconocida: {activation!r}. Disponibles: {sorted(_ACTIVATIONS)}."
            )
        layers: list[nn.Module] = []
        if input_dropout and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPRegressor(BaseModel):
    """Perceptrón multicapa (PyTorch) para regresión de un fenotipo.

    `params` (sub-dict `model.params` del config):
        hidden_dims        : lista de anchos de capas ocultas (default [256]).
        dropout            : dropout tras cada capa oculta (default 0.5).
        input_dropout      : dropout sobre la entrada (default 0.2).
        activation         : "relu"|"gelu"|"tanh"|"leaky_relu"|"elu" (default relu).
        batchnorm          : BatchNorm1d tras cada Linear oculta (default False).
        lr                 : learning rate de Adam (default 1e-3).
        weight_decay       : L2 de Adam (default 1e-4).
        batch_size         : tamaño de minibatch; <=0 o >= n => full-batch (default 64).
        max_epochs         : épocas máximas (default 200).
        loss               : "mse" (default) | "pearson" | "pearson_mse". Pérdida que
            minimiza el optimizador; "pearson" minimiza ``1 - r`` (optimiza la portada,
            regla 5); "pearson_mse" suma ``mse_weight·MSE`` para recalibrar la escala.
        mse_weight         : peso del término MSE en "pearson_mse" (default 0.3). Solo
            se usa con esa pérdida; ignorado con "mse"/"pearson".
        patience           : int opcional. Si está, activa early stopping con un
            inner-val temporal (último año del train). Default: sin early stopping.
        eval_scheme        : "temporal" (default), único esquema soportado.
        refit_full_train   : bool (default True). Con early stopping, reentrena el
            modelo final con todo el train externo por las épocas seleccionadas.
        val_metric         : "loss" (MSE inner-val, default) | "pearson".
        min_delta          : mejora mínima para resetear la paciencia (default 0).
        standardize_inputs : z-score de features train-only (default True).
        standardize_target : z-score del target train-only (default True).
        random_state       : seed del RNG de torch para este ajuste (default 575).
        device             : "cpu" | "cuda" | "auto" (default cpu).
    """

    # --- Resolución de hiperparámetros --------------------------------------

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["hidden_dims"] = [int(h) for h in cfg["hidden_dims"]]
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["eval_scheme"] = str(cfg["eval_scheme"]).lower()
        cfg["loss"] = str(cfg["loss"]).lower()
        cfg["mse_weight"] = float(cfg["mse_weight"])
        cfg["val_metric"] = str(cfg["val_metric"]).lower()
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["patience"] = None if cfg["patience"] is None else int(cfg["patience"])
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
    def _validate_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("MLPRegressor.fit recibió y vacío.")
        return y

    # --- Estandarización train-only -----------------------------------------

    def _compute_scalers(self, X: np.ndarray, y: np.ndarray, cfg: dict[str, Any]):
        """Estadísticos de escala con datos de TRAIN (regla 1)."""
        if cfg["standardize_inputs"]:
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0)
            x_std = np.where(x_std == 0.0, 1.0, x_std)  # SNP constante en train
        else:
            x_mean = np.zeros(X.shape[1])
            x_std = np.ones(X.shape[1])
        if cfg["standardize_target"]:
            y_mean = float(y.mean())
            y_std = float(y.std())
            y_std = y_std if y_std > 0 else 1.0
        else:
            y_mean, y_std = 0.0, 1.0
        return (
            np.asarray(x_mean, dtype=np.float64),
            np.asarray(x_std, dtype=np.float64),
            float(y_mean),
            float(y_std),
        )

    @staticmethod
    def _apply_x(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - x_mean) / x_std

    def _monitor_val(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Val externo del fold (de `set_fit_context`) para MONITOREAR la curva por época.

        Lo estandariza con los scalers de train ya ajustados (``x_mean_``/``y_mean_`` ...) y lo
        devuelve como ``(Xv_std, yv_std)``, o ``None`` si no se pasó o no matchea. Es solo para
        registrar ``val_loss``/``val_pearson`` por época: se usa con ``patience=None`` ⇒ no corta
        ni restaura ⇒ no selecciona nada (sin leakage). Requiere los scalers ya calculados.
        """
        Xe = self._fit_context.get("eval_X")
        ye = self._fit_context.get("eval_y")
        if Xe is None or ye is None:
            return None
        Xe = np.asarray(Xe, dtype=np.float64)
        ye = np.asarray(ye, dtype=np.float64).ravel()
        if Xe.ndim != 2 or Xe.shape[0] == 0 or Xe.shape[1] != self.n_features_in_ or ye.size != Xe.shape[0]:
            return None
        return self._apply_x(Xe, self.x_mean_, self.x_std_), (ye - self.y_mean_) / self.y_std_

    # --- Loop de entrenamiento ----------------------------------------------

    def _build_net(self, in_dim: int, cfg: dict[str, Any]) -> _MLPNet:
        return _MLPNet(
            in_dim,
            cfg["hidden_dims"],
            dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"],
            activation=cfg["activation"],
            batchnorm=cfg["batchnorm"],
        )

    def _train(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray | None,
        yv: np.ndarray | None,
        *,
        cfg: dict[str, Any],
        max_epochs: int,
        patience: int | None = None,
        restore_best: bool = False,
        device: "torch.device",
    ):
        """Entrena la red. Devuelve (net, history, best_epoch, best_val).

        Con ``Xv`` se trackea inner-val por época (loss según ``cfg.loss`` + pearson). Si
        ``patience`` está y la métrica de val no mejora ``patience`` épocas
        seguidas, corta. Con ``restore_best`` deja los pesos de la mejor época.
        """
        net = self._build_net(Xs.shape[1], cfg).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        if cfg["loss"] == "mse":
            loss_fn = nn.MSELoss()
        elif cfg["loss"] == "pearson":
            loss_fn = pearson_loss
        elif cfg["loss"] == "pearson_mse":
            # 1 - r + w·MSE: el término Pearson manda el ranking (portada), el MSE
            # ancla la escala (sobre el target ya estandarizado) ⇒ RMSE/R² vuelven a
            # ser informativos. Reusa pearson_loss y nn.MSELoss.
            _mse = nn.MSELoss()
            _w = cfg["mse_weight"]

            def loss_fn(pred, target):
                return pearson_loss(pred, target) + _w * _mse(pred, target)
        else:
            raise ValueError(
                f"loss desconocida: {cfg['loss']!r}. Disponibles: 'mse', 'pearson', 'pearson_mse'."
            )
        # La pérdida Pearson necesita >=2 muestras por batch (un batch de 1 da r
        # indefinido), igual que BatchNorm: en ambos casos se descarta la cola de 1.
        # pearson_mse incluye el término Pearson ⇒ mismo cuidado.
        drop_singleton = cfg["batchnorm"] or cfg["loss"] in ("pearson", "pearson_mse")

        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        yt = torch.as_tensor(ys, dtype=torch.float32, device=device)
        has_val = Xv is not None and yv is not None
        if has_val:
            Xvt = torch.as_tensor(Xv, dtype=torch.float32, device=device)
            yv_np = np.asarray(yv, dtype=np.float64).ravel()

        n = Xt.shape[0]
        bs = cfg["batch_size"]
        full_batch = bs <= 0 or bs >= n

        history: list[dict[str, float]] = []
        best_score = np.inf      # se minimiza (val_metric="pearson" => -pearson)
        best_epoch = -1
        best_val: dict[str, float] = {}
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0

        for epoch in range(int(max_epochs)):
            net.train()
            if full_batch:
                batches = [torch.arange(n, device=device)]
            else:
                perm = torch.randperm(n, device=device)
                batches = list(torch.split(perm, bs))
                # BatchNorm y la pérdida Pearson rompen con un batch de tamaño 1.
                if drop_singleton and batches and batches[-1].numel() == 1:
                    batches = batches[:-1]
            run_loss = 0.0
            seen = 0
            for idx in batches:
                optim.zero_grad()
                pred = net(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                optim.step()
                run_loss += float(loss.item()) * idx.numel()
                seen += int(idx.numel())
            rec: dict[str, float] = {"epoch": epoch, "train_loss": run_loss / max(seen, 1)}

            if has_val:
                net.eval()
                with torch.no_grad():
                    pv = net(Xvt)
                    v_loss = float(loss_fn(pv, torch.as_tensor(yv, dtype=torch.float32, device=device)).item())
                v_pearson = pearson_r(yv_np, pv.detach().cpu().numpy())
                rec["val_loss"] = v_loss
                rec["val_pearson"] = v_pearson
                if cfg["val_metric"] == "pearson":
                    score = -v_pearson if np.isfinite(v_pearson) else np.inf
                else:
                    score = v_loss
                if score < best_score - cfg["min_delta"]:
                    best_score = score
                    best_epoch = epoch
                    best_val = {"val_loss": v_loss, "val_pearson": v_pearson}
                    best_state = copy.deepcopy(net.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                history.append(rec)
                if patience is not None and no_improve >= patience:
                    break
            else:
                history.append(rec)

        if restore_best and best_state is not None:
            net.load_state_dict(best_state)
        return net, history, best_epoch, best_val

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        cfg = self._resolve_cfg()
        self.cfg_ = cfg
        device = self._resolve_device(cfg["device"])
        X = np.asarray(X, dtype=np.float64)
        y = self._validate_y(y)
        self.n_features_in_ = int(X.shape[1])

        if cfg["patience"] is not None and cfg["eval_scheme"] != "temporal":
            raise ValueError(
                "MLPRegressor solo soporta eval_scheme='temporal' cuando patience "
                "está activo (early stopping con inner-val temporal)."
            )
        split = (
            temporal_inner_split(self._fit_context.get("year"))
            if cfg["patience"] is not None
            else None
        )

        # --- Sin early stopping: entrena max_epochs sobre todo el train ------
        if split is None:
            self._seed(cfg["random_state"])
            self.x_mean_, self.x_std_, self.y_mean_, self.y_std_ = self._compute_scalers(X, y, cfg)
            Xs = self._apply_x(X, self.x_mean_, self.x_std_)
            ys = (y - self.y_mean_) / self.y_std_
            # Val del fold (si run_evaluation lo pasó) SOLO para la curva por época: patience=None
            # y restore_best=False ⇒ no corta ni restaura ⇒ la red final es idéntica (sin leakage).
            mon = self._monitor_val()
            Xv, yv = mon if mon is not None else (None, None)
            net, history, _, _ = self._train(
                Xs, ys, Xv, yv, cfg=cfg, max_epochs=cfg["max_epochs"], device=device
            )
            self.net_ = net
            self.history_ = history
            self.early_stopping_used_ = False
            self.selected_epochs_ = int(cfg["max_epochs"])
            self.best_epoch_ = None
            self.best_val_ = {}
            self.refit_full_train_ = False
            self._finalize(device)
            return self

        # --- Early stopping temporal: selecciona épocas con inner-val --------
        tr, va = split
        # Estandarización SOLO con el inner-train (sin leakage en la selección).
        xm, xs, ym, ys_std = self._compute_scalers(X[tr], y[tr], cfg)
        Xtr = self._apply_x(X[tr], xm, xs)
        ytr = (y[tr] - ym) / ys_std
        Xva = self._apply_x(X[va], xm, xs)
        yva = (y[va] - ym) / ys_std

        self._seed(cfg["random_state"])
        sel_net, history, best_epoch, best_val = self._train(
            Xtr, ytr, Xva, yva,
            cfg=cfg,
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            restore_best=True,
            device=device,
        )
        self.history_ = history
        self.early_stopping_used_ = True
        self.best_epoch_ = int(best_epoch) if best_epoch >= 0 else None
        self.best_val_ = best_val
        selected = max(1, (best_epoch + 1) if best_epoch >= 0 else cfg["max_epochs"])
        self.selected_epochs_ = int(selected)
        self.refit_full_train_ = bool(cfg["refit_full_train"])

        if cfg["refit_full_train"]:
            # Modelo final: todo el train externo, escalado re-ajustado a todo el
            # train, por las épocas elegidas con el split temporal interno.
            self._seed(cfg["random_state"])
            self.x_mean_, self.x_std_, self.y_mean_, self.y_std_ = self._compute_scalers(X, y, cfg)
            Xs = self._apply_x(X, self.x_mean_, self.x_std_)
            ys = (y - self.y_mean_) / self.y_std_
            net, _, _, _ = self._train(
                Xs, ys, None, None, cfg=cfg, max_epochs=selected, device=device
            )
            self.net_ = net
        else:
            # Se conserva la red de selección (entrenada solo con el inner-train)
            # con sus estadísticos de inner-train.
            self.x_mean_, self.x_std_, self.y_mean_, self.y_std_ = xm, xs, ym, ys_std
            self.net_ = sel_net
        self._finalize(device)
        return self

    def _finalize(self, device: "torch.device") -> None:
        self.net_.eval()
        self.device_ = str(device)
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "net_"):
            raise RuntimeError("MLPRegressor.predict llamado antes de fit.")
        device = next(self.net_.parameters()).device
        Xs = self._apply_x(X, self.x_mean_, self.x_std_)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.as_tensor(Xs, dtype=torch.float32, device=device))
        y_hat = out.detach().cpu().numpy().ravel()
        return (y_hat * self.y_std_ + self.y_mean_).astype(float)

    def get_fitted_params(self) -> dict[str, Any]:
        """Diagnóstico del ajuste (para logging/bitácora)."""
        if not hasattr(self, "net_"):
            return {}
        out: dict[str, Any] = {
            "selected_epochs": getattr(self, "selected_epochs_", None),
            "early_stopping_used": int(getattr(self, "early_stopping_used_", False)),
            "refit_full_train": int(getattr(self, "refit_full_train_", False)),
            "n_params": getattr(self, "n_params_", None),
            "hidden_dims": str(self.cfg_["hidden_dims"]),  # no-numérico => metadata
        }
        if getattr(self, "best_epoch_", None) is not None:
            out["best_epoch"] = self.best_epoch_
        if getattr(self, "best_val_", None):
            if np.isfinite(self.best_val_.get("val_loss", np.nan)):
                out["best_val_loss"] = float(self.best_val_["val_loss"])
            if np.isfinite(self.best_val_.get("val_pearson", np.nan)):
                out["best_val_pearson"] = float(self.best_val_["val_pearson"])
        return out

    # --- Persistencia (override torch: state_dict + arquitectura) -----------

    def save(self, path: str | Path) -> Path:
        if not hasattr(self, "net_"):
            raise RuntimeError("MLPRegressor.save llamado antes de fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "config": self.config,
            "cfg_": self.cfg_,
            "state_dict": self.net_.state_dict(),
            "n_features_in_": self.n_features_in_,
            "x_mean_": np.asarray(self.x_mean_),
            "x_std_": np.asarray(self.x_std_),
            "y_mean_": float(self.y_mean_),
            "y_std_": float(self.y_std_),
            "selected_epochs_": getattr(self, "selected_epochs_", None),
            "early_stopping_used_": getattr(self, "early_stopping_used_", False),
            "refit_full_train_": getattr(self, "refit_full_train_", False),
            "best_epoch_": getattr(self, "best_epoch_", None),
            "best_val_": getattr(self, "best_val_", {}),
            "n_params_": getattr(self, "n_params_", None),
            "device_": getattr(self, "device_", "cpu"),
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLPRegressor":
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
        obj.selected_epochs_ = ckpt.get("selected_epochs_")
        obj.early_stopping_used_ = ckpt.get("early_stopping_used_", False)
        obj.refit_full_train_ = ckpt.get("refit_full_train_", False)
        obj.best_epoch_ = ckpt.get("best_epoch_")
        obj.best_val_ = ckpt.get("best_val_", {})
        obj.n_params_ = ckpt.get("n_params_")
        obj.device_ = ckpt.get("device_", "cpu")
        obj.history_ = []
        return obj
