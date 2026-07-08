"""Autoencoder de SNPs binarios: compresor no lineal con pérdida BCE.

SNPAutoencoder es un MLP encoder/decoder (con tied weights opcional en
_TiedDecoder). La salida son logits y la pérdida es BCEWithLogitsLoss, que funde la
sigmoide con la BCE de forma estable (los SNPs son binarios). El latente es lineal
para no saturar el embedding que consume el modelo downstream (GBLUP/XGBoost).

AutoencoderReducer expone la misma interfaz que PCAReducer
(fit/transform/fit_transform/get_fitted_params), así entra en la cadena preprocess:.
fit ajusta el AE con el train del fold y transform proyecta val/test por el encoder
congelado. Entrena con minibatches, early stopping por BCE de validación, un time
budget (max_train_seconds) y callback de curvas por época. Serializa por torch
(state_dict + arquitectura) y es picklable dentro de un PreprocessedModel.

Los hiperparámetros (profundidad/ancho del encoder, latente, activación, dropout,
batchnorm, tied weights, denoising, sparsity L1 del latente, optimizador, lr, weight
decay, batch size, épocas, paciencia, grad clip, scheduler) son barribles por sweep.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LogFn = Callable[[dict[str, Any]], None]

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation={name!r} no soportada. Use una de {tuple(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]()


def _mlp(dims: list[int], activation: str, dropout: float, batchnorm: bool) -> nn.Sequential:
    """MLP nn.Sequential: por capa oculta Linear->[BN]->act->[Dropout]; la última
    capa queda lineal, así sirve para el encoder (a latente) y para el decoder (a logits)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class _TiedDecoder(nn.Module):
    """Decoder con pesos atados a los del encoder (W_dec = W_enc.T; regulariza).

    Reusa las matrices del encoder, así que los gradientes fluyen a sus pesos. Mantiene
    bias y capas post (BN/act/dropout) propios. Encapsula el weight tying para que el
    caso común (sin tying) queden dos nn.Sequential limpios.
    """

    def __init__(self, encoder: nn.Sequential, activation: str, dropout: float, batchnorm: bool) -> None:
        super().__init__()
        self._enc = [m for m in encoder if isinstance(m, nn.Linear)]
        self.biases = nn.ParameterList()
        self.posts = nn.ModuleList()
        L = len(self._enc)
        for i in range(L):
            src = self._enc[L - 1 - i]
            self.biases.append(nn.Parameter(torch.zeros(src.in_features)))
            if i < L - 1:
                post: list[nn.Module] = []
                if batchnorm:
                    post.append(nn.BatchNorm1d(src.in_features))
                post.append(_activation(activation))
                if dropout > 0:
                    post.append(nn.Dropout(float(dropout)))
                self.posts.append(nn.Sequential(*post))
            else:
                self.posts.append(nn.Identity())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        L = len(self._enc)
        for i in range(L):
            src = self._enc[L - 1 - i]
            h = F.linear(h, src.weight.t(), self.biases[i])
            h = self.posts[i](h)
        return h


class SNPAutoencoder(nn.Module):
    """MLP encoder/decoder para SNPs. forward devuelve (logits, z)."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        dims = [int(input_dim), *[int(h) for h in (encoder_hidden or [])], int(latent_dim)]
        self.encoder = _mlp(dims, activation, dropout, batchnorm)
        if tie_weights:
            self.decoder: nn.Module = _TiedDecoder(self.encoder, activation, dropout, batchnorm)
        else:
            self.decoder = _mlp(list(reversed(dims)), activation, dropout, batchnorm)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class _SNPDataset(Dataset):
    """Genotipos binarios para reconstrucción (no supervisado: el target es el input)."""

    def __init__(self, X: np.ndarray) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.X[idx], dtype=torch.float32)




_REDUCER_DEFAULTS: dict[str, Any] = {
    "latent_dim": 64,
    "encoder_hidden": [2048, 512],
    "activation": "relu",
    "dropout": 0.0,
    "batchnorm": False,
    "tie_weights": False,
    "denoising_p": 0.0,
    "l1_latent": 0.0,
    "weight_decay": 0.0,
    "optimizer": "adamw",
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 200,
    "patience": 20,
    "min_delta": 1e-4,
    "grad_clip": 0.0,
    "lr_scheduler": "none",
    "val_frac": 0.1,
    "maf_threshold": 0.0,
    "device": None,
    "seed": 575,
    "max_train_seconds": None,
    "verbose": False,
}


class AutoencoderReducer:
    """Autoencoder como paso de reducción de dimensionalidad, train-only.

    Misma interfaz que PCAReducer para encajar en la cadena preprocess:. La entrada
    deben ser SNPs binarios {0,1}; el embedding de salida es continuo. El val de early
    stopping sale, por prioridad: (1) val externo pasado por set_fit_context; (2)
    held-out por línea del train (val_frac); (3) sin val, sin early stopping.

    El callback log_fn (set_log_fn) recibe por época epoch, ae/train_bce, ae/val_bce y
    ae/val_recon_acc para wandb.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, **overrides: Any) -> None:
        params = {**_REDUCER_DEFAULTS, **(cfg or {}), **overrides}
        unknown = set(params) - set(_REDUCER_DEFAULTS)
        if unknown:
            raise ValueError(f"Hiperparámetros desconocidos del AE: {sorted(unknown)}.")
        self.params = params
        self._fit_context: dict[str, Any] = {}
        self._log_fn: LogFn | None = None


    def set_fit_context(self, **context: Any) -> None:
        """Metadata del fold: line_id/year del train y, opcional, un val
        externo de reconstrucción (X_val, line_id_val)."""
        self._fit_context = context

    def set_log_fn(self, log_fn: LogFn | None) -> None:
        self._log_fn = log_fn


    def _resolve_device(self) -> torch.device:
        dev = self.params.get("device")
        if dev is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev)

    @staticmethod
    def _check_binary(X: np.ndarray) -> None:
        u = np.unique(X[: min(len(X), 256)])
        if not np.all(np.isin(u, (0.0, 1.0))):
            raise ValueError(
                "AutoencoderReducer espera SNPs binarios {0,1} (regla 4: BCE). "
                f"Encontré valores {u[:6]}... ¿pusiste un standardize ANTES del AE? "
                "El z-score va DESPUÉS (lo hace GBLUP internamente), no antes."
            )

    def _maybe_maf(self, fit: bool, X: np.ndarray) -> np.ndarray:
        thr = float(self.params.get("maf_threshold", 0.0) or 0.0)
        if thr <= 0:
            return X
        from src.data.preprocess import SNPPreprocessor

        if fit:
            self._maf = SNPPreprocessor(maf_threshold=thr, standardize=False)
            return self._maf.fit_transform(X)
        return self._maf.transform(X)

    def _split_train_val(self, X: np.ndarray):
        """Devuelve (X_tr, X_val_or_None). Val externo > held-out por línea."""
        ctx = self._fit_context
        if ctx.get("X_val") is not None:
            X_val = self._maybe_maf(False, np.asarray(ctx["X_val"], dtype=np.float32))
            return X, X_val

        val_frac = float(self.params.get("val_frac", 0.0) or 0.0)
        if val_frac <= 0:
            return X, None

        rng = np.random.default_rng(int(self.params.get("seed", 575)))
        line_id = ctx.get("line_id")
        n = X.shape[0]
        if line_id is not None and len(line_id) == n:
            groups = np.asarray(line_id)
            uniq = np.unique(groups)
            n_val = max(1, int(round(len(uniq) * val_frac)))
            val_lines = set(rng.choice(uniq, size=n_val, replace=False).tolist())
            is_val = np.array([g in val_lines for g in groups])
        else:
            idx = rng.permutation(n)
            n_val = max(1, int(round(n * val_frac)))
            is_val = np.zeros(n, bool)
            is_val[idx[:n_val]] = True
        if is_val.all() or not is_val.any():
            return X, None
        return X[~is_val], X[is_val]

    def _make_loaders(self, X_tr: np.ndarray, X_val: np.ndarray | None, gen: torch.Generator):
        bs = int(self.params["batch_size"])
        train_loader = DataLoader(
            _SNPDataset(X_tr), batch_size=bs, shuffle=True,
            drop_last=bool(self.params["batchnorm"]), generator=gen,
        )
        val_loader = (
            DataLoader(_SNPDataset(X_val), batch_size=bs, shuffle=False)
            if X_val is not None else None
        )
        return train_loader, val_loader


    def _build_optimizer(self, model: nn.Module):
        name = str(self.params.get("optimizer", "adamw")).lower()
        lr = float(self.params["lr"])
        wd = float(self.params.get("weight_decay", 0.0) or 0.0)
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"optimizer={name!r} no soportado (adam|adamw|sgd).")

    def _build_scheduler(self, opt):
        name = str(self.params.get("lr_scheduler", "none")).lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(self.params["max_epochs"]))
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
        raise ValueError(f"lr_scheduler={name!r} no soportado (none|cosine|plateau).")


    def _train_epoch(self, model, loader, optimizer, criterion, device, mask_gen) -> float:
        model.train()
        denoise_p = float(self.params.get("denoising_p", 0.0) or 0.0)
        l1 = float(self.params.get("l1_latent", 0.0) or 0.0)
        grad_clip = float(self.params.get("grad_clip", 0.0) or 0.0)
        total, seen = 0.0, 0
        for xb in loader:
            xb = xb.to(device)
            xin = xb
            if denoise_p > 0:
                mask = (torch.rand(xb.shape, generator=mask_gen, device="cpu") < denoise_p).to(device)
                xin = xb * (~mask).to(xb.dtype)
            optimizer.zero_grad(set_to_none=True)
            logits, z = model(xin)
            loss = criterion(logits, xb)
            if l1 > 0:
                loss = loss + l1 * z.abs().mean()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)
        return total / max(1, seen)

    @torch.no_grad()
    def _eval_epoch(self, model, loader, criterion, device) -> tuple[float, float]:
        model.eval()
        total, seen, correct, bits = 0.0, 0, 0, 0
        for xb in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            total += float(criterion(logits, xb)) * xb.size(0)
            seen += xb.size(0)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += int((pred == xb).sum())
            bits += xb.numel()
        return total / max(1, seen), correct / max(1, bits)


    def fit(self, X: np.ndarray, y: Any = None) -> "AutoencoderReducer":
        X = np.asarray(X, dtype=np.float32)
        self._check_binary(X)
        X = self._maybe_maf(True, X)
        X_tr, X_val = self._split_train_val(X)

        device = self._resolve_device()
        seed = int(self.params.get("seed", 575))
        gen = torch.Generator().manual_seed(seed)
        mask_gen = torch.Generator().manual_seed(seed + 1)

        self.input_dim_ = int(X_tr.shape[1])
        model = SNPAutoencoder(
            input_dim=self.input_dim_,
            latent_dim=int(self.params["latent_dim"]),
            encoder_hidden=self.params["encoder_hidden"],
            activation=self.params["activation"],
            dropout=float(self.params["dropout"]),
            batchnorm=bool(self.params["batchnorm"]),
            tie_weights=bool(self.params["tie_weights"]),
        ).to(device)
        self.model_ = model

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.BCEWithLogitsLoss()
        train_loader, val_loader = self._make_loaders(X_tr, X_val, gen)
        eval_loader = val_loader if val_loader is not None else DataLoader(
            _SNPDataset(X_tr), batch_size=int(self.params["batch_size"]), shuffle=False
        )

        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])
        min_delta = float(self.params["min_delta"])
        budget = self.params.get("max_train_seconds")
        budget = float(budget) if budget is not None else None

        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch, epochs_no_improve = -1, 0
        history: list[dict[str, float]] = []
        t0 = time.time()
        stopped_by = "max_epochs"

        for epoch in range(max_epochs):
            train_bce = self._train_epoch(model, train_loader, optimizer, criterion, device, mask_gen)
            val_bce, val_acc = self._eval_epoch(model, eval_loader, criterion, device)
            if scheduler is not None:
                scheduler.step(val_bce) if self.params["lr_scheduler"] == "plateau" else scheduler.step()

            rec = {
                "epoch": epoch,
                "ae/train_bce": train_bce,
                "ae/val_bce": val_bce,
                "ae/val_recon_acc": val_acc,
                "ae/lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(rec)
            if self._log_fn is not None:
                self._log_fn(rec)
            if self.params.get("verbose"):
                print(f"  [ae] epoch {epoch:3d}  train_bce={train_bce:.5f}  val_bce={val_bce:.5f}  acc={val_acc:.4f}")

            if val_bce < best_val - min_delta:
                best_val, best_epoch = val_bce, epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    stopped_by = "early_stopping"
                    break

            if budget is not None and (time.time() - t0) > budget:
                stopped_by = "time_budget"
                break

        model.load_state_dict(best_state)
        _, final_val_acc = self._eval_epoch(model, eval_loader, criterion, device)
        model.eval().to("cpu")

        self.history_ = history
        self.fitted_ = {
            "ae_latent_dim": int(self.params["latent_dim"]),
            "ae_input_dim": self.input_dim_,
            "ae_n_params": int(sum(p.numel() for p in model.parameters())),
            "ae_val_bce_best": float(best_val),
            "ae_val_recon_acc_best": float(final_val_acc),
            "ae_epoch_best": int(best_epoch),
            "ae_epochs_run": int(len(history)),
            "ae_stopped_by": stopped_by,
            "ae_has_val": bool(val_loader is not None),
            "ae_train_seconds": float(time.time() - t0),
        }
        return self


    def _encode_all(self, X: np.ndarray) -> np.ndarray:
        device = self._resolve_device()
        self.model_.to(device).eval()
        loader = DataLoader(_SNPDataset(X), batch_size=int(self.params["batch_size"]), shuffle=False)
        out = []
        with torch.no_grad():
            for xb in loader:
                out.append(self.model_.encode(xb.to(device)).cpu().numpy())
        self.model_.to("cpu")
        if not out:
            return np.empty((0, int(self.params["latent_dim"])), np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("AutoencoderReducer.transform llamado antes de fit.")
        X = self._maybe_maf(False, np.asarray(X, dtype=np.float32))
        return self._encode_all(X)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstrucción (probabilidades sigmoide por SNP). Para inspección/diagnóstico."""
        if not hasattr(self, "model_"):
            raise RuntimeError("AutoencoderReducer.reconstruct llamado antes de fit.")
        X = self._maybe_maf(False, np.asarray(X, dtype=np.float32))
        device = self._resolve_device()
        self.model_.to(device).eval()
        loader = DataLoader(_SNPDataset(X), batch_size=int(self.params["batch_size"]), shuffle=False)
        out = []
        with torch.no_grad():
            for xb in loader:
                logits, _ = self.model_(xb.to(device))
                out.append(torch.sigmoid(logits).cpu().numpy())
        self.model_.to("cpu")
        return np.concatenate(out, axis=0).astype(np.float32)

    def get_fitted_params(self) -> dict[str, Any]:
        return dict(getattr(self, "fitted_", {}))


    def _arch(self) -> dict[str, Any]:
        keys = ("latent_dim", "encoder_hidden", "activation", "dropout", "batchnorm", "tie_weights")
        return {k: self.params[k] for k in keys}

    def save(self, path) -> "Any":
        """Guarda arquitectura + state_dict + métricas en un único .pt (torch)."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ae_reducer_v1",
            "params": self.params,
            "arch": self._arch(),
            "input_dim": int(getattr(self, "input_dim_", 0)),
            "state_dict": self.model_.state_dict(),
            "fitted": getattr(self, "fitted_", {}),
            "maf": getattr(self, "_maf", None),
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path) -> "AutoencoderReducer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(payload["params"])
        obj.input_dim_ = int(payload["input_dim"])
        obj.model_ = SNPAutoencoder(input_dim=obj.input_dim_, **payload["arch"])
        obj.model_.load_state_dict(payload["state_dict"])
        obj.model_.eval()
        obj.fitted_ = payload.get("fitted", {})
        if payload.get("maf") is not None:
            obj._maf = payload["maf"]
        return obj


    def __getstate__(self) -> dict[str, Any]:
        state = {"params": self.params, "fitted": getattr(self, "fitted_", {})}
        if hasattr(self, "model_"):
            state["arch"] = self._arch()
            state["input_dim"] = int(self.input_dim_)
            state["state_dict"] = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            state["maf"] = getattr(self, "_maf", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.params = state["params"]
        self._fit_context = {}
        self._log_fn = None
        self.fitted_ = state.get("fitted", {})
        if "state_dict" in state:
            self.input_dim_ = int(state["input_dim"])
            self.model_ = SNPAutoencoder(input_dim=self.input_dim_, **state["arch"])
            self.model_.load_state_dict(state["state_dict"])
            self.model_.eval()
            if state.get("maf") is not None:
                self._maf = state["maf"]
