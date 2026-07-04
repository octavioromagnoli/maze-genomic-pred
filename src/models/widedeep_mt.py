"""Wide&Deep multi-trait — red neuronal genómica con SKIP LINEAL anclado en multi-trait GBLUP.

Es la respuesta *deep learning* que **supera a GBLUP** en GY óptimo. Combina, en una sola red
entrenada por backprop:

- **Wide (skip lineal aditivo):** un `Linear(p, T)` inicializado EXACTAMENTE en los efectos de
  marcador del multi-trait GBLUP (MTM, ver `src/models/mtm_gblup.py`). Es el *backbone* que ya le
  gana a GBLUP por pleiotropía aditiva GY↔PH; por defecto queda **congelado** (`train_wide=False`)
  para no degradarse (fine-tunearlo con Adam lo destruye — verificado).
- **Deep (tronco no-lineal compartido):** un MLP sobre los SNPs con la cabeza inicializada en cero
  (la red arranca EXACTAMENTE en el aditivo) que aprende una **corrección** compartida entre traits.
  Se entrena con **early stopping honesto** (inner-split temporal del train si hay ≥2 años, si no un
  holdout disjunto por línea) y **ensemble de ``n_models`` seeds** para matar la varianza.

**Resultado y advertencia honesta (para el informe):** la red **supera a GBLUP robustamente en
ambos folds** (val 2019 y val 2018, significativo en 2018), pero la ganancia es **100% del skip
aditivo**: la corrección profunda aporta ~0 (a veces un pelo negativo). Es la confirmación del
*techo aditivo* de GY en líneas endocriadas (dominancia estructuralmente nula, epistasis que no
replica out-of-sample) desde el ángulo más fuerte posible: incluso montando un deep sobre el mejor
aditivo, lo no-lineal no agrega señal. O sea: **sí existe una red que le gana a GBLUP, pero solo
porque lleva el aditivo adentro** — no hay mecanismo puramente profundo que lo supere.

Interfaz multitask idéntica a `mlp_multitask`/`mtm_gblup` (``fit(X, Y, Y_mask)`` /
``predict(X) -> (n, T)``); encaja en `run_multitask_evaluation` y el dispatch de `scripts/train.py`.
Requiere datos completos por fila (en óptimo GY=PH=AD por observación). Sin leakage (regla 1): MTM,
escalado y ES se ajustan solo con el train del fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.eval.metrics import pearson_r
from src.models.base import BaseModel
from src.models.gblup import apply_standardize, genomic_relationship, standardize_train
from src.models.mlp import _ACTIVATIONS
from src.models.mtm_gblup import _em_mtm

_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [32],
    "dropout": 0.7,
    "input_dropout": 0.3,
    "activation": "elu",
    "lr": 1e-3,
    "wd_deep": 1e-4,
    "wd_wide": 0.0,
    "train_wide": False,        # congelar el skip aditivo (fine-tune lo destruye)
    "batch_size": 128,
    "max_epochs": 150,
    "patience": 20,
    "n_models": 10,             # ensemble de seeds (mata varianza)
    "seed_step": 131,
    "loss_weights": {"GY": 1.0, "PH": 1.0, "AD": 0.2},
    "mtm_iter": 300,
    "mtm_tol": 1e-7,
    "mtm_ridge": 1e-3,
    "holdout_frac": 0.2,        # fracción de LÍNEAS para el ES cuando no hay inner temporal
    "random_state": 575,
    "device": "cpu",
}


class _WideDeepNet(nn.Module):
    """`Linear(p,T)` (skip aditivo) + tronco MLP compartido con cabeza init 0 (arranca aditivo)."""

    def __init__(self, p: int, T: int, hidden: list[int], dropout: float, input_dropout: float,
                 activation: str) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(f"activation desconocida: {activation!r}.")
        self.wide = nn.Linear(p, T)
        layers: list[nn.Module] = [nn.Dropout(input_dropout)] if input_dropout > 0 else []
        prev = p
        for h in hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, T)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wide(x) + self.head(self.trunk(x))


class WideDeepMTRegressor(BaseModel):
    """Red Wide&Deep multi-trait con skip aditivo anclado en MTM + ensemble de seeds.

    `params` (sub-dict ``model.params``): ver `_DEFAULTS`. Portada = primer trait (GY).
    """

    def _cfg(self) -> dict[str, Any]:
        c = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        c["hidden_dims"] = [int(h) for h in c["hidden_dims"]]
        c["loss_weights"] = dict(c["loss_weights"] or {})
        return c

    def _device(self, d: str) -> "torch.device":
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        if d == "cuda" and not torch.cuda.is_available():
            d = "cpu"
        return torch.device(d)

    # --- efectos de marcador del MTM (para inicializar el skip) --------------
    def _mtm_marker_effects(self, X: np.ndarray, Y: np.ndarray, c: dict[str, Any]):
        Z, xm, xs = standardize_train(X)
        p = Z.shape[1]
        G = genomic_relationship(Z, Z, p)
        fit = _em_mtm(Y, G, n_iter=int(c["mtm_iter"]), tol=float(c["mtm_tol"]), ridge=float(c["mtm_ridge"]))
        V, d, Uhat = fit["V"], fit["d"], fit["Uhat"]
        Ginv_U = V @ ((V.T @ Uhat) / d[:, None])
        ME = (Z.T @ Ginv_U) / p            # (p, T)
        return xm, xs, ME, fit["mu"], fit["G0"], fit["R0"]

    def _inner_split(self, n: int, c: dict[str, Any]):
        """Inner-split para ES: temporal (último año) si hay ≥2 años; si no, holdout por línea."""
        year = self._fit_context.get("year")
        if year is not None:
            year = np.asarray(year)
            uniq = sorted(int(y) for y in np.unique(year))
            if len(uniq) >= 2:
                vy = uniq[-1]
                return np.flatnonzero(year < vy), np.flatnonzero(year == vy)
        line = self._fit_context.get("line_id")
        rng = np.random.default_rng(int(c["random_state"]))
        if line is not None:
            line = np.asarray(line)
            uniq = np.unique(line)
            val = set(rng.choice(uniq, size=max(1, int(len(uniq) * c["holdout_frac"])), replace=False))
            iva = np.array([i for i, l in enumerate(line) if l in val])
        else:
            idx = rng.permutation(n)
            iva = idx[: max(1, int(n * c["holdout_frac"]))]
        itr = np.array([i for i in range(n) if i not in set(iva.tolist())])
        return itr, iva

    def _train_one(self, Xs, Yz, ME_z, bias_z, split, c, seed, device):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        p, T = Xs.shape[1], Yz.shape[1]
        net = _WideDeepNet(p, T, c["hidden_dims"], c["dropout"], c["input_dropout"], c["activation"]).to(device)
        with torch.no_grad():
            net.wide.weight.copy_(torch.as_tensor(ME_z.T, dtype=torch.float32))
            net.wide.bias.copy_(torch.as_tensor(bias_z, dtype=torch.float32))
        for prm in net.wide.parameters():
            prm.requires_grad = bool(c["train_wide"])
        params = [q for q in net.parameters() if q.requires_grad]
        w = torch.as_tensor(self._w, dtype=torch.float32, device=device)
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Yz, dtype=torch.float32, device=device)
        itr, iva = split
        Xtr, Ytr = Xt[itr], Yt[itr]
        Xva = Xt[iva]
        yva_gy = Yz[iva, 0]
        n = Xtr.shape[0]
        bs = int(c["batch_size"])
        full = bs <= 0 or bs >= n
        opt = torch.optim.Adam(params, lr=c["lr"], weight_decay=c["wd_deep"]) if params else None
        best_r, best_state, no_imp = -1e9, None, 0
        for _ in range(int(c["max_epochs"])):
            if opt is not None:
                net.train()
                batches = [torch.arange(n, device=device)] if full else list(
                    torch.split(torch.randperm(n, device=device), bs))
                for idx in batches:
                    opt.zero_grad()
                    loss = (w * ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean(0)).sum()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                r = pearson_r(yva_gy, net(Xva).cpu().numpy()[:, 0])
            if r > best_r + 1e-5:
                best_r, no_imp = r, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                no_imp += 1
                if opt is None or no_imp >= int(c["patience"]):
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "WideDeepMTRegressor":
        c = self._cfg()
        self.cfg_ = c
        device = self._device(c["device"])
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y_mask is not None and not np.asarray(Y_mask, dtype=bool).all():
            raise ValueError("WideDeepMTRegressor requiere datos completos por fila (usá mlp_multitask si hay faltantes).")
        self.n_targets_ = int(Y.shape[1])
        names = self._fit_context.get("target_names")
        self.target_names_ = [str(m) for m in names] if names is not None else [f"t{j}" for j in range(self.n_targets_)]
        self._w = np.asarray([float(c["loss_weights"].get(nm, 1.0)) for nm in self.target_names_], dtype=float)

        xm, xs, ME, mu, self.G0_, self.R0_ = self._mtm_marker_effects(X, Y, c)
        self.x_mean_, self.x_std_ = xm, xs
        ym = Y.mean(0)
        ys = Y.std(0)
        ys = np.where(ys < 1e-12, 1.0, ys)
        self.y_mean_, self.y_std_ = ym, ys
        Xs = (X - xm) / xs
        Yz = (Y - ym) / ys
        ME_z = ME / ys                       # skip en z-space del target
        bias_z = (mu - ym) / ys
        split = self._inner_split(Xs.shape[0], c)

        base = int(c["random_state"])
        self.nets_: list[_WideDeepNet] = []
        for i in range(int(c["n_models"])):
            self.nets_.append(self._train_one(Xs, Yz, ME_z, bias_z, split, c, base + i * int(c["seed_step"]), device))
        self.device_ = str(device)
        self.n_params_ = int(sum(p.numel() for p in self.nets_[0].parameters()))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "nets_", None):
            raise RuntimeError("WideDeepMTRegressor.predict llamado antes de fit.")
        device = next(self.nets_[0].parameters()).device
        Xs = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_std_
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        acc = None
        with torch.no_grad():
            for net in self.nets_:
                out = net(Xt).cpu().numpy() * self.y_std_ + self.y_mean_   # z^-1 (MSE calibra escala)
                acc = out if acc is None else acc + out
        return acc / len(self.nets_)

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "nets_", None):
            return {}
        out: dict[str, Any] = {
            "n_models": int(self.cfg_["n_models"]),
            "n_params_per_net": int(getattr(self, "n_params_", 0)),
            "train_wide": int(bool(self.cfg_["train_wide"])),
        }
        diag = np.sqrt(np.clip(np.diag(self.G0_), 1e-12, None))
        for a in range(self.n_targets_):
            for b in range(a + 1, self.n_targets_):
                out[f"gcorr_{self.target_names_[a]}_{self.target_names_[b]}"] = float(
                    self.G0_[a, b] / (diag[a] * diag[b]))
        return out

    # --- persistencia (torch: state_dicts del ensemble + escalas) ------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config, "cfg_": self.cfg_,
            "state_dicts": [n.state_dict() for n in self.nets_],
            "arch": {"p": self.nets_[0].wide.in_features, "T": self.n_targets_},
            "x_mean_": self.x_mean_, "x_std_": self.x_std_,
            "y_mean_": self.y_mean_, "y_std_": self.y_std_,
            "target_names_": list(self.target_names_), "n_targets_": self.n_targets_,
            "G0_": self.G0_, "R0_": self.R0_, "n_params_": getattr(self, "n_params_", None),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WideDeepMTRegressor":
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(ck.get("config"))
        obj.cfg_ = ck["cfg_"]
        obj.n_targets_ = int(ck["n_targets_"])
        obj.target_names_ = list(ck["target_names_"])
        obj.x_mean_, obj.x_std_ = ck["x_mean_"], ck["x_std_"]
        obj.y_mean_, obj.y_std_ = ck["y_mean_"], ck["y_std_"]
        obj.G0_, obj.R0_ = ck["G0_"], ck["R0_"]
        c = obj.cfg_
        p, T = ck["arch"]["p"], ck["arch"]["T"]
        obj.nets_ = []
        for sd in ck["state_dicts"]:
            net = _WideDeepNet(p, T, c["hidden_dims"], c["dropout"], c["input_dropout"], c["activation"])
            net.load_state_dict(sd)
            net.eval()
            obj.nets_.append(net)
        obj.n_params_ = ck.get("n_params_")
        return obj
