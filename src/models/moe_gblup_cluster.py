"""Mixture of Experts *hard* multitask con clustering sobre predicciones GBLUP (o PCA+GBLUP),
ENSAMBLADO sobre seeds re-clusterizando.

Motivación. El MoE hard actual (`moe_multitask`) clusteriza el espacio de SNPs reducido por PCA con
UN GMM de seed fija: el ruteo depende de un único clustering, así que un mal reparto de esa seed
sesga TODO el modelo. Acá se atacan dos cosas a la vez:

1. **Espacio de clustering.** En vez de (o además de) la PCA de los SNPs, se clusteriza el espacio
   de **predicciones GBLUP** por trait — el valor de cría genómico predicho (``[GŶ, PĤ, AD̂]``). Es
   un espacio de baja dimensión, supervisado y aditivo, donde cada cluster agrupa líneas de perfil
   fenotípico predicho parecido. Modos (``gating.mode``):
     - ``gblup``     : clusteriza solo las predicciones GBLUP.
     - ``pca_gblup`` : concatena la PCA de los SNPs (whitened) con las predicciones GBLUP
                       z-scoreadas y pesadas por ``gblup_weight`` (combina ambas ideas).
     - ``pca``       : ablación — solo la PCA (como el hard viejo, pero con el ensemble de abajo).

2. **Ensemble sobre clusterings (robustez).** No hay UN clustering fijo. Por cada seed
   ``s in [0, n_seeds)`` se ajusta un GMM propio (``random_state`` distinta) → un clustering
   distinto → **una** red multitask por cluster (+ un fallback global). Se promedian (z-scoreadas
   por columna, como el `EnsembleMultitaskRegressor`) las predicciones de las ``n_seeds`` miembros.
   Así ni el clustering ni la red quedan atados a una sola seed.

Regla 1 (sin leakage). TODO el gating se ajusta SOLO con el train del fold: los GBLUP por trait
(marcadores/λ train-only), la PCA (z-score + componentes train-only), y las medias/desvíos de
estandarización del bloque GBLUP. La predicción GBLUP es la MISMA función lineal ``Z·b`` para train
y val (rrBLUP), así que el espacio de ruteo es consistente entre ajuste (clusterizar el train) e
inferencia (rutear val/test) sin que val entre en ningún ``fit``.

Se enchufa en `run_multitask_evaluation` igual que `moe_multitask` (interfaz ``fit(X, Y, Y_mask)`` /
``predict(X) -> (n, T)``); el dispatch de `scripts/train.py` lo incluye por nombre. Reusa `GBLUP`
(un modelo por trait para el gating), `PCAReducer`/`SNPPreprocessor` (bloque PCA) y
`MLPMultitaskRegressor` (una red por cluster) sin reimplementar nada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import PCAReducer, PreprocChain, SNPPreprocessor
from src.models.base import BaseModel
from src.models.gblup import GBLUP
from src.models.mlp_multitask import MLPMultitaskRegressor

# Defaults del gating (clustering). Los pisa `model.params.gating.*` del config/sweep.
_GATING_DEFAULTS: dict[str, Any] = {
    "mode": "gblup",               # gblup | pca_gblup | pca (espacio a clusterizar)
    # --- bloque GBLUP (modos gblup / pca_gblup) ---
    "gblup_targets": None,         # None => todos los traits del multitask; o lista subset (p.ej. [GY])
    "gblup_lambda_method": "reml", # reml (determinista, barato) | cv (temporal/kfold)
    "gblup_standardize": True,     # z-score por-trait de las predicciones GBLUP (train-only)
    # --- bloque PCA (modos pca / pca_gblup) ---
    "pca_n_components": 50,        # dims retenidas por PCA (HP del sweep)
    "pca_standardize": True,       # z-score por-SNP antes del PCA (train-only)
    "pca_whiten": True,            # componentes a var unitaria (mejor condicionado para GMM 'full')
    "gblup_weight": 3.0,           # peso del bloque GBLUP vs PCA en pca_gblup (compensa la dim de la PCA)
    # --- GMM ---
    "k": 3,                        # nº de clusters GMM (HP del sweep)
    "covariance_type": "full",     # full | diag | tied | spherical
    "reg_covar": 1e-6,             # estabilidad numérica del GMM
    "n_init": 1,                   # reinicios del EM
    "min_cluster_size": 150,       # cluster con menos filas => sin red propia (usa fallback)
    "global_fallback": True,       # red extra sobre TODO el train, para clusters chicos/vacíos
}

# Defaults del ensemble sobre seeds (re-clustering). Los pisa `model.params.ensemble.*`.
_ENSEMBLE_DEFAULTS: dict[str, Any] = {
    "n_seeds": 5,                  # nº de (clustering + red) a promediar
    "seed_step": 1000,             # seed_s = base_seed + s*seed_step
}


class MoEGBLUPClusterRegressor(BaseModel):
    """MoE hard multitask con clustering GBLUP/PCA+GBLUP, ensamblado sobre seeds re-clusterizando.

    `params` (sub-dict ``model.params`` del config):
        gating   : dict con los defaults de ``_GATING_DEFAULTS`` (modo + bloques + GMM + salvaguardas).
        ensemble : dict con ``n_seeds``/``seed_step`` (miembros del ensemble sobre clusterings).
        expert   : dict con ``inner`` = params del `MLPMultitaskRegressor` (una red por cluster). El
                   ``inner.random_state`` es la seed BASE; cada miembro usa ``base + s*seed_step``.

    `fit(X, Y, Y_mask)` recibe ``Y``/``Y_mask`` ``(n, T)``; los nombres de columnas llegan por
    `set_fit_context(target_names=...)`. `predict(X)` devuelve ``(n, T)``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        self.ensemble_params: dict[str, Any] = {**_ENSEMBLE_DEFAULTS, **dict(self.params.get("ensemble", {}) or {})}
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})
        self.inner_params: dict[str, Any] = dict(self.expert_params.get("inner", {}) or {})

    # --- Construcción del gating (train-only) -------------------------------

    def _build_pca_chain(self) -> Any:
        """Cadena de featurización del bloque PCA (train-only): [z-score] -> PCA whitened."""
        g = self.gating_params
        base_seed = int(self.inner_params.get("random_state", 575))
        steps: list = []
        if bool(g["pca_standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(
            PCAReducer(
                n_components=int(g["pca_n_components"]),
                whiten=bool(g["pca_whiten"]),
                svd_solver="full",          # determinista (regla de reproducibilidad)
                random_state=base_seed,
            )
        )
        return steps[0] if len(steps) == 1 else PreprocChain(steps)

    def _fit_gblup_block(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray, target_names, year) -> None:
        """Ajusta un GBLUP por trait del gating (train-only) y guarda medias/desvíos del z-score.

        La predicción GBLUP es ``Z·b`` (rrBLUP): misma función lineal para train y val => el espacio
        de ruteo es consistente y no hay leakage (``b`` sale solo del train).
        """
        g = self.gating_params
        names = list(target_names) if target_names is not None else [f"t{j}" for j in range(Y.shape[1])]
        sel = g["gblup_targets"]
        self.gblup_target_idx_ = (
            list(range(len(names))) if not sel else [names.index(t) for t in sel]
        )
        self.gblup_target_names_ = [names[j] for j in self.gblup_target_idx_]

        self.gblup_models_: list[GBLUP] = []
        for j in self.gblup_target_idx_:
            col = np.asarray(Y_mask[:, j], dtype=bool)
            gb = GBLUP({"params": {"lambda_method": str(g["gblup_lambda_method"])}})
            if year is not None:
                gb.set_fit_context(year=np.asarray(year)[col])
            gb.fit(X[col], Y[col, j])
            self.gblup_models_.append(gb)

        Gtr = self._gblup_features_raw(X)                     # (n, T') predicciones GBLUP
        if bool(g["gblup_standardize"]):
            self.gblup_mean_ = Gtr.mean(axis=0)
            sd = Gtr.std(axis=0)
            self.gblup_std_ = np.where(sd < 1e-12, 1.0, sd)
        else:
            self.gblup_mean_ = np.zeros(Gtr.shape[1])
            self.gblup_std_ = np.ones(Gtr.shape[1])

    def _gblup_features_raw(self, X: np.ndarray) -> np.ndarray:
        """Predicciones GBLUP por trait apiladas ``(n, T')`` (sin z-score)."""
        return np.column_stack([gb.predict(X) for gb in self.gblup_models_])

    def _gating_features(self, X: np.ndarray) -> np.ndarray:
        """Espacio de clustering para un set cualquiera (train o val/test), según ``mode``.

        Reproduce EXACTAMENTE las transformaciones ajustadas en train (PCA whitened + z-score del
        bloque GBLUP), así que clusterizar el train e inferir sobre val usan el MISMO espacio.
        """
        mode = str(self.gating_params["mode"])
        blocks: list[np.ndarray] = []
        if mode in ("pca", "pca_gblup"):
            blocks.append(np.asarray(self.pca_chain_.transform(X), dtype=np.float64))
        if mode in ("gblup", "pca_gblup"):
            Gz = (self._gblup_features_raw(X) - self.gblup_mean_) / self.gblup_std_
            blocks.append(float(self.gating_params["gblup_weight"]) * Gz if mode == "pca_gblup" else Gz)
        if not blocks:
            raise ValueError(f"gating.mode desconocido: {mode!r}. Use gblup | pca_gblup | pca.")
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _make_expert(self, seed: int) -> MLPMultitaskRegressor:
        p = dict(self.inner_params)
        p["random_state"] = int(seed)
        return MLPMultitaskRegressor({"params": p})

    @staticmethod
    def _as_2d(p: Any) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p.reshape(-1, 1) if p.ndim == 1 else p

    def _route_predict(self, X: np.ndarray, labels: np.ndarray, experts: dict[int, Any], fallback: Any) -> np.ndarray:
        """Ruteo hard top-1: cada fila va al expert de su cluster (o al fallback si no tiene)."""
        out = np.empty((X.shape[0], self.n_targets_), dtype=float)
        for c in np.unique(labels):
            rows = np.flatnonzero(labels == c)
            expert = experts.get(int(c), fallback)
            if expert is None:
                raise RuntimeError(
                    f"cluster {int(c)} sin expert ni fallback; activá gating.global_fallback."
                )
            out[rows] = self._as_2d(expert.predict(X[rows]))
        return out

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MoEGBLUPClusterRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("MoEGBLUPClusterRegressor.fit recibió Y vacío.")
        self.n_targets_ = int(Y.shape[1])

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k
        mode = str(g["mode"])
        year = ctx.get("year")
        line_id = ctx.get("line_id")
        target_names = ctx.get("target_names")
        min_size = int(g["min_cluster_size"])

        # --- Gating (train-only, regla 1): bloque PCA y/o bloque GBLUP --------
        if mode in ("pca", "pca_gblup"):
            self.pca_chain_ = self._build_pca_chain()
            self.pca_chain_.fit_transform(X)  # ajusta z-score + PCA solo con el train
        if mode in ("gblup", "pca_gblup"):
            self._fit_gblup_block(X, Y, Y_mask, target_names, year)
        F = self._gating_features(X)          # espacio de clustering del train

        base_seed = int(self.inner_params.get("random_state", 575))
        n_seeds = int(self.ensemble_params["n_seeds"])
        seed_step = int(self.ensemble_params["seed_step"])

        # --- Ensemble sobre clusterings: por seed, un GMM + una red por cluster ----
        self.members_: list[dict[str, Any]] = []
        self.cluster_sizes_all_: list[dict[int, int]] = []
        for s in range(n_seeds):
            seed = base_seed + s * seed_step
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=str(g["covariance_type"]),
                reg_covar=float(g["reg_covar"]),
                n_init=int(g["n_init"]),
                random_state=seed,
            )
            gmm.fit(F)
            labels = gmm.predict(F)  # asignación hard por argmax de responsabilidad

            experts: dict[int, MLPMultitaskRegressor] = {}
            sizes: dict[int, int] = {}
            for c in range(k):
                mask = labels == c
                n_c = int(mask.sum())
                sizes[c] = n_c
                if n_c < min_size:
                    continue  # cluster chico => sin red propia; lo atiende el fallback
                expert = self._make_expert(seed)
                expert.set_fit_context(
                    year=None if year is None else np.asarray(year)[mask],
                    line_id=None if line_id is None else np.asarray(line_id)[mask],
                    target_names=target_names,
                )
                expert.fit(X[mask], Y[mask], Y_mask[mask])
                experts[c] = expert

            # Fallback global (todo el train): atiende clusters chicos/vacíos en inferencia.
            fallback: MLPMultitaskRegressor | None = None
            if bool(g["global_fallback"]) or (len(experts) < k):
                fb = self._make_expert(seed)
                fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
                fb.fit(X, Y, Y_mask)
                fallback = fb

            # Estandarización por columna desde las predicciones de TRAIN de este miembro
            # (sin leakage): la loss Pearson deja la escala arbitraria => sin z-score, la red de
            # mayor varianza domina el promedio entre seeds.
            ptr = self._route_predict(X, labels, experts, fallback)
            mu = ptr.mean(axis=0, keepdims=True)
            sd = ptr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)

            self.members_.append({"gmm": gmm, "experts": experts, "fallback": fallback, "scaler": (mu, sd)})
            self.cluster_sizes_all_.append(sizes)

        # Learning curve representativa (primera red del primer miembro) para el logging.
        first = next(iter(self.members_[0]["experts"].values()), None) or self.members_[0]["fallback"]
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "members_", None):
            raise RuntimeError("MoEGBLUPClusterRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)
        F = self._gating_features(X)
        acc: np.ndarray | None = None
        for m in self.members_:
            labels = m["gmm"].predict(F)
            pred = self._route_predict(X, labels, m["experts"], m["fallback"])
            mu, sd = m["scaler"]
            z = (pred - mu) / sd
            acc = z if acc is None else acc + z
        return acc / len(self.members_)

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "members_", None):
            return {}
        n_experts = [len(m["experts"]) for m in self.members_]
        params: dict[str, Any] = {
            "moe_gblup": 1,
            "k": int(self.k_),
            "n_seeds": int(len(self.members_)),
            "n_experts_mean": float(np.mean(n_experts)),
            "global_fallback_used": int(any(m["fallback"] is not None for m in self.members_)),
            "min_cluster_size": int(self.gating_params["min_cluster_size"]),
            "gating_mode": str(self.gating_params["mode"]),           # no-numérico => metadata
        }
        if getattr(self, "gblup_models_", None):
            for name, gb in zip(self.gblup_target_names_, self.gblup_models_):
                lam = gb.get_fitted_params().get("lambda_0")
                if lam is not None:
                    params[f"gblup_lambda_{name}"] = float(lam)
        if str(self.gating_params["mode"]) in ("pca", "pca_gblup"):
            gfp = getattr(self.pca_chain_, "get_fitted_params", None)
            if callable(gfp):
                params.update(gfp())
            params["gblup_weight"] = float(self.gating_params["gblup_weight"])
        # nº de params entrenables (suma de todas las redes de todos los miembros).
        n_params = 0
        for m in self.members_:
            for e in m["experts"].values():
                n_params += int(getattr(e, "n_params_", 0) or 0)
            if m["fallback"] is not None:
                n_params += int(getattr(m["fallback"], "n_params_", 0) or 0)
        params["n_params"] = int(n_params)
        return params

    # --- Persistencia -------------------------------------------------------
    # GBLUP/GMM/PCA (numpy/sklearn) van en un manifiesto joblib; cada red por
    # cluster (torch) en su propio archivo hermano (evita colisión de state_dict).

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        base = path.parent
        base.mkdir(parents=True, exist_ok=True)

        members_meta: list[dict[str, Any]] = []
        for s, m in enumerate(self.members_):
            expert_files: dict[int, str] = {}
            for c, e in m["experts"].items():
                fname = f"member_{s}_expert_{c}.pt"
                e.save(base / fname)
                expert_files[c] = fname
            fb_file: str | None = None
            if m["fallback"] is not None:
                fb_file = f"member_{s}_fallback.pt"
                m["fallback"].save(base / fb_file)
            members_meta.append(
                {"gmm": m["gmm"], "scaler": m["scaler"], "expert_files": expert_files, "fallback_file": fb_file}
            )

        joblib.dump(
            {
                "config": self.config,
                "gating_params": self.gating_params,
                "ensemble_params": self.ensemble_params,
                "expert_params": self.expert_params,
                "inner_params": self.inner_params,
                "n_targets_": self.n_targets_,
                "k_": self.k_,
                "pca_chain_": getattr(self, "pca_chain_", None),
                "gblup_models_": getattr(self, "gblup_models_", None),
                "gblup_mean_": getattr(self, "gblup_mean_", None),
                "gblup_std_": getattr(self, "gblup_std_", None),
                "gblup_target_idx_": getattr(self, "gblup_target_idx_", None),
                "gblup_target_names_": getattr(self, "gblup_target_names_", None),
                "cluster_sizes_all_": getattr(self, "cluster_sizes_all_", None),
                "members_meta": members_meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoEGBLUPClusterRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.ensemble_params = d["ensemble_params"]
        obj.expert_params = d["expert_params"]
        obj.inner_params = d["inner_params"]
        obj.n_targets_ = int(d["n_targets_"])
        obj.k_ = int(d["k_"])
        obj.pca_chain_ = d["pca_chain_"]
        obj.gblup_models_ = d["gblup_models_"]
        obj.gblup_mean_ = d["gblup_mean_"]
        obj.gblup_std_ = d["gblup_std_"]
        obj.gblup_target_idx_ = d["gblup_target_idx_"]
        obj.gblup_target_names_ = d["gblup_target_names_"]
        obj.cluster_sizes_all_ = d["cluster_sizes_all_"]
        obj.members_ = []
        for meta in d["members_meta"]:
            experts = {
                int(c): MLPMultitaskRegressor.load(base / f) for c, f in meta["expert_files"].items()
            }
            fallback = MLPMultitaskRegressor.load(base / meta["fallback_file"]) if meta["fallback_file"] else None
            obj.members_.append(
                {"gmm": meta["gmm"], "experts": experts, "fallback": fallback, "scaler": meta["scaler"]}
            )
        obj.history_ = []
        return obj
