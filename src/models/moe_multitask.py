"""Mixture of Experts multitask: gating PCA+GMM, un ensemble multitask por cluster.

Idea (mixture of experts con ruteo *hard*):

1. **Gating (train-only, regla 1).** Se reduce la dimensionalidad de los SNPs con PCA y se
   clusteriza el espacio latente con un `GaussianMixture` (GMM). PCA y GMM se ajustan SOLO con el
   train del fold; en inferencia se proyecta val/test sobre los MISMOS componentes y GMM.
2. **Experts.** Un `EnsembleMultitaskRegressor` (varias seeds, tronco compartido + una cabeza por
   trait/ambiente) por cluster, entrenado sobre los **SNPs completos** del subconjunto de ese
   cluster (PCA/GMM se usan solo para rutear, no como features del expert).
3. **Predicción.** Se calcula la responsabilidad de cada muestra bajo el GMM y se rutea (argmax,
   *hard top-1*) al expert del cluster de mayor responsabilidad.

Riesgo del régimen p≫n: partir n≈2319 en K clusters deja cada expert con ~500–800 muestras. Por
eso hay dos salvaguardas: `min_cluster_size` (un cluster con menos filas NO entrena expert propio)
y un **global fallback expert** entrenado sobre TODO el train, que atiende los clusters chicos o
vacíos en inferencia (garantiza que siempre haya predicción). El baseline honesto a batir no es
solo GBLUP sino la red multitask **global** (mismo `expert.inner`, sin gating).

Se enchufa en `run_multitask_evaluation` igual que `mlp_multitask`/`ensemble_multitask`
(interfaz ``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``); el dispatch de `scripts/train.py`
lo incluye por nombre. Reusa `SNPPreprocessor`/`PCAReducer` (gating) y `EnsembleMultitaskRegressor`
(experts) sin reimplementar nada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import PCAReducer, PreprocChain, SNPPreprocessor
from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor

# Defaults del gating (PCA + GMM). Los pisa `model.params.gating.*` del config/sweep.
_GATING_DEFAULTS: dict[str, Any] = {
    "pca_n_components": 50,       # dims retenidas por PCA (HP del sweep)
    "standardize": True,          # z-score por-SNP antes del PCA (train-only)
    "whiten": True,               # componentes a varianza unitaria (mejor condicionado para GMM 'full')
    "k": 3,                       # nº de clusters GMM (HP del sweep)
    "covariance_type": "full",    # full | diag | tied | spherical
    "reg_covar": 1e-6,            # estabilidad numérica del GMM
    "n_init": 1,                  # reinicios del EM (mayor => más estable, más lento)
    "min_cluster_size": 150,      # cluster con menos filas => sin expert propio (usa fallback)
    "global_fallback": True,      # expert extra sobre TODO el train, para clusters chicos/vacíos
    "random_state": 575,
}


class MixtureOfExpertsMultitaskRegressor(BaseModel):
    """Mixture of experts multitask con gating PCA+GMM y ruteo *hard top-1*.

    `params` (sub-dict ``model.params`` del config):
        gating : dict con los defaults de ``_GATING_DEFAULTS`` (PCA + GMM + salvaguardas).
        expert : dict de params de `EnsembleMultitaskRegressor` (``n_models``, ``seed_step`` e
                 ``inner`` con la arquitectura/loss del `MLPMultitaskRegressor`). Es lo que
                 comparten TODOS los experts (uno por cluster + el fallback).

    `fit(X, Y, Y_mask)` recibe ``Y``/``Y_mask`` ``(n, T)``; los nombres de las columnas llegan por
    `set_fit_context(target_names=...)` y se propagan a cada expert. `predict(X)` devuelve ``(n, T)``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        # Params del EnsembleMultitaskRegressor por cluster (arquitectura/loss compartidos).
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})

    # --- Construcción de piezas --------------------------------------------

    def _build_gating(self) -> Any:
        """Cadena de featurización del gating (train-only): [z-score] -> PCA."""
        g = self.gating_params
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(
            PCAReducer(
                n_components=int(g["pca_n_components"]),
                whiten=bool(g["whiten"]),
                svd_solver="full",              # determinista (regla de reproducibilidad)
                random_state=int(g["random_state"]),
            )
        )
        return steps[0] if len(steps) == 1 else PreprocChain(steps)

    def _make_expert(self) -> EnsembleMultitaskRegressor:
        return EnsembleMultitaskRegressor({"params": dict(self.expert_params)})

    def _fit_gating(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray) -> np.ndarray:
        """Ajusta la featurización del gating (train-only) y devuelve las features ``Z`` para el GMM.

        Default no supervisado (PCA): ignora ``Y``/``Y_mask``. Las variantes supervisadas
        (`moe_pls_multitask`, `moe_contrastive_multitask`) lo overridean para orientar el espacio de
        ruteo por el fenotipo o por el parentesco. Guarda la cadena ajustada en ``self.pre_``.
        """
        self.pre_ = self._build_gating()
        return self.pre_.fit_transform(X)

    @staticmethod
    def _as_2d(p: Any) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p.reshape(-1, 1) if p.ndim == 1 else p

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MixtureOfExpertsMultitaskRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("MixtureOfExpertsMultitaskRegressor.fit recibió Y vacío.")
        self.n_targets_ = int(Y.shape[1])

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k

        # --- Gating: featurización (PCA por default) + GMM, train-only (regla 1) ---
        Z = self._fit_gating(X, Y, Y_mask)   # hook: las variantes supervisadas (PLS, contrastivo) lo overridean
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(Z)
        labels = self.gmm_.predict(Z)  # asignación hard por argmax de responsabilidad

        year = ctx.get("year")
        line_id = ctx.get("line_id")
        target_names = ctx.get("target_names")
        min_size = int(g["min_cluster_size"])

        # --- Un expert (ensemble) por cluster con suficientes filas -----------
        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            self.cluster_sizes_[c] = n_c
            if n_c < min_size:
                continue  # cluster chico => sin expert propio; lo atiende el fallback
            expert = self._make_expert()
            expert.set_fit_context(
                year=None if year is None else np.asarray(year)[mask],
                line_id=None if line_id is None else np.asarray(line_id)[mask],
                target_names=target_names,
            )
            expert.fit(X[mask], Y[mask], Y_mask[mask])
            self.experts_[c] = expert

        # --- Global fallback: sobre TODO el train (clusters chicos/vacíos) ----
        # Se entrena si el config lo pide o si algún cluster quedó sin expert (garantiza
        # que toda muestra ruteada tenga a quién ir en inferencia).
        self.fallback_: EnsembleMultitaskRegressor | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < k):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
            fb.fit(X, Y, Y_mask)
            self.fallback_ = fb

        # Learning curve representativa (primer expert, o el fallback) para que run.py loguee algo.
        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("MixtureOfExpertsMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)
        Z = self.pre_.transform(X)
        clusters = self.gmm_.predict(Z)  # cluster de mayor responsabilidad por muestra
        out = np.empty((X.shape[0], self.n_targets_), dtype=float)
        for c in np.unique(clusters):
            rows = np.flatnonzero(clusters == c)
            expert = self.experts_.get(int(c), self.fallback_)
            if expert is None:
                raise RuntimeError(
                    f"cluster {int(c)} sin expert ni fallback; activá gating.global_fallback."
                )
            out[rows] = self._as_2d(expert.predict(X[rows]))
        return out

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "gmm_"):
            return {}
        params: dict[str, Any] = {
            "moe": 1,
            "k": int(self.k_),
            "n_experts": int(len(self.experts_)),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(self.gating_params["min_cluster_size"]),
        }
        for c, s in self.cluster_sizes_.items():
            params[f"cluster_size_{c}"] = int(s)
        gfp = getattr(self.pre_, "get_fitted_params", None)  # pca_n_components + var explicada
        if callable(gfp):
            params.update(gfp())
        n_params = sum(int(getattr(e, "get_fitted_params", dict)().get("n_params", 0) or 0)
                       for e in self.experts_.values())
        if self.fallback_ is not None:
            n_params += int(self.fallback_.get_fitted_params().get("n_params", 0) or 0)
        params["n_params"] = int(n_params)
        return params

    # --- Persistencia (override torch: cada expert es un ensemble de redes) --

    def save(self, path: str | Path) -> Path:
        """Guarda cada expert en su propio subdirectorio (evita colisión de ``submodel_i.pt``)
        + un manifiesto joblib con el gating (PCA/GMM, sklearn) y el ruteo.

        `save_model_artifact` sube el directorio entero, así que al bajar el artifact los
        ``expert_<c>/`` quedan junto al manifiesto y `load` los reencuentra.
        """
        import joblib

        path = Path(path)
        base = path.parent
        base.mkdir(parents=True, exist_ok=True)

        expert_files: dict[int, str] = {}
        for c, e in self.experts_.items():
            sub = base / f"expert_{c}"
            sub.mkdir(parents=True, exist_ok=True)
            e.save(sub / "ensemble.joblib")
            expert_files[c] = f"expert_{c}/ensemble.joblib"

        fallback_file: str | None = None
        if self.fallback_ is not None:
            sub = base / "fallback"
            sub.mkdir(parents=True, exist_ok=True)
            self.fallback_.save(sub / "ensemble.joblib")
            fallback_file = "fallback/ensemble.joblib"

        joblib.dump(
            {
                "config": self.config,
                "gating_params": self.gating_params,
                "expert_params": self.expert_params,
                "pre_": self.pre_,          # SNPPreprocessor/PCAReducer/PreprocChain (numpy/sklearn)
                "gmm_": self.gmm_,          # GaussianMixture (sklearn)
                "k_": self.k_,
                "n_targets_": self.n_targets_,
                "cluster_sizes_": self.cluster_sizes_,
                "expert_files": expert_files,
                "fallback_file": fallback_file,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MixtureOfExpertsMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.pre_ = d["pre_"]
        obj.gmm_ = d["gmm_"]
        obj.k_ = int(d["k_"])
        obj.n_targets_ = int(d["n_targets_"])
        obj.cluster_sizes_ = {int(c): int(s) for c, s in d["cluster_sizes_"].items()}
        obj.experts_ = {
            int(c): EnsembleMultitaskRegressor.load(base / f) for c, f in d["expert_files"].items()
        }
        obj.fallback_ = (
            EnsembleMultitaskRegressor.load(base / d["fallback_file"]) if d["fallback_file"] else None
        )
        obj.history_ = []
        return obj
