"""MoE hard (PCA+GMM, ruteo hard top-1) con GBLUP backbone: los expertos predicen el RESIDUO.

Es el `moe_multitask` (el MoE hard que gana en drought 2019) con UN solo cambio: en vez de predecir
el fenotipo crudo, cada experto predice el residuo ``R = y − û`` del GBLUP y la predicción final es
``û + salida_experto``. El ruteo (PCA de SNPs + GMM, hard top-1) queda IDÉNTICO al hard MoE, así que
esto aísla el efecto de "impulsar la red multitask con el GBLUP como backbone" sobre la arquitectura
que ya funciona (acá se cambia solo el target, no el ruteo).

Pipeline (todo train-only por fold, regla 1):
1. **GBLUP backbone.** Un GBLUP por trait/ambiente ajustado SOLO con el train ⇒ ``û`` ``(n, T)`` y el
   residuo ``R = y − û`` (target de los expertos). En inferencia ``û`` de test sale de aplicar el
   GBLUP ya entrenado (relación genómica G test↔train: genotipos, NO fenotipos de test).
2. **Ruteo PCA+GMM hard top-1** (idéntico a `moe_multitask`): [z-score] → PCA → GMM sobre los SNPs;
   se rutea por argmax de responsabilidad. PCA/GMM se ajustan solo con train.
3. **Un experto (EnsembleMultitaskRegressor) por cluster** que predice ``R`` sobre los SNPs completos,
   + `min_cluster_size` y un global fallback como el hard MoE.
4. **Predicción** = ``û(línea) + experto_del_cluster(línea)``.

**Calibración del residuo.** El ensemble z-scorea sus predicciones (escala arbitraria por la loss
Pearson), así que su salida no está en la escala del residuo; como la suma ``û + salida`` es aditiva,
cada experto se recalibra con un OLS 1D train-only (``R ≈ a·salida + b`` por columna). Un experto sin
correlación real cae a ``a≈0`` ⇒ corrección ≈0 ⇒ la predicción vuelve a ``û`` (GBLUP), el fallback seguro.

Se enchufa en `run_multitask_evaluation` (``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``) igual que
`moe_multitask`; el dispatch de `scripts/train.py` lo incluye por nombre. Reusa el gating de
`moe_multitask` (PCA/GMM), `GBLUP` (backbone) y `EnsembleMultitaskRegressor` (expertos).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import PCAReducer, PreprocChain, SNPPreprocessor
from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP

# Defaults del gating (PCA + GMM), idénticos al hard MoE, más el sub-dict `gblup` del backbone.
_GATING_DEFAULTS: dict[str, Any] = {
    "gblup": {},                  # params por-columna del GBLUP backbone (p.ej. {lambda_method: reml})
    "pca_n_components": 50,       # dims retenidas por PCA (HP del sweep)
    "standardize": True,          # z-score por-SNP antes del PCA (train-only)
    "whiten": True,               # componentes a varianza unitaria (mejor condicionado para GMM 'full')
    "k": 3,                       # nº de clusters GMM (HP del sweep)
    "covariance_type": "full",    # full | diag | tied | spherical
    "reg_covar": 1e-6,            # estabilidad numérica del GMM
    "n_init": 1,                  # reinicios del EM
    "min_cluster_size": 150,      # cluster con menos filas => sin experto propio (usa fallback)
    "global_fallback": True,      # experto extra sobre TODO el train, para clusters chicos/vacíos
    "random_state": 575,
}


class GBLUPResidualHardMoEMultitaskRegressor(BaseModel):
    """MoE hard (PCA+GMM, hard top-1) con GBLUP backbone: los expertos predicen el residuo.

    `params` (sub-dict ``model.params`` del config):
        gating : dict con los defaults de ``_GATING_DEFAULTS`` (backbone GBLUP + PCA + GMM + salvaguardas).
        expert : dict de params de `EnsembleMultitaskRegressor` (``n_models``, ``seed_step``, ``inner``).
                 Igual que el hard MoE.

    `fit(X, Y, Y_mask)` recibe ``Y``/``Y_mask`` ``(n, T)`` (faltantes en 0, mask=True donde hay dato);
    los nombres de columna llegan por `set_fit_context(target_names=...)`. `predict(X)` devuelve ``(n, T)``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})

    # --- Construcción de piezas --------------------------------------------

    def _build_gating(self) -> Any:
        """Cadena de featurización del gating (train-only): [z-score] -> PCA. Idéntica al hard MoE."""
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

    @staticmethod
    def _as_2d(p: Any) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p.reshape(-1, 1) if p.ndim == 1 else p

    # --- Calibración lineal del experto a la escala del residuo -------------

    def _fit_calibration(
        self, expert: EnsembleMultitaskRegressor, X: np.ndarray, R: np.ndarray, M: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """OLS 1D por columna ``R ≈ a·salida + b`` (train-only) que lleva la salida z-scoreada del
        experto a la escala del residuo. Devuelve ``(a, b)`` de shape ``(T,)``. Experto plano
        (varianza ~0) => ``a=0``, ``b=media(R)`` ⇒ corrección nula (la predicción cae a û)."""
        P = self._as_2d(expert.predict(X))
        T = int(R.shape[1])
        slope = np.zeros(T)
        intercept = np.zeros(T)
        for t in range(T):
            m = np.asarray(M[:, t], dtype=bool)
            if int(m.sum()) < 2:
                continue
            p = P[m, t]
            r = R[m, t]
            if float(np.std(p)) < 1e-12:
                intercept[t] = float(r.mean())
                continue
            A = np.vstack([p, np.ones_like(p)]).T
            coef, *_ = np.linalg.lstsq(A, r, rcond=None)
            slope[t], intercept[t] = float(coef[0]), float(coef[1])
        return slope, intercept

    def _calibrated_predict(
        self, model: EnsembleMultitaskRegressor, calib: tuple[np.ndarray, np.ndarray], X: np.ndarray
    ) -> np.ndarray:
        slope, intercept = calib
        return self._as_2d(model.predict(X)) * slope + intercept

    def _model_for_cluster(
        self, c: int
    ) -> tuple[EnsembleMultitaskRegressor, tuple[np.ndarray, np.ndarray]]:
        if c in self.experts_:
            return self.experts_[c], self.calib_experts_[c]
        if self.fallback_ is None:
            raise RuntimeError(f"cluster {c} sin experto ni fallback; activá gating.global_fallback.")
        return self.fallback_, self.calib_fallback_

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(
        self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None
    ) -> "GBLUPResidualHardMoEMultitaskRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("GBLUPResidualHardMoEMultitaskRegressor.fit recibió Y vacío.")
        self.n_targets_ = int(Y.shape[1])

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k

        year = ctx.get("year")
        line_id = ctx.get("line_id")
        target_names = ctx.get("target_names")

        # --- 1) GBLUP backbone por trait/ambiente (train-only) ⇒ û y residuo R --
        self.gblups_: list[GBLUP] = []
        U = np.empty((X.shape[0], self.n_targets_), dtype=float)
        for t in range(self.n_targets_):
            mt = np.asarray(Y_mask[:, t], dtype=bool)
            gb = GBLUP({"params": dict(g["gblup"])})
            gb.set_fit_context(year=None if year is None else np.asarray(year)[mt])
            gb.fit(X[mt], Y[mt, t])
            self.gblups_.append(gb)
            U[:, t] = gb.predict(X)
        R = np.where(Y_mask, Y - U, 0.0)

        # --- 2) Gating PCA + GMM hard top-1 sobre los SNPs (idéntico al hard MoE) --
        self.pre_ = self._build_gating()
        Z = self.pre_.fit_transform(X)
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(Z)
        labels = self.gmm_.predict(Z)  # asignación hard por argmax de responsabilidad

        min_size = int(g["min_cluster_size"])

        # --- 3) Un experto por cluster que predice el RESIDUO + su calibración --
        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.calib_experts_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            self.cluster_sizes_[c] = n_c
            if n_c < min_size:
                continue  # cluster chico => sin experto propio; lo atiende el fallback
            expert = self._make_expert()
            expert.set_fit_context(
                year=None if year is None else np.asarray(year)[mask],
                line_id=None if line_id is None else np.asarray(line_id)[mask],
                target_names=target_names,
            )
            expert.fit(X[mask], R[mask], Y_mask[mask])
            self.experts_[c] = expert
            self.calib_experts_[c] = self._fit_calibration(expert, X[mask], R[mask], Y_mask[mask])

        # --- 4) Global fallback: experto del residuo sobre TODO el train -------
        self.fallback_: EnsembleMultitaskRegressor | None = None
        self.calib_fallback_: tuple[np.ndarray, np.ndarray] | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < k):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
            fb.fit(X, R, Y_mask)
            self.fallback_ = fb
            self.calib_fallback_ = self._fit_calibration(fb, X, R, Y_mask)

        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("GBLUPResidualHardMoEMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)

        # û de test aplicando los GBLUPs ya entrenados (solo genotipos).
        U = np.column_stack([gb.predict(X) for gb in self.gblups_])   # (n, T)
        Z = self.pre_.transform(X)
        clusters = self.gmm_.predict(Z)  # hard top-1 (cluster de mayor responsabilidad)

        out = np.empty((X.shape[0], self.n_targets_), dtype=float)
        for c in np.unique(clusters):
            rows = np.flatnonzero(clusters == c)
            model, calib = self._model_for_cluster(int(c))
            out[rows] = U[rows] + self._calibrated_predict(model, calib, X[rows])
        return out

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "gmm_"):
            return {}
        params: dict[str, Any] = {
            "moe_gblup_residual": 1,
            "k": int(self.k_),
            "n_experts": int(len(self.experts_)),
            "n_targets": int(self.n_targets_),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(self.gating_params["min_cluster_size"]),
        }
        for c, s in self.cluster_sizes_.items():
            params[f"cluster_size_{c}"] = int(s)
        gfp = getattr(self.pre_, "get_fitted_params", None)  # pca_n_components + var explicada
        if callable(gfp):
            params.update(gfp())
        for t, gb in enumerate(self.gblups_):
            lam = gb.get_fitted_params().get("lambda_0")
            if lam is not None:
                params[f"gblup_lambda_0_{t}"] = float(lam)
        n_params = sum(
            int(getattr(e, "get_fitted_params", dict)().get("n_params", 0) or 0)
            for e in self.experts_.values()
        )
        if self.fallback_ is not None:
            n_params += int(self.fallback_.get_fitted_params().get("n_params", 0) or 0)
        params["n_params"] = int(n_params)
        return params

    # --- Persistencia (override: expertos torch en subdirs; resto joblib) ----

    def save(self, path: str | Path) -> Path:
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
                "gblups_": self.gblups_,
                "pre_": self.pre_,          # SNPPreprocessor/PCAReducer/PreprocChain
                "gmm_": self.gmm_,          # GaussianMixture
                "k_": self.k_,
                "n_targets_": self.n_targets_,
                "cluster_sizes_": self.cluster_sizes_,
                "calib_experts_": self.calib_experts_,
                "calib_fallback_": self.calib_fallback_,
                "expert_files": expert_files,
                "fallback_file": fallback_file,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GBLUPResidualHardMoEMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.gblups_ = list(d["gblups_"])
        obj.pre_ = d["pre_"]
        obj.gmm_ = d["gmm_"]
        obj.k_ = int(d["k_"])
        obj.n_targets_ = int(d["n_targets_"])
        obj.cluster_sizes_ = {int(c): int(s) for c, s in d["cluster_sizes_"].items()}
        obj.calib_experts_ = {int(c): v for c, v in d["calib_experts_"].items()}
        obj.calib_fallback_ = d["calib_fallback_"]
        obj.experts_ = {
            int(c): EnsembleMultitaskRegressor.load(base / f) for c, f in d["expert_files"].items()
        }
        obj.fallback_ = (
            EnsembleMultitaskRegressor.load(base / d["fallback_file"]) if d["fallback_file"] else None
        )
        obj.history_ = []
        return obj
