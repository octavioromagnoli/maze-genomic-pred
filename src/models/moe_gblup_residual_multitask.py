"""MoE hard con backbone GBLUP: los expertos predicen el residuo.

Igual que moe_multitask pero, en vez del fenotipo crudo, cada experto predice el
residuo R = y - u_hat del GBLUP y la predicción final es u_hat + salida del experto.
El ruteo (PCA + GMM, hard top-1) queda idéntico al MoE hard, así que esto aísla el
efecto de usar el GBLUP como backbone.

Por fold: se ajusta un GBLUP por trait/ambiente con el train (da u_hat y el residuo
R = y - u_hat, target de los expertos; en inferencia u_hat sale del GBLUP ya
entrenado, sin fenotipos de test), se rutea con PCA+GMM como el MoE hard, y un
experto por cluster predice R sobre los SNPs completos (con min_cluster_size y un
fallback global). La predicción es u_hat + experto del cluster.

El ensemble z-scorea sus salidas (la loss Pearson deja la escala arbitraria), así
que cada experto se recalibra con un OLS 1D por columna (R ~ a*salida + b) para
llevarlas a la escala del residuo. Un experto sin correlación cae a a~0, de modo que
la predicción vuelve a u_hat (GBLUP) como fallback seguro.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import PCAReducer, PreprocChain, SNPPreprocessor
from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP

_GATING_DEFAULTS: dict[str, Any] = {
    "gblup": {},
    "pca_n_components": 50,
    "standardize": True,
    "whiten": True,
    "k": 3,
    "covariance_type": "full",
    "reg_covar": 1e-6,
    "n_init": 1,
    "min_cluster_size": 150,
    "global_fallback": True,
    "random_state": 575,
}


class GBLUPResidualHardMoEMultitaskRegressor(BaseModel):
    """MoE hard con backbone GBLUP: los expertos predicen el residuo.

    params.gating trae los defaults de _GATING_DEFAULTS (backbone GBLUP + PCA + GMM +
    salvaguardas) y params.expert los del EnsembleMultitaskRegressor, como el MoE hard.

    fit(X, Y, Y_mask) recibe Y/Y_mask de shape (n, T) (faltantes en 0, mask=True donde
    hay dato); los nombres de columna llegan por set_fit_context. predict(X) devuelve (n, T).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})


    def _build_gating(self) -> Any:
        """Cadena de featurización del gating: [z-score] -> PCA. Idéntica al MoE hard."""
        g = self.gating_params
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(
            PCAReducer(
                n_components=int(g["pca_n_components"]),
                whiten=bool(g["whiten"]),
                svd_solver="full",
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


    def _fit_calibration(
        self, expert: EnsembleMultitaskRegressor, X: np.ndarray, R: np.ndarray, M: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """OLS 1D por columna (R ~ a*salida + b) que lleva la salida z-scoreada del
        experto a la escala del residuo. Devuelve (a, b) de shape (T,). Un experto plano
        (varianza ~0) da a=0, b=media(R): corrección nula y la predicción cae a u_hat."""
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
        labels = self.gmm_.predict(Z)

        min_size = int(g["min_cluster_size"])

        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.calib_experts_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            self.cluster_sizes_[c] = n_c
            if n_c < min_size:
                continue
            expert = self._make_expert()
            expert.set_fit_context(
                year=None if year is None else np.asarray(year)[mask],
                line_id=None if line_id is None else np.asarray(line_id)[mask],
                target_names=target_names,
            )
            expert.fit(X[mask], R[mask], Y_mask[mask])
            self.experts_[c] = expert
            self.calib_experts_[c] = self._fit_calibration(expert, X[mask], R[mask], Y_mask[mask])

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

        U = np.column_stack([gb.predict(X) for gb in self.gblups_])
        Z = self.pre_.transform(X)
        clusters = self.gmm_.predict(Z)

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
        gfp = getattr(self.pre_, "get_fitted_params", None)
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
                "pre_": self.pre_,
                "gmm_": self.gmm_,
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
