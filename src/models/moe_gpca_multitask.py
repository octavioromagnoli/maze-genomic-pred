"""MoE con clustering por PCA sobre la matriz de parentesco genómico G, expertos sobre el residuo.

Quinta variante de MoE del repo (las otras quedan intactas). Los 4 cambios pedidos:

1. **Espacio de clustering = PCA sobre G_train (no sobre SNPs crudos).** ``G = Z Z' / p`` (marcadores
   estandarizados train-only) es la matriz de parentesco genómico (cuadrada, n_train×n_train). Se
   corre PCA DIRECTAMENTE sobre ``G_train`` (cada fila = el vector de parentescos de una línea contra
   todas las de train) y se toman los primeros ``N`` componentes. El GMM se ajusta sobre ese espacio
   de N dims para asignar cada línea a un cluster/experto.
2. **Target residual.** Un GBLUP por trait/ambiente (train-only) da ``û`` ``(n, T)``; cada experto
   predice el residuo ``R = y − û`` y la predicción final es ``û + salida_MoE``.
3. **Ruteo soft en inferencia (temperatura sobre el GMM).** Para test se calcula
   ``G_test,train = Z_test Z_train' / p`` y se PROYECTA en el espacio de PCA ajustado en train
   (``pca.transform`` = ``(G_test,train − media_train) @ componentes'`` — los autovectores de train).
   Esos componentes proyectados pasan por el GMM entrenado → responsabilidades, a las que se aplica
   una **temperatura τ configurable** antes del softmax: ``w ∝ softmax(log_prob_ponderado / τ)``.
   Como la constante del log-sum-exp por fila se cancela en el softmax, esto equivale exactamente a
   ``w ∝ predict_proba^(1/τ)`` renormalizado (τ→0 = hard argmax, τ=1 = responsabilidades, τ→∞ = uniforme).
4. **Anti-leakage estricto.** GBLUP, ``G_train``, PCA y GMM se ajustan SOLO con train. Las ``û`` de
   test salen de proyectar el GBLUP entrenado sobre ``G_test,train`` (genotipos con las stats de
   estandarización de train, NO fenotipos de test). `predict(X)` recibe solo ``X``.

**Calibración del residuo.** El ensemble z-scorea sus predicciones (escala arbitraria por la loss
Pearson) y la suma ``û + salida`` es aditiva, así que cada experto se recalibra con un OLS 1D
train-only (``R ≈ a·salida + b`` por columna). Experto sin correlación real ⇒ ``a≈0`` ⇒ corrección
≈0 ⇒ la predicción cae a ``û`` (GBLUP), el fallback seguro.

Se enchufa en `run_multitask_evaluation` (``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``); el
dispatch de `scripts/train.py` lo incluye por nombre. Reusa `standardize_train`/`genomic_relationship`
(G, de `gblup.py`), `GBLUP` (backbone) y `EnsembleMultitaskRegressor` (expertos).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP, apply_standardize, genomic_relationship, standardize_train

# Defaults del gating (GBLUP backbone + PCA sobre G + GMM + temperatura). Los pisa `model.params.gating.*`.
_GATING_DEFAULTS: dict[str, Any] = {
    "gblup": {},                  # params por-columna del GBLUP backbone (p.ej. {lambda_method: reml})
    "pca_n_components": 50,       # N componentes de PCA sobre G_train (espacio de clustering)
    "whiten": True,               # componentes a varianza unitaria (mejor condicionado para GMM 'full')
    "k": 4,                       # nº de clusters GMM = nº de expertos
    "covariance_type": "full",    # full | diag | tied | spherical
    "reg_covar": 1e-6,            # estabilidad numérica del GMM
    "n_init": 1,                  # reinicios del EM
    "min_cluster_size": 150,      # cluster con menos filas => sin experto propio (usa fallback)
    "global_fallback": True,      # experto extra sobre TODO el train, para clusters chicos/vacíos
    "random_state": 575,
    "temperature": 1.0,           # τ antes del softmax del GMM (soft routing; >1 = más blando)
}


class GKinshipPCAMoEMultitaskRegressor(BaseModel):
    """MoE con clustering por PCA sobre G, backbone GBLUP + residuo, ruteo soft por temperatura del GMM.

    `params` (sub-dict ``model.params`` del config):
        gating : dict con los defaults de ``_GATING_DEFAULTS`` (backbone GBLUP + PCA-sobre-G + GMM + τ).
        expert : dict de params de `EnsembleMultitaskRegressor` (``n_models``, ``seed_step``, ``inner``).

    `fit(X, Y, Y_mask)` recibe ``Y``/``Y_mask`` ``(n, T)`` (faltantes en 0, mask=True donde hay dato);
    los nombres de columna llegan por `set_fit_context(target_names=...)`. `predict(X)` devuelve ``(n, T)``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})

    # --- Construcción de piezas --------------------------------------------

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
        """OLS 1D por columna ``R ≈ a·salida + b`` (train-only). Devuelve ``(a, b)`` ``(T,)``. Experto
        plano (varianza ~0) => ``a=0``, ``b=media(R)`` ⇒ corrección nula (la predicción cae a û)."""
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

    # --- Ruteo soft: temperatura sobre las responsabilidades del GMM --------

    def _temperature_weights(self, emb: np.ndarray) -> np.ndarray:
        """Responsabilidades del GMM en el embedding PCA(G), suavizadas con temperatura ``τ``.

        ``w ∝ predict_proba^(1/τ)`` renormalizado, que es EXACTAMENTE ``softmax(log_prob_ponderado/τ)``
        (la constante del log-sum-exp por fila se cancela en el softmax). τ→0 = hard argmax, τ=1 =
        responsabilidades del GMM, τ→∞ = uniforme.
        """
        r = self.gmm_.predict_proba(emb)                   # (n, k) responsabilidades
        tau = float(self.gating_params["temperature"])
        logr = np.log(np.clip(r, 1e-12, None)) / tau
        logr -= logr.max(axis=1, keepdims=True)            # estabilidad numérica del softmax
        w = np.exp(logr)
        return w / w.sum(axis=1, keepdims=True)

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(
        self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None
    ) -> "GKinshipPCAMoEMultitaskRegressor":
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("GKinshipPCAMoEMultitaskRegressor.fit recibió Y vacío.")
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

        # --- 2) Matriz de parentesco genómico G del TRAIN (marcadores train-only) --
        # Estandarización sobre TODAS las filas de train (el parentesco usa el genotipo de toda línea,
        # independiente de la máscara de fenotipos). Se guarda Z_train para G_test,train en inferencia.
        Z_tr, self.col_mean_, self.col_std_ = standardize_train(X)
        self.p_ = int(Z_tr.shape[1])
        self.Z_train_ = Z_tr.astype(np.float32)  # (n_train, p); float32 para acotar el artifact
        G_tr = genomic_relationship(Z_tr, Z_tr, self.p_)  # (n_train, n_train), simétrica

        # --- 3) PCA sobre G_train (no sobre SNPs) ⇒ espacio de clustering -----
        n_comp = int(min(int(g["pca_n_components"]), G_tr.shape[0], G_tr.shape[1]))
        self.pca_ = PCA(
            n_components=n_comp,
            whiten=bool(g["whiten"]),
            svd_solver="full",              # determinista (regla de reproducibilidad)
            random_state=int(g["random_state"]),
        )
        emb_tr = self.pca_.fit_transform(G_tr)  # (n_train, N) proyección de las filas de G

        # --- 4) GMM sobre el embedding PCA(G) ---------------------------------
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(emb_tr)
        labels = self.gmm_.predict(emb_tr)  # asignación hard SOLO para definir clusters/entrenar expertos
        self.train_labels_ = np.asarray(labels, dtype=int)

        min_size = int(g["min_cluster_size"])

        # --- 5) Un experto por cluster que predice el RESIDUO + su calibración --
        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.calib_experts_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            mask = self.train_labels_ == c
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

        # --- 6) Global fallback: experto del residuo sobre TODO el train -------
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
            raise RuntimeError("GKinshipPCAMoEMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)

        # û de test aplicando los GBLUPs ya entrenados (solo genotipos).
        U = np.column_stack([gb.predict(X) for gb in self.gblups_])   # (n_test, T)

        # G_test,train con las stats de train (anti-leakage) y proyección al espacio PCA de train.
        Z_te = apply_standardize(X, self.col_mean_, self.col_std_)
        G_ts = genomic_relationship(Z_te, self.Z_train_.astype(np.float64), self.p_)  # (n_test, n_train)
        emb_te = self.pca_.transform(G_ts)                            # (n_test, N) en las componentes de train
        W = self._temperature_weights(emb_te)                        # (n_test, k) pesos blandos (GMM + τ)

        # Mezcla de residuos: Σ_c w_c · experto_c (calibrado). Cachea por modelo (clusters chicos
        # comparten el fallback => una sola pasada).
        resid = np.zeros((X.shape[0], self.n_targets_), dtype=float)
        cache: dict[int, np.ndarray] = {}
        for c in range(self.k_):
            w_c = W[:, c]
            if not np.any(w_c > 0):
                continue
            model, calib = self._model_for_cluster(c)
            key = id(model)
            if key not in cache:
                cache[key] = self._calibrated_predict(model, calib, X)
            resid += w_c[:, None] * cache[key]

        return U + resid

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "gmm_"):
            return {}
        g = self.gating_params
        params: dict[str, Any] = {
            "moe_gpca": 1,
            "k": int(self.k_),
            "n_experts": int(len(self.experts_)),
            "n_targets": int(self.n_targets_),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(g["min_cluster_size"]),
            "temperature": float(g["temperature"]),
            "pca_n_components": int(self.pca_.n_components_),
            "pca_G_var_explained": float(np.sum(self.pca_.explained_variance_ratio_)),
        }
        for c, s in self.cluster_sizes_.items():
            params[f"cluster_size_{c}"] = int(s)
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
                "col_mean_": self.col_mean_,
                "col_std_": self.col_std_,
                "p_": self.p_,
                "Z_train_": self.Z_train_,          # (n_train, p) float32 para G_test,train
                "pca_": self.pca_,                  # PCA sobre G (sklearn)
                "gmm_": self.gmm_,                  # GaussianMixture (sklearn)
                "train_labels_": self.train_labels_,
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
    def load(cls, path: str | Path) -> "GKinshipPCAMoEMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.gblups_ = list(d["gblups_"])
        obj.col_mean_ = d["col_mean_"]
        obj.col_std_ = d["col_std_"]
        obj.p_ = int(d["p_"])
        obj.Z_train_ = d["Z_train_"]
        obj.pca_ = d["pca_"]
        obj.gmm_ = d["gmm_"]
        obj.train_labels_ = np.asarray(d["train_labels_"], dtype=int)
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
