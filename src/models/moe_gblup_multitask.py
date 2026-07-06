"""Mixture of Experts en espacio GBLUP: ruteo por breeding values û, expertos sobre el residuo.

Variante del `moe_multitask` (gating PCA+GMM, ruteo *hard top-1*, expertos sobre el fenotipo crudo).
El `moe_multitask` original queda intacto para comparar; esto es un camino aparte con tres cambios:

1. **Espacio de ruteo = breeding values del GBLUP (no PCA de SNPs).** Se entrena un GBLUP por cada
   trait/ambiente SOLO con el train del fold y se arma la matriz ``û`` ``(n, T)`` donde la fila es la
   línea y la columna ``t`` es la predicción del GBLUP para ese trait/ambiente. El GMM (y el KNN de
   inferencia) corren sobre ``û``, un espacio de pocas dimensiones (= nº de traits/ambientes), así el
   GMM no colapsa como con las 50–100 componentes de PCA.
2. **Target de los expertos = residuo del GBLUP.** Cada experto (un `EnsembleMultitaskRegressor`,
   igual que el MoE hard) predice el residuo ``R = y_obs − û`` del trait focal a partir de los SNPs,
   no el fenotipo crudo. La predicción final de una línea es ``û(línea) + salida_del_experto``: el
   GBLUP es el backbone lineal y los expertos solo corrigen la parte no-lineal encima.
3. **Ruteo soft (no las responsabilidades duras del GMM, que colapsan a one-hot).** En inferencia se
   obtiene ``û`` de la línea de test (aplicando el GBLUP ya entrenado; usa la relación genómica G
   test↔train, que son genotipos, NO fenotipos de test) y se rutea en el mismo espacio ``û`` con
   pesos blandos sobre los K clusters —
     - ``routing="knn"`` (default): KNN sobre ``û`` (K vecinos de train); el peso del cluster c es la
       proporción de esos vecinos que caen en c.
     - ``routing="gmm_temperature"``: responsabilidades del GMM suavizadas con temperatura ``τ``
       (``w ∝ r^(1/τ)`` renormalizado; τ→0 = hard argmax, τ=1 = responsabilidades, τ→∞ = uniforme).
   La predicción del residuo es la mezcla ``Σ_c w_c · experto_c`` y la final es ``û + mezcla``.

**Escala del residuo (calibración).** `EnsembleMultitaskRegressor` z-scorea sus predicciones por
columna (la loss Pearson deja la escala arbitraria), así que su salida NO está en la escala del
residuo. Como la suma ``û + salida`` es ADITIVA, cada experto se recalibra con un ajuste lineal
train-only (OLS 1D ``R ≈ a·salida + b`` por columna) que lo devuelve a la escala del residuo. Es un
stacking aditivo entrenado (como `gblup_residual`), robusto a cualquier escala de salida del experto:
un experto sin correlación real cae a ``a≈0`` ⇒ corrección ≈0 ⇒ la predicción vuelve a ``û`` (GBLUP).

Anti-leakage (regla 1): GBLUPs, ``û`` de train, residuos, GMM, clusters y KNN salen SOLO del train
del fold. El ``û`` de test se obtiene aplicando los GBLUPs ya entrenados (solo genotipos). Ningún
fenotipo de test entra a ningún ajuste — `predict(X)` recibe solo ``X``.

Se enchufa en `run_multitask_evaluation` igual que `moe_multitask`/`moe_soft_multitask` (interfaz
``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``); el dispatch de `scripts/train.py` lo incluye por
nombre. Reusa `GBLUP` (backbone + ruteo) y `EnsembleMultitaskRegressor` (expertos) sin reimplementar.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP

# Defaults del gating (GBLUP → û → GMM/KNN). Los pisa `model.params.gating.*` del config/sweep.
_GATING_DEFAULTS: dict[str, Any] = {
    "gblup": {},                  # params por-columna del GBLUP (p.ej. {lambda_method: reml})
    "standardize": True,          # z-score de las columnas de û (train-only) antes de GMM/KNN
    "k": 4,                       # nº de clusters GMM (HP del sweep)
    "covariance_type": "full",    # full | diag | tied | spherical
    "reg_covar": 1e-6,            # estabilidad numérica del GMM
    "n_init": 1,                  # reinicios del EM
    "min_cluster_size": 150,      # cluster con menos filas => sin experto propio (usa fallback)
    "global_fallback": True,      # experto extra sobre TODO el train, para clusters chicos/vacíos
    "random_state": 575,
    # --- ruteo soft en inferencia ---
    "routing": "knn",             # "knn" | "gmm_temperature"
    "k_neighbors": 30,            # K vecinos de train en el espacio û (routing="knn")
    "temperature": 1.0,           # τ de las responsabilidades del GMM (routing="gmm_temperature")
}


class GBLUPRoutedMoEMultitaskRegressor(BaseModel):
    """MoE multitask ruteado en espacio GBLUP, con expertos sobre el residuo y ruteo soft.

    `params` (sub-dict ``model.params`` del config):
        gating : dict con los defaults de ``_GATING_DEFAULTS`` (GBLUP + GMM + ruteo soft + salvaguardas).
        expert : dict de params de `EnsembleMultitaskRegressor` (``n_models``, ``seed_step`` e
                 ``inner`` con la arquitectura/loss del `MLPMultitaskRegressor`). Igual que el MoE hard.

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
        """OLS 1D por columna ``R ≈ a·ŝalida + b`` (train-only) que lleva la salida del experto a la
        escala del residuo. Necesario porque el ensemble z-scorea sus predicciones (escala arbitraria)
        y la suma ``û + salida`` es aditiva. Devuelve ``(a, b)`` de shape ``(T,)``.

        Ajusta cada columna solo con sus filas reales (``M[:, t]``). Si la predicción del experto en
        esa columna tiene varianza ~0 (experto degenerado), fija ``a=0`` y ``b=media(R)``: corrección
        nula ⇒ la predicción final cae a ``û`` (GBLUP), el fallback seguro.
        """
        P = self._as_2d(expert.predict(X))                 # (n, T) salida z-scoreada del ensemble
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
                intercept[t] = float(r.mean())            # experto plano => solo el offset del residuo
                continue
            A = np.vstack([p, np.ones_like(p)]).T
            coef, *_ = np.linalg.lstsq(A, r, rcond=None)   # [a, b]
            slope[t], intercept[t] = float(coef[0]), float(coef[1])
        return slope, intercept

    def _calibrated_predict(
        self, model: EnsembleMultitaskRegressor, calib: tuple[np.ndarray, np.ndarray], X: np.ndarray
    ) -> np.ndarray:
        slope, intercept = calib
        return self._as_2d(model.predict(X)) * slope + intercept  # (n, T) en escala del residuo

    def _model_for_cluster(
        self, c: int
    ) -> tuple[EnsembleMultitaskRegressor, tuple[np.ndarray, np.ndarray]]:
        """Experto (+ su calibración) del cluster ``c``; los clusters chicos/vacíos caen al fallback."""
        if c in self.experts_:
            return self.experts_[c], self.calib_experts_[c]
        if self.fallback_ is None:
            raise RuntimeError(
                f"cluster {c} sin experto ni fallback; activá gating.global_fallback."
            )
        return self.fallback_, self.calib_fallback_

    # --- Ruteo soft en espacio û -------------------------------------------

    def _routing_weights(self, Uz: np.ndarray) -> np.ndarray:
        """Pesos blandos ``(n, k)`` (filas suman 1) sobre los K clusters, en el espacio û estandarizado."""
        mode = str(self.gating_params["routing"]).lower()
        if mode == "knn":
            return self._knn_weights(Uz)
        if mode == "gmm_temperature":
            return self._temperature_weights(Uz)
        raise ValueError(
            f"gating.routing desconocido: {mode!r}. Use 'knn' o 'gmm_temperature'."
        )

    def _knn_weights(self, Uz: np.ndarray) -> np.ndarray:
        """KNN sobre û: peso del cluster c = proporción de los K vecinos de train que caen en c."""
        _, idx = self.knn_.kneighbors(Uz)                  # (n, K) índices en el train
        neigh_labels = self.train_labels_[idx]             # (n, K) etiqueta de cluster de cada vecino
        W = np.zeros((Uz.shape[0], self.k_))
        for c in range(self.k_):
            W[:, c] = (neigh_labels == c).mean(axis=1)
        return W

    def _temperature_weights(self, Uz: np.ndarray) -> np.ndarray:
        """Responsabilidades del GMM suavizadas con temperatura: ``w ∝ r^(1/τ)`` renormalizado."""
        r = self.gmm_.predict_proba(Uz)                    # (n, k) responsabilidades duras
        tau = float(self.gating_params["temperature"])
        logr = np.log(np.clip(r, 1e-12, None)) / tau
        logr -= logr.max(axis=1, keepdims=True)            # estabilidad numérica del softmax
        w = np.exp(logr)
        return w / w.sum(axis=1, keepdims=True)

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(
        self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None
    ) -> "GBLUPRoutedMoEMultitaskRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("GBLUPRoutedMoEMultitaskRegressor.fit recibió Y vacío.")
        self.n_targets_ = int(Y.shape[1])

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k

        year = ctx.get("year")
        line_id = ctx.get("line_id")
        target_names = ctx.get("target_names")

        # --- 1) GBLUP por trait/ambiente, SOLO train (regla 1) ----------------
        # Cada GBLUP se ajusta con las filas reales de su columna; luego predice û para TODAS las
        # filas de train (genotipo => sin leakage). û es el espacio de ruteo y el backbone lineal.
        self.gblups_: list[GBLUP] = []
        U = np.empty((X.shape[0], self.n_targets_), dtype=float)
        for t in range(self.n_targets_):
            mt = np.asarray(Y_mask[:, t], dtype=bool)
            gb = GBLUP({"params": dict(g["gblup"])})
            gb.set_fit_context(year=None if year is None else np.asarray(year)[mt])
            gb.fit(X[mt], Y[mt, t])
            self.gblups_.append(gb)
            U[:, t] = gb.predict(X)
        self.u_train_ = U

        # --- Residuo del GBLUP = target de los expertos (0 donde no hay dato) --
        R = np.where(Y_mask, Y - U, 0.0)

        # --- 2) Espacio de ruteo: û estandarizado (train-only) ----------------
        if bool(g["standardize"]):
            self.u_mean_ = U.mean(axis=0)
            std = U.std(axis=0)
            self.u_std_ = np.where(std < 1e-12, 1.0, std)
        else:
            self.u_mean_ = np.zeros(self.n_targets_)
            self.u_std_ = np.ones(self.n_targets_)
        Uz = (U - self.u_mean_) / self.u_std_

        # --- 3) GMM sobre û (pocas dims => no colapsa como con PCA) ------------
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(Uz)
        labels = self.gmm_.predict(Uz)                     # asignación hard SOLO para definir clusters
        self.train_labels_ = labels

        # --- 4) Un experto por cluster (predice el residuo) + su calibración ---
        min_size = int(g["min_cluster_size"])
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

        # --- 5) Global fallback: experto del residuo sobre TODO el train -------
        self.fallback_: EnsembleMultitaskRegressor | None = None
        self.calib_fallback_: tuple[np.ndarray, np.ndarray] | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < k):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
            fb.fit(X, R, Y_mask)
            self.fallback_ = fb
            self.calib_fallback_ = self._fit_calibration(fb, X, R, Y_mask)

        # --- 6) KNN sobre û (train-only) para el ruteo soft de inferencia ------
        if str(g["routing"]).lower() == "knn":
            from sklearn.neighbors import NearestNeighbors

            K = min(int(g["k_neighbors"]), Uz.shape[0])
            self.knn_ = NearestNeighbors(n_neighbors=max(K, 1))
            self.knn_.fit(Uz)
        else:
            self.knn_ = None

        # Learning curve representativa (primer experto, o el fallback) para que run.py loguee algo.
        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("GBLUPRoutedMoEMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)

        # û de test aplicando los GBLUPs ya entrenados (solo genotipos, sin fenotipos de test).
        U = np.column_stack([gb.predict(X) for gb in self.gblups_])   # (n, T)
        Uz = (U - self.u_mean_) / self.u_std_
        W = self._routing_weights(Uz)                                 # (n, k) pesos blandos

        # Mezcla de residuos: Σ_c w_c · experto_c (calibrado). Cachea por modelo (los clusters
        # chicos comparten el mismo fallback => una sola pasada).
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
            "moe_gblup": 1,
            "k": int(self.k_),
            "n_experts": int(len(self.experts_)),
            "n_targets": int(self.n_targets_),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(g["min_cluster_size"]),
            "routing": str(g["routing"]),                 # no-numérico => metadata
            "k_neighbors": int(g["k_neighbors"]),
            "temperature": float(g["temperature"]),
        }
        for c, s in self.cluster_sizes_.items():
            params[f"cluster_size_{c}"] = int(s)
        # lambda_0 de cada GBLUP del backbone (uno por trait/ambiente).
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
        """Guarda cada experto (ensemble torch) en su subdirectorio + un manifiesto joblib con los
        GBLUPs (numpy), el GMM/KNN (sklearn), la estandarización de û, las calibraciones y el ruteo.

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
                "gblups_": self.gblups_,          # lista de GBLUP (numpy/sklearn)
                "gmm_": self.gmm_,                # GaussianMixture
                "knn_": self.knn_,                # NearestNeighbors (o None)
                "u_mean_": self.u_mean_,
                "u_std_": self.u_std_,
                "u_train_": self.u_train_,
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
    def load(cls, path: str | Path) -> "GBLUPRoutedMoEMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.gblups_ = list(d["gblups_"])
        obj.gmm_ = d["gmm_"]
        obj.knn_ = d["knn_"]
        obj.u_mean_ = d["u_mean_"]
        obj.u_std_ = d["u_std_"]
        obj.u_train_ = d["u_train_"]
        obj.train_labels_ = d["train_labels_"]
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
