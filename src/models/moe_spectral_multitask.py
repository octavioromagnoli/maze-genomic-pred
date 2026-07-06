"""MoE con Clustering Espectral sobre la matriz de parentesco genómico G, expertos sobre el residuo.

Cuarta variante de MoE del repo (las otras quedan intactas para comparar):
- `moe_multitask`         : PCA+GMM, hard top-1, experto sobre fenotipo crudo (el que gana en drought).
- `moe_gblup_multitask`   : GMM sobre û (breeding values), residuo, ruteo soft KNN en û (descartada).
- `moe_gblup_residual_multitask` : PCA+GMM hard top-1, backbone GBLUP + residuo (recupera 0.323).
- `moe_spectral_multitask` (ESTE) : **Clustering Espectral sobre G_train** para los clusters, backbone
  GBLUP + residuo, y **ruteo soft por KNN sobre la relación genómica G_test↔train**.

Los 4 cambios pedidos:
1. **Clustering Espectral (train).** ``G = Z Z' / p`` (marcadores estandarizados train-only) es la
   matriz de parentesco genómico. Se corre `SpectralClustering(affinity='precomputed')` sobre una
   afinidad no-negativa derivada de ``G_train`` (clip de negativos: relaciones por debajo de la media
   no son aristas) ⇒ cada línea de train cae en uno de ``C`` clusters (un experto por cluster). El
   spectral clustering es transductivo (no tiene `predict`), por eso el ruteo de test es por KNN.
2. **Target residual.** Un GBLUP por trait/ambiente (train-only) da ``û`` ``(n, T)``; cada experto
   predice el residuo ``R = y − û`` y la predicción final es ``û + salida_MoE``.
3. **Ruteo soft por KNN sobre G_test,train (test).** ``G_test,train = Z_test Z_train' / p`` (test
   estandarizado con las stats de train). Los "vecinos más cercanos" de una línea de test son los
   ``K`` train con MAYOR relación genómica (G es similitud: más alto = más pariente). El peso del
   cluster c = proporción de esos K vecinos que caen en c (ej. K=10, 7 en c0 y 3 en c1 ⇒ [0.7, 0.3]).
   ``K`` (``k_neighbors``) es configurable. La salida del MoE es la mezcla ``Σ_c w_c · experto_c``.
4. **Anti-leakage estricto.** GBLUP y Spectral Clustering se ajustan SOLO con train. Las ``û`` de test
   salen de proyectar el GBLUP entrenado sobre la relación genómica test↔train (genotipos, NO
   fenotipos de test). ``G_test,train`` usa las stats de estandarización de train. `predict(X)` recibe
   solo ``X``; ningún fenotipo de test entra a ningún ajuste.

**Calibración del residuo.** El ensemble z-scorea sus predicciones (escala arbitraria por la loss
Pearson) y la suma ``û + salida`` es aditiva, así que cada experto se recalibra con un OLS 1D
train-only (``R ≈ a·salida + b`` por columna) que lo devuelve a la escala del residuo. Experto sin
correlación real ⇒ ``a≈0`` ⇒ corrección ≈0 ⇒ la predicción cae a ``û`` (GBLUP), el fallback seguro.

Se enchufa en `run_multitask_evaluation` (``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``) igual que
los otros MoE; el dispatch de `scripts/train.py` lo incluye por nombre. Reusa `standardize_train`/
`genomic_relationship` (G, de `gblup.py`), `GBLUP` (backbone) y `EnsembleMultitaskRegressor` (expertos).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP, apply_standardize, genomic_relationship, standardize_train

# Defaults del gating (GBLUP backbone + Spectral Clustering + KNN). Los pisa `model.params.gating.*`.
_GATING_DEFAULTS: dict[str, Any] = {
    "gblup": {},                  # params por-columna del GBLUP backbone (p.ej. {lambda_method: reml})
    "residual": True,             # True: expertos sobre el residuo y-û (backbone GBLUP). False: sobre y crudo
    "n_clusters": 4,              # C = nº de clusters espectrales = nº de expertos
    "affinity_clip": True,        # afinidad = clip(G, 0) (relaciones negativas => sin arista)
    "assign_labels": "kmeans",    # SpectralClustering.assign_labels: kmeans | discretize | cluster_qr
    "n_init": 10,                 # reinicios del k-means en el embedding espectral
    "min_cluster_size": 150,      # cluster con menos filas => sin experto propio (usa fallback)
    "global_fallback": True,      # experto extra sobre TODO el train, para clusters chicos/vacíos
    "random_state": 575,
    "k_neighbors": 10,            # K vecinos de train (mayor relación genómica) para el ruteo soft
}


class SpectralMoEGBLUPResidualMultitaskRegressor(BaseModel):
    """MoE con Clustering Espectral sobre G, backbone GBLUP + residuo, ruteo soft por KNN en G_test,train.

    `params` (sub-dict ``model.params`` del config):
        gating : dict con los defaults de ``_GATING_DEFAULTS`` (backbone GBLUP + spectral + KNN + salvaguardas).
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

    # --- Ruteo soft por KNN sobre la relación genómica G_test,train ---------

    def _knn_weights(self, G_ts: np.ndarray) -> np.ndarray:
        """Pesos blandos ``(n_test, C)`` (filas suman 1) por proporción de cluster entre los K vecinos.

        Los K vecinos de una línea de test son los K train con MAYOR ``G_test,train`` (G es similitud:
        más alto = más pariente). Se toma el top-K por fila y se cuenta la proporción por cluster.
        """
        n_train = G_ts.shape[1]
        K = int(min(int(self.gating_params["k_neighbors"]), n_train))
        K = max(K, 1)
        # Índices de los K train con mayor relación genómica por fila (no ordenados, no hace falta).
        idx = np.argpartition(G_ts, n_train - K, axis=1)[:, n_train - K:]  # (n_test, K)
        neigh_labels = self.train_labels_[idx]                            # (n_test, K)
        C = int(self.n_clusters_)
        W = np.zeros((G_ts.shape[0], C), dtype=float)
        for c in range(C):
            W[:, c] = (neigh_labels == c).mean(axis=1)
        return W

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(
        self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None
    ) -> "SpectralMoEGBLUPResidualMultitaskRegressor":
        from sklearn.cluster import SpectralClustering

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        if Y.size == 0:
            raise ValueError("SpectralMoEGBLUPResidualMultitaskRegressor.fit recibió Y vacío.")
        self.n_targets_ = int(Y.shape[1])

        g = self.gating_params
        C = int(g["n_clusters"])
        self.n_clusters_ = C

        year = ctx.get("year")
        line_id = ctx.get("line_id")
        target_names = ctx.get("target_names")

        # --- 1) Target de los expertos: residuo del GBLUP (backbone) o fenotipo crudo --
        # residual=True: un GBLUP por trait/ambiente (train-only) da û y el target es R = y - û; la
        # predicción final suma û. residual=False: sin backbone, los expertos predicen y crudo y la
        # predicción final es solo la mezcla (como el hard MoE crudo, pero con ruteo espectral+soft).
        self.residual_ = bool(g["residual"])
        self.gblups_: list[GBLUP] = []
        if self.residual_:
            U = np.empty((X.shape[0], self.n_targets_), dtype=float)
            for t in range(self.n_targets_):
                mt = np.asarray(Y_mask[:, t], dtype=bool)
                gb = GBLUP({"params": dict(g["gblup"])})
                gb.set_fit_context(year=None if year is None else np.asarray(year)[mt])
                gb.fit(X[mt], Y[mt, t])
                self.gblups_.append(gb)
                U[:, t] = gb.predict(X)
            target = np.where(Y_mask, Y - U, 0.0)   # residuo y - û
        else:
            target = np.where(Y_mask, Y, 0.0)        # fenotipo crudo (faltantes en 0, sin backbone)

        # --- 2) Matriz de parentesco genómico G del TRAIN (marcadores train-only) --
        # Estandarización sobre TODAS las filas de train (los clusters usan el genotipo de toda línea,
        # independiente de la máscara de fenotipos). Se guarda Z_train para G_test,train en inferencia.
        Z_tr, self.col_mean_, self.col_std_ = standardize_train(X)
        self.p_ = int(Z_tr.shape[1])
        self.Z_train_ = Z_tr.astype(np.float32)  # (n_train, p); float32 para acotar el artifact
        G_tr = genomic_relationship(Z_tr, Z_tr, self.p_)  # (n_train, n_train), simétrica

        # --- 3) Clustering Espectral sobre una afinidad no-negativa derivada de G_train --
        affinity = np.clip(G_tr, 0.0, None) if bool(g["affinity_clip"]) else np.abs(G_tr)
        affinity = 0.5 * (affinity + affinity.T)  # simetría exacta (numérica)
        sc = SpectralClustering(
            n_clusters=C,
            affinity="precomputed",
            assign_labels=str(g["assign_labels"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        labels = sc.fit_predict(affinity)  # etiqueta de cluster por línea de train
        self.train_labels_ = np.asarray(labels, dtype=int)

        min_size = int(g["min_cluster_size"])

        # --- 4) Un experto por cluster que predice el RESIDUO + su calibración --
        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.calib_experts_: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(C):
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
            expert.fit(X[mask], target[mask], Y_mask[mask])
            self.experts_[c] = expert
            self.calib_experts_[c] = self._fit_calibration(expert, X[mask], target[mask], Y_mask[mask])

        # --- 5) Global fallback: experto sobre TODO el train (mismo target) ----
        self.fallback_: EnsembleMultitaskRegressor | None = None
        self.calib_fallback_: tuple[np.ndarray, np.ndarray] | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < C):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
            fb.fit(X, target, Y_mask)
            self.fallback_ = fb
            self.calib_fallback_ = self._fit_calibration(fb, X, target, Y_mask)

        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "train_labels_"):
            raise RuntimeError("SpectralMoEGBLUPResidualMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)

        # Relación genómica test↔train con las stats de train (anti-leakage): G_test,train = Z_te Z_tr'/p.
        Z_te = apply_standardize(X, self.col_mean_, self.col_std_)
        G_ts = genomic_relationship(Z_te, self.Z_train_.astype(np.float64), self.p_)  # (n_test, n_train)
        W = self._knn_weights(G_ts)                                   # (n_test, C) pesos blandos

        # Mezcla soft de los expertos (calibrados): Σ_c w_c · experto_c. Cachea por modelo (clusters
        # chicos comparten el fallback => una sola pasada). En escala de residuo (si residual) o de
        # fenotipo crudo (si no).
        mix = np.zeros((X.shape[0], self.n_targets_), dtype=float)
        cache: dict[int, np.ndarray] = {}
        for c in range(self.n_clusters_):
            w_c = W[:, c]
            if not np.any(w_c > 0):
                continue
            model, calib = self._model_for_cluster(c)
            key = id(model)
            if key not in cache:
                cache[key] = self._calibrated_predict(model, calib, X)
            mix += w_c[:, None] * cache[key]

        if self.residual_:
            # û de test aplicando los GBLUPs ya entrenados (solo genotipos): predicción = û + mezcla.
            U = np.column_stack([gb.predict(X) for gb in self.gblups_])   # (n_test, T)
            return U + mix
        return mix                                                       # sin backbone: la mezcla ES la predicción

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "train_labels_"):
            return {}
        g = self.gating_params
        params: dict[str, Any] = {
            "moe_spectral": 1,
            "residual": int(self.residual_),
            "n_clusters": int(self.n_clusters_),
            "n_experts": int(len(self.experts_)),
            "n_targets": int(self.n_targets_),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(g["min_cluster_size"]),
            "k_neighbors": int(g["k_neighbors"]),
            "assign_labels": str(g["assign_labels"]),  # no-numérico => metadata
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
                "residual_": self.residual_,
                "gblups_": self.gblups_,
                "col_mean_": self.col_mean_,
                "col_std_": self.col_std_,
                "p_": self.p_,
                "Z_train_": self.Z_train_,          # (n_train, p) float32 para G_test,train
                "train_labels_": self.train_labels_,
                "n_clusters_": self.n_clusters_,
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
    def load(cls, path: str | Path) -> "SpectralMoEGBLUPResidualMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.residual_ = bool(d.get("residual_", True))
        obj.gblups_ = list(d["gblups_"])
        obj.col_mean_ = d["col_mean_"]
        obj.col_std_ = d["col_std_"]
        obj.p_ = int(d["p_"])
        obj.Z_train_ = d["Z_train_"]
        obj.train_labels_ = np.asarray(d["train_labels_"], dtype=int)
        obj.n_clusters_ = int(d["n_clusters_"])
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
