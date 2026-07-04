"""Sparse genomic prediction — LASSO/ElasticNet con bagging (análogo sparse de BayesB).

Motivación (arquitectura genética). GBLUP asume un modelo **infinitesimal**: prior
gaussiano de igual varianza sobre TODOS los marcadores (ridge sobre el kernel
``G=ZZ'/p``). Es el techo cuando el trait es altamente poligénico (p.ej. GY en ambiente
óptimo: ~2750 marcadores significativos). Pero cuando la arquitectura es **más rala**
—unos pocos loci de efecto grande dominan— un estimador de **selección de variables**
predice mejor. Ese es el caso de **GY bajo sequía** en este dataset (mucho menos
marcadores con señal fuerte que en óptimo): un LASSO (penalidad L1 = MAP de un prior
Laplace, el análogo frecuentista de BayesB) supera a GBLUP de forma amplia y robusta al
predecir el año más reciente del dev desde los anteriores.

Decisiones alineadas con las reglas del proyecto:

- **Sin leakage (regla 1).** Estandarización de marcadores train-only DENTRO de `fit`
  (z-score por SNP con media/desvío del train), igual que GBLUP. Los SNP binarios tienen
  varianza ``p(1-p)`` que varía ~5× entre MAF 0.05 y 0.5; la L1 no es invariante a escala,
  así que estandarizar es necesario para no penalizar de más los SNP de baja MAF. Por eso
  el modelo hace su preprocesamiento adentro y **no** usa el bloque `preprocess:`.
- **Selección de regularización temporal (regla 3).** Si `alpha` no se fija, se elige por
  un inner-split TEMPORAL del train (años previos → último año; p.ej. 2017+2018 ⇒ ajusta
  2017, valida 2018 para el reporte, o 2017+2018 ⇒ valida 2019 al cierre) maximizando el
  Pearson de inner-val (portada, regla 5). Con un solo año de train cae a K-fold por filas.
- **Bagging = análogo del ensemble multi-seed de la red.** Un LASSO single-fit es de ALTA
  varianza ante submuestreo del train (la selección de marcadores en LD es inestable: L1
  elige arbitrariamente uno de varios marcadores correlacionados). Baggear —promediar las
  predicciones de ``n_bags`` LASSO ajustados sobre subsamples del train— reduce esa varianza
  y da el número honesto/robusto (como el proyecto ensambla multi-seed la red para no
  inflar con una sola seed). **Baggear un modelo lineal ≡ promediar sus coeficientes**, así
  que se guarda un único vector ``coef_`` y `predict` es un solo producto ``Z @ coef_``
  (misma idea que la representación rrBLUP de efectos de marcador de GBLUP).

Persistencia: sklearn/numpy puro ⇒ el `save`/`load` por joblib de `BaseModel` sirve sin
override. `get_fitted_params` expone alpha, nº de marcadores seleccionados, l1_ratio y
n_bags para que `run_evaluation`/`train.py` los logueen sin bajar el modelo.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.metrics import pearson_r
from src.models.base import BaseModel
from src.models.gblup import apply_standardize, internal_cv, standardize_train

# Grilla por defecto de alpha (sobre marcadores ESTANDARIZADOS). Rango donde el LASSO va de
# denso (~600 marcadores) a muy ralo (~20); el óptimo de sequía cae en ~0.03–0.05.
_DEFAULT_ALPHA_GRID = np.logspace(-2.5, 0.0, 12)

_DEFAULTS: dict[str, Any] = {
    "l1_ratio": 1.0,            # 1.0 = LASSO puro; <1 = ElasticNet (L2 agrupa marcadores en LD)
    "alpha": None,             # None ⇒ se elige por inner-CV; float ⇒ se fija
    "select_by": "pearson",    # criterio de la inner-CV: "pearson" (portada) | "sse"
    "internal_cv": "temporal", # "temporal" (regla 3) | "kfold"
    "cv_folds": 5,             # pliegues si internal_cv="kfold" o si hay 1 solo año de train
    "bagging": True,           # promedia coeficientes sobre subsamples (reduce varianza de selección)
    "n_bags": 25,
    "subsample_frac": 0.8,
    "subspace_frac": 1.0,      # <1.0 = random-subspace (cada bag ve un subset de MARCADORES) → Random-LASSO
    "random_state": 575,
    "max_iter": 5000,
}


def _make_estimator(alpha: float, l1_ratio: float, max_iter: int):
    """LASSO (l1_ratio=1) o ElasticNet (l1_ratio<1) de sklearn."""
    from sklearn.linear_model import ElasticNet, Lasso

    if l1_ratio >= 1.0:
        return Lasso(alpha=float(alpha), max_iter=int(max_iter))
    return ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=int(max_iter))


class SparseGenomicRegressor(BaseModel):
    """LASSO/ElasticNet sobre marcadores estandarizados, con bagging opcional.

    `params` (sub-dict ``model.params`` del config):
        l1_ratio     : 1.0 (LASSO) | <1 (ElasticNet).
        alpha        : float para fijarlo, o ``null`` para elegirlo por inner-CV.
        alpha_grid   : lista de alphas a probar en la inner-CV (default logspace(-2.5,0,12)).
        select_by    : "pearson" (default, portada) | "sse".
        internal_cv  : "temporal" (default) | "kfold".
        cv_folds     : pliegues de K-fold (default 5).
        bagging      : bool (default True); n_bags / subsample_frac controlan el ensemble.
        subspace_frac: fracción de MARCADORES por bag (default 1.0 = usa todos; <1.0 = random-subspace).
        random_state : seed del submuestreo del bagging (default 575).
    """

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["l1_ratio"] = float(cfg["l1_ratio"])
        cfg["select_by"] = str(cfg["select_by"]).lower()
        cfg["internal_cv"] = str(cfg["internal_cv"]).lower()
        cfg["cv_folds"] = int(cfg["cv_folds"])
        cfg["bagging"] = bool(cfg["bagging"])
        cfg["n_bags"] = int(cfg["n_bags"])
        cfg["subsample_frac"] = float(cfg["subsample_frac"])
        cfg["subspace_frac"] = float(cfg["subspace_frac"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["max_iter"] = int(cfg["max_iter"])
        return cfg

    # --- Selección de alpha por inner-CV temporal (regla 3) -----------------

    def _select_alpha(self, Z: np.ndarray, yc: np.ndarray, cfg: dict[str, Any]) -> float:
        grid = np.asarray(self.params.get("alpha_grid", _DEFAULT_ALPHA_GRID), dtype=float)
        cv = internal_cv(self._fit_context.get("year"), scheme=cfg["internal_cv"], cv_folds=cfg["cv_folds"])
        if isinstance(cv, list):
            splits = cv
        else:
            from sklearn.model_selection import KFold

            splits = list(KFold(n_splits=int(cv)).split(np.arange(Z.shape[0])))

        best_a, best_score = float(grid[0]), -np.inf
        for a in grid:
            score = 0.0
            ok = True
            for tr, va in splits:
                est = _make_estimator(a, cfg["l1_ratio"], cfg["max_iter"])
                est.fit(Z[tr], yc[tr])
                pred = est.predict(Z[va])
                if cfg["select_by"] == "sse":
                    s = -float(np.sum((yc[va] - pred) ** 2))
                else:
                    r = pearson_r(yc[va], pred)
                    if not np.isfinite(r):
                        ok = False
                        break
                    s = r
                score += s
            if ok and score > best_score:
                best_score, best_a = score, float(a)
        return best_a

    # --- Ajuste de coeficientes (single o bagged) ---------------------------

    def _fit_coef(self, Z: np.ndarray, yc: np.ndarray, alpha: float, cfg: dict[str, Any]) -> np.ndarray:
        if not cfg["bagging"]:
            est = _make_estimator(alpha, cfg["l1_ratio"], cfg["max_iter"])
            est.fit(Z, yc)
            return np.asarray(est.coef_, dtype=float)
        # Bagging: promedio de coeficientes sobre subsamples (≡ promedio de predicciones).
        # Con subspace_frac<1.0 cada bag ve además un subset aleatorio de MARCADORES (random-subspace /
        # Random-LASSO): decorrelaciona la selección entre bags y reduce la varianza del estimador sparse
        # MÁS que el bagging por muestras solo (los marcadores en LD dejan de competir siempre juntos). El
        # coeficiente final sigue siendo el promedio de predicciones = suma de coef (0 fuera del subset) / n_bags.
        rng = np.random.default_rng(cfg["random_state"])
        n, p = Z.shape
        m = max(2, int(round(cfg["subsample_frac"] * n)))
        subspace = cfg["subspace_frac"] < 1.0
        kf = max(1, int(round(cfg["subspace_frac"] * p)))
        coef = np.zeros(p, dtype=float)
        self.bag_support_ = np.zeros(p, dtype=float)  # frecuencia de selección por marcador (sobre n_bags)
        for _ in range(cfg["n_bags"]):
            idx = rng.choice(n, m, replace=False)
            if subspace:
                fi = rng.choice(p, kf, replace=False)
                est = _make_estimator(alpha, cfg["l1_ratio"], cfg["max_iter"])
                est.fit(Z[np.ix_(idx, fi)], yc[idx])
                c = np.asarray(est.coef_, dtype=float)
                coef[fi] += c
                self.bag_support_[fi] += (np.abs(c) > 1e-12)
            else:
                est = _make_estimator(alpha, cfg["l1_ratio"], cfg["max_iter"])
                est.fit(Z[idx], yc[idx])
                c = np.asarray(est.coef_, dtype=float)
                coef += c
                self.bag_support_ += (np.abs(c) > 1e-12)
        self.bag_support_ /= cfg["n_bags"]
        return coef / cfg["n_bags"]

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseGenomicRegressor":
        cfg = self._resolve_cfg()
        self.cfg_ = cfg
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            raise ValueError("SparseGenomicRegressor.fit recibió y vacío.")

        Z, self.col_mean_, self.col_std_ = standardize_train(X)
        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_

        forced = self.params.get("alpha", cfg["alpha"])
        self.alpha_ = float(forced) if forced is not None else self._select_alpha(Z, yc, cfg)
        self.coef_ = self._fit_coef(Z, yc, self.alpha_, cfg)
        self.n_selected_ = int(np.sum(np.abs(self.coef_) > 1e-12))
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("SparseGenomicRegressor.predict llamado antes de fit.")
        Z = apply_standardize(np.asarray(X, dtype=float), self.col_mean_, self.col_std_)
        return self.y_mean_ + Z @ self.coef_

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "coef_"):
            return {}
        out: dict[str, Any] = {
            "alpha": getattr(self, "alpha_", None),
            "n_selected": getattr(self, "n_selected_", None),
            "l1_ratio": self.cfg_["l1_ratio"],
            "bagging": int(self.cfg_["bagging"]),
            "n_bags": self.cfg_["n_bags"] if self.cfg_["bagging"] else 1,
            "subspace_frac": self.cfg_["subspace_frac"],
        }
        if hasattr(self, "bag_support_"):
            out["n_markers_freq50"] = int(np.sum(self.bag_support_ >= 0.5))
        return out
