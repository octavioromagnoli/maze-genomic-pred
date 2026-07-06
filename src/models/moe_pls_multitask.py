"""MoE hard con gating PLS (supervisado por fenotipo) en vez de PCA.

Variante del `moe_multitask` (PCA+GMM, hard top-1) con UN cambio: el espacio de ruteo se arma con
**PLS** (Partial Least Squares) en vez de PCA. El PCA es no supervisado (solo mira ``X``); el PLS
busca los componentes que maximizan la **covarianza entre los SNPs ``X`` y el fenotipo ``Y``**, así el
espacio latente queda orientado a separar las líneas por potencial fenotípico (rendimiento) — no solo
por la mayor varianza genómica. El GMM clusteriza ese espacio PLS. Los expertos ven los SNPs completos.

Supervisión: se usan TODAS las cabezas de ``Y`` (multi-target PLS), con los faltantes imputados por la
media de train de cada columna (train-only) — tras el centrado de PLS esos entries aportan covarianza
~0, así que no sesgan. Anti-leakage (regla 1): PLS y GMM se ajustan SOLO con el train del fold; en
inferencia val/test se proyectan sobre las MISMAS direcciones PLS usando SOLO ``X`` (nunca fenotipos
de test).

Implementación: subclase de `MixtureOfExpertsMultitaskRegressor` que overridea `_build_gating`
(PLS en vez de PCA) y `_fit_gating` (pasa la supervisión ``Y`` a la cadena). Hereda todo lo demás
(fit/predict/save/load; el ``pre_`` guardado incluye el PLS ajustado, sklearn ⇒ joblib-serializable).
Se enchufa en `run_multitask_evaluation` igual que `moe_multitask` (incl. `data.multi_trait_env`).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.preprocess import PLSReducer, PreprocChain, SNPPreprocessor
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor


class MoEPLSMultitaskRegressor(MixtureOfExpertsMultitaskRegressor):
    """MoE hard con gating PLS supervisado por fenotipo (+ GMM, hard top-1).

    `params.gating` (además de los del hard MoE): ``pls_n_components`` (default 30; nº de componentes
    latentes del PLS = dimensión del espacio de ruteo). El resto (``k``, ``covariance_type``,
    ``min_cluster_size``, ...) y ``params.expert`` son idénticos a `moe_multitask`. ``pca_n_components``
    se ignora (lo reemplaza ``pls_n_components``).
    """

    def _pls_n_components(self) -> int:
        g = self.gating_params
        return int(g.get("pls_n_components", 30))

    def _build_gating(self) -> Any:
        """Cadena del gating: [z-score] -> PLS (supervisado). El GMM va después, en el MoE base."""
        g = self.gating_params
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(PLSReducer(n_components=self._pls_n_components(), scale=True))
        return PreprocChain(steps)

    def _fit_gating(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray) -> np.ndarray:
        """Ajusta el gating PLS con supervisión = TODAS las cabezas de Y (faltantes imputados por la
        media de train de la columna ⇒ covarianza ~0 tras el centrado del PLS)."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        M = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        # Imputación train-only por columna (media de los datos reales) para el PLS.
        Y_sup = Y.copy()
        for t in range(Y.shape[1]):
            col = M[:, t]
            fill = float(Y[col, t].mean()) if col.any() else 0.0
            Y_sup[~col, t] = fill
        self.pre_ = self._build_gating()
        return self.pre_.fit_transform(X, Y_sup)   # la y fluye hasta el PLSReducer en la cadena

    def get_fitted_params(self) -> dict[str, Any]:
        out = dict(super().get_fitted_params())
        if out:
            out["moe_pls"] = 1
        return out
