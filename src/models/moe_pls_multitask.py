"""MoE hard con gating PLS (supervisado por fenotipo) en vez de PCA.

Igual que moe_multitask pero el espacio de ruteo se arma con PLS en vez de PCA.
El PCA es no supervisado; el PLS busca los componentes que maximizan la covarianza
entre los SNPs y el fenotipo, de modo que el espacio latente separa las líneas por
potencial fenotípico y no solo por varianza genómica. El GMM clusteriza ese espacio
y los expertos ven los SNPs completos.

El PLS usa todas las cabezas de Y (los faltantes se imputan por la media de train
de cada columna; tras el centrado del PLS aportan covarianza ~0). PLS y GMM se
ajustan con el train del fold y en inferencia se proyecta val/test sobre las mismas
direcciones usando solo X.

Es una subclase de MixtureOfExpertsMultitaskRegressor que overridea _build_gating y
_fit_gating; hereda el resto.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.preprocess import PLSReducer, PreprocChain, SNPPreprocessor
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor


class MoEPLSMultitaskRegressor(MixtureOfExpertsMultitaskRegressor):
    """MoE hard con gating PLS supervisado por fenotipo (+ GMM, hard top-1).

    En params.gating agrega pls_n_components (default 30, la dimensión del espacio
    de ruteo) y reemplaza a pca_n_components. El resto de los params son los del
    moe_multitask.
    """

    def _pls_n_components(self) -> int:
        g = self.gating_params
        return int(g.get("pls_n_components", 30))

    def _build_gating(self) -> Any:
        """Cadena del gating: [z-score] -> PLS. El GMM va después, en el MoE base."""
        g = self.gating_params
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(PLSReducer(n_components=self._pls_n_components(), scale=True))
        return PreprocChain(steps)

    def _fit_gating(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray) -> np.ndarray:
        """Ajusta el PLS con todas las cabezas de Y; imputa los faltantes por la
        media de train de cada columna (covarianza ~0 tras el centrado del PLS)."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        M = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        Y_sup = Y.copy()
        for t in range(Y.shape[1]):
            col = M[:, t]
            fill = float(Y[col, t].mean()) if col.any() else 0.0
            Y_sup[~col, t] = fill
        self.pre_ = self._build_gating()
        return self.pre_.fit_transform(X, Y_sup)

    def get_fitted_params(self) -> dict[str, Any]:
        out = dict(super().get_fitted_params())
        if out:
            out["moe_pls"] = 1
        return out
