"""MoE hard con LD Pruning antes del PCA en el gating.

Variante del `moe_multitask` (PCA+GMM, hard top-1) con UN cambio: antes de la PCA del gating se poda
el desequilibrio de ligamiento (LD) de los SNPs de train (`LDPruner`), y la **PCA y el GMM se ajustan
exclusivamente sobre la matriz podada**. Los expertos siguen viendo los SNPs COMPLETOS (la poda solo
afecta el espacio de ruteo, igual que la PCA/GMM en el hard MoE original).

Motivación: los SNPs en fuerte LD son redundantes y el PCA sobre la matriz cruda queda dominado por
bloques de marcadores co-heredados (estructura local repetida), no por la estructura de subpoblación
que interesa para el ruteo. Podar por LD (r² > umbral en una ventana deslizante) deja un conjunto
casi independiente ⇒ las componentes principales capturan mejor la estructura poblacional.

Anti-leakage (regla 1): la máscara de SNPs conservados se aprende SOLO con el train del fold
(`LDPruner.fit`) y se aplica idéntica a val/test (`transform` selecciona las mismas columnas), así las
dimensiones coinciden al proyectar en el PCA. Todo (poda, PCA, GMM) es train-only.

Implementación: **subclase de `MixtureOfExpertsMultitaskRegressor`** que solo overridea `_build_gating`
para anteponer `LDPruner`. Hereda fit/predict/save/load/get_fitted_params (el `pre_` guardado incluye
la máscara del `LDPruner`, numpy ⇒ joblib-serializable; sus stats se logean por fold). Se enchufa en
`run_multitask_evaluation` igual que `moe_multitask` (incl. `data.multi_trait_env` para 6 cabezas).
"""
from __future__ import annotations

from typing import Any

from src.data.preprocess import LDPruner, PCAReducer, PreprocChain, SNPPreprocessor
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor

# Defaults de la poda LD; los pisa `model.params.gating.ld_prune.*` del config/sweep.
_LD_DEFAULTS: dict[str, Any] = {
    "window_size": 50,     # ventana deslizante en nº de SNPs
    "r2_threshold": 0.5,   # r² por encima del cual un par es redundante
}


class MoELDPruneMultitaskRegressor(MixtureOfExpertsMultitaskRegressor):
    """MoE hard (PCA+GMM) con LD Pruning train-only antes del PCA del gating.

    `params.gating` (además de los del hard MoE): sub-dict ``ld_prune`` con ``window_size`` (default
    50) y ``r2_threshold`` (default 0.5). El resto (``pca_n_components``, ``k``, ``covariance_type``,
    ``min_cluster_size``, ...) y ``params.expert`` son idénticos a `moe_multitask`.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        ld = dict(self.gating_params.get("ld_prune") or {})
        self.gating_params["ld_prune"] = {
            "window_size": int(ld.get("window_size", _LD_DEFAULTS["window_size"])),
            "r2_threshold": float(ld.get("r2_threshold", _LD_DEFAULTS["r2_threshold"])),
        }

    def _build_gating(self) -> Any:
        """Cadena de featurización del gating: [LD pruning] -> [z-score] -> PCA (todo train-only)."""
        g = self.gating_params
        ld = g["ld_prune"]
        steps: list = [
            LDPruner(window_size=int(ld["window_size"]), r2_threshold=float(ld["r2_threshold"]))
        ]
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
        return PreprocChain(steps)
