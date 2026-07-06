"""MoE hard con gating por metric learning contrastivo (proyector lineal + triplet loss) en vez de PCA.

Variante del `moe_multitask` (PCA+GMM, hard top-1) con UN cambio: el espacio de ruteo se arma con un
**proyector lineal de una sola capa entrenado con triplet loss** (`ContrastiveProjector`) que fuerza a
juntar a las líneas emparentadas (hermanas) y alejar a las de familias distintas, usando el parentesco
genómico ``G`` para definir positivos/negativos. El GMM clusteriza ese espacio latente. Los expertos
ven los SNPs completos. Es la alternativa "deep learning superficial" al PCA/AE para el ruteo.

Anti-leakage (regla 1): el proyector (y su ``G``, estandarización) y el GMM se ajustan SOLO con el
train del fold; en inferencia val/test se proyectan con la matriz lineal aprendida (solo ``X``).

Implementación: subclase de `MixtureOfExpertsMultitaskRegressor` que solo overridea `_build_gating`
(proyector contrastivo en vez de PCA). Hereda todo lo demás; el ``pre_`` guardado incluye la matriz
lineal ``W_`` del proyector (numpy ⇒ joblib-serializable, sin torch en el artifact).
"""
from __future__ import annotations

from typing import Any

from src.data.preprocess import PreprocChain, SNPPreprocessor
from src.models.contrastive_gating import ContrastiveProjector
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor

# Defaults del proyector contrastivo; los pisa `model.params.gating.contrastive.*`.
_CONTRASTIVE_DEFAULTS: dict[str, Any] = {
    "latent_dim": 20,
    "n_pos": 5,
    "n_neg": 20,
    "margin": 1.0,
    "epochs": 150,
    "lr": 1e-2,
}


class MoEContrastiveMultitaskRegressor(MixtureOfExpertsMultitaskRegressor):
    """MoE hard con gating contrastivo (proyector lineal + triplet loss sobre G) + GMM, hard top-1.

    `params.gating` (además de los del hard MoE): sub-dict ``contrastive`` con ``latent_dim`` (20),
    ``n_pos`` (5), ``n_neg`` (20), ``margin`` (1.0), ``epochs`` (150), ``lr`` (0.01). El resto (``k``,
    ``covariance_type``, ``min_cluster_size``, ...) y ``params.expert`` son idénticos a `moe_multitask`.
    ``pca_n_components`` se ignora (lo reemplaza ``latent_dim`` del proyector).
    """

    def _build_gating(self) -> Any:
        """Cadena del gating: [z-score] -> proyector contrastivo. El GMM va después, en el MoE base."""
        g = self.gating_params
        c = {**_CONTRASTIVE_DEFAULTS, **dict(g.get("contrastive") or {})}
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(
            ContrastiveProjector(
                latent_dim=int(c["latent_dim"]),
                n_pos=int(c["n_pos"]),
                n_neg=int(c["n_neg"]),
                margin=float(c["margin"]),
                epochs=int(c["epochs"]),
                lr=float(c["lr"]),
                random_state=int(g["random_state"]),
            )
        )
        return PreprocChain(steps)

    def get_fitted_params(self) -> dict[str, Any]:
        out = dict(super().get_fitted_params())
        if out:
            out["moe_contrastive"] = 1
        return out
