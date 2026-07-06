"""Registry de modelos: mapea `model.name` del config a una clase.

`build_model` también compone el preprocesamiento: si el config trae un bloque
`preprocess:`, envuelve el modelo en `PreprocessedModel`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.preprocess import build_preprocessor
from src.models.base import BaseModel
from src.models.cbeta_vae import CBetaVAERegressor
from src.models.deep_kernel_attn import DeepKernelAttentionRegressor
from src.models.dummy import MeanPredictor
from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.gblup import GBLUP
from src.models.gblup_residual import GBLUPResidualRegressor
from src.models.mlp import MLPRegressor
from src.models.mlp_multitask import MLPMultitaskRegressor
from src.models.moe_gblup_multitask import GBLUPRoutedMoEMultitaskRegressor
from src.models.moe_gblup_residual_multitask import GBLUPResidualHardMoEMultitaskRegressor
from src.models.moe_gpca_multitask import GKinshipPCAMoEMultitaskRegressor
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor
from src.models.moe_singletask import MixtureOfExpertsSingleTaskRegressor
from src.models.moe_soft_multitask import SoftMoEMultitaskRegressor
from src.models.moe_spectral_multitask import SpectralMoEGBLUPResidualMultitaskRegressor
from src.models.xgb import XGBoostRegressor

REGISTRY: dict[str, type[BaseModel]] = {
    "mean_predictor": MeanPredictor,
    "gblup": GBLUP,
    "gblup_residual": GBLUPResidualRegressor,
    "xgboost": XGBoostRegressor,
    "mlp": MLPRegressor,
    "mlp_multitask": MLPMultitaskRegressor,
    "ensemble_multitask": EnsembleMultitaskRegressor,
    "moe_multitask": MixtureOfExpertsMultitaskRegressor,
    "moe_gblup_multitask": GBLUPRoutedMoEMultitaskRegressor,
    "moe_gblup_residual_multitask": GBLUPResidualHardMoEMultitaskRegressor,
    "moe_gpca_multitask": GKinshipPCAMoEMultitaskRegressor,
    "moe_spectral_multitask": SpectralMoEGBLUPResidualMultitaskRegressor,
    "moe_singletask": MixtureOfExpertsSingleTaskRegressor,
    "moe_soft_multitask": SoftMoEMultitaskRegressor,
    "cbeta_vae": CBetaVAERegressor,
    "deep_kernel_attn": DeepKernelAttentionRegressor,
}

class PreprocessedModel(BaseModel):
    """Compone un preprocesador train-only de SNPs con un modelo interno.

    Respeta la interfaz `fit`/`predict`(sirve para todos los modelos). El
    preprocesamiento (máscara de columnas, stats de escala) **viaja dentro del
    modelo guardado**. `save`/`load` usan el joblib del objeto
    entero (default de `BaseModel`), que persiste `pre` + `inner` juntos.
    """

    def __init__(self, pre: Any, inner: BaseModel) -> None:
        super().__init__(getattr(inner, "config", {}))
        self.pre = pre
        self.inner = inner

    def set_fit_context(self, **context: Any) -> None:
        # Propaga el contexto del fold (year/line_id) al modelo interno: lo usa
        # para su selección de HP temporal-aware. Sin
        # esto el inner no vería los años y caería a K-fold aleatorio
        super().set_fit_context(**context)
        self.inner.set_fit_context(**context)
        # También al preprocesador si lo acepta (p.ej. el autoencoder lo usa para
        # su held-out por línea train-only, sin tocar el val externo del fold).
        pre_setter = getattr(self.pre, "set_fit_context", None)
        if callable(pre_setter):
            pre_setter(**context)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreprocessedModel":
        if hasattr(self.inner, "fit_with_preprocessor"):
            self.inner.fit_with_preprocessor(self.pre, X, y)
            return self
        Xt = self.pre.fit_transform(X)   # fit del preprocesador solo con train
        self.inner.fit(Xt, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.inner.predict(self.pre.transform(X))   # mismas columnas/stats de train

    def get_fitted_params(self) -> dict[str, Any]:
        # Estimaciones del modelo interno (alpha, lambda_0, ...) + del
        # preprocesador (n_components/varianza del PCA) → se logean por fold.
        out = dict(self.inner.get_fitted_params())
        gfp = getattr(self.pre, "get_fitted_params", None)
        if callable(gfp):
            out.update(gfp())
        return out


def build_model(config: dict[str, Any]) -> BaseModel:
    """Instancia el modelo indicado en `config['model']['name']`.

    Si el config trae un bloque `preprocess:`, envuelve el modelo en
    `PreprocessedModel` (preprocesamiento train-only de SNPs, común a todos los
    modelos). El loop de entrenamiento no cambia: sigue llamando `fit`/`predict`.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name")
    if name is None:
        raise ValueError("Falta `model.name` en el config.")
    if name not in REGISTRY:
        raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {sorted(REGISTRY)}.")
    model: BaseModel = REGISTRY[name](model_cfg)

    pre = build_preprocessor(config.get("preprocess"))
    if pre is not None:
        model = PreprocessedModel(pre, model)
    return model
