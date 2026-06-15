"""Registry de modelos: mapea `model.name` del config a una clase.

`build_model` también compone el preprocesamiento: si el config trae un bloque
`preprocess:`, envuelve el modelo en `PreprocessedModel`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.preprocess import SNPPreprocessor, build_preprocessor
from src.models.base import BaseModel
from src.models.dummy import MeanPredictor

REGISTRY: dict[str, type[BaseModel]] = {
    "mean_predictor": MeanPredictor,
}

class PreprocessedModel(BaseModel):
    """Compone un preprocesador train-only de SNPs con un modelo interno.

    Respeta la interfaz `fit`/`predict`(sirve para todos los modelos). El
    preprocesamiento (máscara de columnas, stats de escala) **viaja dentro del
    modelo guardado**. `save`/`load` usan el joblib del objeto
    entero (default de `BaseModel`), que persiste `pre` + `inner` juntos.
    """

    def __init__(self, pre: SNPPreprocessor, inner: BaseModel) -> None:
        super().__init__(getattr(inner, "config", {}))
        self.pre = pre
        self.inner = inner

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PreprocessedModel":
        Xt = self.pre.fit_transform(X)   # fit del preprocesador solo con train
        self.inner.fit(Xt, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.inner.predict(self.pre.transform(X))   # mismas columnas/stats de train


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
