"""Registry de modelos: mapea `model.name` del config a una clase.

Agregar un modelo = registrar su clase acá. `train.py` instancia vía
`build_model(config)`, nunca importando clases a mano. Así el modelo se elige
desde el YAML, sin tocar código.
"""
from __future__ import annotations

from typing import Any

from src.models.base import BaseModel
from src.models.dummy import MeanPredictor
from src.models.gblup import GBLUP

REGISTRY: dict[str, type[BaseModel]] = {
    "mean_predictor": MeanPredictor,
    "gblup": GBLUP,
}


def build_model(config: dict[str, Any]) -> BaseModel:
    """Instancia el modelo indicado en `config['model']['name']`."""
    model_cfg = config.get("model", {})
    name = model_cfg.get("name")
    if name is None:
        raise ValueError("Falta `model.name` en el config.")
    if name not in REGISTRY:
        raise KeyError(f"Modelo '{name}' no registrado. Disponibles: {sorted(REGISTRY)}.")
    return REGISTRY[name](model_cfg)
