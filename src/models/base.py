"""Interfaz común de modelos.

Todo modelo del proyecto respeta `fit(X, y) -> self` y `predict(X) -> np.ndarray`.
Esto permite que el harness (`src/training/run.py`) trate a cualquier modelo de
forma uniforme.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Clase base abstracta. Recibe el sub-dict `model` del config."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.params: dict[str, Any] = self.config.get("params", {}) if isinstance(self.config, dict) else {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Ajusta el modelo SOLO con datos de train. Devuelve self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice un vector (n_samples,) de valores continuos."""

    # --- Persistencia -------------------------------------------------------
    # Default: serializa el objeto entero con joblib (sirve para modelos
    # sklearn/numpy como los baselines). Los modelos torch
    # deben overridear `save`/`load` para guardar `state_dict` + arquitectura.
    # `src/training/save_model.py` los usa para los wandb Artifacts.

    def save(self, path: str | Path) -> Path:
        """Persiste el modelo entrenado en `path`. Devuelve la ruta escrita."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Carga un modelo persistido con `save`. Inversa del default joblib."""
        import joblib

        return joblib.load(path)
