"""Interfaz común de modelos.

Todo modelo del proyecto respeta `fit(X, y) -> self` y `predict(X) -> np.ndarray`.
Esto permite que el harness (`src/training/run.py`) trate a cualquier modelo de
forma uniforme.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
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
