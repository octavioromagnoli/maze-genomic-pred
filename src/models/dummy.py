"""Predictor por la media, como piso de comparación.

fit guarda la media de y_train y predict la devuelve para toda fila.
"""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModel


class MeanPredictor(BaseModel):
    """Predice siempre la media del target de train."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanPredictor":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("MeanPredictor.fit recibió y vacío.")
        self.mean_: float = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("MeanPredictor.predict llamado antes de fit.")
        n = np.asarray(X).shape[0]
        return np.full(n, self.mean_, dtype=float)
