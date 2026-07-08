"""Interfaz común de modelos.

Todo modelo del proyecto respeta fit(X, y) -> self y predict(X) -> np.ndarray.
Esto permite que el harness (src/training/run.py) trate a cualquier modelo de
forma uniforme.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Clase base abstracta. Recibe el sub-dict model del config."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.params: dict[str, Any] = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        self._fit_context: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Ajusta el modelo con el train del fold. Devuelve self."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice un vector (n_samples,) de valores continuos."""


    def set_fit_context(self, **context: Any) -> None:
        """Metadata del fold (year, line_id de las filas de train).

        La fija run_evaluation antes de fit. Sirve para que un modelo respete la
        estructura temporal en su selección interna de hiperparámetros (CV
        temporal 2017-2018 en vez de K-fold aleatorio). Los que no la usan la ignoran.
        """
        self._fit_context = context

    def get_fitted_params(self) -> dict[str, Any]:
        """Hiperparámetros/estimaciones recuperadas en el ajuste.

        Vacío por default; los modelos que estiman algo (lambda_0, varianzas) lo
        overridean. Se loguea a wandb y se guarda en la metadata del artifact.
        """
        return {}


    def save(self, path: str | Path) -> Path:
        """Persiste el modelo entrenado en path. Devuelve la ruta escrita."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Carga un modelo persistido con save. Inversa del default joblib."""
        import joblib

        return joblib.load(path)
