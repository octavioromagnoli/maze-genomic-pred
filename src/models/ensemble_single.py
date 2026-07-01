"""Ensemble de seeds de un modelo single-task (p.ej. XGBoost): entrena K y promedia.

Mismo motivo que `ensemble_multitask` pero para la ruta single-task: el Pearson de UNA seed en
el inner-val es ruidoso, así que un sweep que selecciona HP por ese número **overfittea el
inner-val** (winner's curse). Acá cada trial del sweep evalúa un ENSEMBLE de K seeds: el
``pearson_r_mean`` que optimiza el sweep pasa a ser el del ensemble (estable), y el modelo final
desplegado ES ese ensemble.

Envuelve cualquier modelo del registry (por ``inner_name``, p.ej. ``"xgboost"``): entrena
``n_models`` copias con seeds ``inner.random_state + i*seed_step`` y promedia sus predicciones.
Para modelos CALIBRADOS (XGBoost, loss MSE) usa promedio crudo (``standardize: false``, default),
que preserva la escala (RMSE/R² siguen informativos); para modelos de escala arbitraria (red con
loss pearson) poné ``standardize: true`` (z-score por modelo con stats de TRAIN antes de promediar).

Single-task ⇒ se enchufa en `run_evaluation` (``fit(X, y)`` / ``predict(X) -> (n,)``) como
cualquier otro modelo. El preprocesamiento (p.ej. filtro MAF) va en el bloque ``preprocess:`` del
config y `build_model` lo aplica UNA vez a nivel del ensemble (cada sub-modelo recibe X ya filtrado).
Persistencia: el default joblib de `BaseModel` sirve si el inner es picklable (XGBoost lo es).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import BaseModel


class EnsembleRegressor(BaseModel):
    """Ensemble de ``n_models`` modelos single-task del registry (distintas seeds, promedio).

    `params` (sub-dict ``model.params`` del config):
        inner_name : clave del registry del modelo interno (default ``"xgboost"``).
        inner      : dict de params del modelo interno (lo que barre el sweep: ``inner.*``).
        n_models   : cantidad de seeds del ensemble (default 5).
        seed_step  : separación entre seeds; seed_i = inner.random_state + i*seed_step (default 1000).
        standardize: z-score por modelo (stats de train) antes de promediar (default False).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.inner_name: str = str(self.params.get("inner_name", "xgboost"))
        self.inner_params: dict[str, Any] = dict(self.params.get("inner", {}) or {})
        self.n_models: int = int(self.params.get("n_models", 5))
        self.seed_step: int = int(self.params.get("seed_step", 1000))
        self.standardize: bool = bool(self.params.get("standardize", False))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleRegressor":
        # Import tardío: registry.py importa este módulo, así que importar REGISTRY arriba sería
        # circular. Acá (en fit) el registry ya está completamente construido.
        from src.models.registry import REGISTRY

        cls = REGISTRY[self.inner_name]
        base_seed = int(self.inner_params.get("random_state", 575))
        self.models_: list[BaseModel] = []
        self.scalers_: list[tuple[float, float]] = []
        for i in range(self.n_models):
            p = dict(self.inner_params)
            p["random_state"] = base_seed + i * self.seed_step
            m = cls({"params": p})
            m.set_fit_context(**self._fit_context)  # year/line_id (early stopping del inner), etc.
            m.fit(X, y)
            if self.standardize:
                ptr = np.asarray(m.predict(X), dtype=float).ravel()
                mu = float(ptr.mean())
                sd = float(ptr.std())
                sd = sd if sd > 1e-12 else 1.0
            else:
                mu, sd = 0.0, 1.0
            self.models_.append(m)
            self.scalers_.append((mu, sd))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "models_", None):
            raise RuntimeError("EnsembleRegressor.predict llamado antes de fit.")
        acc: np.ndarray | None = None
        for m, (mu, sd) in zip(self.models_, self.scalers_):
            z = (np.asarray(m.predict(X), dtype=float).ravel() - mu) / sd
            acc = z if acc is None else acc + z
        return acc / len(self.models_)

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "models_", None):
            return {}
        return {
            "ensemble": 1,
            "n_models": int(self.n_models),
            "seed_step": int(self.seed_step),
            "inner_name": self.inner_name,
        }
