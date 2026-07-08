"""Ensemble de seeds de la red multitask: entrena K redes y promedia sus predicciones.

El Pearson de una sola seed en el inner-val es ruidoso, así que un sweep que elige HP
por ese número tiende a overfittear el inner-val. Evaluar cada trial como un ensemble
de K seeds estabiliza la métrica, y el modelo final desplegado es ese mismo ensemble.

Envuelve un modelo multitarea inner (por default mlp_multitask, configurable por
inner_name a cualquier subclase de MLPMultitaskRegressor del REGISTRY, p.ej.
moe_soft_multitask). Entrena n_models copias con seeds random_state + i*seed_step y
promedia las predicciones estandarizadas por columna: la loss Pearson deja la escala
de cada red arbitraria, así que sin estandarizar la de mayor varianza dominaría. La
estandarización usa los estadísticos de las predicciones de train de cada sub-modelo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.mlp_multitask import MLPMultitaskRegressor


class EnsembleMultitaskRegressor(BaseModel):
    """Ensemble de n_models MLPMultitaskRegressor (promedio z-scoreado por columna).

    params:
        inner      : params del modelo inner (arquitectura, loss, loss_weights,
                     random_state base). Es lo que barre el sweep.
        inner_name : nombre en el REGISTRY del modelo a ensemblar (default mlp_multitask;
                     debe ser subclase de MLPMultitaskRegressor).
        n_models   : cantidad de seeds del ensemble (default 5).
        seed_step  : separación entre seeds; seed_i = random_state + i*seed_step (default 1000).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.inner_params: dict[str, Any] = dict(self.params.get("inner", {}) or {})
        self.inner_name: str = str(self.params.get("inner_name", "mlp_multitask"))
        self.n_models: int = int(self.params.get("n_models", 5))
        self.seed_step: int = int(self.params.get("seed_step", 1000))

    def _resolve_inner_cls(self) -> type[MLPMultitaskRegressor]:
        """Clase inner del ensemble, resuelta por nombre desde el REGISTRY.

        Import perezoso del registry para evitar el ciclo (registry importa este módulo).
        Se exige subclase de MLPMultitaskRegressor porque el promedio z-scoreado asume la
        interfaz multitarea fit(X, Y, mask) / predict -> (n, T).
        """
        from src.models.registry import REGISTRY

        try:
            cls = REGISTRY[self.inner_name]
        except KeyError as exc:
            raise ValueError(
                f"inner_name '{self.inner_name}' no está en REGISTRY {sorted(REGISTRY)}."
            ) from exc
        if not issubclass(cls, MLPMultitaskRegressor):
            raise ValueError(
                f"inner_name '{self.inner_name}' ({cls.__name__}) debe ser subclase de "
                "MLPMultitaskRegressor (interfaz multitarea fit(X,Y,mask)/predict->(n,T))."
            )
        return cls


    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "EnsembleMultitaskRegressor":
        ctx = self._fit_context
        inner_cls = self._resolve_inner_cls()
        base_seed = int(self.inner_params.get("random_state", 575))
        self.models_: list[MLPMultitaskRegressor] = []
        self.scalers_: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n_models):
            p = dict(self.inner_params)
            p["random_state"] = base_seed + i * self.seed_step
            m = inner_cls({"params": p})
            m.set_fit_context(
                year=ctx.get("year"), line_id=ctx.get("line_id"), target_names=ctx.get("target_names")
            )
            m.fit(X, Y, Y_mask)
            ptr = self._as_2d(m.predict(X))
            mu = ptr.mean(axis=0, keepdims=True)
            sd = ptr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            self.models_.append(m)
            self.scalers_.append((mu, sd))
        self.history_ = getattr(self.models_[0], "history_", [])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "models_", None):
            raise RuntimeError("EnsembleMultitaskRegressor.predict llamado antes de fit.")
        acc: np.ndarray | None = None
        for m, (mu, sd) in zip(self.models_, self.scalers_):
            z = (self._as_2d(m.predict(X)) - mu) / sd
            acc = z if acc is None else acc + z
        return acc / len(self.models_)

    @staticmethod
    def _as_2d(p: Any) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p.reshape(-1, 1) if p.ndim == 1 else p

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "models_", None):
            return {}
        first = self.models_[0]
        return {
            "ensemble": 1,
            "n_models": int(self.n_models),
            "seed_step": int(self.seed_step),
            "n_params": int(getattr(first, "n_params_", 0) or 0) * self.n_models,
            "inner_name": self.inner_name,
            "inner_hidden_dims": str(self.inner_params.get("hidden_dims")),
        }


    def save(self, path: str | Path) -> Path:
        """Guarda cada sub-red (torch nativo) como archivo hermano + un manifiesto joblib.

        El artifact sube el directorio entero, así que al bajarlo los submodel_i.pt
        quedan junto al manifiesto y load los reencuentra.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        for i, m in enumerate(self.models_):
            fname = f"submodel_{i}.pt"
            m.save(path.parent / fname)
            files.append(fname)
        joblib.dump(
            {
                "config": self.config,
                "inner_name": self.inner_name,
                "n_models": self.n_models,
                "seed_step": self.seed_step,
                "files": files,
                "scalers": self.scalers_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleMultitaskRegressor":
        import joblib

        path = Path(path)
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.inner_name = str(d.get("inner_name", "mlp_multitask"))
        obj.n_models = int(d["n_models"])
        obj.seed_step = int(d["seed_step"])
        inner_cls = obj._resolve_inner_cls()
        obj.models_ = [inner_cls.load(path.parent / f) for f in d["files"]]
        obj.scalers_ = list(d["scalers"])
        obj.history_ = []
        return obj
