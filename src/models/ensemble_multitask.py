"""Ensemble de seeds del MLP multitask: entrena K redes y promedia sus predicciones.

Motivación (winner's curse): el Pearson de UNA sola seed en el inner-val es ruidoso, así que un
sweep que selecciona HP por ese número **overfittea el inner-val y no generaliza** (verificado en
este proyecto para el multienv: el mejor single-seed falla la regla de oro). La solución es que
**cada trial del sweep evalúe un ENSEMBLE de K seeds**: el ``pearson_r_GY`` que optimiza el sweep
pasa a ser el del ensemble (estable), y el modelo final desplegado ES ese mismo ensemble. Así la
ventaja que elige el sweep se sostiene.

Envuelve `MLPMultitaskRegressor` (no reimplementa la red). Entrena ``n_models`` copias con seeds
``inner.random_state + i*seed_step`` y promedia las predicciones **estandarizadas por columna**
(la pérdida Pearson deja la escala de cada red arbitraria ⇒ sin estandarizar, la red de mayor
varianza domina y el "ensemble" colapsa a ella). La estandarización usa los estadísticos de las
predicciones de TRAIN de cada sub-modelo (sin leakage, independiente del batch de inferencia).

Se enchufa en `run_multitask_evaluation` (interfaz ``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``)
igual que `mlp_multitask`; el dispatch multitask de `scripts/train.py` lo incluye por nombre.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.mlp_multitask import MLPMultitaskRegressor


class EnsembleMultitaskRegressor(BaseModel):
    """Ensemble de ``n_models`` `MLPMultitaskRegressor` (promedio z-scoreado por columna).

    `params` (sub-dict ``model.params`` del config):
        inner     : dict de params de `MLPMultitaskRegressor` (arquitectura, loss, loss_weights,
                    random_state base, ...). Es lo que barre el sweep (``model.params.inner.*``).
        n_models  : cantidad de seeds del ensemble (default 5).
        seed_step : separación entre seeds; seed_i = inner.random_state + i*seed_step (default 1000).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.inner_params: dict[str, Any] = dict(self.params.get("inner", {}) or {})
        self.n_models: int = int(self.params.get("n_models", 5))
        self.seed_step: int = int(self.params.get("seed_step", 1000))

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "EnsembleMultitaskRegressor":
        ctx = self._fit_context
        base_seed = int(self.inner_params.get("random_state", 575))
        self.models_: list[MLPMultitaskRegressor] = []
        self.scalers_: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.n_models):
            p = dict(self.inner_params)
            p["random_state"] = base_seed + i * self.seed_step
            m = MLPMultitaskRegressor({"params": p})
            # Solo propaga target_names (alinea loss_weights con las columnas de Y) + year/line_id;
            # NO el val de monitoreo (eval_*): el ensemble no lo necesita y así es más rápido.
            m.set_fit_context(
                year=ctx.get("year"), line_id=ctx.get("line_id"), target_names=ctx.get("target_names")
            )
            m.fit(X, Y, Y_mask)
            # Estadísticos de estandarización por columna desde las predicciones de TRAIN (sin leakage).
            ptr = self._as_2d(m.predict(X))
            mu = ptr.mean(axis=0, keepdims=True)
            sd = ptr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            self.models_.append(m)
            self.scalers_.append((mu, sd))
        # Expone la learning curve de la primera red (para que run.py la loguee).
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
            "inner_hidden_dims": str(self.inner_params.get("hidden_dims")),  # no-numérico => metadata
        }

    # --- Persistencia (override: cada sub-modelo es torch) ------------------

    def save(self, path: str | Path) -> Path:
        """Guarda cada sub-red (torch nativo) como archivo hermano + un manifiesto joblib.

        `save_model_artifact` sube el directorio entero, así que al bajar el artifact los
        ``submodel_i.pt`` quedan junto al manifiesto y `load` los reencuentra.
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
        obj.n_models = int(d["n_models"])
        obj.seed_step = int(d["seed_step"])
        obj.models_ = [MLPMultitaskRegressor.load(path.parent / f) for f in d["files"]]
        obj.scalers_ = list(d["scalers"])
        obj.history_ = []
        return obj
