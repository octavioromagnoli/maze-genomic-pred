"""Stacking aditivo GBLUP + red sobre el residual.

GBLUP captura la señal lineal (efectos de marcador aditivos); una red predice lo que
GBLUP deja sin explicar, el residual ``y - ŷ_GBLUP``. La predicción final es la suma
``ŷ_GBLUP + ŷ_red``. Es un stacking ADITIVO ENTRENADO (la red corrige a GBLUP), distinto
del blend convexo post-hoc ``α·MLP + (1-α)·GBLUP`` de ``src/eval/ensemble.py`` (la "A3").

El GBLUP se reentrena por fold con el train del fold y el residual se calcula sobre ese
mismo train (in-sample): no es leakage (el val externo nunca entra a ningún ajuste), es
optimismo estadístico moderado — en p≫n GBLUP es un shrinker fuerte y su residual
in-sample conserva señal real. Todo lo controla ``cv.train_years``/``cv.val_year`` del
config: si train=2017→val 2018 entrena el GBLUP en 2017; si train=2017+2018→val 2019
entrena el GBLUP en 2017+2018; la red y la suma se acomodan solas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.gblup import GBLUP
from src.models.mlp import MLPRegressor


class GBLUPResidualRegressor(BaseModel):
    """Compone un GBLUP (base lineal) con un MLP que modela su residual.

    `params` (sub-dict `model.params` del config):
        gblup : dict de params de `GBLUP` (p.ej. ``{lambda_method: reml}``). Default {}.
        mlp   : dict de params de `MLPRegressor` (hidden_dims, lr, loss, ...). Default {}.

    La predicción es ``gblup.predict(X) + mlp.predict(X)``. Como la suma es ADITIVA, la red
    debe predecir el residual en escala calibrada ⇒ el MLP tiene que usar loss "mse" o
    "pearson_mse", NUNCA "pearson" pura (invariante a afines ⇒ descalibra la suma).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        gblup_params = self.params.get("gblup", {}) or {}
        mlp_params = self.params.get("mlp", {}) or {}
        self.gblup = GBLUP({"params": dict(gblup_params)})
        self.mlp = MLPRegressor({"params": dict(mlp_params)})

    # --- Contexto del fold --------------------------------------------------

    def set_fit_context(self, **context: Any) -> None:
        # Guarda el contexto del fold y lo propaga al GBLUP (usa `year` para su CV de
        # lambda). El contexto del MLP se arma en `fit`, una vez entrenado el GBLUP: su
        # monitoreo necesita el residual del val (eval_y - gblup.predict(eval_X)).
        super().set_fit_context(**context)
        self.gblup.set_fit_context(**context)

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBLUPResidualRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # 1) Base lineal: GBLUP ajustado SOLO con el train del fold (regla 1).
        self.gblup.fit(X, y)

        # 2) Residual in-sample del train = target de la red (lo no-lineal que GBLUP no
        #    explicó). El val externo no interviene acá (sin leakage).
        r = y - self.gblup.predict(X)
        self.resid_std_ = float(np.std(r))

        # 3) Contexto del MLP: copia el del fold y, si run_evaluation pasó el val para
        #    MONITOREAR la curva por época, lo convierte a residual del val
        #    (eval_y - gblup.predict(eval_X)). GBLUP ajustado en train predice el val ⇒ es
        #    la misma operación que en `predict`, sin leakage; con patience=null el MLP no
        #    selecciona ni restaura sobre el val ⇒ la red final es idéntica.
        mlp_ctx = dict(self._fit_context)
        eval_X = mlp_ctx.get("eval_X")
        eval_y = mlp_ctx.get("eval_y")
        if eval_X is not None and eval_y is not None:
            eval_X = np.asarray(eval_X, dtype=np.float64)
            eval_y = np.asarray(eval_y, dtype=np.float64).ravel()
            mlp_ctx["eval_y"] = eval_y - self.gblup.predict(eval_X)
        self.mlp.set_fit_context(**mlp_ctx)

        # 4) Red sobre el residual (z-score de inputs y del residual train-only adentro
        #    del MLP).
        self.mlp.fit(X, r)

        # 5) Expone la learning curve de la red para que run.py la loguee (history_).
        self.history_ = getattr(self.mlp, "history_", [])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.gblup.predict(X) + self.mlp.predict(X)

    def get_fitted_params(self) -> dict[str, Any]:
        """Estimaciones de la base y de la red (prefijadas) + diagnóstico del residual.

        `run_evaluation` copia los escalares numéricos a las métricas del fold (wandb) y
        `save_model_artifact` preserva también los strings (p.ej. ``gblup_lambda_method``).
        """
        out: dict[str, Any] = {}
        for k, v in self.gblup.get_fitted_params().items():
            out[f"gblup_{k}"] = v
        for k, v in self.mlp.get_fitted_params().items():
            out[f"mlp_{k}"] = v
        if hasattr(self, "resid_std_"):
            out["resid_std"] = self.resid_std_
        return out

    # --- Persistencia (override: GBLUP es numpy/joblib, MLP es torch) -------

    def save(self, path: str | Path) -> Path:
        """Guarda GBLUP (joblib) + MLP (torch nativo) como dos archivos hermanos.

        `save_model_artifact` sube el directorio entero (``model.bin`` + ``mlp.pt``), así
        que al bajar el artifact ambos quedan lado a lado y `load` los reencuentra. No se
        joblib-ea el objeto entero (picklearía el ``nn.Module`` del MLP, justo lo que el
        MLP evita con su propio save/load torch); se delega en el save de cada sub-modelo.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mlp_file = "mlp.pt"
        self.mlp.save(path.parent / mlp_file)  # state_dict + arquitectura + scalers
        joblib.dump(
            {"config": self.config, "gblup": self.gblup, "mlp_file": mlp_file},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GBLUPResidualRegressor":
        import joblib

        path = Path(path)
        d = joblib.load(path)
        obj = cls(d.get("config"))           # reconstruye sub-modelos vacíos…
        obj.gblup = d["gblup"]               # …y los pisa con los ya entrenados
        obj.mlp = MLPRegressor.load(path.parent / d["mlp_file"])
        obj.history_ = getattr(obj.mlp, "history_", [])
        return obj
