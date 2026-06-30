"""Modelo de DATA AUGMENTATION con el C-β-VAE, sweepable como cualquier otro modelo.

`AugmentedRegressor` envuelve todo el pipeline de augmentation dentro de la interfaz
común (`fit`/`predict`), así entra a `run_evaluation` → `scripts/train.py` → `wandb
sweep` sin un entrypoint nuevo. En cada `fit` (es decir, en cada fold/trial):

1. Entrena un **C-β-VAE generador** SOLO con el train del fold (regla 1).
2. Genera ``n_synth`` pares (X, y) sintéticos (posterior-jitter conservando el label
   real, o prior-sampling con label de la cabeza de regresión).
3. Reentrena un modelo **downstream configurable** (cualquiera del registry: `mlp`,
   `gblup`, ...) con train real + sintéticos.

`predict` delega en el downstream. Como es un modelo más, el sweep
`configs/sweep_cbeta_vae_augment.yaml` puede barrer a la vez los HP del generador, los
de la generación (estrategia, cantidad, jitter) y los del downstream, optimizando
`pearson_r_mean` en el val del fold (selección honesta en 2018). Después, en el
notebook, el config ganador se reentrena con ``train_years=[2017, 2018]`` y
``val_year=2019`` (reporte): el modelo entrena el generador con 2017+2018, agrega los
sintéticos y testea en 2019.

Reglas: el generador y los sintéticos se derivan SOLO del train del fold (regla 1); las
filas sintéticas reciben `line_id` centinela que nunca cruza a val (regla 2); la entrada
queda binaria para la BCE del generador (regla 4); el sweep optimiza Pearson (regla 5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.eval.augment import build_augmented_train, synthetic_pairs
from src.models.base import BaseModel
from src.models.cbeta_vae import CBetaVAERegressor

# Defaults del bloque `augment` (cantidad/estrategia de la generación).
_AUG_DEFAULTS: dict[str, Any] = {
    "strategy": "posterior",   # "posterior" (label real) | "prior" (label de la reg-head)
    "n_synth_mult": 1.0,        # n_synth = round(mult * n_train); 0 => sin sintéticos (baseline)
    "jitter": 1.0,              # escala del ruido latente en posterior-jitter
    "temperature": 1.0,         # escala de la prior en prior-sampling
    "binarize": "bernoulli",    # "bernoulli" | "threshold"
    "seed": 575,
}


class AugmentedRegressor(BaseModel):
    """Generador C-β-VAE + downstream reentrenado con train + sintéticos.

    `params` (sub-dict `model.params` del config):
        generator : dict de params del `CBetaVAERegressor` (latent_dim, beta, lambda_aux, ...).
        downstream: ``{"name": <registry>, "params": {...}}`` del modelo a reentrenar
            (default ``{"name": "mlp"}``).
        augment   : ``{strategy, n_synth_mult, jitter, temperature, binarize, seed}``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gen_params = dict(self.params.get("generator", {}) or {})
        ds = self.params.get("downstream", {}) or {}
        self.downstream_name = ds.get("name", "mlp")
        self.downstream_params = dict(ds.get("params", {}) or {})
        self.aug = {**_AUG_DEFAULTS, **(self.params.get("augment", {}) or {})}

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AugmentedRegressor":
        from src.models.registry import REGISTRY  # lazy: evita el ciclo registry<->augmented

        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        ctx = dict(self._fit_context)
        # year/line_id del train del fold (los pasa run_evaluation). Fallback defensivo si
        # se llama el modelo a mano sin contexto.
        year = ctx.get("year")
        line_id = ctx.get("line_id")
        year = np.asarray(year, dtype=int) if year is not None else np.zeros(len(X), dtype=int)
        line_id = np.asarray(line_id, dtype=object) if line_id is not None else np.arange(len(X)).astype(object)

        # 1) Generador C-β-VAE, ajustado SOLO con el train del fold (regla 1).
        self.generator_ = CBetaVAERegressor({"params": self.gen_params})
        self.generator_.set_fit_context(**ctx)
        self.generator_.fit(X, y)

        # 2) Sintéticos (derivados solo del train).
        mult = float(self.aug["n_synth_mult"])
        n_synth = max(0, int(round(mult * len(X))))
        if n_synth > 0:
            X_syn, y_syn = synthetic_pairs(
                self.generator_,
                strategy=str(self.aug["strategy"]),
                n=n_synth,
                X_ref=X,
                y_ref=y,
                jitter=float(self.aug["jitter"]),
                temperature=float(self.aug["temperature"]),
                binarize=str(self.aug["binarize"]),
                seed=int(self.aug["seed"]),
            )
            X_aug, y_aug, line_aug, year_aug = build_augmented_train(X, y, line_id, year, X_syn, y_syn)
        else:  # sin sintéticos => baseline downstream con el train real
            X_aug, y_aug, line_aug, year_aug = X, y, line_id, year
        self.n_synth_ = int(n_synth)

        # 3) Downstream reentrenado con train real + sintéticos.
        if self.downstream_name not in REGISTRY:
            raise KeyError(
                f"downstream '{self.downstream_name}' no está en el registry: {sorted(REGISTRY)}."
            )
        self.downstream_ = REGISTRY[self.downstream_name]({"params": self.downstream_params})
        down_ctx = dict(ctx)
        down_ctx["year"] = year_aug          # años del train aumentado (sintéticos = último año real)
        down_ctx["line_id"] = line_aug       # line_id centinela en las filas sintéticas
        self.downstream_.set_fit_context(**down_ctx)
        self.downstream_.fit(X_aug, y_aug)

        # La curva que loguea run.py: la del downstream (el predictor final).
        self.history_ = getattr(self.downstream_, "history_", []) or getattr(self.generator_, "history_", [])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "downstream_"):
            raise RuntimeError("AugmentedRegressor.predict llamado antes de fit.")
        return self.downstream_.predict(np.asarray(X))

    def get_fitted_params(self) -> dict[str, Any]:
        """Diagnóstico: cantidad de sintéticos + estimaciones del generador y del downstream."""
        out: dict[str, Any] = {
            "n_synth": getattr(self, "n_synth_", 0),
            "n_synth_mult": float(self.aug["n_synth_mult"]),
            "aug_strategy_prior": int(str(self.aug["strategy"]).lower() == "prior"),
        }
        if hasattr(self, "generator_"):
            for k, v in self.generator_.get_fitted_params().items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"gen_{k}"] = v
        if hasattr(self, "downstream_"):
            for k, v in self.downstream_.get_fitted_params().items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"down_{k}"] = v
        return out

    # --- Persistencia (generador torch + downstream según su formato) -------

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.generator_.save(path.parent / "generator.pt")
        self.downstream_.save(path.parent / "downstream.bin")
        joblib.dump(
            {"config": self.config, "downstream_name": self.downstream_name, "n_synth_": getattr(self, "n_synth_", 0)},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AugmentedRegressor":
        import joblib

        from src.models.registry import REGISTRY

        path = Path(path)
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.generator_ = CBetaVAERegressor.load(path.parent / "generator.pt")
        obj.downstream_ = REGISTRY[d["downstream_name"]].load(path.parent / "downstream.bin")
        obj.n_synth_ = d.get("n_synth_", 0)
        obj.history_ = getattr(obj.downstream_, "history_", [])
        return obj
