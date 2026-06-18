"""XGBoost — gradient boosting de árboles para regresión genómica.

Wrapper fino sobre `xgboost.XGBRegressor` que respeta la interfaz común
(`fit`/`predict`) y se enchufa en `run_evaluation` como cualquier otro modelo.

Decisiones de diseño alineadas con las reglas del proyecto:

- **Sin leakage (regla 1).** No hace preprocesamiento propio: si el YAML trae un
  bloque `preprocess:`, `build_model` lo envuelve en `PreprocessedModel`, que
  ajusta el scaler/máscara MAF solo con el train del fold. Acá `fit` recibe el
  train ya transformado.
- **Early stopping temporal opcional (regla 3).** Con
  `params.early_stopping_rounds`, se reserva el ÚLTIMO año del train como
  inner-val (p.ej. 2017→2018) solo para elegir cuántos árboles usar. Luego, por
  default, se reentrena un modelo final desde cero con TODO el train del fold
  (p.ej. 2017+2018) y `n_estimators=best_iteration+1`. El val/test externo no se
  toca. Si hay un solo año en train, se usa `n_estimators` fijo.
- **Reproducibilidad.** `random_state` se fija desde el config (default 575) para
  controlar el RNG interno de XGBoost, que la seed global de numpy/torch no
  alcanza.

`get_fitted_params` expone `best_iteration`/`best_score` y los HP efectivos para
que `run_evaluation`/`train.py` los logeen a wandb sin bajar el modelo.

Persistencia: `XGBRegressor` es picklable, así que el `save`/`load` por joblib
de `BaseModel` sirve sin override.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.models.base import BaseModel

# Hiperparámetros por defecto: un punto de partida razonable para p >> n
# (árboles poco profundos, learning rate chico, submuestreo de filas/columnas
# como regularización). Se overridean por `model.params` en el YAML.
_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
}

# Claves de `params` que NO son hiperparámetros de XGBoost (control del wrapper).
_WRAPPER_KEYS = {"early_stopping_rounds", "eval_scheme", "random_state", "refit_full_train"}


def temporal_inner_split(years: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Inner-split temporal dentro del train: años previos vs. último año.

    Devuelve ``(idx_inner_train, idx_inner_val)`` con el último año como inner-val,
    o ``None`` si no hay al menos 2 años distintos (no se puede partir por año).
    """
    if years is None:
        return None
    years = np.asarray(years)
    uniq = sorted(int(y) for y in np.unique(years))
    if len(uniq) < 2:
        return None
    vy = uniq[-1]
    return np.flatnonzero(years < vy), np.flatnonzero(years == vy)


class XGBoostRegressor(BaseModel):
    """Gradient boosting (XGBoost) para regresión de un fenotipo.

    `params` (sub-dict `model.params` del config) — además de los HP de
    `xgboost.XGBRegressor` (n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, reg_lambda, reg_alpha, min_child_weight, gamma, ...):
        early_stopping_rounds : int opcional. Si está, activa early stopping con
            un inner-val temporal (último año del train) para elegir el número de
            árboles. Default: sin early stopping.
        eval_scheme           : "temporal" (default) — único esquema soportado para
            el early stopping, coherente con la regla 3.
        refit_full_train      : bool (default True). Si hay early stopping temporal,
            reentrena el modelo final con todo el train externo y el número de
            árboles seleccionado.
        random_state          : seed del RNG interno de XGBoost (default 575).
    """

    @staticmethod
    def _as_xgb_matrix(X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X, dtype=np.float32)

    def _validate_y(self, y: np.ndarray) -> np.ndarray:
        y = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        if y.size == 0:
            raise ValueError("XGBoostRegressor.fit recibió y vacío.")
        return y

    def _effective_hp(self) -> dict[str, Any]:
        # HP del booster: defaults + override del YAML (sin las claves del wrapper).
        hp = {**_DEFAULTS}
        hp.update({k: v for k, v in self.params.items() if k not in _WRAPPER_KEYS})
        hp["random_state"] = int(self.params.get("random_state", 575))
        # En el entorno local, XGBoost 3.2.0 puede segfaultear con n_jobs=-1
        # sobre esta matriz SNP grande. Default conservador; el YAML puede overridear.
        hp.setdefault("n_jobs", 1)
        hp["objective"] = "reg:squarederror"
        hp["n_estimators"] = int(hp["n_estimators"])
        return hp

    def _early_stopping_config(self) -> tuple[int | None, tuple[np.ndarray, np.ndarray] | None]:
        early = self.params.get("early_stopping_rounds")
        if early is None:
            return None, None
        eval_scheme = str(self.params.get("eval_scheme", "temporal")).lower()
        if eval_scheme != "temporal":
            raise ValueError(
                "XGBoostRegressor solo soporta eval_scheme='temporal' cuando "
                "early_stopping_rounds está activo."
            )
        return int(early), temporal_inner_split(self._fit_context.get("year"))

    def _select_n_estimators(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        *,
        hp: dict[str, Any],
        early_stopping_rounds: int,
    ):
        import xgboost as xgb

        selector = xgb.XGBRegressor(early_stopping_rounds=early_stopping_rounds, **hp)
        selector.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        best_iteration = int(getattr(selector, "best_iteration", -1))
        selected = best_iteration + 1 if best_iteration >= 0 else int(hp["n_estimators"])
        selected = max(1, min(int(hp["n_estimators"]), int(selected)))

        self.early_stopping_used_ = True
        self.best_iteration_ = best_iteration
        self.best_score_ = float(getattr(selector, "best_score", float("nan")))
        self.selected_n_estimators_ = selected
        self.selector_n_train_ = int(X_tr.shape[0])
        self.selector_n_val_ = int(X_va.shape[0])
        return selected, selector

    def _fit_final(self, X: np.ndarray, y: np.ndarray, *, hp: dict[str, Any], n_estimators: int) -> None:
        import xgboost as xgb

        final_hp = {**hp, "n_estimators": int(n_estimators)}
        self.model_ = xgb.XGBRegressor(**final_hp)
        self.model_.fit(X, y, verbose=False)
        self.n_features_in_: int = X.shape[1]

    def _fit_without_selection(self, X: np.ndarray, y: np.ndarray, *, hp: dict[str, Any]) -> None:
        # Sin early stopping: n_estimators fijo sobre todo el train del fold.
        self.early_stopping_used_ = False
        self.best_iteration_ = None
        self.best_score_ = float("nan")
        self.selected_n_estimators_ = int(hp["n_estimators"])
        self.refit_full_train_ = False
        self._fit_final(X, y, hp=hp, n_estimators=int(hp["n_estimators"]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        X = self._as_xgb_matrix(X)
        y = self._validate_y(y)
        hp = self._effective_hp()
        early, split = self._early_stopping_config()
        refit_full_train = bool(self.params.get("refit_full_train", True))

        if early is None or split is None:
            self._fit_without_selection(X, y, hp=hp)
            return self

        tr, va = split
        selected, selector = self._select_n_estimators(
            X[tr], y[tr], X[va], y[va],
            hp=hp,
            early_stopping_rounds=early,
        )
        self.refit_full_train_ = refit_full_train
        if refit_full_train:
            # Modelo final: todo el train externo, con el número de árboles elegido
            # por el split temporal interno.
            self._fit_final(X, y, hp=hp, n_estimators=selected)
        else:
            self.model_ = selector
            self.n_features_in_ = X.shape[1]
        return self

    def fit_with_preprocessor(self, pre: Any, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        """Ajusta XGBoost cuando vive dentro de `PreprocessedModel`.

        Para la selección interna, el preprocesador se ajusta solo con el
        inner-train. Para el modelo final refitteado, el preprocesador persistido
        en el wrapper se ajusta con todo el train externo.
        """
        X_raw = np.asarray(X)
        y = self._validate_y(y)
        hp = self._effective_hp()
        early, split = self._early_stopping_config()
        refit_full_train = bool(self.params.get("refit_full_train", True))

        if early is None or split is None:
            Xt = self._as_xgb_matrix(pre.fit_transform(X_raw))
            self._fit_without_selection(Xt, y, hp=hp)
            return self

        tr, va = split
        if refit_full_train:
            selector_pre = deepcopy(pre)
            X_tr = self._as_xgb_matrix(selector_pre.fit_transform(X_raw[tr]))
            X_va = self._as_xgb_matrix(selector_pre.transform(X_raw[va]))
        else:
            X_tr = self._as_xgb_matrix(pre.fit_transform(X_raw[tr]))
            X_va = self._as_xgb_matrix(pre.transform(X_raw[va]))

        selected, selector = self._select_n_estimators(
            X_tr, y[tr], X_va, y[va],
            hp=hp,
            early_stopping_rounds=early,
        )
        self.refit_full_train_ = refit_full_train
        if refit_full_train:
            Xt = self._as_xgb_matrix(pre.fit_transform(X_raw))
            self._fit_final(Xt, y, hp=hp, n_estimators=selected)
        else:
            self.model_ = selector
            self.n_features_in_ = X_tr.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("XGBoostRegressor.predict llamado antes de fit.")
        X = self._as_xgb_matrix(X)
        return np.asarray(self.model_.predict(X), dtype=float)

    def get_fitted_params(self) -> dict[str, Any]:
        """HP efectivos y diagnóstico del ajuste (para logging/bitácora)."""
        if not hasattr(self, "model_"):
            return {}
        out: dict[str, Any] = {
            "n_estimators": int(self.model_.get_params().get("n_estimators")),
            "max_depth": self.model_.get_params().get("max_depth"),
            "learning_rate": self.model_.get_params().get("learning_rate"),
            "early_stopping_used": int(getattr(self, "early_stopping_used_", False)),
            "selected_n_estimators": getattr(self, "selected_n_estimators_", None),
            "refit_full_train": int(getattr(self, "refit_full_train_", False)),
        }
        if getattr(self, "best_iteration_", None) is not None:
            out["best_iteration"] = self.best_iteration_
        if getattr(self, "selector_n_train_", None) is not None:
            out["selector_n_train"] = self.selector_n_train_
            out["selector_n_val"] = self.selector_n_val_
        if not np.isnan(getattr(self, "best_score_", float("nan"))):
            out["best_score"] = self.best_score_
        return out
