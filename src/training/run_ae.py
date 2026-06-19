"""Entrenamiento del autoencoder de SNPs (no supervisado), separado del supervisado.

Por qué un runner propio y no `run_evaluation`: el AE se optimiza por **BCE de
reconstrucción** (no por la métrica downstream), es no supervisado y no tiene folds
de target. `run_evaluation` es el harness supervisado (Pearson por fold); meter ahí
el AE lo desvirtuaría. El downstream (GBLUP/XGBoost sobre el embedding) sí sigue
usando `scripts/train.py` + `preprocess.autoencoder`. Así que el único entrypoint
supervisado sigue siendo `train.py`; `scripts/train_ae.py` es el del compresor.

Régimen temporal (reglas 1 y 3): el AE se entrena con ``cv.train_years`` (p.ej.
2017) y se valida la reconstrucción en un holdout que NO es el val externo del
downstream:
- ``cv.val_scheme: temporal`` → BCE sobre ``cv.val_year`` (p.ej. 2018), el año
  siguiente al train (mismo split 2017→2018 que usan GBLUP/XGB internamente). Deja
  2019/2020 intactos.
- ``cv.val_scheme: holdout_lines`` → held-out por LÍNEA dentro del train (lo hace el
  propio reducer con ``autoencoder.val_frac``). Útil al cierre, cuando el train es
  todo el dev y no hay "año siguiente" disponible.
``holdout_test_year`` (2020) se excluye siempre.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data.loader import load_dataset
from src.eval.splits import assert_no_line_leakage, fixed_temporal_split
from src.models.autoencoder import AutoencoderReducer

LogFn = Callable[[dict[str, Any]], None]


def _train_only_idx(year: np.ndarray, train_years, holdout_test_year) -> np.ndarray:
    """Índices de filas cuyos años están en ``train_years`` (excluye el holdout)."""
    year = np.asarray(year)
    sel = {int(y) for y in train_years}
    if holdout_test_year is not None:
        sel.discard(int(holdout_test_year))
    return np.flatnonzero(np.isin(year, list(sel)))


def run_ae_training(
    config: dict[str, Any],
    log_fn: LogFn | None = None,
    *,
    return_reducer: bool = False,
) -> dict[str, Any]:
    """Entrena el AE y devuelve métricas de reconstrucción (+ el reducer si se pide).

    Config consumido: ``data.{geno_csv,pheno_csv,target,env}``,
    ``cv.{train_years,val_scheme,val_year,holdout_test_year}``,
    ``preprocess.autoencoder`` (HP del AE), ``seed``.

    Devuelve ``{fitted, n_train, n_val, train_years, val_year, val_scheme}`` y, con
    ``return_reducer=True``, además ``reducer`` (el ``AutoencoderReducer`` entrenado).
    """
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})
    ae_cfg = (config.get("preprocess", {}) or {}).get("autoencoder")
    if not ae_cfg:
        raise ValueError("Falta el bloque `preprocess.autoencoder` en el config del AE.")
    seed = int(config.get("seed", 575))

    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    arr = dataset.to_modeling_arrays(
        target=data_cfg.get("target", "GY"), env=data_cfg.get("env")
    )

    train_years = cv_cfg.get("train_years")
    if not train_years:
        raise ValueError("cv.train_years es obligatorio para el AE (p.ej. [2017]).")
    htest = cv_cfg.get("holdout_test_year")
    val_scheme = str(cv_cfg.get("val_scheme", "temporal")).lower()

    # El reducer hereda la seed del experimento si su sub-bloque no la fija.
    ae_params = dict(ae_cfg)
    ae_params.setdefault("seed", seed)
    reducer = AutoencoderReducer(ae_params)
    if log_fn is not None:
        reducer.set_log_fn(log_fn)

    if val_scheme == "temporal":
        val_year = cv_cfg.get("val_year")
        if val_year is None:
            raise ValueError("cv.val_year es obligatorio con val_scheme='temporal'.")
        splits = fixed_temporal_split(
            arr.year, arr.line_id,
            val_year=int(val_year),
            train_years=[int(y) for y in train_years],
            holdout_test_year=htest,
            enforce="val",
        )
        if not splits:
            raise ValueError(
                f"Split vacío: train_years={train_years} no es anterior a val_year={val_year}."
            )
        (tr, va), = splits
        assert_no_line_leakage(tr, va, arr.line_id)  # guard duro (regla 2)
        reducer.set_fit_context(
            line_id=arr.line_id[tr], year=arr.year[tr],
            X_val=arr.X[va], line_id_val=arr.line_id[va],
        )
        reducer.fit(arr.X[tr])
        n_train, n_val = int(len(tr)), int(len(va))
    elif val_scheme == "holdout_lines":
        tr = _train_only_idx(arr.year, train_years, htest)
        if len(tr) == 0:
            raise ValueError(f"Sin filas de train para train_years={train_years}.")
        reducer.set_fit_context(line_id=arr.line_id[tr], year=arr.year[tr])
        reducer.fit(arr.X[tr])  # el reducer hace su held-out por línea (val_frac)
        n_train = int(len(tr))
        n_val = None  # lo reporta fitted['ae_has_val'] / val_frac
        val_year = None
    else:
        raise ValueError(f"cv.val_scheme desconocido: {val_scheme!r} (temporal|holdout_lines).")

    fitted = reducer.get_fitted_params()
    result: dict[str, Any] = {
        "fitted": fitted,
        "n_train": n_train,
        "n_val": n_val,
        "train_years": sorted(int(y) for y in train_years),
        "val_year": int(val_year) if val_year is not None else None,
        "val_scheme": val_scheme,
        "target": data_cfg.get("target", "GY"),
        "env": data_cfg.get("env"),
    }
    if return_reducer:
        result["reducer"] = reducer
    return result
