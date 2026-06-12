"""Loop de cross-validation reutilizable (sin dependencia dura de wandb).

`run_cv` carga datos, arma los folds temporales agrupados por línea, corre el
guard antileakage en cada fold, entrena un modelo fresco por fold y calcula las
métricas. El logging se inyecta como callback `log_fn`, así el harness es testeable
sin wandb y `scripts/train.py` solo cablea `wandb.log`.

el año `cv.holdout_test_year` (2020 por defecto) se excluye
de la CV y NO se evalúa acá — es el test final intocable hasta el cierre.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data.loader import load_dataset
from src.eval.metrics import regression_metrics, summarize_folds
from src.eval.splits import assert_no_line_leakage, temporal_grouped_folds
from src.models.registry import build_model

LogFn = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


def run_cv(config: dict[str, Any], log_fn: LogFn | None = None) -> dict[str, Any]:
    """Corre la CV forward-chaining agrupada por línea y devuelve resultados.

    Estructura del config consumida:
        data.geno_csv, data.pheno_csv, data.target, data.env
        cv.holdout_test_year, cv.min_train_years, cv.enforce_disjoint_lines
        model.name (+ model.params)

    Devuelve dict con ``per_fold`` (lista de métricas) y ``summary`` (agregado).
    """
    log = log_fn or _noop
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})

    # --- Carga directa de los CSV + arrays de modelado -----------------------
    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    target = data_cfg.get("target", "GY")
    env = data_cfg.get("env")  # None => apila optimal + drought
    arrays = dataset.to_modeling_arrays(target=target, env=env)

    # --- Folds temporales + disjuntos por línea ------------------------------
    enforce = "val" if cv_cfg.get("enforce_disjoint_lines", True) else "train"
    folds = temporal_grouped_folds(
        arrays.year,
        arrays.line_id,
        holdout_test_year=cv_cfg.get("holdout_test_year"),
        min_train_years=int(cv_cfg.get("min_train_years", 1)),
        enforce=enforce,
    )
    if not folds:
        raise RuntimeError(
            "No se generó ningún fold de CV. Revisá que haya >= 2 años de dev "
            f"(holdout_test_year={cv_cfg.get('holdout_test_year')}) en los datos."
        )

    per_fold: list[dict[str, float]] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        # Guard duro: ninguna línea cruza train/val (regla 2).
        assert_no_line_leakage(train_idx, val_idx, arrays.line_id)
        if len(val_idx) == 0:
            # Tras hacer líneas disjuntas puede quedar val vacío; se reporta y se saltea.
            log({"fold": i, "skipped_empty_val": 1})
            continue

        X_tr, y_tr = arrays.X[train_idx], arrays.y[train_idx]
        X_va, y_va = arrays.X[val_idx], arrays.y[val_idx]

        model = build_model(config)  # modelo fresco por fold (sin fuga de estado)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_va)

        m = regression_metrics(y_va, y_hat)
        train_years = sorted(set(int(y) for y in arrays.year[train_idx]))
        val_years = sorted(set(int(y) for y in arrays.year[val_idx]))
        m.update(
            fold=i,
            n_train=int(len(train_idx)),
            n_val=int(len(val_idx)),
            train_years=str(train_years),
            val_year=val_years[0] if len(val_years) == 1 else str(val_years),
        )
        per_fold.append(m)
        log({f"fold/{k}": v for k, v in m.items()})

    summary = summarize_folds([{k: v for k, v in f.items() if k in ("pearson_r", "rmse", "r2")} for f in per_fold])
    log({f"summary/{k}": v for k, v in summary.items()})

    return {
        "per_fold": per_fold,
        "summary": summary,
        "target": target,
        "env": env,
        "n_obs": int(arrays.X.shape[0]),
        "n_lines": int(dataset.n_lines),
        "holdout_test_year": cv_cfg.get("holdout_test_year"),
    }
