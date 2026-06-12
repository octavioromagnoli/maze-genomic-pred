from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data.loader import load_dataset
from src.eval.metrics import (
    bootstrap_ci,
    pearson_r,
    r2,
    regression_metrics,
    rmse,
    summarize_folds,
)
from src.eval.splits import assert_no_line_leakage, fixed_temporal_split
from src.models.registry import build_model

LogFn = Callable[[dict[str, Any]], None]

# Métricas a las que se les calcula IC por bootstrap (mismo resample para las 3).
_CI_METRICS = {"pearson_r": pearson_r, "rmse": rmse, "r2": r2}
# Bookkeeping que NO entra al resumen; los bordes _lo/_hi tampoco (se setean
# aparte con nombres limpios). Todo otro escalar numérico sí se resume.
_SUMMARY_EXCLUDE = {"fold", "n", "n_train", "n_val", "val_year", "train_years"}


def _noop(_: dict[str, Any]) -> None:
    pass


def _is_summary_metric(k: str, v: Any) -> bool:
    return (
        k not in _SUMMARY_EXCLUDE
        and not k.endswith(("_lo", "_hi"))
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    )


def _build_folds(cv_cfg: dict[str, Any], year, line_id) -> list:
    """Arma el split fijo train/val. Devuelve lista de (train_idx, val_idx)."""
    scheme = cv_cfg.get("scheme", "fixed_temporal")
    if scheme != "fixed_temporal":
        raise ValueError(
            f"cv.scheme desconocido: {scheme!r}. Único soportado: 'fixed_temporal'."
        )
    enforce = "val" if cv_cfg.get("enforce_disjoint_lines", True) else "train"
    return fixed_temporal_split(
        year,
        line_id,
        val_year=cv_cfg.get("val_year"),
        holdout_test_year=cv_cfg.get("holdout_test_year"),
        enforce=enforce,
    )


def run_evaluation(config: dict[str, Any], log_fn: LogFn | None = None) -> dict[str, Any]:
    """Corre la evaluación temporal y devuelve resultados.

    Estructura del config consumida:
        data.geno_csv, data.pheno_csv, data.target, data.env
        cv.scheme, cv.val_year, cv.holdout_test_year, cv.enforce_disjoint_lines
        bootstrap.enabled, bootstrap.n_boot, bootstrap.alpha (IC opcional)
        model.name (+ model.params)

    Devuelve dict con ``per_fold`` (lista de métricas) y ``summary`` (agregado).
    Con el split fijo ``per_fold`` tiene un único elemento.
    """
    log = log_fn or _noop
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})
    boot_cfg = config.get("bootstrap", {})
    seed = int(config.get("seed", 42))

    # --- Carga directa de los CSV + arrays de modelado -----------------------
    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    target = data_cfg.get("target", "GY")
    env = data_cfg.get("env")  # None => apila optimal + drought
    arrays = dataset.to_modeling_arrays(target=target, env=env)

    # --- Split(s) temporal(es) + disjuntos por línea -------------------------
    folds = _build_folds(cv_cfg, arrays.year, arrays.line_id)
    if not folds:
        raise RuntimeError(
            "No se generó ningún split. Revisá cv.scheme / cv.val_year / "
            f"cv.holdout_test_year={cv_cfg.get('holdout_test_year')} contra los "
            "años presentes en los datos (se necesita >= 1 año de train antes del val)."
        )

    per_fold: list[dict[str, float]] = []
    boot_samples: list[dict[str, Any]] = []
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

        if boot_cfg.get("enabled", False):
            # IC por bootstrap a nivel línea (cluster) sobre el set de validación.
            ci = bootstrap_ci(
                y_va,
                y_hat,
                _CI_METRICS,
                group=arrays.line_id[val_idx],
                n_boot=int(boot_cfg.get("n_boot", 1000)),
                alpha=float(boot_cfg.get("alpha", 0.05)),
                seed=seed + i,
            )
            for name, b in ci.items():
                m[f"{name}_lo"], m[f"{name}_hi"] = b["lo"], b["hi"]
            boot_samples.append({"fold": i, **{name: b["samples"] for name, b in ci.items()}})

        per_fold.append(m)
        log({f"fold/{k}": v for k, v in m.items()})

    if not per_fold:
        raise RuntimeError(
            "Todos los splits quedaron con val vacío tras enforce_disjoint_lines. "
            "Con un split fijo esto suele significar que casi todas las líneas del "
            "año de validación ya estaban en train; probá enforce_disjoint_lines: false."
        )

    summary = summarize_folds(
        [{k: v for k, v in f.items() if _is_summary_metric(k, v)} for f in per_fold]
    )
    log({f"summary/{k}": v for k, v in summary.items()})

    return {
        "per_fold": per_fold,
        "summary": summary,
        "bootstrap": boot_samples,
        "target": target,
        "env": env,
        "n_obs": int(arrays.X.shape[0]),
        "n_lines": int(dataset.n_lines),
        "holdout_test_year": cv_cfg.get("holdout_test_year"),
    }
