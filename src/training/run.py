from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data.loader import load_dataset
from src.eval.metrics import (
    bootstrap_ci,
    coincidence_index,
    pearson_r,
    r2,
    regression_metrics,
    rmse,
    summarize_folds,
)
from src.eval.splits import (
    assert_no_line_leakage,
    expanding_temporal_split,
    fixed_temporal_split,
)
from src.models.registry import build_model

LogFn = Callable[[dict[str, Any]], None]

_CI_METRICS = {"pearson_r": pearson_r, "rmse": rmse, "r2": r2, "matching20": coincidence_index}
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
    """Arma los folds temporales. Devuelve lista de (train_idx, val_idx, scoring).

    scoring=False marca folds de DIAGNÓSTICO: se evalúan y loguean igual, pero
    NO entran en el resumen (pearson_r_mean) que optimiza el sweep.

    Schemes:
    - fixed_temporal: un único fold de scoring (split fijo train/val).
    - expanding_temporal: folds de scoring por val_years (expanding window,
      train = años previos) + folds de diagnóstico opcionales en report_only
      (lista de {val_year, train_years?}; sin train_years usa expanding).
    """
    scheme = cv_cfg.get("scheme", "fixed_temporal")
    enforce = "val" if cv_cfg.get("enforce_disjoint_lines", True) else "train"
    htest = cv_cfg.get("holdout_test_year")

    if scheme == "fixed_temporal":
        fs = fixed_temporal_split(
            year, line_id,
            val_year=cv_cfg.get("val_year"),
            holdout_test_year=htest,
            train_years=cv_cfg.get("train_years"),
            enforce=enforce,
        )
        return [(tr, va, True) for tr, va in fs]

    if scheme == "expanding_temporal":
        folds: list = [
            (tr, va, True)
            for tr, va in expanding_temporal_split(
                year, line_id,
                val_years=cv_cfg.get("val_years", []),
                holdout_test_year=htest,
                enforce=enforce,
            )
        ]
        for spec in cv_cfg.get("report_only", []):
            fs = fixed_temporal_split(
                year, line_id,
                val_year=spec["val_year"],
                holdout_test_year=htest,
                train_years=spec.get("train_years"),
                enforce=enforce,
            )
            folds += [(tr, va, False) for tr, va in fs]
        return folds

    raise ValueError(
        f"cv.scheme desconocido: {scheme!r}. Soportados: "
        "'fixed_temporal', 'expanding_temporal'."
    )


def run_evaluation(
    config: dict[str, Any],
    log_fn: LogFn | None = None,
    *,
    return_models: bool = False,
) -> dict[str, Any]:
    """Corre la evaluación temporal y devuelve resultados.

    Estructura del config consumida:
        data.geno_csv, data.pheno_csv, data.target, data.env
        cv.scheme, cv.val_year, cv.holdout_test_year, cv.enforce_disjoint_lines
        bootstrap.enabled, bootstrap.n_boot, bootstrap.alpha (IC opcional)
        model.name (+ model.params)

    Devuelve dict con per_fold (lista de métricas) y summary (agregado).
    Con el split fijo per_fold tiene un único elemento.
    """
    log = log_fn or _noop
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})
    boot_cfg = config.get("bootstrap", {})
    seed = int(config.get("seed", 575))

    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    target = data_cfg.get("target", "GY")
    env = data_cfg.get("env")
    arrays = dataset.to_modeling_arrays(target=target, env=env)

    folds = _build_folds(cv_cfg, arrays.year, arrays.line_id)
    if not folds:
        raise RuntimeError(
            "No se generó ningún split. Revisá cv.scheme / cv.val_year / "
            f"cv.holdout_test_year={cv_cfg.get('holdout_test_year')} contra los "
            "años presentes en los datos (se necesita >= 1 año de train antes del val)."
        )

    per_fold: list[dict[str, float]] = []
    boot_samples: list[dict[str, Any]] = []
    trained: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    for i, (train_idx, val_idx, scoring) in enumerate(folds):
        assert_no_line_leakage(train_idx, val_idx, arrays.line_id)
        if len(val_idx) == 0:
            log({"fold": i, "skipped_empty_val": 1})
            continue

        X_tr, y_tr = arrays.X[train_idx], arrays.y[train_idx]
        X_va, y_va = arrays.X[val_idx], arrays.y[val_idx]

        model = build_model(config)
        model.set_fit_context(
            year=arrays.year[train_idx],
            line_id=arrays.line_id[train_idx],
            eval_X=X_va,
            eval_y=y_va,
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_va)

        m = regression_metrics(y_va, y_hat)
        m["matching20"] = coincidence_index(y_va, y_hat, 0.2)
        train_years = sorted(set(int(y) for y in arrays.year[train_idx]))
        val_years = sorted(set(int(y) for y in arrays.year[val_idx]))
        m.update(
            fold=i,
            role="scoring" if scoring else "report",
            n_train=int(len(train_idx)),
            n_val=int(len(val_idx)),
            train_years=str(train_years),
            val_year=val_years[0] if len(val_years) == 1 else str(val_years),
        )
        fitted_params = model.get_fitted_params()
        for k, v in fitted_params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                m[k] = v

        if boot_cfg.get("enabled", False):
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
        preds.append({
            "fold": i,
            "role": m["role"],
            "val_year": m["val_year"],
            "line_id": np.asarray(arrays.line_id[val_idx]),
            "y_true": np.asarray(y_va, dtype=float).ravel(),
            "y_pred": np.asarray(y_hat, dtype=float).ravel(),
        })
        hist = getattr(model, "history_", None) or getattr(getattr(model, "inner", None), "history_", None)
        if hist:
            histories.append({"fold": i, "history": list(hist)})
        if return_models:
            trained.append({
                "fold": i,
                "model": model,
                "train_years": train_years,
                "fitted_params": fitted_params,
            })
        tag = f"f{i}_{m['role']}_{m['val_year']}"
        log({f"{tag}/{k}": v for k, v in m.items()})

    if not per_fold:
        raise RuntimeError(
            "Todos los splits quedaron con val vacío tras enforce_disjoint_lines. "
            "Con un split fijo esto suele significar que casi todas las líneas del "
            "año de validación ya estaban en train; probá enforce_disjoint_lines: false."
        )

    scoring_folds = [f for f in per_fold if f.get("role") == "scoring"]
    summary = summarize_folds(
        [{k: v for k, v in f.items() if _is_summary_metric(k, v)} for f in scoring_folds]
    )
    log({f"summary/{k}": v for k, v in summary.items()})

    result: dict[str, Any] = {
        "per_fold": per_fold,
        "summary": summary,
        "bootstrap": boot_samples,
        "histories": histories,
        "predictions": preds,
        "target": target,
        "env": env,
        "n_obs": int(arrays.X.shape[0]),
        "n_lines": int(dataset.n_lines),
        "holdout_test_year": cv_cfg.get("holdout_test_year"),
    }
    if return_models:
        result["models"] = trained
    return result
