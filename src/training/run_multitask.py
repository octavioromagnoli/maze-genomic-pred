"""Evaluación temporal multitask: una red, varias cabezas, métricas por trait.

Espeja run_evaluation pero para la red multitask: entrena una sola red (tronco
compartido + cabeza Linear(prev, n_targets)) y al evaluar itera por trait aplicando su
máscara, de modo que métricas e IC se calculan solo con los datos reales de cada trait.
Usa los mismos splits temporales y guards anti-leakage que la ruta single-task.

Diferencias con run_evaluation: data.targets (lista) en vez de data.target; el loader
devuelve Y/Y_mask (n_obs, n_targets); per_fold tiene un dict por (fold, trait); y
summary agrega por trait y expone pearson_r_GY (el Pearson de GY de los folds de
scoring) como métrica que optimiza el sweep, no el promedio simple de los traits.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.loader import ENVIRONMENTS, TARGETS, load_dataset
from src.eval.metrics import (
    bootstrap_ci,
    coincidence_index,
    regression_metrics,
    summarize_folds,
)
from src.eval.splits import assert_no_line_leakage
from src.models.registry import build_model
from src.training.run import LogFn, _build_folds, _CI_METRICS, _is_summary_metric, _noop


def run_multitask_evaluation(
    config: dict[str, Any],
    log_fn: LogFn | None = None,
    *,
    return_models: bool = False,
) -> dict[str, Any]:
    """Corre la evaluación temporal multitask y devuelve resultados por trait.

    Estructura del config consumida (además de la común):
        data.targets (lista de traits), data.env
        model.name = "mlp_multitask" (+ model.params, incl. loss_weights)

    Devuelve dict con per_fold (un elemento por fold×trait), summary (agregado
    por trait + pearson_r_GY), predictions (por fold×trait, enmascaradas) y el
    resto del bookkeeping (bootstrap, histories, models).
    """
    log = log_fn or _noop
    data_cfg = config.get("data", {})
    cv_cfg = config.get("cv", {})
    boot_cfg = config.get("bootstrap", {})
    seed = int(config.get("seed", 575))

    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    env = data_cfg.get("env")
    if data_cfg.get("multi_trait_env"):
        trait_list = list(data_cfg.get("targets") or list(TARGETS))
        envs = list(data_cfg.get("envs") or list(ENVIRONMENTS))
        arrays = dataset.to_modeling_arrays_multi_trait_env(targets=trait_list, envs=envs)
        targets = [f"{t}_{e}" for t in trait_list for e in envs]
    elif data_cfg.get("multienv"):
        target = data_cfg.get("target", "GY")
        envs = list(data_cfg.get("envs") or list(ENVIRONMENTS))
        arrays = dataset.to_modeling_arrays_multienv(target=target, envs=envs)
        targets = [f"{target}_{e}" for e in envs]
    else:
        targets = list(data_cfg.get("targets") or [])
        if not targets:
            raise ValueError(
                "run_multitask_evaluation necesita `data.targets` (lista de traits, p.ej. "
                "[GY, PH, AD]) o `data.multienv: true` con `data.target`/`data.envs`."
            )
        arrays = dataset.to_modeling_arrays_multitask(targets=targets, env=env)

    folds = _build_folds(cv_cfg, arrays.year, arrays.line_id)
    if not folds:
        raise RuntimeError(
            "No se generó ningún split. Revisá cv.scheme / cv.val_year / "
            f"cv.holdout_test_year={cv_cfg.get('holdout_test_year')} contra los años presentes."
        )

    per_fold: list[dict[str, Any]] = []
    boot_samples: list[dict[str, Any]] = []
    trained: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []

    for i, (train_idx, val_idx, scoring) in enumerate(folds):
        assert_no_line_leakage(train_idx, val_idx, arrays.line_id)
        if len(val_idx) == 0:
            log({"fold": i, "skipped_empty_val": 1})
            continue

        X_tr, Y_tr, M_tr = arrays.X[train_idx], arrays.Y[train_idx], arrays.Y_mask[train_idx]
        X_va, Y_va, M_va = arrays.X[val_idx], arrays.Y[val_idx], arrays.Y_mask[val_idx]

        model = build_model(config)
        model.set_fit_context(
            year=arrays.year[train_idx],
            line_id=arrays.line_id[train_idx],
            target_names=targets,
            eval_X=X_va,
            eval_Y=Y_va,
            eval_Ymask=M_va,
        )
        model.fit(X_tr, Y_tr, M_tr)
        Yhat = np.asarray(model.predict(X_va), dtype=float)
        fitted_params = model.get_fitted_params()

        train_years = sorted(set(int(y) for y in arrays.year[train_idx]))
        val_years = sorted(set(int(y) for y in arrays.year[val_idx]))
        val_year = val_years[0] if len(val_years) == 1 else str(val_years)
        role = "scoring" if scoring else "report"

        for j, trait in enumerate(targets):
            col = np.asarray(M_va[:, j], dtype=bool)
            yt = Y_va[col, j]
            yp = Yhat[col, j]
            m: dict[str, Any] = regression_metrics(yt, yp)
            m["matching20"] = coincidence_index(yt, yp, 0.2)
            m.update(
                fold=i,
                trait=trait,
                role=role,
                n_train=int(M_tr[:, j].sum()),
                n_val=int(col.sum()),
                train_years=str(train_years),
                val_year=val_year,
            )
            for k, v in fitted_params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    m[k] = v

            if boot_cfg.get("enabled", False) and int(col.sum()) >= 2:
                ci = bootstrap_ci(
                    yt, yp, _CI_METRICS,
                    group=np.asarray(arrays.line_id[val_idx])[col],
                    n_boot=int(boot_cfg.get("n_boot", 1000)),
                    alpha=float(boot_cfg.get("alpha", 0.05)),
                    seed=seed + i,
                )
                for name, b in ci.items():
                    m[f"{name}_lo"], m[f"{name}_hi"] = b["lo"], b["hi"]
                boot_samples.append(
                    {"fold": i, "trait": trait, **{name: b["samples"] for name, b in ci.items()}}
                )

            per_fold.append(m)
            preds.append({
                "fold": i,
                "trait": trait,
                "role": role,
                "val_year": val_year,
                "line_id": np.asarray(arrays.line_id[val_idx])[col],
                "y_true": np.asarray(yt, dtype=float).ravel(),
                "y_pred": np.asarray(yp, dtype=float).ravel(),
            })
            tag = f"f{i}_{trait}_{role}_{val_year}"
            log({f"{tag}/{k}": v for k, v in m.items()})

        hist = getattr(model, "history_", None)
        if hist:
            histories.append({"fold": i, "history": list(hist)})
        if return_models:
            trained.append({
                "fold": i,
                "model": model,
                "train_years": train_years,
                "fitted_params": fitted_params,
            })

    if not per_fold:
        raise RuntimeError(
            "Todos los splits quedaron con val vacío tras enforce_disjoint_lines. "
            "Probá enforce_disjoint_lines: false."
        )

    summary: dict[str, float] = {}
    for trait in targets:
        scoring_t = [f for f in per_fold if f.get("trait") == trait and f.get("role") == "scoring"]
        s = summarize_folds(
            [{k: v for k, v in f.items() if _is_summary_metric(k, v)} for f in scoring_t]
        )
        for k, v in s.items():
            summary[f"{trait}/{k}"] = v
    if data_cfg.get("multi_trait_env"):
        primary = f"GY_{data_cfg.get('cover_env', 'drought')}"
        if primary not in targets:
            primary = targets[0]
    elif data_cfg.get("multienv"):
        primary = f"{data_cfg.get('target', 'GY')}_{data_cfg.get('cover_env', 'drought')}"
        if primary not in targets:
            primary = targets[0]
    else:
        primary = "GY" if "GY" in targets else targets[0]
    summary["pearson_r_GY"] = summary.get(f"{primary}/pearson_r_mean", float("nan"))
    log({f"summary/{k}": v for k, v in summary.items()})

    result: dict[str, Any] = {
        "per_fold": per_fold,
        "summary": summary,
        "bootstrap": boot_samples,
        "histories": histories,
        "predictions": preds,
        "targets": targets,
        "env": env,
        "n_obs": int(arrays.X.shape[0]),
        "n_lines": int(dataset.n_lines),
        "holdout_test_year": cv_cfg.get("holdout_test_year"),
    }
    if return_models:
        result["models"] = trained
    return result
