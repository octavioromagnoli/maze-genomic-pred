"""Evaluadores para el notebook de comparación (cachean en disco, no tocan wandb).

`make_evaluators` arma los closures ``evaluate_model`` / ``eval_with_preds`` que usan los
notebooks: corren `run_evaluation` con bootstrap forzado, cachean por
``(config, seed, n_boot, alpha)`` y devuelven filas tidy + las muestras del bootstrap (y,
``eval_with_preds``, además stashea las predicciones del fold de val para los ensembles).

La lógica vive acá (no inline en el notebook) para que **cada sección pueda levantarla con
un solo import y correr de forma independiente**: con el caché en disco
(``outputs/eval_cache``) una sección que se vuelve a correr es un *cache hit* ⇒ no reentrena.
Mantener la misma `results_cache_key` que el resto del proyecto garantiza que comparta caché.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from src.eval.ensemble import fold_predictions
from src.training.run import run_evaluation
from src.training.save_model import (
    load_cached_results,
    results_cache_key,
    save_cached_results,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def make_evaluators(
    metrics: Sequence[str],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
    curves_by_model: dict | None = None,
    preds_by_model: dict | None = None,
):
    """Devuelve ``(evaluate_model, eval_with_preds, curves_by_model, preds_by_model)``.

    Args:
        metrics: nombres de las métricas a reportar en las filas tidy (las del proyecto).
        n_boot, alpha, use_cache: defaults del bootstrap/caché (override por llamada).
        curves_by_model / preds_by_model: dicts donde acumular las learning curves y las
            predicciones del fold de val; se crean vacíos si no se pasan. Pasá los mismos
            entre secciones para acumular sin pisar.

    `evaluate_model(name, config)` ⇒ ``(rows, samples)`` con
    ``rows = [{model, metric, point, lo, hi, n}]`` y ``samples = {metric: muestras_boot}``.
    `eval_with_preds(...)` hace lo mismo y además guarda ``preds_by_model[name]`` (line_id /
    y_true / y_pred del fold de val) para alinear y combinar en los ensembles.
    """
    metrics = tuple(metrics)
    curves: dict = {} if curves_by_model is None else curves_by_model
    preds: dict = {} if preds_by_model is None else preds_by_model

    def evaluate_model(name, config, *, n_boot=n_boot, alpha=alpha, use_cache=use_cache):
        cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
        set_seed(int(cfg.get("seed", 575)))
        cfg.setdefault("bootstrap", {})
        cfg["bootstrap"].update(enabled=True, n_boot=n_boot, alpha=alpha)  # IC siempre on

        key = results_cache_key(cfg, n_boot=n_boot, alpha=alpha)
        res = load_cached_results(key) if use_cache else None
        if res is None:
            res = run_evaluation(cfg)              # sin log_fn => no toca wandb
            save_cached_results(key, res)          # cachea para las próximas corridas
        else:
            print(f"  [cache hit] {name} ({key})")

        fold = res["per_fold"][-1]                 # split fijo: el (único) fold de val
        boot = res["bootstrap"][-1] if res["bootstrap"] else {}
        curves[name] = res.get("histories", [])    # learning curves (si el modelo las expone)

        rows, samples = [], {}
        for m in metrics:
            rows.append({
                "model": name, "metric": m,
                "point": float(fold.get(m, np.nan)),
                "lo": float(fold.get(f"{m}_lo", np.nan)),
                "hi": float(fold.get(f"{m}_hi", np.nan)),
                "n": int(fold.get("n", 0)),
            })
            samples[m] = np.asarray(boot.get(m, []), dtype=float)
        return rows, samples

    def eval_with_preds(name, config, *, role=None, val_year=None, **kw):
        rows, samples = evaluate_model(name, config, **kw)
        cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
        cfg.setdefault("bootstrap", {})
        nb, al = kw.get("n_boot", n_boot), kw.get("alpha", alpha)
        cfg["bootstrap"].update(enabled=True, n_boot=nb, alpha=al)
        res = load_cached_results(results_cache_key(cfg, n_boot=nb, alpha=al))
        preds[name] = fold_predictions(res, role=role, val_year=val_year)
        return rows, samples

    return evaluate_model, eval_with_preds, curves, preds
