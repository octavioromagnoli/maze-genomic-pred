"""Traer el resultado de un sweep de wandb al análisis local.

Un sweep deja, por cada trial, los HP barridos como claves planas con punto
(p.ej. ``"preprocess.pca.n_components"``) en ``run.config`` y las métricas en
``run.summary`` (`scripts/train.py` loguea ``pearson_r_mean`` ahí).
`best_run_from_sweep` busca el trial que optimiza la métrica del sweep;
`best_config_from_sweep` reconstruye el config de ese trial (mismo merge
dot-notation que `train.py`) para re-evaluarlo localmente.

Para GBLUP (determinista y barato) "el mejor modelo" queda **totalmente
reproducido por su config + seed**: no hace falta bajar un pickle, se re-fittea y
da bit-a-bit lo mismo que la corrida del sweep. Para modelos caros/estocásticos
(NN/AE), guardá el modelo con ``artifacts.save_model: true`` y bajá el Artifact
con `load_model_artifact` (ver `src/training/save_model.py`).
"""
from __future__ import annotations

import copy
from typing import Any

from src.utils.config import load_config, set_nested


def best_run_from_sweep(
    sweep_path: str,
    *,
    metric: str | None = None,
    goal: str | None = None,
) -> tuple[dict[str, Any], float, str]:
    """Mejor trial del sweep: ``(overrides, metric_value, run_id)``.

    Args:
        sweep_path: ``"entity/project/sweep_id"``.
        metric / goal: por defecto se leen del config del sweep
            (``metric.name`` / ``metric.goal``); pasarlos los pisa.

    ``overrides`` son los HP barridos del mejor run (claves planas con punto,
    listas para `set_nested`). Recorre ``sweep.runs`` y elige por ``run.summary``,
    así no depende del estado interno de `best_run()` de wandb.
    """
    import wandb

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    mcfg = (getattr(sweep, "config", {}) or {}).get("metric", {})
    metric = metric or mcfg.get("name", "pearson_r_mean")
    maximize = not (goal or mcfg.get("goal", "maximize")).lower().startswith("min")

    best_run, best_val = None, None
    for run in sweep.runs:
        val = run.summary.get(metric)
        if val is None:
            continue
        val = float(val)
        if best_val is None or (val > best_val if maximize else val < best_val):
            best_val, best_run = val, run
    if best_run is None:
        raise ValueError(
            f"Ningún run del sweep {sweep_path} tiene '{metric}' en summary. "
            "¿La corrida la loguea? (train.py la deja en run.summary)."
        )

    overrides = {k: v for k, v in dict(best_run.config).items() if "." in k}
    if not overrides:
        raise ValueError(
            f"El mejor run {best_run.id} no tiene HP barridos (claves con punto) "
            "en run.config; ¿este sweep barre algo con notación dot?"
        )
    return overrides, best_val, best_run.id


def best_config_from_sweep(
    sweep_path: str,
    base_config: str | dict[str, Any],
    *,
    metric: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Config base + overrides del mejor trial del sweep (listo para evaluar).

    Reproduce el trial ganador localmente (mismo merge dot-notation que
    `scripts/train.py`). Pasáselo a `run_evaluation` / `evaluate_model`.
    """
    cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    overrides, _, _ = best_run_from_sweep(sweep_path, metric=metric, goal=goal)
    for dotted, value in overrides.items():
        set_nested(cfg, dotted, value)
    return cfg
