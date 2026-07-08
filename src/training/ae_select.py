"""Recuperar el mejor autoencoder de un sweep (para el downstream GBLUP/XGBoost).

Un sweep del AE corre un trial por combinación de HP a dimensión latente fija. Cada
trial deja val_bce_best en run.summary y sube su AE como Artifact. Este módulo encuentra
el trial de menor val_bce_best y baja su Artifact, así el downstream compara
GBLUP/XGBoost sobre el mejor AE de cada dimensión sin re-tunear.

No reentrena nada: el AE bajado está entrenado con cv.train_years, listo para la fase de
selección de dimensión. Para reportar en val 2019 (train 2017+2018) el AE se re-ajusta
train-only con los mismos HP (viven en meta['ae_params'], recuperables con
best_ae_hparams_from_sweep).
"""
from __future__ import annotations

from typing import Any


def _best_run(sweep, metric: str, minimize: bool):
    best_run, best_val = None, None
    for run in sweep.runs:
        val = run.summary.get(metric)
        if val is None:
            continue
        val = float(val)
        if best_val is None or (val < best_val if minimize else val > best_val):
            best_val, best_run = val, run
    if best_run is None:
        raise ValueError(
            f"Ningún run del sweep tiene '{metric}' en summary. "
            "¿Corriste scripts/train_ae.py (deja val_bce_best en run.summary)?"
        )
    return best_run, best_val


def best_ae_from_sweep(
    sweep_path: str,
    *,
    metric: str = "val_bce_best",
    goal: str = "minimize",
    mark_best_alias: bool = False,
) -> dict[str, Any]:
    """Baja el AE del mejor trial del sweep. Devuelve un dict con el reducer cargado.

    Args:
        sweep_path: "entity/project/sweep_id".
        metric / goal: criterio de selección (default: val_bce_best mínimo).
        mark_best_alias: si True, agrega el alias best al Artifact ganador
            (cómodo para luego hacer use_artifact('ae-snp-d64:best')).

    Returns: ``{"reducer", "meta", "run_id", "metric", "metric_value",
               "artifact"}`. reducer es un AutoencoderReducer` en eval, listo
            para .transform(X).
    """
    import wandb

    from src.training.save_model import load_ae_artifact

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    minimize = not str(goal).lower().startswith("max")
    best_run, best_val = _best_run(sweep, metric, minimize)

    arts = [a for a in best_run.logged_artifacts() if a.type == "autoencoder"]
    if not arts:
        raise ValueError(
            f"El mejor run {best_run.id} no logueó ningún Artifact de tipo "
            "'autoencoder'. ¿El config tenía artifacts.save_model: true?"
        )
    art = arts[0]
    local_dir = art.download()
    reducer, meta = load_ae_artifact(local_dir)

    if mark_best_alias:
        if "best" not in art.aliases:
            art.aliases.append("best")
            art.save()

    return {
        "reducer": reducer,
        "meta": meta,
        "run_id": best_run.id,
        "metric": metric,
        "metric_value": best_val,
        "artifact": f"{art.name}",
    }


def best_ae_hparams_from_sweep(
    sweep_path: str,
    *,
    metric: str = "val_bce_best",
    goal: str = "minimize",
) -> dict[str, Any]:
    """HP del AE ganador (sin bajar pesos), para re-entrenar train-only en otro fold.

    Devuelve meta['ae_params'] del Artifact del mejor trial: el bloque
    preprocess.autoencoder con el que se reproduce el AE en el split que toque
    (p.ej. re-ajustar con train 2017+2018 para reportar en val 2019).
    """
    res = best_ae_from_sweep(sweep_path, metric=metric, goal=goal)
    return dict(res["meta"].get("ae_params", {}))
