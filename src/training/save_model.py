from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib

from src.eval.splits import LeakageError
from src.models.registry import REGISTRY

# --- Caché de resultados de evaluación (para el notebook) -------------------


def _default_cache_dir() -> Path:
    return Path(os.environ.get("EVAL_CACHE_DIR", "outputs/eval_cache"))


# Versión del set de métricas de run_evaluation. Bumpear esto invalida el caché
# viejo cuando cambia QUÉ se calcula (no solo los HP): v2 agrega matching20
# (coincidence index al top-20%) al per_fold y su IC por bootstrap.
_METRICS_VERSION = 2


def results_cache_key(config: dict[str, Any], *, n_boot: int, alpha: float) -> str:
    """Hash estable de lo que determina el resultado de `run_evaluation`.

    Incluye data/cv/model/seed + (n_boot, alpha) del bootstrap + la versión del set
    de métricas (`_METRICS_VERSION`), para que agregar/cambiar una métrica invalide
    los resultados cacheados automáticamente (sin borrar el caché a mano).
    """
    payload = {
        "data": config.get("data"),
        "cv": config.get("cv"),
        "model": config.get("model"),
        "preprocess": config.get("preprocess"),
        "seed": config.get("seed"),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "metrics_v": _METRICS_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def load_cached_results(key: str, *, cache_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Devuelve el dict de resultados cacheado para `key`, o ``None`` si no existe."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = cache_dir / f"{key}.joblib"
    return joblib.load(path) if path.exists() else None


def save_cached_results(
    key: str, results: dict[str, Any], *, cache_dir: str | Path | None = None
) -> Path:
    """Persiste `results` bajo `key`. Devuelve la ruta del archivo escrito."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.joblib"
    joblib.dump(results, path)
    return path


# --- Artifacts de modelo ----------------------------------------------------

_MODEL_FILE = "model.bin"
_META_FILE = "meta.json"


def _default_checkpoints_dir() -> Path:
    return Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))


def _normalize_years(years: Any) -> list[int]:
    return sorted(int(y) for y in years)


def save_model_artifact(
    model: Any,
    *,
    name: str,
    model_name: str,
    train_years: Any,
    target: str,
    env: str | None,
    seed: int,
    val_metrics: dict[str, float] | None = None,
    fitted_params: dict[str, Any] | None = None,
    run: Any | None = None,
    aliases: tuple[str, ...] = ("latest",),
    checkpoints_dir: str | Path | None = None,
) -> Path:
    """Guarda el modelo entrenado: copia local estable + (si hay `run`) wandb Artifact.

    `model_name` es la clave del registry, necesaria para reconstruir en `load`.
    `train_years` ata el modelo a su régimen de train (clave anti-leakage): se
    guarda en metadata y `load_model_artifact` lo verifica. Devuelve el directorio
    local del modelo (contiene ``model.bin`` + ``meta.json``).
    """
    base = (Path(checkpoints_dir) if checkpoints_dir is not None else _default_checkpoints_dir()) / name
    base.mkdir(parents=True, exist_ok=True)

    # El modelo decide su propio formato (joblib por default; torch para el AE).
    model.save(base / _MODEL_FILE)

    meta: dict[str, Any] = {
        "model_name": model_name,
        "target": target,
        "env": env,
        "train_years": _normalize_years(train_years),
        "seed": int(seed),
    }
    if val_metrics:
        meta["val_metrics"] = {
            k: (float(v) if math.isfinite(v) else None)
            for k, v in val_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    if fitted_params:
        # Estimaciones del modelo (lambda_0, varianzas REML, esquema de CV interna)
        # quedan en la metadata para inspeccionarlas sin recargar el modelo.
        meta["fitted_params"] = {
            k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in fitted_params.items()
            if v is not None
        }
    (base / _META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if run is not None and hasattr(run, "log_artifact"):
        import wandb

        art = wandb.Artifact(name=name, type="model", metadata=meta)
        art.add_dir(str(base))
        run.log_artifact(art, aliases=list(aliases))

    return base


_AE_FILE = "ae.pt"


def save_ae_artifact(
    reducer: Any,
    *,
    name: str,
    train_years: Any,
    target: str,
    env: str | None,
    seed: int,
    metrics: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    run: Any | None = None,
    aliases: tuple[str, ...] = ("latest",),
    checkpoints_dir: str | Path | None = None,
) -> Path:
    """Guarda un `AutoencoderReducer` entrenado: checkpoint local + wandb Artifact.

    El AE es un preprocesador (no está en el REGISTRY de modelos), así que tiene su
    propio formato torch (``ae.pt`` vía ``reducer.save``). `train_years` ata el AE a
    su régimen de train (clave anti-leakage; el embedding se ajustó SOLO con esos
    años). La metadata guarda los HP del AE + las métricas de reconstrucción para
    inspeccionar sin bajar el modelo. Devuelve el directorio local del AE.
    """
    base = (Path(checkpoints_dir) if checkpoints_dir is not None else _default_checkpoints_dir()) / name
    base.mkdir(parents=True, exist_ok=True)
    reducer.save(base / _AE_FILE)

    meta: dict[str, Any] = {
        "artifact_kind": "autoencoder",
        "target": target,
        "env": env,
        "train_years": _normalize_years(train_years),
        "seed": int(seed),
        "ae_params": getattr(reducer, "params", {}),
    }
    if metrics:
        meta["metrics"] = {
            k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in metrics.items()
            if v is not None
        }
    if extra_meta:
        meta.update(extra_meta)
    (base / _META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if run is not None and hasattr(run, "log_artifact"):
        import wandb

        art = wandb.Artifact(name=name, type="autoencoder", metadata=meta)
        art.add_dir(str(base))
        run.log_artifact(art, aliases=list(aliases))

    return base


def load_ae_artifact(
    ref: str | Path,
    *,
    expected_train_years: Any | None = None,
    run: Any | None = None,
    checkpoints_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Carga un AE guardado con `save_ae_artifact`. Devuelve ``(reducer, meta)``.

    Como en `load_model_artifact`, si se pasa `expected_train_years` verifica que
    coincidan con los del artifact (anti-leakage: un AE entrenado con otros años
    proyectaría datos que el fold no debe ver). `ref` puede ser un dir local, un
    nombre bajo checkpoints/, o ``name:alias`` con un run de wandb activo.
    """
    from src.models.autoencoder import AutoencoderReducer

    base = _resolve_model_dir(ref, run, checkpoints_dir)
    meta_path = base / _META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Falta {_META_FILE} en {base}; ¿se guardó con save_ae_artifact?")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if expected_train_years is not None:
        got = _normalize_years(meta.get("train_years", []))
        exp = _normalize_years(expected_train_years)
        if got != exp:
            raise LeakageError(
                f"El AE '{ref}' se entrenó con train_years={got} pero el fold actual "
                f"entrena con {exp}. Reusarlo metería leakage (reglas #1/#3). "
                "Reentrená el AE para este split (mismos HP, otro train)."
            )

    reducer = AutoencoderReducer.load(base / _AE_FILE)
    return reducer, meta


def _resolve_model_dir(ref: str | Path, run: Any | None, checkpoints_dir: str | Path | None) -> Path:
    """Resuelve `ref` a un directorio local con ``model.bin`` + ``meta.json``.

    `ref` puede ser un directorio/archivo local, un nombre bajo el dir de
    checkpoints, o (si hay `run`) una referencia de artifact ``name:alias`` que se
    baja con ``use_artifact``.
    """
    p = Path(ref)
    if p.is_dir():
        return p
    if p.exists():  # archivo suelto => su carpeta
        return p.parent

    base = Path(checkpoints_dir) if checkpoints_dir is not None else _default_checkpoints_dir()
    local = base / str(ref)
    if local.is_dir():
        return local

    if run is not None and ":" in str(ref) and hasattr(run, "use_artifact"):
        art = run.use_artifact(str(ref))
        return Path(art.download())

    raise FileNotFoundError(
        f"No pude resolver el modelo '{ref}'. Probá un directorio local, un nombre "
        f"bajo {base}/, o 'name:alias' con un run de wandb activo."
    )


def load_model_artifact(
    ref: str | Path,
    *,
    expected_train_years: Any | None = None,
    run: Any | None = None,
    checkpoints_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Carga un modelo guardado con `save_model_artifact`. Devuelve ``(model, meta)``.

    Si `expected_train_years` se pasa (los años de train del fold donde se va a
    reusar el modelo), verifica que coincidan con los del artifact; si no, levanta
    `LeakageError`.
    """
    base = _resolve_model_dir(ref, run, checkpoints_dir)
    meta_path = base / _META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Falta {_META_FILE} en {base}; ¿se guardó con save_model_artifact?"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if expected_train_years is not None:
        got = _normalize_years(meta.get("train_years", []))
        exp = _normalize_years(expected_train_years)
        if got != exp:
            raise LeakageError(
                f"El modelo '{ref}' se entrenó con train_years={got} pero el fold "
                f"actual entrena con {exp}. Reusarlo proyectaría datos que ese fold "
                "no debe ver (leakage; reglas #1/#3). Reentrená el modelo para este split."
            )

    model_name = meta.get("model_name")
    if model_name not in REGISTRY:
        raise KeyError(
            f"model_name '{model_name}' del artifact no está en el registry: {sorted(REGISTRY)}."
        )
    model = REGISTRY[model_name].load(base / _MODEL_FILE)
    return model, meta
