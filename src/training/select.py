"""Selección de hiperparámetros por holdout temporal INTERNO (nested), sin sesgo.

El problema: si elegís un HP (p.ej. la dim de PCA) maximizando la métrica sobre el
mismo val externo (2019) con el que después comparás/reportás, el número del
ganador está inflado (winner's curse): lo elegiste para que le vaya bien ahí.

La solución honesta: elegir el HP en un holdout **anterior** (inner-val), entrenando
con los años previos a ese, y recién evaluar el HP elegido en el val externo (que
esta función NO toca). Respeta el orden temporal (regla 3) y deja el val externo
limpio para una comparación justa contra modelos sin ese HP (p.ej. el baseline).

Reutiliza el splitter del proyecto (`fixed_temporal_split`) y el preprocesamiento
train-only vía `build_model`, así no se reimplementa nada ni se mete leakage.
General para cualquier HP entero (PCA dim, latente del AE, ...) vía `dim_key`.
"""
from __future__ import annotations

import copy
from typing import Any

from src.utils.config import load_config, set_nested
from src.utils.seed import set_seed


def nested_temporal_dim_select(
    base_config: str | dict[str, Any],
    dim_key: str,
    dims: list[int],
    *,
    inner_val_year: int,
    metric: str = "pearson_r",
    skip_dims_geq_inner_n: bool = True,
) -> dict[str, Any]:
    """Elige el mejor valor de `dim_key` en un inner-val temporal, sin tocar el val externo.

    Args:
        base_config: ruta o dict del experimento (de ahí salen data/seed/preprocess/model).
        dim_key: clave dot-notation del HP a elegir (p.ej. ``"preprocess.pca.n_components"``).
        dims: valores candidatos.
        inner_val_year: año que se usa como inner-val (train = años previos). P.ej. 2018
            ⇒ train 2017 → val 2018, dejando 2019 (val externo) intocado.
        metric: ``"pearson_r"`` (max), ``"r2"`` (max) o ``"rmse"`` (min).
        skip_dims_geq_inner_n: descarta dims ``>= n_train`` del inner. Para PCA es clave
            (pedir más componentes que muestras degenera: techo de rango p≫n). Para HPs
            que no son dim de PCA (latente chico del AE) nunca aplica.

    Returns:
        ``{"best", "scores": {dim: score}, "inner_val_year", "inner_train_n",
           "skipped", "metric"}``.
    """
    from src.data.loader import load_dataset
    from src.eval import metrics as M
    from src.eval.splits import assert_no_line_leakage, fixed_temporal_split
    from src.models.registry import build_model

    cfg0 = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    data = cfg0.get("data", {})
    seed = int(cfg0.get("seed", 575))
    metric_fn = {"pearson_r": M.pearson_r, "rmse": M.rmse, "r2": M.r2}[metric]

    ds = load_dataset(data["geno_csv"], data["pheno_csv"])
    arr = ds.to_modeling_arrays(target=data.get("target", "GY"), env=data.get("env"))
    splits = fixed_temporal_split(
        arr.year, arr.line_id,
        val_year=int(inner_val_year),
        holdout_test_year=cfg0.get("cv", {}).get("holdout_test_year"),
        enforce="val",
    )
    if not splits:
        raise ValueError(
            f"No hay año de train anterior a inner_val_year={inner_val_year} "
            "(¿pediste el año más viejo del dev como inner-val?)."
        )
    (tr, va), = splits
    assert_no_line_leakage(tr, va, arr.line_id)   # guard duro (regla 2)
    n_in = int(len(tr))

    scores: dict[int, float] = {}
    skipped: list[int] = []
    for d in dims:
        if skip_dims_geq_inner_n and int(d) >= n_in:
            skipped.append(int(d))                # techo de rango: no es elegible honestamente
            continue
        cfg = copy.deepcopy(cfg0)
        set_nested(cfg, dim_key, int(d))
        set_seed(seed)
        m = build_model(cfg)
        m.set_fit_context(year=arr.year[tr], line_id=arr.line_id[tr])
        m.fit(arr.X[tr], arr.y[tr])
        scores[int(d)] = float(metric_fn(arr.y[va], m.predict(arr.X[va])))

    if not scores:
        raise ValueError(
            f"Ninguna dim quedó evaluable (n_train inner={n_in}; descartadas {skipped}). "
            "Pasá dims más chicas que el techo de rango."
        )
    best = (min if metric == "rmse" else max)(scores, key=scores.get)
    return {
        "best": best,
        "scores": scores,
        "inner_val_year": int(inner_val_year),
        "inner_train_n": n_in,
        "skipped": skipped,
        "metric": metric,
    }


def config_with_nested_dim(
    base_config: str | dict[str, Any],
    dim_key: str,
    dims: list[int],
    *,
    inner_val_year: int,
    metric: str = "pearson_r",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Igual que `nested_temporal_dim_select`, pero devuelve también el config listo.

    El config conserva el split EXTERNO del `base_config` (p.ej. val 2019) y le
    aplica la dim elegida en el inner. Pasáselo a `run_evaluation`/`evaluate_model`
    para evaluarlo en el val externo de forma honesta (la dim no se eligió ahí).

    Returns: ``(config, selection_info)``.
    """
    cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    sel = nested_temporal_dim_select(
        cfg, dim_key, dims, inner_val_year=inner_val_year, metric=metric
    )
    set_nested(cfg, dim_key, sel["best"])
    return cfg, sel
