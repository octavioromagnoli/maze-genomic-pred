"""Cierre 2020 con spike-in: rescatar la predicción de rinde del colapso cold-start.

Predecir 2020 con un modelo entrenado solo en dev (2017-2019) COLAPSA para GY (Pearson negativo):
2020 no comparte líneas con dev (cero overlap de GID) y la sequía derivó. El *spike-in* modela el
escenario real de despliegue: fenotipar una **fracción chica** de líneas del ciclo nuevo (2020) y
sumarlas al train. Este harness mide cuánto rescata.

Protocolo (toca 2020 — es el CIERRE, regla 3): para una fracción ``s``, se eligen al azar (semilla
``rep``) ``s``·|2020| líneas de 2020, se agregan al train CON su fenotipo, y se predicen las líneas
de 2020 **restantes** (held-out, nunca vistas). Preprocesamiento del modelo ajustado train-only sobre
el train aumentado (regla 1). ``s=0`` es el cold-start puro (determinista ⇒ una sola realización).

Reusa los helpers de `noise.py` (`_build_arrays`/`_fit`/`_predict_heads`) para despachar single-task
(GBLUP) vs multitask (MoE). Devuelve filas ``{model, train, spike, rep, pearson, n_held, n_spike}``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.data.loader import load_dataset
from src.eval.metrics import pearson_r
from src.eval.noise import _build_arrays, _fit, _predict_heads
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def spike_closure(
    config: str | dict[str, Any],
    *,
    label: str,
    train_years: Sequence[int] = (2017, 2018, 2019),
    test_year: int = 2020,
    spike_fracs: Sequence[float] = (0.0, 0.05, 0.10),
    reps: Sequence[int] = (1, 42, 2024),
    target_head: str = "GY_drought",
    seed: int = 575,
) -> list[dict[str, Any]]:
    """Curva de rescate por spike-in para UN modelo. Ver docstring del módulo."""
    set_seed(seed)
    cfg = load_config(config) if isinstance(config, str) else dict(config)
    ds = load_dataset(cfg["data"]["geno_csv"], cfg["data"]["pheno_csv"])
    arrays, heads, multitask = _build_arrays(cfg, ds)
    year = np.asarray(arrays.year, dtype=int)
    X = np.asarray(arrays.X, dtype=float)
    dev = np.isin(year, list(train_years))
    test = year == int(test_year)
    test_idx = np.flatnonzero(test)
    if target_head not in heads and multitask:
        raise ValueError(f"{label}: cabeza {target_head} no está en {heads}")
    col = heads.index(target_head) if multitask else 0
    train_lab = "-".join(str(y) for y in train_years)

    rows: list[dict[str, Any]] = []
    for s in spike_fracs:
        reps_use = (reps[0],) if s <= 0 else reps   # s=0 es determinista (nadie spikeado)
        n_spike = int(round(float(s) * test_idx.size))
        for rep in reps_use:
            rng = np.random.default_rng(int(rep))
            spiked = rng.permutation(test_idx)[:n_spike]
            spiked_mask = np.zeros(year.size, dtype=bool)
            spiked_mask[spiked] = True
            held_mask = test & ~spiked_mask
            tr_mask = dev | spiked_mask               # dev + líneas 2020 spikeadas (con fenotipo)

            model = build_model(cfg)
            _fit(model, X[tr_mask], arrays, tr_mask, heads, multitask)   # preproc train-only (aumentado)
            yp = _predict_heads(model, X[held_mask], heads, multitask, [target_head])[target_head]
            if multitask:
                yt = arrays.Y[held_mask][:, col]
                m = np.asarray(arrays.Y_mask[held_mask][:, col], dtype=bool)
                yt, yp = yt[m], np.asarray(yp)[m]
            else:
                yt = np.asarray(arrays.y)[held_mask]
            rows.append({
                "model": label, "train": train_lab, "spike": float(s), "rep": int(rep),
                "pearson": float(pearson_r(yt, yp)), "n_held": int(np.size(yt)), "n_spike": int(n_spike),
            })
    return rows
