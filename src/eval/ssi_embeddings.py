"""Embeddings del tronco compartido de la red MTL, para el kernel aprendido del SSI.

Entrena (o reusa) UNA red multitask sobre el training de dev (year < target_year, p.ej. 2017–2019),
la congela, y extrae el embedding del TRONCO compartido (penultima capa, antes de las cabezas
por-trait) para cualquier matriz de marcadores.

============================ FRONTERA DE LEAKAGE (leer) ============================
La red se entrena SOLO con marcadores -> fenotipos de dev; NUNCA ve fenotipos de 2020. El embedding
de un individuo es una funcion determinista de SUS MARCADORES, asi que calcularlo para los individuos
de 2020 (incluido el prediction set) NO filtra su fenotipo: define la SIMILITUD, no la prediccion.
La prediccion final del SSI usa solo fenotipos del training (ver `run_ssi.py`).
===================================================================================
"""
from __future__ import annotations

import numpy as np

from src.data.loader import Dataset
from src.models.registry import build_model


def _trunk_of(net) -> "object":
    """Devuelve el tronco (Sequential sin la cabeza Linear final) de un `_MLPNet`."""
    import torch.nn as nn

    return nn.Sequential(*list(net.net.children())[:-1])  # todo menos el Linear(out_dim) final


def _embed_single(model, Xs_tensor):
    """Embedding del tronco de UN `MLPMultitaskRegressor` ya entrenado (net en eval)."""
    import torch

    net = model.net_
    net.eval()
    trunk = _trunk_of(net)
    device = next(net.parameters()).device
    with torch.no_grad():
        return trunk(Xs_tensor.to(device)).cpu().numpy()


def trunk_embedding(mtl_model, X: np.ndarray) -> np.ndarray:
    """Embedding del tronco compartido para los marcadores ``X`` ``(n, n_snps)``.

    Aplica el z-score de entrada train-only del modelo (``x_mean_``/``x_std_``) y corre el tronco en
    modo eval (sin dropout ⇒ determinista). Para un `EnsembleMultitaskRegressor` promedia los
    embeddings de sus sub-redes (misma arquitectura ⇒ dims alineadas).
    """
    import torch

    X = np.asarray(X, dtype=np.float64)
    subs = getattr(mtl_model, "models_", None)  # ensemble -> lista de MLPMultitaskRegressor
    members = list(subs) if subs else [mtl_model]
    accum = None
    for m in members:
        Xs = (X - m.x_mean_) / m.x_std_
        emb = _embed_single(m, torch.as_tensor(Xs, dtype=torch.float32))
        accum = emb if accum is None else accum + emb
    return accum / len(members)


def train_mtl_on_dev(dataset: Dataset, mtl_config: dict, *, target_year: int = 2020):
    """Entrena la red MTL del `mtl_config` con el training de DEV (year < target_year).

    Reusa `build_model` (mlp_multitask / ensemble_multitask) y su `fit(X, Y, Y_mask)`. Los targets y
    el ambiente salen del propio `mtl_config` (bloque `data`). Devuelve el modelo entrenado.
    """
    data_cfg = mtl_config.get("data", {})
    if data_cfg.get("multienv"):
        target = data_cfg.get("target", "GY")
        envs = list(data_cfg.get("envs") or ["optimal", "drought"])
        arr = dataset.to_modeling_arrays_multienv(target=target, envs=envs)
        target_names = [f"{target}_{e}" for e in envs]
    else:
        target_names = list(data_cfg.get("targets") or ["GY", "PH", "AD"])
        arr = dataset.to_modeling_arrays_multitask(targets=target_names, env=data_cfg.get("env"))

    dev = np.asarray(arr.year) < int(target_year)  # <-- dev only: la red nunca ve 2020
    model = build_model(mtl_config)
    model.set_fit_context(
        year=arr.year[dev], line_id=arr.line_id[dev], target_names=target_names
    )
    model.fit(arr.X[dev], arr.Y[dev], arr.Y_mask[dev])
    return model


def compute_mtl_embeddings(
    dataset: Dataset, mtl_config: dict, X_target: np.ndarray, *, target_year: int = 2020
) -> np.ndarray:
    """Entrena la MTL en dev y devuelve el embedding del tronco para los marcadores ``X_target``.

    ``X_target`` son los marcadores de los individuos del SSI (todos los años, incluido 2020). El
    embedding sale solo de esos marcadores (ver frontera de leakage arriba).
    """
    model = train_mtl_on_dev(dataset, mtl_config, target_year=target_year)
    return trunk_embedding(model, X_target)
