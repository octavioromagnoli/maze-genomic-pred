"""Evaluacion del Sparse Selection Index (SSI) con el protocolo del paper (Lopez-Cruz 2021).

Hermano de `run_evaluation`/`run_multitask_evaluation`, pero para el SSI: NO ajusta un modelo
`fit`/`predict`, sino que por cada (particion de 2020 x composicion de training x kernel) resuelve la
Eq. 3 (via `src.eval.ssi_solver`) y predice ``u_i = sum_j b_ij y_j`` con fenotipos crudos del TS.

Reusa sin reescribir:
- `lambda_via_reml` de `src/models/gblup.py` para ``lambda0 = sigma_e^2/sigma_u^2`` por training.
- kernels (`src/eval/kernels.py`), splits (`src/eval/ssi_splits.py`), embeddings (`ssi_embeddings.py`).
- `pearson_r` de `src/eval/metrics.py` para la accuracy.

Baselines apples-to-apples: G aditiva (GSSI/lasso-GBLUP del compañero), Gaussiano sobre SNPs (KSSI) y
el aporte (K_learned Gram/Gaussiano) corren con el MISMO solver, splits y protocolo de lambda.

===================== FRONTERA DE LEAKAGE =====================
Kernels: SOLO marcadores/embeddings (nunca fenotipos). Embedding MTL: entrenado en dev, aplicado a
2020 solo por marcadores. Prediccion: SOLO fenotipos del TS. Fenotipos del prediction set 2020: held out
(solo se usan para medir accuracy al final). lambda0: REML sobre fenotipos del TS unicamente.
==============================================================
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data.loader import load_dataset
from src.eval.kernels import build_kernels
from src.eval.metrics import pearson_r
from src.eval.ssi_embeddings import compute_mtl_embeddings
from src.eval.ssi_solver import (
    lambda_grid,
    predict_with_training_phenotypes,
    select_lambda_cv,
    ssi_solve_batch,
)
from src.eval.ssi_splits import compose_training, iter_settings, partition_target_cycle
from src.models.gblup import lambda_via_reml
from src.utils.config import load_config, set_nested

LogFn = Any


def _noop(_: dict) -> None:
    return None


def ssi_fit_predict(
    K_TS: np.ndarray, K_PS_TS: np.ndarray, y_TS: np.ndarray, lambda0: float, lam: float,
    *, max_iter: int, tol: float,
) -> np.ndarray:
    """Resuelve la Eq. 3 para todo el prediction set y predice ``u = B' y_TS``."""
    n_ts = K_TS.shape[0]
    P = K_TS + float(lambda0) * np.eye(n_ts)
    C = np.ascontiguousarray(K_PS_TS.T)          # (n_TS, n_PS): columna = K_0i por individuo
    B, _ = ssi_solve_batch(P, C, float(lam), max_iter=max_iter, tol=tol)
    return predict_with_training_phenotypes(B, y_TS)


def _run_partition(p, *, arr_year, arr_y, kernels, settings, ssi_cfg, seed) -> list[dict]:
    """Todas las (composicion x kernel) de UNA particion de 2020. Aislado para paralelizar con joblib."""
    lam_cfg = ssi_cfg.get("lambda", {})
    sol_cfg = ssi_cfg.get("solver", {})
    n_lambda = int(lam_cfg.get("n_lambda", 50))
    lmr = float(lam_cfg.get("lambda_min_ratio", 0.01))
    cv_folds = int(lam_cfg.get("cv_folds", 10))
    max_iter = int(sol_cfg.get("max_iter", 1000))
    tol = float(sol_cfg.get("tol", 1e-7))
    target_year = int(ssi_cfg.get("target_year", 2020))
    frac_predict = float(ssi_cfg.get("frac_predict", 0.85))
    max_train_size = ssi_cfg.get("max_train_size")
    max_predict_size = ssi_cfg.get("max_predict_size")

    available, predict = partition_target_cycle(
        arr_year, target_year=target_year, frac_predict=frac_predict, seed=seed + p
    )
    if max_predict_size is not None and predict.size > int(max_predict_size):
        rng = np.random.default_rng(seed + p * 100_003)
        predict = np.sort(rng.choice(predict, size=int(max_predict_size), replace=False))
    y_ps = arr_y[predict]
    records: list[dict] = []
    for si, (label, cycle_years, pct) in enumerate(settings):
        train_idx = compose_training(
            arr_year, available, cycle_years=cycle_years, pct_2020=pct,
            target_year=target_year, seed=seed + p * 1000 + si,
        )
        if max_train_size is not None and train_idx.size > int(max_train_size):
            rng = np.random.default_rng(seed + p * 1000 + si)
            train_idx = np.sort(rng.choice(train_idx, size=int(max_train_size), replace=False))
        if train_idx.size < max(cv_folds, 3) or predict.size < 2:
            continue
        y_tr = arr_y[train_idx]
        for kname, K in kernels.items():
            K_TS = K[np.ix_(train_idx, train_idx)]
            K_PS_TS = K[np.ix_(predict, train_idx)]
            lambda0 = float(lambda_via_reml(K_TS, y_tr)[0])          # sigma_e^2/sigma_u^2 (REML, TS)
            offdiag = K_TS.copy()
            np.fill_diagonal(offdiag, 0.0)
            lambdas = lambda_grid(offdiag, n_lambda=n_lambda, lambda_min_ratio=lmr)
            lam, _ = select_lambda_cv(
                K_TS, y_tr, lambda0, lambdas, n_folds=cv_folds, seed=seed + p, max_iter=max_iter, tol=tol
            )
            u = ssi_fit_predict(K_TS, K_PS_TS, y_tr, lambda0, lam, max_iter=max_iter, tol=tol)
            records.append({
                "partition": int(p), "kernel": kname, "composition": label,
                "accuracy": float(pearson_r(y_ps, u)), "lambda": float(lam),
                "lambda0": lambda0, "n_train": int(train_idx.size), "n_predict": int(predict.size),
            })
    return records


def run_ssi_evaluation(config: dict[str, Any], log_fn: LogFn | None = None) -> dict[str, Any]:
    """Corre el protocolo SSI del paper y devuelve accuracy media (±SD) por (kernel, composicion)."""
    log = log_fn or _noop
    data_cfg = config.get("data", {})
    ssi_cfg = config.get("ssi", {})
    target = ssi_cfg.get("target", "GY")
    env = ssi_cfg.get("env", "drought")
    target_year = int(ssi_cfg.get("target_year", 2020))
    seed = int(ssi_cfg.get("seed", config.get("seed", 575)))
    n_partitions = int(ssi_cfg.get("n_partitions", 100))
    n_jobs = int(ssi_cfg.get("n_jobs", 1))

    # --- Datos del trait×ambiente objetivo (una fila por (GID, Year); todos los años incl. 2020) ---
    dataset = load_dataset(data_cfg["geno_csv"], data_cfg["pheno_csv"])
    arr = dataset.to_modeling_arrays(target=target, env=env)
    n_2020 = int(np.sum(np.asarray(arr.year) == target_year))
    if n_2020 == 0:
        raise RuntimeError(f"No hay individuos de {target_year} para {target}×{env}.")

    # --- Embedding MTL (solo si algun kernel lo pide). Frontera de leakage en ssi_embeddings.py ---
    kernels_spec = list(ssi_cfg.get("kernels", [{"kind": "G"}]))
    needs_emb = any(str(k["kind"]).endswith("emb") for k in kernels_spec)
    embedding = None
    if needs_emb:
        emb_cfg = ssi_cfg["embedding"]
        mtl_config = load_config(emb_cfg["mtl_config"])
        for dotted, val in (emb_cfg.get("overrides") or {}).items():  # p.ej. bajar epochs en el smoke
            set_nested(mtl_config, dotted, val)
        embedding = compute_mtl_embeddings(dataset, mtl_config, arr.X, target_year=target_year)
        log({"embedding_dim": int(embedding.shape[1])})

    # --- Kernels sobre TODOS los individuos (marcadores/embeddings; sin fenotipos) ---
    kernels = build_kernels(arr.X, embedding, kernels_spec)
    log({"n_all": int(arr.X.shape[0]), "n_2020": n_2020, "n_kernels": len(kernels)})

    settings = list(iter_settings(ssi_cfg.get("compositions"), ssi_cfg.get("pcts")))

    # --- 100 particiones (paralelizables: cada una es independiente) ---
    from joblib import Parallel, delayed

    part_args = dict(arr_year=np.asarray(arr.year), arr_y=np.asarray(arr.y, dtype=float),
                     kernels=kernels, settings=settings, ssi_cfg=ssi_cfg, seed=seed)
    if n_jobs == 1:
        all_records = [rec for p in range(n_partitions) for rec in _run_partition(p, **part_args)]
    else:
        batches = Parallel(n_jobs=n_jobs)(
            delayed(_run_partition)(p, **part_args) for p in range(n_partitions)
        )
        all_records = [rec for batch in batches for rec in batch]

    # --- Resumen por (kernel, composicion): media ± SD de accuracy sobre las particiones ---
    per_setting: list[dict] = []
    keys = sorted({(r["kernel"], r["composition"]) for r in all_records})
    for kname, comp in keys:
        accs = np.array([r["accuracy"] for r in all_records
                         if r["kernel"] == kname and r["composition"] == comp], dtype=float)
        accs = accs[np.isfinite(accs)]
        lams = np.array([r["lambda"] for r in all_records
                         if r["kernel"] == kname and r["composition"] == comp], dtype=float)
        row = {
            "kernel": kname, "composition": comp,
            "acc_mean": float(np.mean(accs)) if accs.size else float("nan"),
            "acc_sd": float(np.std(accs, ddof=1)) if accs.size > 1 else 0.0,
            "n_valid": int(accs.size), "lambda_mean": float(np.mean(lams)) if lams.size else float("nan"),
        }
        per_setting.append(row)
        log({f"acc/{kname}/{comp}": row["acc_mean"]})

    return {
        "target": target, "env": env, "target_year": target_year,
        "n_all": int(arr.X.shape[0]), "n_2020": n_2020,
        "kernels": list(kernels.keys()), "n_partitions": n_partitions,
        "per_setting": per_setting, "records": all_records,
    }
