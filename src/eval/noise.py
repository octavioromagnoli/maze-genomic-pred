"""Robustez al ruido: degradación de Pearson bajo corrupción de SNPs.

Reusa el loader, el split temporal del config (train_years / val_year) y pearson_r; lo
único que agrega es la perturbación de X. Los SNPs son binarios, así que el ruido es un
flip Bernoulli: cada entrada se invierte con probabilidad eps (modela error de
genotipado). El preproceso del modelo se ajusta siempre train-only.

En mode="inference" se entrena con el train limpio una vez y se evalúa sobre X_val
corrupto a cada eps (robustez de despliegue; un fit, muchos predicts). En mode="train"
se entrena con X_train corrupto y se evalúa sobre val limpio (ruido como
regularización; un fit por eps).

noise_curve devuelve filas tidy {model, mode, eps, realization, head, pearson, n}; para
cada eps promedia n_realizations realizaciones. eps=0 es determinista y ancla la curva.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.data.loader import load_dataset
from src.eval.metrics import pearson_r
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def flip_snps(X: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Flip Bernoulli de bits sobre una matriz de SNPs binarios {0,1}.

    Cada entrada se invierte (x -> 1-x) con probabilidad eps. eps<=0 devuelve X intacto
    (copia). No asume que X sea entera: opera sobre {0.0, 1.0}.
    """
    if eps <= 0:
        return np.asarray(X)
    Xn = np.array(X, dtype=float, copy=True)
    mask = rng.random(Xn.shape) < float(eps)
    Xn[mask] = 1.0 - Xn[mask]
    return Xn


def _build_arrays(cfg: dict[str, Any], ds: Any) -> tuple[Any, list[str], bool]:
    """Devuelve (arrays, heads, multitask) según el modo del config (multi_trait_env / multienv /
    single-task). heads son los nombres de cabeza (GY_optimal, ...)."""
    d = cfg["data"]
    if d.get("multi_trait_env"):
        traits = list(d.get("targets") or ["GY", "PH", "AD"])
        envs = list(d.get("envs") or ["optimal", "drought"])
        a = ds.to_modeling_arrays_multi_trait_env(targets=traits, envs=envs)
        return a, [f"{t}_{e}" for t in traits for e in envs], True
    if d.get("multienv"):
        t = d.get("target", "GY")
        envs = list(d.get("envs") or ["optimal", "drought"])
        a = ds.to_modeling_arrays_multienv(target=t, envs=envs)
        return a, [f"{t}_{e}" for e in envs], True
    a = ds.to_modeling_arrays(d["target"], d.get("env"))
    return a, [f'{d["target"]}_{d.get("env")}'], False


def _fit(model: Any, X_tr: np.ndarray, arrays: Any, tr: np.ndarray,
         heads: list[str], multitask: bool) -> None:
    """Fittea el modelo (fresco) sobre el train tr con X_tr (limpio o corrupto)."""
    year_tr = np.asarray(arrays.year)[tr]
    line_tr = np.asarray(arrays.line_id)[tr]
    if multitask:
        model.set_fit_context(year=year_tr, line_id=line_tr, target_names=heads)
        model.fit(X_tr, arrays.Y[tr], arrays.Y_mask[tr])
    else:
        model.set_fit_context(year=year_tr, line_id=line_tr)
        model.fit(X_tr, np.asarray(arrays.y)[tr])


def _predict_heads(model: Any, X_va: np.ndarray, heads: list[str], multitask: bool,
                   want: Sequence[str]) -> dict[str, np.ndarray]:
    """Predice y devuelve {head: y_pred} para las cabezas pedidas que el modelo expone."""
    if multitask:
        Yhat = np.asarray(model.predict(X_va), dtype=float)
        return {h: Yhat[:, heads.index(h)] for h in want if h in heads}
    yp = np.asarray(model.predict(X_va), dtype=float).ravel()
    return {heads[0]: yp}


def _pearson_head(arrays: Any, va: np.ndarray, head: str, heads: list[str],
                  multitask: bool, y_pred: np.ndarray) -> tuple[float, int]:
    """Pearson de una cabeza sobre val, aplicando su máscara (solo datos reales)."""
    if multitask:
        j = heads.index(head)
        yt = arrays.Y[va][:, j]
        m = np.asarray(arrays.Y_mask[va][:, j], dtype=bool)
        yt, yp = yt[m], np.asarray(y_pred)[m]
    else:
        yt, yp = np.asarray(arrays.y)[va], np.asarray(y_pred)
    return float(pearson_r(yt, yp)), int(np.size(yt))


def noise_curve(
    config: str | dict[str, Any],
    *,
    label: str,
    epsilons: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
    mode: str = "inference",
    n_realizations: int = 5,
    heads: Sequence[str] = ("GY_optimal", "GY_drought"),
    seed: int = 575,
) -> list[dict[str, Any]]:
    """Curva de degradación de Pearson vs ε para UN modelo (config), en el fold honesto val.

    - mode="inference": un fit limpio + predicts sobre val corrupto a cada ε (barato).
    - mode="train": un fit por (ε, realización) sobre train corrupto, predict limpio (caro).

    Devuelve filas {model, mode, eps, realization, head, pearson, n} (una por ε×realización×cabeza).
    Para ε=0 hay una sola realización (sin ruido, determinista).
    """
    set_seed(seed)
    cfg = load_config(config) if isinstance(config, str) else dict(config)
    ds = load_dataset(cfg["data"]["geno_csv"], cfg["data"]["pheno_csv"])
    arrays, all_heads, multitask = _build_arrays(cfg, ds)
    want = [h for h in heads if (h in all_heads or not multitask)]

    cv = cfg["cv"]
    year = np.asarray(arrays.year)
    tr = np.isin(year, list(cv["train_years"]))
    va = year == int(cv["val_year"])
    X = np.asarray(arrays.X, dtype=float)
    rng_master = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []

    def _emit(eps, real, preds):
        for h, yp in preds.items():
            p, n = _pearson_head(arrays, va, h, all_heads, multitask, yp)
            rows.append({"model": label, "mode": mode, "eps": float(eps),
                         "realization": int(real), "head": h, "pearson": p, "n": n})

    if mode == "inference":
        model = build_model(cfg)
        _fit(model, X[tr], arrays, tr, all_heads, multitask)
        _emit(0.0, 0, _predict_heads(model, X[va], all_heads, multitask, want))
        for eps in [e for e in epsilons if e > 0]:
            for r in range(n_realizations):
                rng = np.random.default_rng(rng_master.integers(1 << 32))
                Xva_n = flip_snps(X[va], eps, rng)
                _emit(eps, r, _predict_heads(model, Xva_n, all_heads, multitask, want))
    elif mode == "train":
        model = build_model(cfg)
        _fit(model, X[tr], arrays, tr, all_heads, multitask)
        _emit(0.0, 0, _predict_heads(model, X[va], all_heads, multitask, want))
        for eps in [e for e in epsilons if e > 0]:
            for r in range(n_realizations):
                rng = np.random.default_rng(rng_master.integers(1 << 32))
                Xtr_n = flip_snps(X[tr], eps, rng)
                m = build_model(cfg)
                _fit(m, Xtr_n, arrays, tr, all_heads, multitask)
                _emit(eps, r, _predict_heads(m, X[va], all_heads, multitask, want))
    else:
        raise ValueError(f"mode debe ser 'inference' o 'train', no {mode!r}")

    return rows
