"""
Métricas de regresión, centralizadas.
Las métricas individuales (pearson_r, rmse, r2) devuelven `nan` si no se pueden calcular
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes distintos: y_true {yt.shape} vs y_pred {yp.shape}.")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def pearson_r(y_true, y_pred) -> float:
    """Correlación de Pearson. `nan` si n<2 o alguna serie tiene varianza ~0."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    st, sp = yt.std(), yp.std()  # ddof=0, consistente
    if st < _EPS or sp < _EPS:
        return float("nan")
    return float(np.mean((yt - yt.mean()) * (yp - yp.mean())) / (st * sp))


def rmse(y_true, y_pred) -> float:
    """Raíz del error cuadrático medio."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coeficiente de determinación. `nan` si la varianza de y_true es ~0."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < _EPS:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Devuelve las tres métricas + n en un dict (orden: portada primero)."""
    yt, _ = _clean_pair(y_true, y_pred)
    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "n": int(yt.size),
    }


def summarize_folds(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Agrega métricas por fold en mean/std (ignorando `nan`).

    Para cada métrica ``m`` agrega ``m_mean`` y ``m_std``. Útil para el resumen
    de CV que se loguea a ``wandb.summary``.
    """
    if not per_fold:
        return {}
    keys = [k for k in per_fold[0] if k != "n"]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([f.get(k, np.nan) for f in per_fold], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        out[f"{k}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan")
    out["n_folds"] = len(per_fold)
    return out
