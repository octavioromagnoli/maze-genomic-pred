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


def coincidence_index(y_true, y_pred, frac: float = 0.2, *, higher_is_better: bool = True) -> float:
    """Coincidence index (*matching*) para selección: aciertos en el top-``frac``.

    Mide la utilidad del modelo para **seleccionar líneas**, no para predecir el valor
    exacto: de las líneas que realmente caen en el mejor ``frac`` por fenotipo
    **observado**, ¿qué proporción el modelo también pone en su mejor ``frac`` por valor
    **predicho**? Es la métrica de selección truncada estándar en mejoramiento genómico.

    Ojo: se evalúa sobre **líneas/genotipos** (las observaciones del fold), no sobre
    "genes" ni marcadores. ``frac=0.2`` ⇒ matching20 (top-20%).

    Args:
        frac: fracción de corte en (0, 1]. 0.2 = top-20%.
        higher_is_better: ``True`` (p.ej. GY: más rinde = mejor) toma los valores más
            altos; ``False`` (p.ej. días a floración más cortos) invierte el orden.

    Returns:
        Proporción de aciertos en ``[0, 1]`` = ``|top_obs ∩ top_pred| / k`` con
        ``k = round(frac·n)``. **Referencia**: una selección al azar acierta ~``frac``
        (0.2 para el top-20%); 1.0 = recupera exactamente el top observado. ``nan`` si
        ``n < 2`` o si alguna serie es ~constante (sin ranking posible, igual que
        `pearson_r`). El conteo "a cuántos les pegó" se recupera como ``round(ci·k)``.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    n = yt.size
    if n < 2:
        return float("nan")
    if yt.std() < _EPS or yp.std() < _EPS:   # predicción/target constante => top ambiguo
        return float("nan")
    k = max(1, int(round(frac * n)))
    if k >= n:                       # frac ~1: el top es todo => coincidencia trivial
        return 1.0
    if not higher_is_better:
        yt, yp = -yt, -yp
    top_true = np.argpartition(yt, -k)[-k:]   # índices de los k mayores (sin ordenar)
    top_pred = np.argpartition(yp, -k)[-k:]
    return float(np.intersect1d(top_true, top_pred).size / k)


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


def bootstrap_ci(
    y_true,
    y_pred,
    metrics: dict,
    *,
    group=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """IC percentil por bootstrap para varias métricas a la vez.

    Args:
        metrics: dict ``{nombre: fn(y_true, y_pred) -> float}``.
        group: etiqueta de grupo por observación para el cluster bootstrap;
            ``None`` = bootstrap de filas independientes.
        n_boot: cantidad de réplicas.
        alpha: nivel; el IC es ``[alpha/2, 1-alpha/2]`` (0.05 => IC 95%).
        seed: semilla del RNG (reproducible).

    Devuelve ``{nombre: {"lo": float, "hi": float, "samples": np.ndarray}}``.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    rng = np.random.default_rng(int(seed))
    names = list(metrics)
    draws = {name: np.empty(int(n_boot)) for name in names}

    if group is None:
        n = yt.size
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            for name in names:
                draws[name][b] = metrics[name](yt[idx], yp[idx])
    else:
        group = np.asarray(group)
        members = [np.flatnonzero(group == g) for g in np.unique(group)]
        K = len(members)
        for b in range(int(n_boot)):
            idx = np.concatenate([members[j] for j in rng.integers(0, K, K)])
            for name in names:
                draws[name][b] = metrics[name](yt[idx], yp[idx])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict] = {}
    for name in names:
        s = draws[name]
        if np.isfinite(s).any():
            lo, hi = np.nanpercentile(s, [lo_q, hi_q])
        else:
            lo = hi = float("nan")  # métrica degenerada (p. ej. predicción constante)
        out[name] = {"lo": float(lo), "hi": float(hi), "samples": s}
    return out
