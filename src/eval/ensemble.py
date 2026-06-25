"""Ensembles de predicciones para el reporte: blend MLP+GBLUP (A3) y promedio de redes (A4).

Trabaja sobre las predicciones por fold que devuelve `run_evaluation`
(``result["predictions"]``): cada entrada trae ``line_id`` + ``y_true`` + ``y_pred`` del
val de un fold. Estas utilidades:

- **alinean por línea** las predicciones de modelos distintos (mismo split temporal ⇒ en la
  práctica las mismas filas, pero se alinea por `line_id` por las dudas);
- **eligen el peso del blend** ``alpha`` en un inner-val ANTERIOR (regla 3: nunca en el set
  de reporte, evita winner's curse);
- arman el **IC bootstrap pareado** de (modelo − referencia) reusando
  `src.eval.metrics.bootstrap_ci` con la misma seed/grupo, así el ruido compartido del val
  se cancela (mismo criterio que la celda pareada del notebook).

La lógica vive acá (no en el notebook) para respetar la convención del proyecto
(notebooks solo importan y llaman funciones de `src/`) y para poder testearla.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.eval.metrics import bootstrap_ci, coincidence_index, pearson_r


def fold_predictions(
    result: dict[str, Any], *, role: str | None = None, val_year: int | None = None
) -> dict[str, Any]:
    """Devuelve la entrada de ``result['predictions']`` que matchea (por defecto la última).

    Con split fijo hay un solo fold de val ⇒ ``[-1]`` alcanza. ``role`` ("scoring"/"report")
    y ``val_year`` filtran cuando hay varios folds (expanding window).
    """
    preds = result.get("predictions") or []
    if not preds:
        raise ValueError(
            "El resultado no trae 'predictions'. ¿run_evaluation viejo o caché v2? "
            "Subí _METRICS_VERSION / recomputá (USE_CACHE=False)."
        )
    cand = preds
    if role is not None:
        cand = [p for p in cand if p["role"] == role]
    if val_year is not None:
        cand = [p for p in cand if str(p["val_year"]) == str(val_year)]
    if not cand:
        raise ValueError(f"Sin predicciones para role={role!r} val_year={val_year!r}.")
    return cand[-1]


def align_by_line(*preds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Alinea N dicts de predicción por ``line_id`` común.

    Devuelve ``(line_id, y_true, [y_pred_0, y_pred_1, ...])`` ordenados igual en todos. Toma
    la intersección de líneas (mismo split ⇒ normalmente idénticas) y verifica que el
    ``y_true`` observado coincida entre modelos (misma línea/año/ambiente ⇒ mismo fenotipo).
    """
    if not preds:
        raise ValueError("align_by_line necesita al menos una predicción.")

    def _index(p: dict[str, Any]) -> dict[Any, int]:
        ids = np.asarray(p["line_id"]).ravel()
        idx = {l: i for i, l in enumerate(ids.tolist())}
        if len(idx) != ids.size:
            raise ValueError("line_id repetido dentro de un fold (no debería pasar por env/año).")
        return idx

    indexes = [_index(p) for p in preds]
    common = set(indexes[0])
    for idx in indexes[1:]:
        common &= set(idx)
    if not common:
        raise ValueError("Las predicciones no comparten ninguna línea (¿splits distintos?).")
    line_id = np.array(sorted(common, key=lambda v: str(v)))

    sel = [np.array([idx[l] for l in line_id.tolist()]) for idx in indexes]
    y_true = np.asarray(preds[0]["y_true"], dtype=float).ravel()[sel[0]]
    for p, s in zip(preds[1:], sel[1:]):
        yt_other = np.asarray(p["y_true"], dtype=float).ravel()[s]
        if not np.allclose(y_true, yt_other, equal_nan=True):
            raise ValueError("y_true no coincide entre modelos para las mismas líneas.")
    yhats = [np.asarray(p["y_pred"], dtype=float).ravel()[s] for p, s in zip(preds, sel)]
    return line_id, y_true, yhats


def blend(yhat_a, yhat_b, alpha: float) -> np.ndarray:
    """Combinación convexa ``alpha·a + (1-alpha)·b`` (a = primer modelo, p.ej. el MLP)."""
    a = float(alpha)
    return a * np.asarray(yhat_a, dtype=float) + (1.0 - a) * np.asarray(yhat_b, dtype=float)


def select_blend_alpha(
    y_inner,
    yhat_a_inner,
    yhat_b_inner,
    *,
    metric: str = "pearson",
    grid: np.ndarray | None = None,
    frac: float = 0.2,
) -> tuple[float, list[tuple[float, float]]]:
    """Elige ``alpha`` en el INNER-val maximizando la métrica del blend (regla 3).

    ``metric``: ``"pearson"`` (default, portada) o ``"matching20"`` (coincidence index al
    top-``frac``). Devuelve ``(alpha, scores)`` con ``scores = [(alpha, valor), ...]`` para
    graficar la curva. El ``alpha`` elegido se aplica DESPUÉS al set de reporte, que esta
    función NO toca.
    """
    grid_arr = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, dtype=float)
    if metric == "matching20":
        fn: Callable = lambda yt, yp: coincidence_index(yt, yp, frac)
    elif metric == "pearson":
        fn = pearson_r
    else:
        raise ValueError(f"metric desconocida: {metric!r}. Use 'pearson' o 'matching20'.")
    scores = [(float(a), float(fn(y_inner, blend(yhat_a_inner, yhat_b_inner, a)))) for a in grid_arr]
    finite = [s for s in scores if np.isfinite(s[1])]
    if not finite:
        raise ValueError("Ningún alpha dio una métrica finita (¿predicción constante?).")
    best_alpha = max(finite, key=lambda t: t[1])[0]
    return best_alpha, scores


def average_predictions(yhats: list) -> np.ndarray:
    """Promedio simple de una lista de vectores YA alineados (A4: ensemble de redes)."""
    if not yhats:
        raise ValueError("average_predictions recibió una lista vacía.")
    return np.mean(np.vstack([np.asarray(y, dtype=float).ravel() for y in yhats]), axis=0)


def paired_bootstrap_diff(
    y_true,
    yhat,
    yhat_ref,
    metrics: dict[str, Callable],
    *,
    group=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """IC bootstrap PAREADO de (modelo − referencia) por métrica.

    Corre `bootstrap_ci` para los dos modelos con la MISMA ``seed`` y ``group`` (cluster por
    línea) ⇒ ``samples_modelo[b] − samples_ref[b]`` es un bootstrap pareado válido (el ruido
    de qué líneas caen en val se cancela). Devuelve ``{name: {"point","lo","hi","samples"}}``
    de la diferencia; si el IC no cruza 0, la diferencia es significativa (signo: + favorece
    al modelo en pearson_r/r2/matching20, lo contrario en rmse).
    """
    b_mod = bootstrap_ci(y_true, yhat, metrics, group=group, n_boot=n_boot, alpha=alpha, seed=seed)
    b_ref = bootstrap_ci(y_true, yhat_ref, metrics, group=group, n_boot=n_boot, alpha=alpha, seed=seed)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, Any]] = {}
    for name, fn in metrics.items():
        d = b_mod[name]["samples"] - b_ref[name]["samples"]
        d = d[np.isfinite(d)]
        point = float(fn(y_true, yhat) - fn(y_true, yhat_ref))
        if d.size:
            lo, hi = np.nanpercentile(d, [lo_q, hi_q])
        else:
            lo = hi = float("nan")
        out[name] = {"point": point, "lo": float(lo), "hi": float(hi), "samples": d}
    return out
