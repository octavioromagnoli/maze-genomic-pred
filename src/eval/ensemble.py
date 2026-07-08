"""Ensembles de predicciones para el reporte: blend MLP+GBLUP y promedio de redes.

Trabaja sobre las predicciones por fold de run_evaluation (result["predictions"]):
cada entrada trae line_id + y_true + y_pred del val de un fold. Alinea por línea las
predicciones de modelos distintos, elige el peso alpha del blend en un inner-val
anterior (nunca en el set de reporte) y arma el IC bootstrap pareado de
(modelo - referencia) con bootstrap_ci, con la misma seed/grupo para que el ruido
compartido del val se cancele. Vive acá y no en el notebook para poder testearlo.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.eval.metrics import bootstrap_ci, coincidence_index, pearson_r


def fold_predictions(
    result: dict[str, Any], *, role: str | None = None, val_year: int | None = None
) -> dict[str, Any]:
    """Devuelve la entrada de result['predictions'] que matchea (por defecto la última).

    Con split fijo hay un solo fold de val => [-1] alcanza. role ("scoring"/"report")
    y val_year filtran cuando hay varios folds (expanding window).
    """
    preds = result.get("predictions") or []
    if not preds:
        raise ValueError(
            "El resultado no trae 'predictions'. ¿caché viejo, de antes de que "
            "run_evaluation devolviera predicciones? Recomputá (USE_CACHE=False) o "
            "borrá outputs/eval_cache."
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
    """Alinea N dicts de predicción por line_id común.

    Devuelve (line_id, y_true, [y_pred_0, y_pred_1, ...]) ordenados igual en todos. Toma
    la intersección de líneas (mismo split => normalmente idénticas) y verifica que el
    y_true observado coincida entre modelos (misma línea/año/ambiente => mismo fenotipo).
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
    """Combinación convexa alpha*a + (1-alpha)*b (a = primer modelo, p.ej. el MLP)."""
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
    """Elige alpha en el inner-val maximizando la métrica del blend.

    metric: "pearson" (default, portada) o "matching20" (coincidence index al
    top-frac). Devuelve (alpha, scores) con scores = [(alpha, valor), ...] para
    graficar la curva. El alpha elegido se aplica DESPUÉS al set de reporte, que esta
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


def _zfit(yhat) -> tuple[float, float]:
    """Media y desvío de un vector de predicción (desvío 0 -> 1, evita dividir por 0)."""
    yhat = np.asarray(yhat, dtype=float)
    mu, sd = float(np.mean(yhat)), float(np.std(yhat))
    return mu, (sd if sd > 1e-12 else 1.0)


def _zapply(yhat, params: tuple[float, float]) -> np.ndarray:
    mu, sd = params
    return (np.asarray(yhat, dtype=float) - mu) / sd


def average_predictions(yhats: list, *, standardize: bool = False) -> np.ndarray:
    """Promedio de una lista de vectores YA alineados (A4: ensemble de redes).

    Con standardize=True z-scorea cada predictor por su propia media/desvío ANTES de
    promediar: necesario si los modelos tienen escalas dispares (p.ej. MLP con
    loss=pearson), porque si no el de mayor varianza domina el promedio y el "ensemble"
    colapsa a ese modelo. No usa el target => no introduce leakage.
    """
    if not yhats:
        raise ValueError("average_predictions recibió una lista vacía.")
    arrs = [np.asarray(y, dtype=float).ravel() for y in yhats]
    if standardize:
        arrs = [_zapply(a, _zfit(a)) for a in arrs]
    return np.mean(np.vstack(arrs), axis=0)


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

    Corre bootstrap_ci para los dos modelos con la MISMA seed y group (cluster por
    línea) => samples_modelo[b] − samples_ref[b] es un bootstrap pareado válido (el ruido
    de qué líneas caen en val se cancela). Devuelve {name: {"point","lo","hi","samples"}}
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


def bootstrap_table(
    y_true,
    preds_by_name: dict[str, Any],
    metrics: dict[str, Callable],
    *,
    group=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    n: int | None = None,
    standardize: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """IC bootstrap de varias predicciones YA alineadas, en el formato de los gráficos.

    preds_by_name = {nombre: y_pred} (todas alineadas al mismo y_true/group).
    Corre bootstrap_ci por modelo con la MISMA seed y group => las diferencias
    muestras_x − muestras_ref quedan pareadas (mismo resample). Devuelve:

    - rows: lista tidy [{model, metric, point, lo, hi, n}] (igual que evaluate_model)
      => se arma un DataFrame y se pasa a :func:src.eval.report_plots.plot_ci / tablas.
    - samples_by_model: {nombre: {metrica: muestras}} => :func:plot_box /
      :func:plot_paired_diff.

    Con standardize=True z-scorea cada predictor por su propia media/desvío antes de las
    métricas: pone a todos en escala comparable (útil cuando hay modelos de escala arbitraria,
    p.ej. MLP con loss=pearson). Pearson y matching20 son invariantes; RMSE/R² pasan a
    escala estandarizada (no en unidades del target => solo orientativos). No usa el target.

    Útil para los ensembles del notebook (A3/A4), donde el "modelo" es un vector de
    predicciones combinado (blend, promedio) que no pasa por run_evaluation.
    """
    n = int(np.size(np.asarray(y_true))) if n is None else int(n)
    rows: list[dict[str, Any]] = []
    samples_by_model: dict[str, dict[str, np.ndarray]] = {}
    for name, yhat in preds_by_name.items():
        if standardize:
            yhat = _zapply(yhat, _zfit(yhat))
        b = bootstrap_ci(y_true, yhat, metrics, group=group, n_boot=n_boot, alpha=alpha, seed=seed)
        s: dict[str, np.ndarray] = {}
        for m, fn in metrics.items():
            rows.append({
                "model": name, "metric": m,
                "point": float(fn(y_true, yhat)),
                "lo": float(b[m]["lo"]), "hi": float(b[m]["hi"]), "n": n,
            })
            s[m] = b[m]["samples"]
        samples_by_model[name] = s
    return rows, samples_by_model


def paired_diff_summary(
    samples_by_model: dict[str, dict[str, np.ndarray]],
    ref: str,
    metrics,
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Resumen de (modelo − ref) por métrica desde las muestras pareadas (texto/significancia).

    Mismo criterio que las celdas pareadas del notebook: IC95% de la resta sobre el mismo
    resample; si no cruza 0, la diferencia es significativa. Devuelve
    {modelo: {metrica: {"delta", "lo", "hi", "sig"}}} (excluye la referencia).
    """
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, s in samples_by_model.items():
        if name == ref:
            continue
        per: dict[str, dict[str, Any]] = {}
        for m in metrics:
            sx = np.asarray(s[m], dtype=float)
            sr = np.asarray(samples_by_model[ref][m], dtype=float)
            k = min(len(sx), len(sr))
            d = (sx[:k] - sr[:k])
            d = d[np.isfinite(d)]
            if d.size:
                lo, hi = np.nanpercentile(d, [lo_q, hi_q])
                per[m] = {"delta": float(np.mean(d)), "lo": float(lo), "hi": float(hi),
                          "sig": bool(lo > 0 or hi < 0)}
            else:
                per[m] = {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
        out[name] = per
    return out
