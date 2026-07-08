"""Diagnóstico de un sweep: ¿el ranking en val (2018) se sostiene en el reporte (2019)?

Un sweep rankea sus trials por la métrica sobre val 2018 (train 2017), pero el reporte
reentrena en 2017+2018 y mide sobre 2019. Que el top-1 en 2018 sea el mejor en 2019 no
está garantizado, y quedarse con "el k-ésimo porque gana en 2019" es seleccionar sobre
el set de reporte (winner's curse). Este módulo arma la evidencia para elegir sin sesgo:

- Selección (val 2018), multi-seed: reevalúa el top-k variando la seed y reporta
  media ± sd; un trial cuya ventaja sobrevive al cambio de seed es robusto. No mira 2019.
- Reporte (2019): reentrena cada trial en 2017+2018 y mide Pearson + IC95% (bootstrap
  cluster por línea) sobre 2019. Es un análisis de sensibilidad, no la excusa para
  quedarse con el que más sube.
- HP: devuelve los HP barridos de cada trial para leer qué tienen en común los buenos.

Sirve para cualquier sweep (multitask o single-task) y cualquier env: detecta el tipo
desde model.name y rutea a run_multitask_evaluation / run_evaluation. El año de reporte
se deriva del de selección o se pasa explícito. Reusa el caché de save_model. 2020 nunca
se evalúa acá.
"""
from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.training.run import run_evaluation
from src.training.run_multitask import run_multitask_evaluation
from src.training.save_model import (
    load_cached_results,
    results_cache_key,
    save_cached_results,
)
from src.training.sweep import top_runs_from_sweep
from src.utils.config import load_config, set_nested
from src.utils.seed import set_seed

_DEFAULT_SEEDS: tuple[int, ...] = (575, 576, 577, 578, 579)
_HP_PREFIX = "model.params."



def _is_multitask(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("model", {}).get("name", "")).strip() == "mlp_multitask"


def _select_value(res: dict[str, Any], multitask: bool, primary_trait: str) -> float:
    """Pearson del trait primario en el set de selección, leído del summary.

    Multitask expone "<trait>/pearson_r_mean" (y el alias pearson_r_GY);
    single-task expone pearson_r_mean. Un único fold de scoring => es el valor del fold.
    """
    s = res.get("summary", {})
    if multitask:
        return float(s.get(f"{primary_trait}/pearson_r_mean", s.get("pearson_r_GY", float("nan"))))
    return float(s.get("pearson_r_mean", float("nan")))


def _report_fold_metrics(
    res: dict[str, Any], multitask: bool, primary_trait: str
) -> tuple[float, float, float]:
    """(pearson, lo, hi) del trait primario en el fold de scoring del reporte.

    En multitask filtra trait == primary_trait y role == "scoring"; en single-task
    toma el último fold de scoring (ignora trait). Si no hay IC (bootstrap off) => NaN.
    """
    pf = res.get("per_fold", [])
    if multitask:
        cands = [f for f in pf if f.get("trait") == primary_trait and f.get("role") == "scoring"]
    else:
        cands = [f for f in pf if f.get("role") == "scoring"]
    if not cands:
        return float("nan"), float("nan"), float("nan")
    f = cands[-1]
    return (
        float(f.get("pearson_r", float("nan"))),
        float(f.get("pearson_r_lo", float("nan"))),
        float(f.get("pearson_r_hi", float("nan"))),
    )



def derive_report_cv(base_cv: dict[str, Any]) -> dict[str, Any]:
    """Deriva el cv de REPORTE corriendo el reloj un año respecto del de selección.

    El val de selección pasa a ser train y el año siguiente es el nuevo val
    (p.ej. {train [2017], val 2018} => {train [2017, 2018], val 2019}). Conserva
    scheme, holdout_test_year y enforce_disjoint_lines. Levanta error si el
    nuevo val caería sobre el holdout_test_year (no se reporta sobre el set sellado):
    en ese caso pasá report_cv explícito.
    """
    val = base_cv.get("val_year")
    if val is None:
        raise ValueError(
            "El cv base no tiene `val_year`; no puedo derivar el reporte. Pasá `report_cv` explícito."
        )
    trains = sorted({int(t) for t in (base_cv.get("train_years") or [])} | {int(val)})
    new_val = int(val) + 1
    htest = base_cv.get("holdout_test_year")
    if htest is not None and new_val == int(htest):
        raise ValueError(
            f"El reporte derivado caería sobre el año sellado (holdout_test_year={htest}). "
            "Pasá `report_cv` explícito o revisá el cv base."
        )
    cv = copy.deepcopy(base_cv)
    cv["scheme"] = base_cv.get("scheme", "fixed_temporal")
    cv["train_years"] = trains
    cv["val_year"] = new_val
    return cv


def _apply(base_cfg: dict[str, Any], overrides: dict[str, Any], cv: dict[str, Any]) -> dict[str, Any]:
    """Config = base + overrides del trial (claves con punto) + cv dado; sin guardar modelo."""
    cfg = copy.deepcopy(base_cfg)
    for dotted, value in overrides.items():
        set_nested(cfg, dotted, value)
    cfg["cv"] = copy.deepcopy(cv)
    cfg["artifacts"] = {"save_model": False}
    return cfg


def _cached_eval(
    cfg: dict[str, Any], *, multitask: bool, n_boot: int, alpha: float, use_cache: bool
) -> dict[str, Any]:
    """Corre la evaluación (multitask o single-task) con el caché en disco del proyecto.

    n_boot <= 0 => bootstrap apagado (selección rápida, sin IC). La key del caché
    incluye seed/cv/model/n_boot, así cada (trial, seed, split) cachea por separado.
    """
    cfg = copy.deepcopy(cfg)
    enabled = int(n_boot) > 0
    cfg.setdefault("bootstrap", {})
    cfg["bootstrap"].update(enabled=enabled, n_boot=int(max(n_boot, 0)), alpha=float(alpha))
    key = results_cache_key(cfg, n_boot=int(max(n_boot, 0)), alpha=float(alpha))
    res = load_cached_results(key) if use_cache else None
    if res is None:
        runner = run_multitask_evaluation if multitask else run_evaluation
        res = runner(cfg)
        save_cached_results(key, res)
    return res



def evaluate_report_point(
    config: str | dict[str, Any],
    *,
    report_cv: dict[str, Any] | None = None,
    primary_trait: str = "GY",
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
) -> tuple[float, float, float]:
    """(pearson, lo, hi) del trait primario para un config cualquiera en el reporte.

    Útil para traer un baseline (GBLUP, univariado) al MISMO split de reporte y la MISMA
    métrica que el top-k del sweep, para graficarlos juntos. Si se pasa report_cv, le
    pisa el cv al config; si no, usa el del propio config. Detecta multitask/single-task.
    """
    cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
    multitask = _is_multitask(cfg)
    if report_cv is not None:
        cfg["cv"] = copy.deepcopy(report_cv)
    cfg.setdefault("artifacts", {})["save_model"] = False
    res = _cached_eval(cfg, multitask=multitask, n_boot=n_boot, alpha=alpha, use_cache=use_cache)
    return _report_fold_metrics(res, multitask, primary_trait)


def diagnose_sweep(
    sweep_path: str,
    base_config: str | dict[str, Any],
    *,
    k: int = 5,
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    report_seeds: Sequence[int] | None = None,
    select_cv: dict[str, Any] | None = None,
    report_cv: dict[str, Any] | None = None,
    primary_trait: str = "GY",
    metric: str | None = None,
    goal: str | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tabla diagnóstica del top-k de un sweep: estabilidad en 2018 vs reporte en 2019.

    Para cada uno de los k mejores trials (rankeados por la métrica del sweep sobre
    val 2018) devuelve una fila con:

    - sweep_2018_1seed: la métrica que reportó el trial en el sweep (1 sola seed).
    - sel_2018_mean / sel_2018_sd / _min / _max: Pearson del trait primario
      en val 2018 reevaluado con seeds (robustez de la selección, sin mirar 2019).
    - report_2019 / report_lo / report_hi: Pearson + IC95% en el reporte (2019),
      reentrenando en train+val. Es el número honesto del trial (no elijas por él).
    - una columna por HP barrido (lo que sigue a model.params., p.ej. dropout,
      weight_decay, hidden_dims, loss_weights.AD).

    Args:
        sweep_path: "entity/project/sweep_id".
        base_config: config base del sweep (de ahí salen data/model/cv de selección).
        k: cuántos trials del top mirar.
        seeds: seeds para el multi-seed de selección (la 1ª debería ser la del base).
        report_seeds: si se pasa, además del punto+IC (seed base) reentrena el reporte 2019
            con estas seeds -> columnas report_2019_mean/sd/min/max. Es el test decisivo:
            separa una mejora real (media alta, sd baja) de un pico de una sola seed (el IC
            bootstrap NO captura la varianza de reentrenar).
        select_cv: cv de selección; default = el cv del base_config (val 2018).
        report_cv: cv de reporte; default = derive_report_cv del de selección (val 2019).
        primary_trait: trait de portada (Pearson que se mide en selección y reporte).
        metric / goal: métrica del sweep para rankear; default se lee del config del sweep.
        n_boot / alpha: bootstrap del IC del reporte. use_cache: caché en disco.

    Returns:
        DataFrame (una fila por rank, 1 = mejor en val 2018), columnas ordenadas:
        identificación, selección, reporte y luego los HP.
    """
    base_cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    multitask = _is_multitask(base_cfg)
    sel_cv = copy.deepcopy(select_cv) if select_cv is not None else copy.deepcopy(base_cfg.get("cv", {}))
    rep_cv = copy.deepcopy(report_cv) if report_cv is not None else derive_report_cv(sel_cv)
    seeds = [int(s) for s in seeds]

    if verbose:
        kind = "multitask" if multitask else "single-task"
        print(f"[diagnose_sweep] {sweep_path}  ({kind}, trait primario={primary_trait})")
        print(f"  selección: cv={ {k: sel_cv.get(k) for k in ('train_years', 'val_year')} }  seeds={seeds}")
        print(f"  reporte:   cv={ {k: rep_cv.get(k) for k in ('train_years', 'val_year')} }  n_boot={n_boot}")

    top = top_runs_from_sweep(sweep_path, k=int(k), metric=metric, goal=goal)
    if verbose:
        print(f"  top-{len(top)} trials traídos del sweep.\n")

    rows: list[dict[str, Any]] = []
    for rank, (overrides, sweep_val, run_id) in enumerate(top, start=1):
        if verbose:
            print(f"  [#{rank}] run {run_id}  (sweep val2018={sweep_val:.4f})")

        sel_vals: list[float] = []
        for s in seeds:
            cfg_s = _apply(base_cfg, overrides, sel_cv)
            set_nested(cfg_s, "seed", s)
            set_nested(cfg_s, "model.params.random_state", s)
            set_seed(s)
            res_s = _cached_eval(cfg_s, multitask=multitask, n_boot=0, alpha=alpha, use_cache=use_cache)
            sel_vals.append(_select_value(res_s, multitask, primary_trait))
        sel_arr = np.asarray(sel_vals, dtype=float)

        cfg_r = _apply(base_cfg, overrides, rep_cv)
        set_seed(int(base_cfg.get("seed", 575)))
        res_r = _cached_eval(cfg_r, multitask=multitask, n_boot=n_boot, alpha=alpha, use_cache=use_cache)
        rep_point, rep_lo, rep_hi = _report_fold_metrics(res_r, multitask, primary_trait)

        row: dict[str, Any] = {
            "rank": rank,
            "run_id": run_id,
            "sweep_2018_1seed": float(sweep_val),
            "sel_2018_mean": float(np.nanmean(sel_arr)),
            "sel_2018_sd": float(np.nanstd(sel_arr, ddof=1)) if sel_arr.size > 1 else 0.0,
            "sel_2018_min": float(np.nanmin(sel_arr)),
            "sel_2018_max": float(np.nanmax(sel_arr)),
            "report_2019": rep_point,
            "report_lo": rep_lo,
            "report_hi": rep_hi,
        }

        if report_seeds:
            rep_vals: list[float] = []
            for s in report_seeds:
                cfg_rs = _apply(base_cfg, overrides, rep_cv)
                set_nested(cfg_rs, "seed", int(s))
                set_nested(cfg_rs, "model.params.random_state", int(s))
                set_seed(int(s))
                res_rs = _cached_eval(cfg_rs, multitask=multitask, n_boot=0, alpha=alpha, use_cache=use_cache)
                rep_vals.append(_select_value(res_rs, multitask, primary_trait))
            rep_marr = np.asarray(rep_vals, dtype=float)
            row["report_2019_mean"] = float(np.nanmean(rep_marr))
            row["report_2019_sd"] = float(np.nanstd(rep_marr, ddof=1)) if rep_marr.size > 1 else 0.0
            row["report_2019_min"] = float(np.nanmin(rep_marr))
            row["report_2019_max"] = float(np.nanmax(rep_marr))

        for dotted, value in overrides.items():
            if dotted.startswith(_HP_PREFIX):
                row[dotted[len(_HP_PREFIX):]] = str(value) if isinstance(value, list) else value
        rows.append(row)
        if verbose:
            msg = (
                f"        sel2018 = {row['sel_2018_mean']:.4f} ± {row['sel_2018_sd']:.4f}"
                f"  |  report2019 = {rep_point:.4f}  IC95%[{rep_lo:.4f}, {rep_hi:.4f}]"
            )
            if report_seeds:
                msg += (
                    f"  |  report2019 multi-seed = {row['report_2019_mean']:.4f}"
                    f" ± {row['report_2019_sd']:.4f}  [{row['report_2019_min']:.4f}, {row['report_2019_max']:.4f}]"
                )
            print(msg + "\n")

    lead = ["rank", "run_id", "sweep_2018_1seed", "sel_2018_mean", "sel_2018_sd",
            "sel_2018_min", "sel_2018_max", "report_2019", "report_lo", "report_hi"]
    if report_seeds:
        lead += ["report_2019_mean", "report_2019_sd", "report_2019_min", "report_2019_max"]
    df = pd.DataFrame(rows)
    hp_cols = [c for c in df.columns if c not in lead]
    return df[[*lead, *hp_cols]]


def plot_select_vs_report(
    df: pd.DataFrame,
    *,
    baseline: tuple[float, float, float] | None = None,
    baseline_label: str = "GBLUP",
    primary_trait: str = "GY",
    title: str | None = None,
):
    """Figura: por rank del sweep, val 2018 (media±sd, multi-seed) vs reporte 2019 (punto±IC95%).

    Si el ranking de 2018 fuera predictivo, los puntos azules (2019) caerían en el mismo
    orden que los grises (2018). Que NO pase es justo el motivo por el que "el k-ésimo"
    gana en 2019 por azar. La banda roja (baseline = (point, lo, hi) de
    evaluate_report_point) deja ver si TODO el top-k le gana al baseline en 2019 — el
    argumento de robustez honesto, en vez de cherry-pickear el que más sube. Devuelve el
    Figure (el notebook decide dónde hacer savefig).
    """
    import matplotlib.pyplot as plt

    ranks = df["rank"].to_numpy()
    x = np.arange(len(ranks))
    sel = df["sel_2018_mean"].to_numpy(float)
    selsd = df["sel_2018_sd"].to_numpy(float)
    rep = df["report_2019"].to_numpy(float)
    rlo = df["report_lo"].to_numpy(float)
    rhi = df["report_hi"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(1.7 * len(ranks) + 2.5, 4.5))
    has_ms = "report_2019_mean" in df.columns
    dx = 0.15 if has_ms else 0.08
    ax.errorbar(x - dx, sel, yerr=selsd, fmt="s", capsize=4, lw=1.4, color="tab:gray",
                label="val 2018 (media ± sd, multi-seed)")
    yerr = np.vstack([np.maximum(rep - rlo, 0), np.maximum(rhi - rep, 0)])
    ax.errorbar(x, rep, yerr=yerr, fmt="o", capsize=5, lw=1.6, color="tab:blue",
                label="reporte 2019 (1 seed, punto ± IC95%)")
    if has_ms:
        rm = df["report_2019_mean"].to_numpy(float)
        rsd = df["report_2019_sd"].to_numpy(float)
        ax.errorbar(x + dx, rm, yerr=rsd, fmt="^", capsize=4, lw=1.4, color="tab:green",
                    label="reporte 2019 (media ± sd, multi-seed)")
    if baseline is not None:
        bp, blo, bhi = baseline
        if np.isfinite(blo) and np.isfinite(bhi):
            ax.axhspan(blo, bhi, color="tab:red", alpha=0.12)
        ax.axhline(bp, color="tab:red", ls="--", lw=1.3, label=f"{baseline_label} 2019 ({bp:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r}" for r in ranks])
    ax.set_xlabel("rank del sweep (por Pearson en val 2018)")
    ax.set_ylabel(f"Pearson {primary_trait}")
    ax.set_title(title or "Selección (val 2018) vs reporte (2019) por rank del sweep")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def scan_hp(
    base_config: str | dict[str, Any],
    hp_key: str,
    values: Sequence[Any],
    *,
    sweep_path: str | None = None,
    sweep_rank: int = 1,
    metric: str | None = None,
    goal: str | None = None,
    overrides: dict[str, Any] | None = None,
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    report_seeds: Sequence[int] | None = None,
    select_cv: dict[str, Any] | None = None,
    report_cv: dict[str, Any] | None = None,
    primary_trait: str = "GY",
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Barrido CONTROLADO de UN hiperparámetro: su efecto en selección (2018) y reporte (2019).

    Fija todos los HP (en base_config, y/o los del trial sweep_rank de sweep_path,
    y/o overrides) y varía SOLO hp_key sobre values. Para cada valor mide el Pearson
    del trait primario en val de selección (multi-seed) y en el reporte (punto + IC con seed
    base, y multi-seed si report_seeds). A diferencia de leer trials de un sweep —donde los
    HP varían juntos y el efecto queda confundido—, esto aísla el efecto del HP barrido.

    Args:
        base_config: config base (de ahí salen data/model/cv y el resto de los HP).
        hp_key: clave dot-notation a barrer (p.ej. "model.params.loss_weights.AD").
        values: valores a probar.
        sweep_path / sweep_rank: si se pasan, hornea los HP del trial sweep_rank (1 = mejor)
            de ese sweep como base (útil para "los HP del mejor modelo"). hp_key se ignora
            de esos overrides (lo controla values).
        overrides: HP extra a fijar (claves dot, pisan a los del sweep).
        seeds / report_seeds: multi-seed de selección / de reporte (ver diagnose_sweep).
        select_cv / report_cv: cv de selección (default = el del base) / de reporte (default =
            derive_report_cv). primary_trait, n_boot, alpha, use_cache: como en diagnose_sweep.

    Returns:
        DataFrame con una fila por valor: la columna del HP (nombre = hp_key sin el
        prefijo model.params.), sel_mean/sd/min/max (selección) y report_point/lo/hi
        (+ report_mean/sd/min/max si report_seeds).
    """
    base_cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    multitask = _is_multitask(base_cfg)
    sel_cv = copy.deepcopy(select_cv) if select_cv is not None else copy.deepcopy(base_cfg.get("cv", {}))
    rep_cv = copy.deepcopy(report_cv) if report_cv is not None else derive_report_cv(sel_cv)
    seeds = [int(s) for s in seeds]
    hp_col = hp_key[len(_HP_PREFIX):] if hp_key.startswith(_HP_PREFIX) else hp_key

    base_ov: dict[str, Any] = {}
    if sweep_path:
        top = top_runs_from_sweep(sweep_path, k=int(sweep_rank), metric=metric, goal=goal)
        base_ov = dict(top[int(sweep_rank) - 1][0])
    if overrides:
        base_ov.update(overrides)
    base_ov.pop(hp_key, None)

    if verbose:
        print(f"[scan_hp] {hp_key} sobre {list(values)}  (trait primario={primary_trait})")
        print(f"  selección: cv={ {k: sel_cv.get(k) for k in ('train_years', 'val_year')} }  seeds={seeds}")
        print(f"  reporte:   cv={ {k: rep_cv.get(k) for k in ('train_years', 'val_year')} }  "
              f"report_seeds={list(report_seeds) if report_seeds else None}\n")

    rows: list[dict[str, Any]] = []
    for v in values:
        ov = dict(base_ov)
        ov[hp_key] = v

        sel_vals: list[float] = []
        for s in seeds:
            cfg_s = _apply(base_cfg, ov, sel_cv)
            set_nested(cfg_s, "seed", s)
            set_nested(cfg_s, "model.params.random_state", s)
            set_seed(s)
            res_s = _cached_eval(cfg_s, multitask=multitask, n_boot=0, alpha=alpha, use_cache=use_cache)
            sel_vals.append(_select_value(res_s, multitask, primary_trait))
        sel_arr = np.asarray(sel_vals, dtype=float)

        cfg_r = _apply(base_cfg, ov, rep_cv)
        set_seed(int(base_cfg.get("seed", 575)))
        res_r = _cached_eval(cfg_r, multitask=multitask, n_boot=n_boot, alpha=alpha, use_cache=use_cache)
        rp, rlo, rhi = _report_fold_metrics(res_r, multitask, primary_trait)

        row: dict[str, Any] = {
            hp_col: v,
            "sel_mean": float(np.nanmean(sel_arr)),
            "sel_sd": float(np.nanstd(sel_arr, ddof=1)) if sel_arr.size > 1 else 0.0,
            "sel_min": float(np.nanmin(sel_arr)),
            "sel_max": float(np.nanmax(sel_arr)),
            "report_point": rp,
            "report_lo": rlo,
            "report_hi": rhi,
        }
        if report_seeds:
            rep_vals: list[float] = []
            for s in report_seeds:
                cfg_rs = _apply(base_cfg, ov, rep_cv)
                set_nested(cfg_rs, "seed", int(s))
                set_nested(cfg_rs, "model.params.random_state", int(s))
                set_seed(int(s))
                res_rs = _cached_eval(cfg_rs, multitask=multitask, n_boot=0, alpha=alpha, use_cache=use_cache)
                rep_vals.append(_select_value(res_rs, multitask, primary_trait))
            rep_marr = np.asarray(rep_vals, dtype=float)
            row["report_mean"] = float(np.nanmean(rep_marr))
            row["report_sd"] = float(np.nanstd(rep_marr, ddof=1)) if rep_marr.size > 1 else 0.0
            row["report_min"] = float(np.nanmin(rep_marr))
            row["report_max"] = float(np.nanmax(rep_marr))
        rows.append(row)
        if verbose:
            extra = (f"  |  report multi = {row['report_mean']:.4f} ± {row['report_sd']:.4f}"
                     if report_seeds else "")
            print(f"  {hp_col}={str(v):<6}: sel = {row['sel_mean']:.4f} ± {row['sel_sd']:.4f}"
                  f"  |  report = {rp:.4f}  IC95%[{rlo:.4f}, {rhi:.4f}]{extra}")
    return pd.DataFrame(rows)


def plot_hp_scan(
    df: pd.DataFrame,
    hp_name: str,
    *,
    baseline: tuple[float, float, float] | None = None,
    baseline_label: str = "GBLUP",
    primary_trait: str = "GY",
    title: str | None = None,
):
    """Figura del barrido de un HP: hp_name (x) vs Pearson en selección (2018) y reporte (2019).

    Las dos curvas (gris = val 2018, verde/azul = reporte 2019) dejan ver si la tendencia del HP
    aparece en AMBOS años (seleccionable honestamente) o solo en el reporte (efecto del año).
    Usa report_mean ± sd si está (multi-seed), si no el punto ± IC95%. baseline =
    (point, lo, hi) de evaluate_report_point. Devuelve el Figure.
    """
    import matplotlib.pyplot as plt

    x = df[hp_name].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(x, df["sel_mean"].to_numpy(float), yerr=df["sel_sd"].to_numpy(float),
                fmt="s-", capsize=4, lw=1.5, color="tab:gray", label="selección val 2018 (media ± sd)")
    if "report_mean" in df.columns:
        ax.errorbar(x, df["report_mean"].to_numpy(float), yerr=df["report_sd"].to_numpy(float),
                    fmt="^-", capsize=4, lw=1.5, color="tab:green", label="reporte 2019 (media ± sd, multi-seed)")
    else:
        rp = df["report_point"].to_numpy(float)
        yerr = np.vstack([np.maximum(rp - df["report_lo"].to_numpy(float), 0),
                          np.maximum(df["report_hi"].to_numpy(float) - rp, 0)])
        ax.errorbar(x, rp, yerr=yerr, fmt="o-", capsize=4, lw=1.5, color="tab:blue",
                    label="reporte 2019 (punto ± IC95%)")
    if baseline is not None:
        bp, blo, bhi = baseline
        if np.isfinite(blo) and np.isfinite(bhi):
            ax.axhspan(blo, bhi, color="tab:red", alpha=0.12)
        ax.axhline(bp, color="tab:red", ls="--", lw=1.3, label=f"{baseline_label} 2019 ({bp:.3f})")
    ax.set_xlabel(hp_name)
    ax.set_ylabel(f"Pearson {primary_trait}")
    ax.set_title(title or f"Barrido de {hp_name}: selección (2018) vs reporte (2019)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _scoring_predictions(res: dict[str, Any], multitask: bool, primary_trait: str) -> dict[str, Any]:
    """Predicciones (line_id/y_true/y_pred) del trait primario en el fold de scoring."""
    preds = res.get("predictions") or []
    if multitask:
        cand = [p for p in preds if p.get("trait") == primary_trait and p.get("role") == "scoring"]
    else:
        cand = [p for p in preds if p.get("role") == "scoring"]
    if not cand:
        raise ValueError(
            f"Sin predicciones scoring para trait={primary_trait!r} "
            "(¿caché viejo sin 'predictions'? recomputá con use_cache=False)."
        )
    return cand[-1]


def _format_compare(rows, deltas, ref, metrics, n_lines) -> str:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        by.setdefault(r["metric"], {})[r["model"]] = r
    out = [f"Delta pareado vs {ref} (bootstrap cluster por línea, n={n_lines} líneas; "
           f"* = IC95% no cruza 0):", ""]
    for m in metrics:
        out.append(f"[{m}]")
        ref_row = by.get(m, {}).get(ref)
        if ref_row:
            out.append(f"    {ref:34s} {ref_row['point']:+.4f}")
        for model, r in by.get(m, {}).items():
            if model == ref:
                continue
            d = deltas.get(model, {}).get(m, {})
            star = "  *" if d.get("sig") else ""
            out.append(f"    {model:34s} {r['point']:+.4f}   Δ={d.get('delta', float('nan')):+.4f}  "
                       f"IC95%[{d.get('lo', float('nan')):+.4f}, {d.get('hi', float('nan')):+.4f}]{star}")
        out.append("")
    return "\n".join(out)


def compare_trial_vs_baseline(
    sweep_path: str,
    base_config: str | dict[str, Any],
    baseline_config: str | dict[str, Any],
    *,
    rank: int = 1,
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    model_label: str = "modelo",
    baseline_label: str = "GBLUP",
    report_cv: dict[str, Any] | None = None,
    primary_trait: str = "GY",
    metric: str | None = None,
    goal: str | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """¿El trial rank del sweep le gana al baseline en el reporte? Delta pareado, cluster por línea.

    Trae las predicciones del trial en el reporte (2019) con la seed base y como
    ensemble de seeds (promedio z-scoreado de las predicciones de seeds, alineadas por
    línea — un modelo legítimo y de menor varianza), y las del baseline; alinea los tres por
    línea y calcula el delta pareado (modelo − baseline) por bootstrap cluster por línea.
    Es el test correcto para "¿le gana?": compara el MISMO resample, en vez de IC marginales
    sueltos (el ensemble reduce el ruido de seed; el cluster por línea, el de muestreo).

    Returns:
        dict con table (tidy [model, metric, point, lo, hi, n]), deltas
        ({model: {metric: {delta, lo, hi, sig}}}, sig = IC95% no cruza 0), samples
        (bootstrap pareado), text (resumen formateado), fig (Pearson ± IC por modelo) y
        n_lines.
    """
    from src.eval.ensemble import (align_by_line, average_predictions, bootstrap_table,
                                    paired_diff_summary)
    from src.eval.metrics import coincidence_index, pearson_r

    base_cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
    multitask = _is_multitask(base_cfg)
    rep_cv = copy.deepcopy(report_cv) if report_cv is not None else derive_report_cv(base_cfg.get("cv", {}))
    seeds = [int(s) for s in seeds]
    base_seed = int(base_cfg.get("seed", 575))
    if base_seed not in seeds:
        seeds = [base_seed, *seeds]

    overrides = dict(top_runs_from_sweep(sweep_path, k=int(rank), metric=metric, goal=goal)[int(rank) - 1][0])
    if verbose:
        print(f"[compare] trial #{rank} de {sweep_path} vs {baseline_label}  "
              f"(reporte val_year={rep_cv.get('val_year')}, seeds={seeds})\n")

    preds_by_seed: dict[int, dict[str, Any]] = {}
    for s in seeds:
        cfg_s = _apply(base_cfg, overrides, rep_cv)
        set_nested(cfg_s, "seed", s)
        set_nested(cfg_s, "model.params.random_state", s)
        set_seed(s)
        res_s = _cached_eval(cfg_s, multitask=multitask, n_boot=0, alpha=alpha, use_cache=use_cache)
        preds_by_seed[s] = _scoring_predictions(res_s, multitask, primary_trait)

    line_ens, ytrue_ens, yhats_seeds = align_by_line(*[preds_by_seed[s] for s in seeds])
    yhat_ens = average_predictions(yhats_seeds, standardize=True)
    pred_ens = {"line_id": line_ens, "y_true": ytrue_ens, "y_pred": yhat_ens}

    base_b = load_config(baseline_config) if isinstance(baseline_config, str) else copy.deepcopy(baseline_config)
    base_b["cv"] = copy.deepcopy(rep_cv)
    base_b.setdefault("artifacts", {})["save_model"] = False
    res_b = _cached_eval(base_b, multitask=_is_multitask(base_b), n_boot=0, alpha=alpha, use_cache=use_cache)
    pred_b = _scoring_predictions(res_b, _is_multitask(base_b), primary_trait)

    line_id, y_true, (yh_base, yh_seed, yh_ens) = align_by_line(pred_b, preds_by_seed[base_seed], pred_ens)
    seed_name = f"{model_label} (seed {base_seed})"
    ens_name = f"{model_label} (ensemble {len(seeds)})"
    preds_by_name = {baseline_label: yh_base, seed_name: yh_seed, ens_name: yh_ens}
    metrics = {"pearson_r": pearson_r, "matching20": coincidence_index}
    rows, samples = bootstrap_table(y_true, preds_by_name, metrics, group=line_id,
                                    n_boot=n_boot, alpha=alpha, seed=0)
    deltas = paired_diff_summary(samples, baseline_label, list(metrics), alpha=alpha)
    text = _format_compare(rows, deltas, baseline_label, ("pearson_r", "matching20"), int(len(line_id)))

    fig = None
    try:
        from src.eval.report_plots import plot_ci
        df_pr = pd.DataFrame([r for r in rows if r["metric"] == "pearson_r"])
        fig = plot_ci(df_pr, metrics=("pearson_r",),
                      title=f"Pearson_r {primary_trait} - {model_label} vs {baseline_label} "
                            f"(reporte {rep_cv.get('val_year')})")
    except Exception as e:
        if verbose:
            print(f"  [warn] no pude armar la figura ({e})")

    if verbose:
        print(text)
    return {"table": pd.DataFrame(rows), "deltas": deltas, "samples": samples,
            "text": text, "fig": fig, "n_lines": int(len(line_id))}
