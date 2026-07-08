"""Evaluadores para el notebook de comparación (cachean en disco, no tocan wandb).

make_evaluators arma los closures evaluate_model / eval_with_preds que usan los
notebooks: corren run_evaluation con bootstrap forzado, cachean por
(config, seed, n_boot, alpha) y devuelven filas tidy + las muestras del bootstrap
(eval_with_preds además guarda las predicciones del fold de val para los ensembles).

Vive acá para que cada sección del notebook se levante con un solo import y corra
independiente: con el caché en disco (outputs/eval_cache) volver a correr una sección es
un cache hit. Usa la misma results_cache_key que el resto del proyecto para compartir caché.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import pandas as pd

from src.eval.ensemble import (
    align_by_line,
    bootstrap_table,
    fold_predictions,
    paired_diff_summary,
)
from src.eval.report_plots import plot_ci
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


def make_evaluators(
    metrics: Sequence[str],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    use_cache: bool = True,
    curves_by_model: dict | None = None,
    preds_by_model: dict | None = None,
):
    """Devuelve (evaluate_model, eval_with_preds, curves_by_model, preds_by_model).

    Args:
        metrics: nombres de las métricas a reportar en las filas tidy (las del proyecto).
        n_boot, alpha, use_cache: defaults del bootstrap/caché (override por llamada).
        curves_by_model / preds_by_model: dicts donde acumular las learning curves y las
            predicciones del fold de val; se crean vacíos si no se pasan. Pasá los mismos
            entre secciones para acumular sin pisar.

    evaluate_model(name, config) => (rows, samples) con
    rows = [{model, metric, point, lo, hi, n}] y samples = {metric: muestras_boot}.
    eval_with_preds(...) hace lo mismo y además guarda preds_by_model[name] (line_id /
    y_true / y_pred del fold de val) para alinear y combinar en los ensembles.
    """
    metrics = tuple(metrics)
    curves: dict = {} if curves_by_model is None else curves_by_model
    preds: dict = {} if preds_by_model is None else preds_by_model

    def evaluate_model(name, config, *, n_boot=n_boot, alpha=alpha, use_cache=use_cache):
        cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
        set_seed(int(cfg.get("seed", 575)))
        cfg.setdefault("bootstrap", {})
        cfg["bootstrap"].update(enabled=True, n_boot=n_boot, alpha=alpha)

        key = results_cache_key(cfg, n_boot=n_boot, alpha=alpha)
        res = load_cached_results(key) if use_cache else None
        if res is None:
            res = run_evaluation(cfg)
            save_cached_results(key, res)
        else:
            print(f"  [cache hit] {name} ({key})")

        fold = res["per_fold"][-1]
        boot = res["bootstrap"][-1] if res["bootstrap"] else {}
        curves[name] = res.get("histories", [])

        rows, samples = [], {}
        for m in metrics:
            rows.append({
                "model": name, "metric": m,
                "point": float(fold.get(m, np.nan)),
                "lo": float(fold.get(f"{m}_lo", np.nan)),
                "hi": float(fold.get(f"{m}_hi", np.nan)),
                "n": int(fold.get("n", 0)),
            })
            samples[m] = np.asarray(boot.get(m, []), dtype=float)
        return rows, samples

    def eval_with_preds(name, config, *, role=None, val_year=None, **kw):
        rows, samples = evaluate_model(name, config, **kw)
        cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
        cfg.setdefault("bootstrap", {})
        nb, al = kw.get("n_boot", n_boot), kw.get("alpha", alpha)
        cfg["bootstrap"].update(enabled=True, n_boot=nb, alpha=al)
        res = load_cached_results(results_cache_key(cfg, n_boot=nb, alpha=al))
        preds[name] = fold_predictions(res, role=role, val_year=val_year)
        return rows, samples

    return evaluate_model, eval_with_preds, curves, preds




def make_multitask_evaluator(metrics: Sequence[str], *, n_boot: int = 2000,
                             alpha: float = 0.05, use_cache: bool = True):
    """Closure eval_multitask(name, config) -> dict (run_multitask_evaluation cacheado)."""
    def eval_multitask(name, config, *, n_boot=n_boot, alpha=alpha, use_cache=use_cache):
        cfg = load_config(config) if isinstance(config, str) else copy.deepcopy(config)
        set_seed(int(cfg.get("seed", 575)))
        cfg.setdefault("bootstrap", {})
        cfg["bootstrap"].update(enabled=True, n_boot=n_boot, alpha=alpha)
        key = results_cache_key(cfg, n_boot=n_boot, alpha=alpha)
        res = load_cached_results(key) if use_cache else None
        if res is None:
            res = run_multitask_evaluation(cfg)
            save_cached_results(key, res)
        else:
            print(f"  [cache hit] {name} ({key})")
        return res
    return eval_multitask


def sweep_overrides(sweep_id, *, rank: int = 1, metric=None, goal=None) -> dict:
    """HP barridos (claves con punto) del modelo en posición rank del sweep (1 = mejor).

    Devuelve {} si no hay sweep_id o si falla la lectura (=> se usan los HP base del
    config, con aviso). rank permite elegir el 2º/3º mejor, etc. (reusa top_runs_from_sweep).
    """
    if not sweep_id:
        return {}
    try:
        top = top_runs_from_sweep(sweep_id, k=int(rank), metric=metric, goal=goal)
    except Exception as e:
        print(f"  [warn] no pude leer sweep {sweep_id} ({e}); uso HP base")
        return {}
    if len(top) < rank:
        print(f"  [warn] sweep {sweep_id}: solo {len(top)} runs (<{rank}); uso el #{len(top)}")
    return top[min(int(rank), len(top)) - 1][0]


def _report_cfg(base, overrides: dict, report_cv: dict) -> dict:
    """Config de REPORTE: base + overrides del sweep, con el cv de reporte y sin guardar modelo."""
    cfg = load_config(base) if isinstance(base, str) else copy.deepcopy(base)
    for dotted, value in overrides.items():
        set_nested(cfg, dotted, value)
    cfg["cv"] = dict(report_cv)
    cfg["artifacts"] = {"save_model": False}
    return cfg


def gblup_report_config(gblup_base, trait: str, env: str, report_cv: dict) -> dict:
    """GBLUP del (trait, env) en el reporte (override target/env de un base reml)."""
    cfg = _report_cfg(gblup_base, {}, report_cv)
    cfg["data"]["target"] = trait
    cfg["data"]["env"] = env
    return cfg


def univ_report_config(mlp_base, trait: str, env: str, report_cv: dict, *,
                       gy_sweep: str = "", rank: int = 1) -> dict:
    """MLP single-task del (trait, env) en el reporte. GY usa su sweep (rank); PH/AD HP base."""
    overrides = sweep_overrides(gy_sweep, rank=rank) if trait == "GY" else {}
    cfg = _report_cfg(mlp_base, overrides, report_cv)
    cfg["data"]["target"] = trait
    cfg["data"]["env"] = env
    return cfg


def format_multitask_deltas(paired: dict, traits: Sequence[str],
                            refs: Sequence[str] = ("GBLUP", "Univariado"),
                            metric: str = "pearson_r") -> str:
    """Texto de los deltas pareados por trait (*  = IC95% no cruza 0)."""
    out: list[str] = []
    for trait in traits:
        out.append(f"[{trait}] deltas {metric} (* = IC95% no cruza 0):")
        for ref in refs:
            for model, per in paired.get(trait, {}).get(ref, {}).items():
                d = per.get(metric)
                if d:
                    star = "  *" if d["sig"] else ""
                    out.append(f"    {model:18s} vs {ref:11s}: {d['delta']:+.4f}  "
                               f"IC95%[{d['lo']:+.4f}, {d['hi']:+.4f}]{star}")
        out.append("")
    return "\n".join(out)


def make_multitask_comparison(*, eval_with_preds, preds_by_model, report_cv: dict,
                              metrics: Sequence[str], ens_metrics: dict,
                              traits: Sequence[str] = ("GY", "PH", "AD"),
                              n_boot: int = 2000, alpha: float = 0.05, use_cache: bool = True):
    """Devuelve compare_env(env, ...) para la tabla por trait Multitask vs Univariado vs GBLUP.

    eval_with_preds/preds_by_model son los de make_evaluators (single-task: univariado y
    GBLUP). compare_env evalúa los arms multitask (cacheado), alinea por línea y saca IC95% del
    MISMO resample (deltas pareados). Devuelve un dict con tidy (tidy pearson_r),
    wide (pivot trait×modelo), samples, paired, deltas (texto),
    fig_match_gy (Figure de matching20 de GY, mismo estilo que figs: punto ± IC95% por
    modelo) y figs ({trait: Figure} de pearson_r). ranks elige qué modelo de cada
    sweep usar: (1,) el mejor,
    (1, 2, 3) el top-3 (la tabla se ensancha), (2,) solo el segundo.
    """
    eval_multitask = make_multitask_evaluator(metrics, n_boot=n_boot, alpha=alpha, use_cache=use_cache)
    traits = tuple(traits)

    def _label(arm, rank):
        return arm if rank == 1 else f"{arm} #{rank}"

    def compare_env(env, *, mt_bases, mt_sweeps, gblup_base, mlp_base, gy_sweep,
                    arms_for_trait=None, ranks=(1,)):
        ranks = tuple(ranks)
        base_arms = list(mt_bases)
        if arms_for_trait is None:
            arms_for_trait = {t: [a for a in base_arms
                                  if t in load_config(mt_bases[a])["data"]["targets"]]
                              for t in traits}

        mt_preds: dict = {}
        for arm in base_arms:
            for rank in ranks:
                cfg = _report_cfg(mt_bases[arm], sweep_overrides(mt_sweeps.get(arm, ""), rank=rank), report_cv)
                res = eval_multitask(f"{_label(arm, rank)} {env}", cfg)
                mt_preds[_label(arm, rank)] = {p["trait"]: p for p in res["predictions"]
                                               if p["role"] == "scoring"}

        for trait in traits:
            eval_with_preds(f"Univariado {trait} {env}",
                            univ_report_config(mlp_base, trait, env, report_cv, gy_sweep=gy_sweep))
            eval_with_preds(f"GBLUP {trait} {env}",
                            gblup_report_config(gblup_base, trait, env, report_cv))

        tidy, samples_by_trait, paired, rows_by_trait = [], {}, {}, {}
        for trait in traits:
            named = [("GBLUP", preds_by_model[f"GBLUP {trait} {env}"]),
                     ("Univariado", preds_by_model[f"Univariado {trait} {env}"])]
            for arm in arms_for_trait[trait]:
                for rank in ranks:
                    lab = _label(arm, rank)
                    if trait in mt_preds.get(lab, {}):
                        named.append((lab, mt_preds[lab][trait]))
            line_id, y_true, yhats = align_by_line(*[p for _, p in named])
            preds_by_name = {nm: yh for (nm, _), yh in zip(named, yhats)}
            rows, samples = bootstrap_table(y_true, preds_by_name, ens_metrics,
                                            group=line_id, n_boot=n_boot, alpha=alpha, seed=0)
            samples_by_trait[trait] = samples
            rows_by_trait[trait] = rows
            paired[trait] = {ref: paired_diff_summary(samples, ref, ens_metrics, alpha=alpha)
                             for ref in ("GBLUP", "Univariado")}
            for r in rows:
                if r["metric"] == "pearson_r":
                    tidy.append({"trait": trait, "model": r["model"], "metric": "pearson_r",
                                 "point": r["point"], "lo": r["lo"], "hi": r["hi"], "n": r["n"]})

        tidy = pd.DataFrame(tidy)
        wide = tidy.pivot(index="trait", columns="model", values="point").reindex(traits)
        figs = {t: plot_ci(tidy[tidy.trait == t], metrics=("pearson_r",),
                           title=f"Pearson_r - {t} ({env}, reporte 2019)") for t in traits}
        gy_match_rows = [r for r in rows_by_trait.get("GY", []) if r["metric"] == "matching20"]
        fig_match_gy = (plot_ci(pd.DataFrame(gy_match_rows), metrics=("matching20",),
                                title=f"matching20 - GY ({env}, reporte 2019)")
                        if gy_match_rows else None)
        return {"env": env, "tidy": tidy, "wide": wide, "samples": samples_by_trait,
                "paired": paired, "deltas": format_multitask_deltas(paired, traits),
                "fig_match_gy": fig_match_gy, "figs": figs}

    return compare_env
