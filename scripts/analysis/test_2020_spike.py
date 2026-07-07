"""Test FINAL en 2020: cold-start + spike-in de todos los modelos, sin leakage.

Pregunta: ¿cuánta muestra del ciclo objetivo (2020) hace falta para rescatar la accuracy
cross-cycle, que colapsa entrenando solo con años históricos? (ver memoria
`cierre-2020-no-predecible`: GY colapsa a negativo sin spike). Diseño acordado con el usuario:

  - 3 sets de train históricos:  2017-2019 · 2018-2019 · 2019 solo.
  - 3 niveles de spike:          0% · +5% · +10% de las LÍNEAS de 2020 metidas en train.
  - HOLDOUT COMÚN FIJO:          el mismo 90% de las líneas de 2020 es el test para los 3
                                 niveles (así el tamaño/composición del test no influye en
                                 los Δ). El spike sale del 10% complementario ⇒ nunca se
                                 predice una línea usada en train (sin leakage, regla 2).
  - 7 modelos:                   GBLUP, Ensemble (sin gating), MoE hard, MoE soft, MoE GxE,
                                 MoE PLS, MoE contrastivo.
  - REPS:                        cada rep resortea el barajado de 2020 (holdout/spike) Y
                                 re-seedea la red; se reporta media ± sd sobre reps.

Reusa el harness (nada ad-hoc): cv `spike_in` (src/eval/splits.spike_in_split, ya con guard
anti-leakage) + run_multitask_evaluation (neuronales, 6 cabezas) / run_evaluation (GBLUP por
trait×env). Todo cacheado por hash de config ⇒ resumible (si muere, retoma de cache).

Salidas (en esta carpeta): test2020_spike_long.csv (fila por modelo×train×spike×rep×fenotipo),
test2020_spike_summary.csv (media±sd + Δ vs 0% pareado), test2020_spike.json.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import pearson_r
from src.eval.nb_eval import make_multitask_evaluator, gblup_report_config
from src.eval.ensemble import fold_predictions
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.training.run import run_evaluation
from src.training.save_model import load_cached_results, results_cache_key, save_cached_results

OUT = Path(__file__).resolve().parent
HOLDOUT_YEAR = 2020
HOLDOUT_FRAC = 0.9            # 90% de 2020 = test común fijo; 10% = pool de spike
N_BOOT, ALPHA = 1000, 0.05   # IC off en este test (la dispersión viene de las reps)

TRAIN_SETS = {                # nombre -> años históricos de train
    "2017-2019": [2017, 2018, 2019],
    "2018-2019": [2018, 2019],
    "2019":      [2019],
}
SPIKES = [0.0, 0.05, 0.10]
REPS_FAST = [1, 42, 2024]    # 3 reps (uniforme para TODOS los modelos)
REPS_SLOW = [1, 42, 2024]    # mismos 3 reps para gxe/pls/contr (comparación pareja)

GBLUP_BASE = "configs/exp_gblup_2017_2018_to_2019.yaml"
# Ordenados esencial-primero (los resultados útiles caen antes en el stream).
ARMS = {
    "Ensemble (sin gating)":   ("configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml", REPS_FAST),
    "MoE hard (PCA gating)":   ("configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",    REPS_FAST),
    "MoE soft (gate aprend.)": ("configs/exp_moe_soft_6head_2017_2018_to_2019.yaml",    REPS_FAST),
    "MoE PLS (gating superv.)":("configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",     REPS_SLOW),
    "MoE contrastivo":         ("configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml", REPS_SLOW),
    "MoE GxE (ruteo x amb.)":  ("configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",     REPS_SLOW),  # último: cola larga
}
TRAITS, ENVS = ["GY", "PH", "AD"], ["optimal", "drought"]
HEADS = [f"{t}_{e}" for t in TRAITS for e in ENVS]
LONG_CSV = OUT / "test2020_spike_long.csv"


def spike_cv(train_years, spike_pct, spike_seed):
    return {
        "scheme": "spike_in",
        "holdout_year": HOLDOUT_YEAR,
        "train_years": list(train_years),
        "spike_pct": float(spike_pct),
        "holdout_frac": HOLDOUT_FRAC,
        "spike_seed": int(spike_seed),
        "enforce_disjoint_lines": True,
    }


def reseed(cfg: dict, s: int, cv: dict) -> dict:
    """Deep-copy con TODOS los seed/random_state en `s`, el cv de spike y sin guardar modelo."""
    cfg = copy.deepcopy(cfg)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("seed", "random_state") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    o[k] = int(s)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    cfg["seed"] = int(s)
    cfg["cv"] = dict(cv)
    cfg["artifacts"] = {"save_model": False}
    cfg.setdefault("bootstrap", {})
    cfg["bootstrap"].update(enabled=False)   # sin IC por fold: la dispersión viene de las reps
    return cfg


def head_pred_mt(res, head):
    cand = [p for p in res["predictions"] if p.get("trait") == head and p["role"] == "scoring"]
    return cand[-1] if cand else None


def eval_gblup(trait, env, cv, rep):
    """GBLUP de un (trait,env) con el cv de spike. Determinista, pero el split cambia por rep."""
    cfg = gblup_report_config(GBLUP_BASE, trait, env, cv)
    cfg["seed"] = int(rep)
    set_seed(int(rep))
    cfg.setdefault("bootstrap", {}); cfg["bootstrap"].update(enabled=False)
    key = results_cache_key(cfg, n_boot=N_BOOT, alpha=ALPHA)
    res = load_cached_results(key)
    if res is None or not res.get("predictions"):
        res = run_evaluation(cfg); save_cached_results(key, res)
    return fold_predictions(res, role="scoring", val_year=HOLDOUT_YEAR)


def append_rows(rows, csv_path=None):
    """Stream incremental al CSV largo (resumible / inspeccionable a mitad de corrida)."""
    path = csv_path or LONG_CSV
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def gblup_unit(ts_name, ts, spk, rep):
    """Una unidad de trabajo GBLUP = (train,spike,rep) sobre las 6 cabezas (trait×env)."""
    cv = spike_cv(ts, spk, rep)
    rows = []
    for t in TRAITS:
        for e in ENVS:
            gp = eval_gblup(t, e, cv, rep)
            rows.append({"model": "GBLUP", "train_set": ts_name, "spike_pct": spk, "rep": rep,
                         "pheno": f"{t}_{e}", "pearson_r": float(pearson_r(gp["y_true"], gp["y_pred"])),
                         "n": int(len(gp["y_true"]))})
    return rows


def neural_unit(evalmt, base_cfgs, arm, ts_name, ts, spk, rep):
    """Una unidad de trabajo neuronal = una corrida multitask (6 cabezas de una vez)."""
    cfg = reseed(base_cfgs[arm], rep, spike_cv(ts, spk, rep))
    res = evalmt(f"{arm}·{ts_name}·s{spk}·r{rep}", cfg)
    rows = []
    for head in HEADS:
        pr = head_pred_mt(res, head)
        if pr is None:
            continue
        rows.append({"model": arm, "train_set": ts_name, "spike_pct": spk, "rep": rep,
                     "pheno": head, "pearson_r": float(pearson_r(pr["y_true"], pr["y_pred"])),
                     "n": int(len(pr["y_true"]))})
    return rows


def build_jobs():
    """Lista plana de unidades de trabajo (GBLUP primero, luego neuronales por arm)."""
    jobs = [("gblup", "GBLUP", n, ts, spk, rep)
            for n, ts in TRAIN_SETS.items() for spk in SPIKES for rep in REPS_FAST]
    for arm, (_, reps) in ARMS.items():
        jobs += [("neural", arm, n, ts, spk, rep)
                 for n, ts in TRAIN_SETS.items() for spk in SPIKES for rep in reps]
    return jobs


def main():
    # Paralelismo DATA-PARALLEL entre procesos (las redes son chicas ⇒ los hilos no escalan;
    # el cache es por-config y cada shard computa configs disjuntos ⇒ sin colisiones).
    #   SHARD="k/N"  -> este proceso computa solo la tajada k (llena cache, log per-shard).
    #   sin SHARD    -> colector: recorre TODO (cache hits tras los shards) y arma CSV+resumen.
    shard_env = os.environ.get("SHARD")
    evalmt = make_multitask_evaluator(list({"pearson_r": pearson_r}), n_boot=N_BOOT, alpha=ALPHA)
    base_cfgs = {arm: load_config(path) for arm, (path, _) in ARMS.items()}
    jobs = build_jobs()

    def run_job(kind, arm, ts_name, ts, spk, rep):
        return (gblup_unit(ts_name, ts, spk, rep) if kind == "gblup"
                else neural_unit(evalmt, base_cfgs, arm, ts_name, ts, spk, rep))

    if shard_env:
        k, nsh = (int(x) for x in shard_env.split("/"))
        out_csv = OUT / f"test2020_spike_long.shard{k}.csv"
        if out_csv.exists():
            out_csv.unlink()
        mine = [(i, j) for i, j in enumerate(jobs) if i % nsh == k]
        for i, (kind, arm, ts_name, ts, spk, rep) in mine:
            rows = run_job(kind, arm, ts_name, ts, spk, rep)
            append_rows(rows, out_csv)
            gyd = next((x["pearson_r"] for x in rows if x["pheno"] == "GY_drought"), float("nan"))
            print(f"[s{k} {arm[:20]:20s}] {ts_name:10s} spike={spk:.2f} rep{rep:<5d} GY_drought={gyd:+.3f}", flush=True)
        print(f"[shard {k}/{nsh}] DONE — {len(mine)} jobs", flush=True)
        return

    # ---------------- Colector: recorre todo (cache hits) + resumen ----------------
    if LONG_CSV.exists():
        LONG_CSV.unlink()
    long_rows = []
    for kind, arm, ts_name, ts, spk, rep in jobs:
        rows = run_job(kind, arm, ts_name, ts, spk, rep)
        append_rows(rows); long_rows += rows
        gyd = next((x["pearson_r"] for x in rows if x["pheno"] == "GY_drought"), float("nan"))
        print(f"[{(arm)[:22]:22s}] {ts_name:10s} spike={spk:.2f} rep{rep:<5d} GY_drought={gyd:+.3f}", flush=True)

    # ---------------- Resumen: media ± sd + Δ vs 0% pareado ----------------
    long = pd.DataFrame(long_rows)
    sum_rows = []
    for model in ["GBLUP"] + list(ARMS):
        for ts_name in TRAIN_SETS:
            for head in HEADS:
                base = long[(long.model == model) & (long.train_set == ts_name) & (long.pheno == head)]
                if base.empty:
                    continue
                # r(0%) por rep para el Δ pareado (mismo holdout dentro de la rep).
                r0 = base[base.spike_pct == 0.0].set_index("rep")["pearson_r"]
                for spk in SPIKES:
                    v = base[base.spike_pct == spk]
                    if v.empty:
                        continue
                    rr = v.set_index("rep")["pearson_r"]
                    common = rr.index.intersection(r0.index)
                    dvals = (rr.loc[common] - r0.loc[common]).values if len(common) else np.array([])
                    sum_rows.append({
                        "model": model, "train_set": ts_name, "pheno": head, "spike_pct": spk,
                        "mean": round(float(rr.mean()), 4),
                        "sd": round(float(rr.std(ddof=1)), 4) if len(rr) > 1 else 0.0,
                        "n_reps": int(len(rr)),
                        "delta_vs0_mean": round(float(dvals.mean()), 4) if dvals.size else 0.0,
                        "delta_vs0_sd": round(float(dvals.std(ddof=1)), 4) if dvals.size > 1 else 0.0,
                    })
    summ = pd.DataFrame(sum_rows)
    summ.to_csv(OUT / "test2020_spike_summary.csv", index=False)
    payload = {"config": {"holdout_year": HOLDOUT_YEAR, "holdout_frac": HOLDOUT_FRAC,
                          "train_sets": TRAIN_SETS, "spikes": SPIKES,
                          "reps_fast": REPS_FAST, "reps_slow": REPS_SLOW},
               "summary": sum_rows}
    (OUT / "test2020_spike.json").write_text(json.dumps(payload, indent=2, default=float))

    # ---------------- Impresión: GY_drought (la portada) por train×spike ----------------
    print("\n" + "=" * 92)
    print("GY_drought en 2020 — media ± sd sobre reps  |  columnas: spike 0% / +5% / +10%")
    print("=" * 92)
    for ts_name in TRAIN_SETS:
        print(f"\n-- train {ts_name} --")
        print(f"  {'modelo':<26}" + "".join(f"{f'spike {int(s*100)}%':>16}" for s in SPIKES))
        for model in ["GBLUP"] + list(ARMS):
            cells = []
            for spk in SPIKES:
                r = summ[(summ.model == model) & (summ.train_set == ts_name)
                         & (summ.pheno == "GY_drought") & (summ.spike_pct == spk)]
                cells.append("-".rjust(16) if r.empty
                             else f"{r['mean'].values[0]:+.3f}±{r['sd'].values[0]:.3f}".rjust(16))
            print(f"  {model:<26}" + "".join(cells))
    print("\nArchivos: test2020_spike_long.csv · test2020_spike_summary.csv · test2020_spike.json")


if __name__ == "__main__":
    main()
