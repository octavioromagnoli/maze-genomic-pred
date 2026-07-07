"""Mejor MoE partition-ensemble (sweep partens vgwxnu7z): train 2019 -> predecir 2020, 6 fenotipos.

Cold-start puro (sin spike): 2019 no tiene overlap de línea con 2020. Se corre en 6 procesos
data-parallel (una SEED por core, 1 hilo c/u — las redes no escalan con hilos) y se reporta
media ± sd sobre 6 seeds. GBLUP (determinista) va de referencia.

  SEED=<n>  -> worker: entrena esa seed, escribe partens2020_seed<n>.csv.
  sin SEED  -> agregador: junta los 6 seeds, agrega media±sd, corre GBLUP, imprime + guarda summary.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import pearson_r
from src.eval.nb_eval import gblup_report_config
from src.eval.ensemble import fold_predictions
from src.training.run import run_evaluation
from src.training.run_multitask import run_multitask_evaluation
from src.utils.config import load_config, set_nested
from src.utils.seed import set_seed
from src.training.save_model import load_cached_results, results_cache_key, save_cached_results

OUT = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-501/-Users-octavio-Documents-GitHub-maze-genomic-pred/"
               "dad59702-7f4c-444b-87a6-9b87caffc3f3/scratchpad")
BASE = "configs/exp_moe_partens_6head_2017_to_2018.yaml"
GBLUP_BASE = "configs/exp_gblup_2017_2018_to_2019.yaml"
# train 2019 -> val 2020 (TODAS las líneas de 2020; sin holdout_test_year para no excluir 2020).
CV = {"scheme": "fixed_temporal", "val_year": 2020, "train_years": [2019],
      "holdout_test_year": None, "enforce_disjoint_lines": True}
TRAITS, ENVS = ["GY", "PH", "AD"], ["optimal", "drought"]
HEADS = [f"{t}_{e}" for t in TRAITS for e in ENVS]
SEEDS = [1, 42, 2024, 7, 123, 2025]           # 6 seeds -> 6 cores
N_BOOT, ALPHA = 2000, 0.05


def reseed(cfg, s):
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
    return cfg


def build_cfg(seed):
    hp = json.loads((SCRATCH / "partens_hp.json").read_text())["hp"]
    cfg = load_config(BASE)
    for dotted, v in hp.items():
        set_nested(cfg, dotted, v)
    cfg = reseed(cfg, seed)
    cfg["cv"] = dict(CV)
    cfg["artifacts"] = {"save_model": False}
    cfg.setdefault("bootstrap", {}); cfg["bootstrap"].update(enabled=True, n_boot=N_BOOT, alpha=ALPHA)
    return cfg


def worker(seed):
    res = run_multitask_evaluation(build_cfg(seed))
    pf = {f["trait"]: f for f in res["per_fold"] if f["role"] == "scoring"}
    rows = []
    for h in HEADS:
        m = pf[h]
        rows.append({"model": "partens", "pheno": h, "seed": seed,
                     "pearson_r": float(m["pearson_r"]), "lo": float(m.get("pearson_r_lo", np.nan)),
                     "hi": float(m.get("pearson_r_hi", np.nan)), "n": int(m["n_val"])})
    pd.DataFrame(rows).to_csv(OUT / f"partens2020_seed{seed}.csv", index=False)
    gyd = next(r["pearson_r"] for r in rows if r["pheno"] == "GY_drought")
    print(f"[seed {seed}] listo — GY_drought={gyd:+.3f}", flush=True)


def eval_gblup(trait, env):
    gc = gblup_report_config(GBLUP_BASE, trait, env, CV)
    set_seed(int(gc.get("seed", 575)))
    gc.setdefault("bootstrap", {}); gc["bootstrap"].update(enabled=True, n_boot=N_BOOT, alpha=ALPHA)
    key = results_cache_key(gc, n_boot=N_BOOT, alpha=ALPHA)
    r = load_cached_results(key)
    if r is None or not r.get("predictions"):
        r = run_evaluation(gc); save_cached_results(key, r)
    p = fold_predictions(r, role="scoring", val_year=2020)
    return float(pearson_r(p["y_true"], p["y_pred"])), int(len(p["y_true"]))


def aggregator():
    parts = [pd.read_csv(OUT / f"partens2020_seed{s}.csv") for s in SEEDS if (OUT / f"partens2020_seed{s}.csv").exists()]
    long = pd.concat(parts, ignore_index=True)
    long.to_csv(OUT / "partens2020_from2019_long.csv", index=False)

    rows = []
    for h in HEADS:
        v = long[long.pheno == h]["pearson_r"].values
        rg, n = eval_gblup(*h.split("_"))
        rows.append({"pheno": h, "partens_mean": round(float(v.mean()), 4),
                     "partens_sd": round(float(v.std(ddof=1)), 4), "n_seeds": int(v.size),
                     "gblup_r": round(rg, 4), "delta_vs_gblup": round(float(v.mean()) - rg, 4), "n_val": n})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "partens2020_from2019_summary.csv", index=False)

    print("\n" + "=" * 74)
    print("MEJOR partition-ensemble — train 2019 -> predecir 2020 (COLD-START, sin spike)")
    print(f"media ± sd sobre {len(parts)} seeds · GBLUP determinista de referencia")
    print("=" * 74)
    print("%-13s%22s%10s%13s" % ("fenotipo", "partens (media±sd)", "GBLUP", "Δ vs GBLUP"))
    for r in rows:
        cell = "%+.3f±%.3f" % (r["partens_mean"], r["partens_sd"])
        print("%-13s%22s%+10.3f%+13.3f" % (r["pheno"], cell, r["gblup_r"], r["delta_vs_gblup"]))
    print("\nGuardado: partens2020_from2019_{long,summary}.csv")


def main():
    seed_env = os.environ.get("SEED")
    if seed_env:
        worker(int(seed_env))
    else:
        aggregator()


if __name__ == "__main__":
    main()
