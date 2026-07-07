"""Blindaje del benchmark de 6 fenotipos: robustez a la seed + corrección por multiplicidad.

Dos análisis, ambos reusando el harness (sin eval ad-hoc):
  (1) ROBUSTEZ A LA SEED — corre cada modelo neuronal sobre N seeds (re-seedea seed + todos los
      random_state; el gating GMM y el ensemble de expertos cambian de semilla) y reporta media ± sd
      del Pearson por fenotipo. GBLUP es determinista (REML) ⇒ un solo valor.
  (2) MULTIPLICIDAD — sobre los 36 tests (6 modelos × 6 fenotipos) vs GBLUP calcula un p-valor bootstrap
      pareado de dos colas (mismo resample, cluster-línea) y aplica Benjamini-Hochberg (FDR 5%): marca
      qué diferencias sobreviven la corrección.

Salidas: robustez_seeds_long.csv, robustez_summary.csv, robustez_fdr.csv, robustez_6pheno.json
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import pearson_r, coincidence_index
from src.eval.nb_eval import make_multitask_evaluator, gblup_report_config
from src.eval.ensemble import align_by_line, bootstrap_table, fold_predictions
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.training.run import run_evaluation
from src.training.save_model import load_cached_results, results_cache_key, save_cached_results

OUT = Path(__file__).resolve().parent
REPORT_CV = {"scheme": "fixed_temporal", "val_year": 2019, "train_years": [2017, 2018],
             "holdout_test_year": 2020, "enforce_disjoint_lines": True}
GBLUP_BASE = "configs/exp_gblup_2017_2018_to_2019.yaml"
ARMS = {
    "MoE hard (PCA gating)": "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft (gate aprend.)": "configs/exp_moe_soft_6head_2017_2018_to_2019.yaml",
    "Ensemble (sin gating)": "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE (ruteo x amb.)": "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS (gating superv.)": "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
    "MoE contrastivo": "configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml",
}
TRAITS, ENVS = ["GY", "PH", "AD"], ["optimal", "drought"]
HEADS = [f"{t}_{e}" for t in TRAITS for e in ENVS]
SEEDS_CORE = [575, 1, 42, 2024, 7]   # 5 seeds para los contendientes de GY_drought (hard/soft/ensemble)
SEEDS_SEC = [575, 1, 42]             # 3 seeds para los secundarios (gxe/pls/contr): pierden en sequia y gxe es lento
ARM_SEEDS = {
    "MoE hard (PCA gating)": SEEDS_CORE,
    "MoE soft (gate aprend.)": SEEDS_CORE,
    "Ensemble (sin gating)": SEEDS_CORE,
    "MoE GxE (ruteo x amb.)": SEEDS_SEC,
    "MoE PLS (gating superv.)": SEEDS_SEC,
    "MoE contrastivo": SEEDS_SEC,
}
N_BOOT, ALPHA = 2000, 0.05
METRICS = {"pearson_r": pearson_r, "matching20": lambda yt, yp: coincidence_index(yt, yp, 0.2)}


def reseed(cfg: dict, s: int) -> dict:
    """Deep-copy del config con TODOS los `seed`/`random_state` seteados a `s` (seed_step intacto)."""
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
    cfg["cv"] = dict(REPORT_CV)
    cfg["artifacts"] = {"save_model": False}
    return cfg


def head_pred(res, head):
    cand = [p for p in res["predictions"] if p["trait"] == head and p["role"] == "scoring"]
    return cand[-1] if cand else None


def eval_gblup(trait, env):
    cfg = gblup_report_config(GBLUP_BASE, trait, env, REPORT_CV)
    set_seed(int(cfg.get("seed", 575)))
    cfg.setdefault("bootstrap", {}); cfg["bootstrap"].update(enabled=True, n_boot=N_BOOT, alpha=ALPHA)
    key = results_cache_key(cfg, n_boot=N_BOOT, alpha=ALPHA)
    res = load_cached_results(key)
    if res is None or not res.get("predictions"):
        res = run_evaluation(cfg); save_cached_results(key, res)
    return fold_predictions(res, role="scoring", val_year=2019)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg: devuelve (adjusted_p, reject) alineados al orden de entrada."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]        # monotonía BH
    adj = np.clip(adj, 0, 1)
    out = np.empty(m); out[order] = adj
    return out, out <= alpha


def main():
    evalmt = make_multitask_evaluator(list(METRICS), n_boot=N_BOOT, alpha=ALPHA)
    base_cfgs = {arm: load_config(path) for arm, path in ARMS.items()}

    # ---------- (1) Robustez a la seed ----------
    long_rows, preds575 = [], {}   # preds575[arm][head] = pred (seed 575, para el test pareado)
    for arm in ARMS:
        preds575[arm] = {}
        for s in ARM_SEEDS[arm]:
            cfg = reseed(base_cfgs[arm], s)
            res = evalmt(f"{arm} · seed{s}", cfg)
            for head in HEADS:
                pr = head_pred(res, head)
                if pr is None:
                    continue
                r = float(pearson_r(pr["y_true"], pr["y_pred"]))
                long_rows.append({"pheno": head, "model": arm, "seed": s, "pearson_r": r})
                if s == 575:
                    preds575[arm][head] = pr
            print(f"  {arm} seed{s}: GY_drought={pearson_r(*[head_pred(res,'GY_drought')[k] for k in ('y_true','y_pred')]):.4f}", flush=True)

    long = pd.DataFrame(long_rows)
    long.to_csv(OUT / "robustez_seeds_long.csv", index=False)

    # GBLUP determinista
    gblup_pt, gblup_pred = {}, {}
    for t in TRAITS:
        for e in ENVS:
            head = f"{t}_{e}"
            gp = eval_gblup(t, e)
            gblup_pred[head] = gp
            gblup_pt[head] = float(pearson_r(gp["y_true"], gp["y_pred"]))

    # Resumen media ± sd por (pheno, model)
    sum_rows = []
    for head in HEADS:
        sum_rows.append({"pheno": head, "model": "GBLUP", "mean": round(gblup_pt[head], 4),
                         "sd": 0.0, "min": round(gblup_pt[head], 4), "max": round(gblup_pt[head], 4),
                         "n_seeds": 1, "delta_mean": 0.0, "sign_stable": True})
        for arm in ARMS:
            v = long[(long.pheno == head) & (long.model == arm)]["pearson_r"].values
            if v.size == 0:
                continue
            dvals = v - gblup_pt[head]
            sum_rows.append({
                "pheno": head, "model": arm, "mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
                "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n_seeds": int(v.size),
                "delta_mean": round(float(dvals.mean()), 4),
                "sign_stable": bool(np.all(dvals > 0) or np.all(dvals < 0)),   # el signo del Δ no cambia con la seed
            })
    summ = pd.DataFrame(sum_rows)
    summ.to_csv(OUT / "robustez_summary.csv", index=False)

    # ---------- (2) Multiplicidad: p-valores bootstrap pareados + BH-FDR ----------
    fdr_rows = []
    for head in HEADS:
        named = [("GBLUP", gblup_pred[head])] + [(arm, preds575[arm][head]) for arm in ARMS if head in preds575[arm]]
        line_id, y_true, yhats = align_by_line(*[p for _, p in named])
        pbn = {nm: yh for (nm, _), yh in zip(named, yhats)}
        _, samples = bootstrap_table(y_true, pbn, METRICS, group=line_id, n_boot=N_BOOT, alpha=ALPHA, seed=0)
        sr = np.asarray(samples["GBLUP"]["pearson_r"], dtype=float)
        for arm in [a for a in ARMS if a in pbn]:
            sx = np.asarray(samples[arm]["pearson_r"], dtype=float)
            k = min(len(sx), len(sr)); d = sx[:k] - sr[:k]; d = d[np.isfinite(d)]
            frac_pos = float(np.mean(d > 0))
            p = 2.0 * min(frac_pos, 1 - frac_pos)
            p = max(p, 1.0 / N_BOOT)                      # piso: no puede ser 0 con muestreo finito
            fdr_rows.append({"pheno": head, "model": arm, "delta": round(float(d.mean()), 4),
                             "p_boot": round(p, 4)})
    fdr = pd.DataFrame(fdr_rows)
    adj, rej = bh_fdr(fdr["p_boot"].values, alpha=ALPHA)
    fdr["p_bh"] = np.round(adj, 4)
    fdr["sig_raw_05"] = fdr["p_boot"] < ALPHA
    fdr["sig_bh_05"] = rej
    fdr.to_csv(OUT / "robustez_fdr.csv", index=False)

    payload = {"seeds_core": SEEDS_CORE, "seeds_sec": SEEDS_SEC, "n_boot": N_BOOT, "alpha": ALPHA,
               "summary": sum_rows, "fdr": fdr.to_dict("records"), "gblup_point": gblup_pt}
    (OUT / "robustez_6pheno.json").write_text(json.dumps(payload, indent=2, default=float))

    # ---------- Impresión ----------
    print("\n" + "=" * 96)
    print(f"ROBUSTEZ A LA SEED — Pearson r media ± sd  (core {SEEDS_CORE} · secundarios {SEEDS_SEC} · GBLUP determinista)")
    print("=" * 96)
    models = ["GBLUP"] + list(ARMS)
    print(f"{'fenotipo':<13}" + "".join(f"{m[:18]:>21}" for m in models))
    for head in HEADS:
        cells = []
        for m in models:
            r = summ[(summ.pheno == head) & (summ.model == m)]
            if r.empty:
                cells.append(f"{'-':>21}"); continue
            mu, sd = r["mean"].values[0], r["sd"].values[0]
            cells.append(f"{mu:>6.3f}±{sd:.3f}".rjust(21) if m != "GBLUP" else f"{mu:>6.3f}".rjust(21))
        print(f"{head:<13}" + "".join(cells))

    print("\n" + "=" * 96)
    print("MULTIPLICIDAD — Δ vs GBLUP (seed 575): supervivencia a Benjamini-Hochberg FDR 5% sobre 36 tests")
    print("=" * 96)
    n_raw, n_bh = int(fdr["sig_raw_05"].sum()), int(fdr["sig_bh_05"].sum())
    print(f"Significativos sin corregir: {n_raw}/36   ·   tras BH-FDR 5%: {n_bh}/36")
    lost = fdr[fdr.sig_raw_05 & ~fdr.sig_bh_05]
    if not lost.empty:
        print("Pierden significancia al corregir (eran marginales):")
        for _, r in lost.iterrows():
            print(f"   {r['pheno']:<12} {r['model']:<26} Δ={r['delta']:+.3f}  p={r['p_boot']:.3f} → p_BH={r['p_bh']:.3f}")
    print("\nHeadlines que SÍ sobreviven BH-FDR:")
    for head in ["GY_drought", "AD_drought"]:
        for _, r in fdr[(fdr.pheno == head) & fdr.sig_bh_05].iterrows():
            print(f"   {head:<12} {r['model']:<26} Δ={r['delta']:+.3f}  p_BH={r['p_bh']:.4f}")
    print("\nArchivos: robustez_seeds_long.csv · robustez_summary.csv · robustez_fdr.csv · robustez_6pheno.json")


if __name__ == "__main__":
    main()
