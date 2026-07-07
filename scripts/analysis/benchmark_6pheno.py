"""Benchmark de los 6 fenotipos (GY/PH/AD x optimal/drought) vs GBLUP en el fold honesto (val 2019).

Reusa el harness del proyecto (nada de loops de eval ad-hoc):
- Modelos multitask (multi_trait_env, 6 cabezas): `make_multitask_evaluator` -> `run_multitask_evaluation` cacheado.
- GBLUP por (trait, env): `gblup_report_config` + `run_evaluation` cacheado.
- IC pareado vs GBLUP: `align_by_line` (por linea) + `bootstrap_table` (mismo resample) + `paired_diff_summary`.

Salidas en el scratchpad: benchmark_6pheno_tidy.csv, benchmark_6pheno_pearson_wide.csv, benchmark_6pheno.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import pearson_r, coincidence_index
from src.eval.nb_eval import make_multitask_evaluator, gblup_report_config
from src.eval.ensemble import align_by_line, bootstrap_table, paired_diff_summary, fold_predictions
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.training.run import run_evaluation
from src.training.save_model import load_cached_results, results_cache_key, save_cached_results

OUT = Path(__file__).resolve().parent

REPORT_CV = {
    "scheme": "fixed_temporal",
    "val_year": 2019,
    "train_years": [2017, 2018],
    "holdout_test_year": 2020,
    "enforce_disjoint_lines": True,
}
GBLUP_BASE = "configs/exp_gblup_2017_2018_to_2019.yaml"
ARMS = {
    "MoE hard (PCA gating)": "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft (gate aprend.)": "configs/exp_moe_soft_6head_2017_2018_to_2019.yaml",
    "Ensemble (sin gating)": "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE (ruteo x amb.)": "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS (gating superv.)": "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
    "MoE contrastivo": "configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml",
}
TRAITS = ["GY", "PH", "AD"]
ENVS = ["optimal", "drought"]
N_BOOT = 2000
ALPHA = 0.05
METRICS = {"pearson_r": pearson_r, "matching20": lambda yt, yp: coincidence_index(yt, yp, 0.2)}


def eval_gblup_cached(trait: str, env: str) -> dict:
    cfg = gblup_report_config(GBLUP_BASE, trait, env, REPORT_CV)
    set_seed(int(cfg.get("seed", 575)))
    cfg.setdefault("bootstrap", {})
    cfg["bootstrap"].update(enabled=True, n_boot=N_BOOT, alpha=ALPHA)
    key = results_cache_key(cfg, n_boot=N_BOOT, alpha=ALPHA)
    res = load_cached_results(key)
    if res is None or not res.get("predictions"):  # caché viejo sin predicciones -> recomputar
        res = run_evaluation(cfg)
        save_cached_results(key, res)
        print(f"  [computed] GBLUP {trait}_{env}", flush=True)
    else:
        print(f"  [cache hit] GBLUP {trait}_{env}", flush=True)
    return fold_predictions(res, role="scoring", val_year=2019)


def report_cfg_arm(path: str) -> dict:
    cfg = load_config(path)
    cfg["cv"] = dict(REPORT_CV)
    cfg["artifacts"] = {"save_model": False}
    return cfg


def main() -> None:
    eval_multitask = make_multitask_evaluator(list(METRICS), n_boot=N_BOOT, alpha=ALPHA)

    # 1) Corre cada modelo multitask UNA vez -> predicciones de las 6 cabezas.
    arm_preds: dict[str, dict[str, dict]] = {}
    for arm, path in ARMS.items():
        print(f"[arm] {arm} <- {path}", flush=True)
        res = eval_multitask(arm, report_cfg_arm(path))
        arm_preds[arm] = {p["trait"]: p for p in res["predictions"] if p["role"] == "scoring"}
        gyd = arm_preds[arm].get("GY_drought")
        if gyd is not None:
            print(f"    GY_drought pearson = {pearson_r(gyd['y_true'], gyd['y_pred']):.4f}  (n={len(gyd['y_true'])})", flush=True)

    # 2) Por fenotipo: alinea GBLUP + arms por linea, IC pareado vs GBLUP.
    rows: list[dict] = []
    paired_all: dict[str, dict] = {}
    for trait in TRAITS:
        for env in ENVS:
            head = f"{trait}_{env}"
            print(f"[pheno] {head}", flush=True)
            gb_pred = eval_gblup_cached(trait, env)
            named = [("GBLUP", gb_pred)]
            for arm in ARMS:
                if head in arm_preds[arm]:
                    named.append((arm, arm_preds[arm][head]))
            line_id, y_true, yhats = align_by_line(*[p for _, p in named])
            preds_by_name = {nm: yh for (nm, _), yh in zip(named, yhats)}
            rws, samples = bootstrap_table(
                y_true, preds_by_name, METRICS, group=line_id, n_boot=N_BOOT, alpha=ALPHA, seed=0
            )
            paired = paired_diff_summary(samples, "GBLUP", METRICS, alpha=ALPHA)
            paired_all[head] = paired
            for r in rws:
                is_gblup = r["model"] == "GBLUP"
                pr = paired.get(r["model"], {}).get(r["metric"], {})
                rows.append({
                    "trait": trait, "env": env, "pheno": head, "model": r["model"],
                    "metric": r["metric"], "point": r["point"], "lo": r["lo"], "hi": r["hi"],
                    "delta_vs_gblup": (0.0 if is_gblup else pr.get("delta", np.nan)),
                    "sig_vs_gblup": (False if is_gblup else pr.get("sig", False)),
                    "n": r["n"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "benchmark_6pheno_tidy.csv", index=False)

    pear = df[df.metric == "pearson_r"].copy()
    wide = pear.pivot(index="pheno", columns="model", values="point")
    order = ["GBLUP"] + list(ARMS)
    wide = wide[[c for c in order if c in wide.columns]]
    wide = wide.reindex([f"{t}_{e}" for t in TRAITS for e in ENVS])
    wide.to_csv(OUT / "benchmark_6pheno_pearson_wide.csv")

    # JSON completo (para el artifact / informe).
    payload = {
        "config": {"report_cv": REPORT_CV, "gblup_base": GBLUP_BASE, "arms": ARMS,
                   "n_boot": N_BOOT, "alpha": ALPHA},
        "rows": rows,
        "paired_vs_gblup": {
            head: {model: {m: pr[m] for m in pr} for model, pr in paired.items()}
            for head, paired in paired_all.items()
        },
    }
    (OUT / "benchmark_6pheno.json").write_text(json.dumps(payload, indent=2, default=float))

    # --- Impresion en consola: tabla pearson_r con marca de significancia -----
    print("\n" + "=" * 88, flush=True)
    print("PEARSON r (val 2019)  |  * = supera a GBLUP con IC95% de la diferencia sin cruzar 0", flush=True)
    print("=" * 88, flush=True)
    models = [c for c in order if c in pear.model.unique()]
    hdr = f"{'fenotipo':<14}" + "".join(f"{m[:20]:>22}" for m in models)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for head in [f"{t}_{e}" for t in TRAITS for e in ENVS]:
        sub = pear[pear.pheno == head].set_index("model")
        cells = []
        for m in models:
            if m not in sub.index:
                cells.append(f"{'-':>22}")
                continue
            r = sub.loc[m]
            star = "*" if bool(r["sig_vs_gblup"]) else " "
            cells.append(f"{r['point']:>7.3f} [{r['lo']:.2f},{r['hi']:.2f}]{star:>3}")
        print(f"{head:<14}" + "".join(cells), flush=True)
    print("\nArchivos: benchmark_6pheno_tidy.csv | benchmark_6pheno_pearson_wide.csv | benchmark_6pheno.json", flush=True)


if __name__ == "__main__":
    main()
