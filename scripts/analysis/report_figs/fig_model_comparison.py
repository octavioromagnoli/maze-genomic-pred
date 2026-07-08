"""FIGURA: comparativa de los 6 modelos del informe en GY_drought y GY_optimal (val 2019).

Portada = Pearson (regla 5), con IC95% por bootstrap cluster-linea (ya calculado por el harness).
Reusa el camino cacheado del proyecto (make_evaluators / make_multitask_evaluator, nb_eval) => no
reentrena si el resultado ya esta en outputs/eval_cache. 2020 sigue sellado (solo se toca dev/2019).

Los 6 modelos MoE/ensemble son multitask (run_multitask_evaluation, cabezas GY_optimal/GY_drought);
GBLUP es single-task por ambiente (run_evaluation). Guarda PNG + imprime la tabla de numeros.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.eval.nb_eval import make_evaluators, make_multitask_evaluator

N_BOOT, ALPHA = 1000, 0.05
OUTDIR = "experiments/figs/report"
os.makedirs(OUTDIR, exist_ok=True)

# --- Mapeo modelo -> config(s) val2019 --------------------------------------
# multitask: (config, es_multienv?) ; GBLUP: (config_drought, config_optimal)
MULTITASK = {   # label -> config (todos exponen cabezas GY_optimal / GY_drought)
    "MoE hard":    "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft":    "configs/exp_moe_soft_multienv_confirm_2019.yaml",
    "Ensemble":    "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE":     "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS":     "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
}
GBLUP = {"drought": "configs/exp_gblup_2017_2018_to_2019_drought.yaml",
         "optimal": "configs/exp_gblup_2017_2018_to_2019.yaml"}

evaluate_model, _, _, _ = make_evaluators(("pearson_r",), n_boot=N_BOOT, alpha=ALPHA)
eval_multitask = make_multitask_evaluator(("pearson_r",), n_boot=N_BOOT, alpha=ALPHA)


def mt_head(res, head):
    rows = [f for f in res["per_fold"] if f.get("trait") == head and f.get("role") == "scoring"]
    r = rows[-1]
    return (r["pearson_r"], r.get("pearson_r_lo", np.nan), r.get("pearson_r_hi", np.nan), int(r.get("n_val", 0)))


def gblup_head(cfg):
    rows, _ = evaluate_model("GBLUP", cfg)
    r = rows[0]
    return (r["point"], r["lo"], r["hi"], r["n"])


records = []  # {model, env, point, lo, hi}
for label, cfg in MULTITASK.items():
    print(f"[eval] {label}")
    res = eval_multitask(label, cfg)
    for env, head in (("drought", "GY_drought"), ("optimal", "GY_optimal")):
        p, lo, hi, n = mt_head(res, head)
        records.append({"model": label, "env": env, "point": p, "lo": lo, "hi": hi, "n": n})

print("[eval] GBLUP")
for env in ("drought", "optimal"):
    p, lo, hi, n = gblup_head(GBLUP[env])
    records.append({"model": "GBLUP", "env": env, "point": p, "lo": lo, "hi": hi, "n": n})

df = pd.DataFrame(records)
print("\n=== TABLA Pearson val 2019 (punto [IC95%]) ===")
for env in ("drought", "optimal"):
    print(f"\n-- GY_{env} --")
    sub = df[df.env == env].sort_values("point", ascending=False)
    for _, r in sub.iterrows():
        print(f"  {r['model']:12s}  {r['point']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  n={r['n']}")

# --- Plot: dos paneles (drought | optimal), punto +- IC95%, ordenado por drought ---
order = df[df.env == "drought"].sort_values("point").set_index("model").index.tolist()
COL = {"GBLUP": "#888888", "MoE hard": "#2a9d8f", "MoE soft": "#8ab17d",
       "Ensemble": "#457b9d", "MoE GxE": "#e9c46a", "MoE PLS": "#e76f51"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, env, title in zip(axes, ("drought", "optimal"), ("GY sequía (portada)", "GY óptimo")):
    sub = df[df.env == env].set_index("model").reindex(order)
    y = np.arange(len(order))
    for yi, m in zip(y, order):
        r = sub.loc[m]
        ax.errorbar(r["point"], yi, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                    fmt="o", ms=7, capsize=3, color=COL.get(m, "#333"),
                    ecolor=COL.get(m, "#333"), elinewidth=1.6, zorder=3)
        ax.annotate(f"{r['point']:.3f}", (r["point"], yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#222")
    gb = df[(df.env == env) & (df.model == "GBLUP")]["point"].iloc[0]
    ax.axvline(gb, ls="--", lw=1.2, color="#888888", zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Pearson r (val 2019)")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
axes[0].set_yticks(np.arange(len(order)))
axes[0].set_yticklabels(order)
fig.suptitle("Comparativa de modelos — Pearson predicho vs. observado, validación 2019 (dev)\n"
             "línea punteada = GBLUP (baseline aditivo)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(OUTDIR, "fig2_model_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nGUARDADO: {out}")
df.to_csv(os.path.join(OUTDIR, "fig2_model_comparison.csv"), index=False)
