"""Fase TRAIN-noise del análisis de ruido (fig 5), standalone (no recomputa la inferencia).

Entrena sobre SNPs corruptos (flip Bernoulli a ε) y evalúa sobre val 2019 LIMPIO → ¿el ruido en
train regulariza al modelo? Subset MoE hard vs Ensemble (sin gating) vs GBLUP, en GY_drought.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.eval.noise import noise_curve

OUTDIR = "experiments/figs/report"
EPS_T = (0.0, 0.05, 0.10)
RT = 2
COL = {"GBLUP": "#8a8a8a", "MoE hard": "#1f8a7c", "Ensemble": "#3f6f8f"}
SUB = {
    "MoE hard": "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "Ensemble": "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "GBLUP":    "configs/exp_gblup_2017_2018_to_2019_drought.yaml",
}

print("=== TRAIN-NOISE ===", flush=True)
rows = []
for lab, cfg in SUB.items():
    print(f"[train] {lab}", flush=True)
    rows += noise_curve(cfg, label=lab, epsilons=EPS_T, mode="train", n_realizations=RT, heads=("GY_drought",))
dt = pd.DataFrame(rows)
dt.to_csv(f"{OUTDIR}/fig5_noise_train.csv", index=False)
agg = dt.groupby(["model", "eps"])["pearson"].agg(["mean", "std"]).reset_index()

mods = ["MoE hard", "Ensemble", "GBLUP"]
fig, ax = plt.subplots(figsize=(7.4, 4.5))
w = 0.26
for i, eps in enumerate(EPS_T):
    xs = np.arange(len(mods)) + (i - 1) * w
    means = [agg[(agg["model"] == m) & (agg["eps"] == eps)]["mean"].iloc[0] for m in mods]
    sds = [np.nan_to_num(agg[(agg["model"] == m) & (agg["eps"] == eps)]["std"].iloc[0]) for m in mods]
    shade = 1 - 0.26 * i
    cols = [tuple(min(1, c * shade) for c in matplotlib.colors.to_rgb(COL[m])) for m in mods]
    ax.bar(xs, means, w, yerr=sds, capsize=3, color=cols,
           label=("limpio (ε=0)" if eps == 0 else f"train ε={eps:g}"))
    for x, mn in zip(xs, means):
        ax.annotate(f"{mn:.3f}", (x, mn), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
ax.set_xticks(np.arange(len(mods))); ax.set_xticklabels(mods)
ax.set_ylabel("Pearson r — GY sequía (val 2019 LIMPIO)")
ax.set_title("Ruido en TRAIN como regularización\n(fit sobre SNPs corruptos → val limpio; ¿sube o baja?)", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig5_noise_train.png", dpi=150, bbox_inches="tight")
print("GUARDADO fig5_noise_train.png", flush=True)
print(agg.pivot(index="model", columns="eps", values="mean").round(3).to_string(), flush=True)
print("TRAIN-NOISE DONE", flush=True)
