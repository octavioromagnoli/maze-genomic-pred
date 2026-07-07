"""Análisis de robustez al ruido de los modelos del informe (fig 4 y 5).

Inferencia (fit limpio, val corrupto a cada ε; R realizaciones) para los 7 modelos → curva de
degradación de Pearson vs ε en GY_drought (portada) y GY_optimal. Train-noise (fit sobre train
corrupto, val limpio) para un subset → ¿el ruido en train regulariza? Reusa src.eval.noise.
Guarda CSVs + PNGs en experiments/figs/report/. 2020 SELLADO (solo dev/2019).
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
os.makedirs(OUTDIR, exist_ok=True)
EPS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
R = 5
COL = {"GBLUP": "#8a8a8a", "MoE hard": "#1f8a7c", "MoE soft": "#8ab17d", "Ensemble": "#3f6f8f",
       "MoE GxE": "#c9a227", "MoE PLS": "#d9563a", "Contrastivo": "#9d4edd"}

MT = {  # multitask: exponen GY_optimal y GY_drought
    "MoE hard":    "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft":    "configs/exp_moe_soft_multienv_confirm_2019.yaml",
    "Ensemble":    "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE":     "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS":     "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
    "Contrastivo": "configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml",
}
GBLUP_DRO = "configs/exp_gblup_2017_2018_to_2019_drought.yaml"
GBLUP_OPT = "configs/exp_gblup_2017_2018_to_2019.yaml"

# ====================== FASE 1: INFERENCIA (todos) ======================
print("=== INFERENCIA ===", flush=True)
rows = []
for lab, cfg in MT.items():
    print(f"[inf] {lab}", flush=True)
    rows += noise_curve(cfg, label=lab, epsilons=EPS, mode="inference", n_realizations=R,
                        heads=("GY_optimal", "GY_drought"))
print("[inf] GBLUP (dro + opt)", flush=True)
rows += noise_curve(GBLUP_DRO, label="GBLUP", epsilons=EPS, mode="inference", n_realizations=R, heads=("GY_drought",))
rows += noise_curve(GBLUP_OPT, label="GBLUP", epsilons=EPS, mode="inference", n_realizations=R, heads=("GY_optimal",))
df = pd.DataFrame(rows)
df.to_csv(f"{OUTDIR}/fig4_noise_inference.csv", index=False)

agg = df.groupby(["model", "head", "eps"])["pearson"].agg(["mean", "std"]).reset_index()


def plot_inf(ax, head, title, order):
    for m in order:
        s = agg[(agg["model"] == m) & (agg["head"] == head)].sort_values("eps")
        if s.empty:
            continue
        mean = s["mean"].to_numpy(); sd = np.nan_to_num(s["std"].to_numpy())
        x = s["eps"].to_numpy()
        ls = "--" if m == "GBLUP" else "-"
        ax.plot(x, mean, ls, marker="o", ms=4, lw=2, color=COL[m], label=m, zorder=3)
        ax.fill_between(x, mean - sd, mean + sd, color=COL[m], alpha=0.13, lw=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("ε — prob. de flip por SNP")
    ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)


# orden por Pearson a ε=0 (limpio), por cabeza
def order_by_clean(head):
    c = agg[(agg["head"] == head) & (agg["eps"] == 0.0)].sort_values("mean", ascending=False)
    return c["model"].tolist()


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
plot_inf(axes[0], "GY_drought", "GY sequía (portada)", order_by_clean("GY_drought"))
plot_inf(axes[1], "GY_optimal", "GY óptimo", order_by_clean("GY_optimal"))
axes[0].set_ylabel("Pearson r (val 2019)")
axes[1].legend(fontsize=8, loc="upper right", framealpha=0.92)
fig.suptitle("Robustez al ruido en INFERENCIA — flip Bernoulli de SNPs en val 2019\n"
             f"línea = media de {R} realizaciones · banda = ±1 sd · GBLUP punteado", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUTDIR}/fig4_noise_inference.png", dpi=150, bbox_inches="tight")
print("GUARDADO fig4_noise_inference.png", flush=True)

# --- tabla resumen: caída relativa a ε=0.05 y 0.10 (GY_drought) ---
print("\n-- GY_drought: Pearson por ε (media) --", flush=True)
piv = agg[agg["head"] == "GY_drought"].pivot(index="model", columns="eps", values="mean").round(3)
print(piv.to_string(), flush=True)

# ====================== FASE 2: TRAIN-NOISE (subset) ======================
print("\n=== TRAIN-NOISE (subset) ===", flush=True)
EPS_T = (0.0, 0.05, 0.10); RT = 3
SUB = {"MoE hard": MT["MoE hard"], "Ensemble": MT["Ensemble"], "GBLUP": GBLUP_DRO}
trows = []
for lab, cfg in SUB.items():
    print(f"[train] {lab}", flush=True)
    trows += noise_curve(cfg, label=lab, epsilons=EPS_T, mode="train", n_realizations=RT, heads=("GY_drought",))
dt = pd.DataFrame(trows)
dt.to_csv(f"{OUTDIR}/fig5_noise_train.csv", index=False)
aggt = dt.groupby(["model", "eps"])["pearson"].agg(["mean", "std"]).reset_index()

fig2, ax = plt.subplots(figsize=(7.2, 4.4))
mods = ["MoE hard", "Ensemble", "GBLUP"]
w = 0.26
for i, eps in enumerate(EPS_T):
    xs = np.arange(len(mods)) + (i - 1) * w
    means = [aggt[(aggt.model == m) & (aggt.eps == eps)]["mean"].iloc[0] for m in mods]
    sds = [np.nan_to_num(aggt[(aggt.model == m) & (aggt.eps == eps)]["std"].iloc[0]) for m in mods]
    shade = 1 - 0.28 * i
    ax.bar(xs, means, w, yerr=sds, capsize=3,
           color=[tuple(min(1, c * shade) for c in matplotlib.colors.to_rgb(COL[m])) for m in mods],
           label=f"ε={eps:g}" + (" (limpio)" if eps == 0 else ""))
    for x, mn in zip(xs, means):
        ax.annotate(f"{mn:.3f}", (x, mn), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
ax.set_xticks(np.arange(len(mods))); ax.set_xticklabels(mods)
ax.set_ylabel("Pearson r — GY sequía (val 2019 limpio)")
ax.set_title("Ruido en TRAIN como regularización\n(fit sobre SNPs corruptos → val limpio; ¿sube o baja?)", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(f"{OUTDIR}/fig5_noise_train.png", dpi=150, bbox_inches="tight")
print("GUARDADO fig5_noise_train.png", flush=True)
print("\n-- TRAIN-noise GY_drought (media) --", flush=True)
print(aggt.pivot(index="model", columns="eps", values="mean").round(3).to_string(), flush=True)
print("\nNOISE ANALYSIS DONE", flush=True)
