"""Genera las figuras del informe IEEE (sin contrastivo) a partir de los CSVs ya reproducidos.
Salida: informe/figs/{fig_val2019,fig_noise,fig_moe_diagram,fig_eda_drift}.png
El test 2020 (fig_test2020) se arma aparte cuando termina el job de spike.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REP = "experiments/figs/report"
FIG = "informe/figs"
FI = "figuras_informe"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})

# paleta consistente (6 modelos del informe; contrastivo excluido)
COL = {"GBLUP": "#6b7280", "Red multitarea": "#3f6f8f", "Ensemble": "#3f6f8f",
       "MoE hard": "#12897a", "MoE soft": "#82b366", "MoE GxE": "#c9a227", "MoE PLS": "#d9563a"}
DROP = "Contrastivo"

# ============ FIG val 2019: comparativa 6 modelos, 2 paneles ============
df = pd.read_csv(f"{REP}/fig2_model_comparison.csv")
df = df[df.model != DROP].copy()
df["model"] = df["model"].replace({"Ensemble": "Red multitarea"})
order = df[df.env == "drought"].sort_values("point")["model"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
for ax, env, title in zip(axes, ("drought", "optimal"), ("GY sequía (portada)", "GY óptimo")):
    sub = df[df.env == env].set_index("model").reindex(order)
    gb = df[(df.env == env) & (df.model == "GBLUP")]["point"].iloc[0]
    ax.axvline(gb, ls="--", lw=1.1, color="#6b7280", zorder=1, alpha=0.8)
    for yi, m in enumerate(order):
        r = sub.loc[m]
        ax.errorbar(r["point"], yi, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                    fmt="o", ms=7, capsize=3, color=COL[m], ecolor=COL[m], elinewidth=1.7, zorder=3)
        ax.annotate(f"{r['point']:.3f}", (r["point"], yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color="#222")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Pearson r (val 2019)")
    ax.grid(axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)
    ax.set_xlim(0.0, 0.52)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_val2019.png", bbox_inches="tight")
print("OK fig_val2019.png")

# ============ FIG ruido inferencia: GY sequía, 6 modelos ============
dn = pd.read_csv(f"{REP}/fig4_noise_inference.csv")
dn = dn[dn.model != DROP].copy()
dn["model"] = dn["model"].replace({"Ensemble": "Red multitarea"})
agg = dn[dn["head"] == "GY_drought"].groupby(["model", "eps"]).pearson.agg(["mean", "std"]).reset_index()
order_n = agg[agg.eps == 0.0].sort_values("mean", ascending=False)["model"].tolist()

fig, ax = plt.subplots(figsize=(6.6, 4.2))
for m in order_n:
    s = agg[agg.model == m].sort_values("eps")
    mean = s["mean"].to_numpy(); sd = np.nan_to_num(s["std"].to_numpy()); x = s["eps"].to_numpy()
    ls = "--" if m == "GBLUP" else "-"
    lw = 2.4 if m == "MoE hard" else 1.7
    ax.plot(x, mean, ls, marker="o", ms=4.5, lw=lw, color=COL[m], label=m, zorder=3)
    ax.fill_between(x, mean - sd, mean + sd, color=COL[m], alpha=0.12, lw=0)
ax.set_xlabel(r"$\varepsilon$ — prob. de flip por SNP")
ax.set_ylabel("Pearson r — GY sequía (val 2019)")
ax.set_title("Robustez al ruido en inferencia", fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.93, ncol=2)
ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_noise.png", bbox_inches="tight")
print("OK fig_noise.png")

# ============ MoE diagrama: recortar whitespace ============
im = Image.open(f"{FI}/MoE diagrama principal.png").convert("RGB")
arr = np.asarray(im)
nonwhite = np.any(arr < 245, axis=2)
ys, xs = np.where(nonwhite)
pad = 25
box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
       min(arr.shape[1], xs.max() + pad), min(arr.shape[0], ys.max() + pad))
im.crop(box).save(f"{FIG}/fig_moe_diagram.png")
print("OK fig_moe_diagram.png crop", box)

# ============ EDA drift: copiar ============
shutil.copy(f"{FI}/fig1_eda_pca3d_matplotlib.png", f"{FIG}/fig_eda_drift.png")
print("OK fig_eda_drift.png")
