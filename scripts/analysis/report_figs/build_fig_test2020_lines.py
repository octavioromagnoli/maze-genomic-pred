"""Figura + tabla del cierre 2020 (train 2018+2019, spike-in) para el informe.
Lee el CSV que produce spike_2018_2019.py y genera informe/figs/fig_test2020.png (2 paneles:
GY sequía | GY óptimo) + imprime el pivote de medias para llenar la tabla del .tex.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "experiments/figs/report/test2020_spike_2018_2019.csv"
OUT = "informe/figs/fig_test2020.png"
COL = {"GBLUP": "#6b7280", "Red multitarea": "#3f6f8f", "MoE hard": "#12897a",
       "MoE soft": "#82b366", "MoE GxE": "#c9a227", "MoE PLS": "#d9563a"}

d = pd.read_csv(CSV)
d["model"] = d["model"].replace({"Ensemble": "Red multitarea"})
agg = d.groupby(["head", "model", "spike"]).pearson.agg(["mean", "std"]).reset_index()

order = ["MoE hard", "MoE PLS", "MoE GxE", "MoE soft", "Red multitarea", "GBLUP"]
order = [m for m in order if m in agg.model.unique()]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
for ax, h, title in zip(axes, ("GY_drought", "GY_optimal"), ("GY sequía (portada)", "GY óptimo")):
    ax.axhline(0, color="#b00", lw=1.1, ls="-", alpha=0.55, zorder=1)
    for m in order:
        s = agg[(agg["head"] == h) & (agg.model == m)].sort_values("spike")
        if s.empty:
            continue
        x = s["spike"].to_numpy() * 100
        mean = s["mean"].to_numpy(); sd = np.nan_to_num(s["std"].to_numpy())
        ls = "--" if m == "GBLUP" else "-"
        lw = 2.4 if m == "MoE hard" else 1.7
        ax.plot(x, mean, ls, marker="o", ms=5.5, lw=lw, color=COL[m], label=m, zorder=3)
        ax.fill_between(x, mean - sd, mean + sd, color=COL[m], alpha=0.11, lw=0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("% de líneas de 2020 fenotipadas (spike-in)")
    ax.set_xticks([0, 5, 10])
    ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("Pearson r — 2020 held-out")
axes[1].legend(fontsize=8.5, loc="lower right", framealpha=0.93)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("OK", OUT)

print("\n=== MEDIAS para la tabla (Pearson, media de reps) ===")
for h in ("GY_drought", "GY_optimal"):
    print(f"\n-- {h} --")
    piv = agg[agg["head"] == h].pivot(index="model", columns="spike", values="mean").round(3)
    print(piv.reindex([m for m in ["GBLUP", "Red multitarea", "MoE hard", "MoE soft", "MoE GxE", "MoE PLS"] if m in piv.index]).to_string())
