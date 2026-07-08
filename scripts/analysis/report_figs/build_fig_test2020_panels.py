"""Figura alternativa del test 2020 (sugerencia del compañero): 3 paneles forest-plot
(sin spike | 5% | 10%) al estilo de la comparativa 2019, GY sequía, 6 modelos, media ± sd (3 reps).
NO se referencia en el .tex todavía. Guarda informe/figs/fig_test2020_panels.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "experiments/figs/report/test2020_spike_2018_2019.csv"
OUT = "informe/figs/fig_test2020_panels.png"
COL = {"GBLUP": "#6b7280", "Red multitarea": "#3f6f8f", "MoE hard": "#12897a",
       "MoE soft": "#82b366", "MoE GxE": "#c9a227", "MoE PLS": "#d9563a"}

d = pd.read_csv(CSV)
d["model"] = d["model"].replace({"Ensemble": "Red multitarea"})
d = d[d["head"] == "GY_drought"]
agg = d.groupby(["model", "spike"]).pearson.agg(["mean", "std"]).reset_index()

# orden: por el rescate a 10% (mejor arriba)
order = (agg[agg.spike == 0.10].sort_values("mean")["model"].tolist())  # ascendente => mejor arriba en y
SPIKES = [(0.0, "Sin spike-in (0%)"), (0.05, "Spike-in 5%"), (0.10, "Spike-in 10%")]

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
for ax, (s, title) in zip(axes, SPIKES):
    ax.axvline(0, ls="-", lw=1.2, color="#b00", alpha=0.5, zorder=1)
    gb = agg[(agg.spike == s) & (agg.model == "GBLUP")]["mean"].iloc[0]
    ax.axvline(gb, ls="--", lw=1.0, color="#6b7280", alpha=0.8, zorder=1)
    for yi, m in enumerate(order):
        r = agg[(agg.spike == s) & (agg.model == m)].iloc[0]
        sd = 0.0 if np.isnan(r["std"]) else r["std"]
        ax.errorbar(r["mean"], yi, xerr=sd, fmt="o", ms=7, capsize=3,
                    color=COL[m], ecolor=COL[m], elinewidth=1.7, zorder=3)
        ax.annotate(f"{r['mean']:.3f}", (r["mean"], yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#222")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Pearson r — GY sequía (2020)")
    ax.set_xlim(-0.24, 0.60)
    ax.grid(axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
fig.suptitle("Cierre 2020 — rescate por spike-in (train 2018–2019). Media ± sd sobre 3 reps · "
             "línea roja = 0 · línea gris punteada = GBLUP", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("OK", OUT)
print(agg.pivot(index="model", columns="spike", values="mean").round(3).reindex(order[::-1]).to_string())
