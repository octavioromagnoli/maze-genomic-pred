"""Cierre 2020 con spike-in (fig 6): rescate del colapso cold-start para los modelos del informe.

Train = dev 2017-2019 + fracción s de líneas de 2020 (con fenotipo) → predice el 2020 restante.
Métrica: Pearson GY_drought en el 2020 held-out. spike ∈ {0, 5%, 10%}, reps {1,42,2024}.
Guarda CSV incremental + PNG. TOCA 2020 (es el cierre, regla 3). Reusa src.eval.spike.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.eval.spike import spike_closure

OUTDIR = "experiments/figs/report"
os.makedirs(OUTDIR, exist_ok=True)
CSV = f"{OUTDIR}/fig6_spike_2020.csv"
SPIKES = (0.0, 0.05, 0.10)
REPS = (1, 42, 2024)
COL = {"GBLUP": "#8a8a8a", "MoE hard": "#1f8a7c", "Ensemble": "#3f6f8f",
       "MoE PLS": "#d9563a", "Contrastivo": "#9d4edd"}

# GBLUP primero (rápido → resultado parcial temprano), después los pesados
MODELS = [
    ("GBLUP",       "configs/exp_gblup_2017_2018_to_2019_drought.yaml"),
    ("MoE hard",    "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml"),
    ("Ensemble",    "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml"),
    ("MoE PLS",     "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml"),
    ("Contrastivo", "configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml"),
]

all_rows = []
for label, cfg in MODELS:
    print(f"\n=== {label} ===", flush=True)
    try:
        rows = spike_closure(cfg, label=label, train_years=(2017, 2018, 2019),
                             spike_fracs=SPIKES, reps=REPS, target_head="GY_drought")
        for r in rows:
            print(f"  {label:12s} spike={r['spike']:.2f} rep{r['rep']:<5d} "
                  f"GY_drought={r['pearson']:+.3f} (held={r['n_held']}, spike={r['n_spike']})", flush=True)
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(CSV, index=False)   # guardado incremental
    except Exception as e:
        print(f"  [ERROR] {label}: {e}", flush=True)

df = pd.DataFrame(all_rows)
df.to_csv(CSV, index=False)
print(f"\nGUARDADO {CSV}", flush=True)

# --- figura: Pearson vs spike, línea por modelo ---
agg = df.groupby(["model", "spike"])["pearson"].agg(["mean", "std"]).reset_index()
order = [m for m, _ in MODELS if m in agg["model"].unique()]
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.axhline(0, color="#b00", lw=1.2, ls="-", alpha=0.6, zorder=1)
ax.text(0.001, 0.01, "colapso cold-start", color="#b00", fontsize=8, va="bottom")
for m in order:
    s = agg[agg["model"] == m].sort_values("spike")
    x = s["spike"].to_numpy() * 100
    mean = s["mean"].to_numpy(); sd = np.nan_to_num(s["std"].to_numpy())
    lw = 2.6 if m in ("MoE hard", "GBLUP") else 1.8
    ls = "--" if m == "GBLUP" else "-"
    ax.plot(x, mean, ls, marker="o", ms=6, lw=lw, color=COL.get(m, "#333"), label=m, zorder=3)
    ax.fill_between(x, mean - sd, mean + sd, color=COL.get(m, "#333"), alpha=0.12, lw=0)
ax.set_xlabel("% de líneas de 2020 fenotipadas y agregadas al train (spike-in)")
ax.set_ylabel("Pearson r — GY sequía en 2020 held-out")
ax.set_xticks([0, 5, 10])
ax.set_title("Cierre 2020: el spike-in rescata la predicción de rinde\n"
             "train = dev 2017-2019 + spike · media de 3 reps · banda ±1 sd", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.93)
ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig6_spike_2020.png", dpi=150, bbox_inches="tight")
print("GUARDADO fig6_spike_2020.png", flush=True)

print("\n-- resumen (Pearson medio por spike) --", flush=True)
print(agg.pivot(index="model", columns="spike", values="mean").round(3).to_string(), flush=True)
print("SPIKE 2020 DONE", flush=True)
