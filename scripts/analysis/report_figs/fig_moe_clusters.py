"""FIGURA: análisis de clusters de la solución del MoE (gating PCA+GMM del campeón, 6 cabezas).

Replica EXACTAMENTE el ruteo del MoE hard campeón (z-score -> PCA-100 whiten -> GMM k=4 full,
min_cluster_size=250), ajustado train-only sobre 2017+2018, y proyecta val 2019. NO entrena los
expertos (el ruteo lo define el gating; los expertos son downstream). Muestra:
  (A) el espacio PCA del gating coloreado por cluster (train) + val 2019 superpuesto,
  (B) tamaño de cada cluster en train (umbral min_cluster_size => expert propio vs fallback),
  (C) GY_drought OBSERVADO medio por cluster => las subpoblaciones difieren en potencial de rinde,
      que es lo que un experto por cluster puede explotar.
2020 SELLADO (no entra).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.getcwd())
from src.utils.config import load_config
from src.data.loader import load_dataset
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor

OUTDIR = "experiments/figs/report"
os.makedirs(OUTDIR, exist_ok=True)
CFG = "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml"
cfg = load_config(CFG)

# --- Arrays dev (6 cabezas). Head order: [GY_opt, GY_dro, PH_opt, PH_dro, AD_opt, AD_dro] ---
ds = load_dataset(cfg["data"]["geno_csv"], cfg["data"]["pheno_csv"])
arr = ds.to_modeling_arrays_multi_trait_env(targets=["GY", "PH", "AD"], envs=["optimal", "drought"])
X, year, Y, M = arr.X, np.asarray(arr.year, int), arr.Y, arr.Y_mask
GY_DRO = 1  # indice de la cabeza GY_drought
dev = np.isin(year, [2017, 2018, 2019])
X, year, Y, M = X[dev], year[dev], Y[dev], M[dev]
tr = np.isin(year, [2017, 2018]); va = year == 2019
print(f"train(17-18)={tr.sum()}  val(19)={va.sum()}")

# --- Gating del campeón, train-only (idéntico a MixtureOfExpertsMultitaskRegressor.fit) ---
model = MixtureOfExpertsMultitaskRegressor({"params": cfg["model"]["params"]})
g = model.gating_params
Z_tr = np.asarray(model._fit_gating(X[tr], Y[tr], M[tr]))   # ajusta self.pre_ SOLO en train
gmm = GaussianMixture(n_components=int(g["k"]), covariance_type=str(g["covariance_type"]),
                      reg_covar=float(g["reg_covar"]), n_init=int(g["n_init"]),
                      random_state=int(g["random_state"]))
gmm.fit(Z_tr)
lab_tr = gmm.predict(Z_tr)
Z_va = np.asarray(model.pre_.transform(X[va]))
lab_va = gmm.predict(Z_va)
K = int(g["k"]); MIN = int(g["min_cluster_size"])

sizes_tr = np.array([(lab_tr == c).sum() for c in range(K)])
sizes_va = np.array([(lab_va == c).sum() for c in range(K)])
own_expert = sizes_tr >= MIN
print("cluster sizes train:", sizes_tr.tolist(), " val:", sizes_va.tolist(),
      " expert propio:", own_expert.tolist(), f"(min_cluster_size={MIN})")

# Composición por año de cada cluster (train) => ¿subpoblaciones genéticas o proxies del año?
yr_tr = year[tr]
YEARS = [2017, 2018]  # train
comp = np.zeros((K, len(YEARS)))
for c in range(K):
    for j, y in enumerate(YEARS):
        comp[c, j] = ((lab_tr == c) & (yr_tr == y)).sum()
comp_frac = comp / comp.sum(axis=1, keepdims=True)
print("composición por año (train) por cluster:\n", comp.astype(int))

# GY_drought observado medio por cluster (para texto/csv, no como panel)
gy = Y[tr][:, GY_DRO]; gym = M[tr][:, GY_DRO].astype(bool)
per_cluster_gy = [((lab_tr == c) & gym).sum() and gy[(lab_tr == c) & gym].mean() for c in range(K)]
print("GY_drought obs medio por cluster:", np.round(per_cluster_gy, 3).tolist())

PAL = ["#e76f51", "#2a9d8f", "#457b9d", "#e9c46a", "#9d4edd", "#8ab17d"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})

# (A) espacio PCA del gating: train por cluster + val superpuesto
ax = axes[0]
for c in range(K):
    m = lab_tr == c
    ax.scatter(Z_tr[m, 0], Z_tr[m, 1], s=12, alpha=0.45, color=PAL[c % len(PAL)],
               label=f"cluster {c} (n={sizes_tr[c]}{'' if own_expert[c] else ', →fallback'})")
ax.scatter(Z_va[:, 0], Z_va[:, 1], s=16, facecolors="none", edgecolors="black",
           linewidths=0.5, alpha=0.5, label=f"val 2019 (n={va.sum()})")
# centroides GMM
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=220, marker="X",
           c=[PAL[c % len(PAL)] for c in range(K)], edgecolor="black", linewidth=1.6, zorder=10)
ax.set_xlabel("PC1 del gating (whitened)"); ax.set_ylabel("PC2 del gating (whitened)")
ax.set_title("(A) Ruteo GMM sobre PCA — subpoblaciones\ntrain 2017-2018 (color) + val 2019 (○)", fontsize=10)
ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
ax.grid(ls=":", alpha=0.4); ax.set_axisbelow(True)

# (B) tamaños de cluster train/val + umbral
ax = axes[1]
xk = np.arange(K)
ax.bar(xk - 0.19, sizes_tr, 0.38, color=[PAL[c % len(PAL)] for c in range(K)], label="train 17-18")
ax.bar(xk + 0.19, sizes_va, 0.38, color=[PAL[c % len(PAL)] for c in range(K)], alpha=0.45, label="val 19")
ax.axhline(MIN, ls="--", color="#b00", lw=1.3)
ax.text(K - 0.5, MIN + 8, f"min_cluster_size={MIN}", color="#b00", ha="right", fontsize=8)
for c in range(K):
    ax.annotate(str(sizes_tr[c]), (c - 0.19, sizes_tr[c]), textcoords="offset points",
                xytext=(0, 2), ha="center", fontsize=7.5)
ax.set_xticks(xk); ax.set_xticklabels([f"c{c}" for c in range(K)])
ax.set_ylabel("nº de líneas"); ax.set_title("(B) Tamaño de subpoblación\n(≥umbral = experto propio)", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True)

# (C) composición por año de cada cluster (train) — ¿subpoblación o proxy del año?
ax = axes[2]
YRCOL = {2017: "#1b6ca8", 2018: "#e07a2f"}
bottom = np.zeros(K)
for j, y in enumerate(YEARS):
    ax.bar(xk, comp_frac[:, j], 0.6, bottom=bottom, color=YRCOL[y], label=str(y))
    bottom += comp_frac[:, j]
ax.set_xticks(xk); ax.set_xticklabels([f"c{c}" for c in range(K)])
ax.set_ylim(0, 1); ax.set_ylabel("fracción de líneas del cluster")
ax.set_title("(C) Composición por año de cada cluster\n(c2 cruza campañas; c1/c3 son 2018 → gating ligado en parte al drift)", fontsize=10)
ax.legend(fontsize=8, title="campaña"); ax.grid(axis="y", ls=":", alpha=0.4); ax.set_axisbelow(True)

fig.suptitle("Análisis de clusters de la solución del MoE — gating PCA+GMM (k=4) del campeón, train-only",
             fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(OUTDIR, "fig3_moe_clusters.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"GUARDADO: {out}")
