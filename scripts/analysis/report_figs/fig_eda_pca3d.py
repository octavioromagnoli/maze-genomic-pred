"""FIGURA EDA: PCA 3D de los SNPs coloreado por año (solo dev 2017-2019; 2020 SELLADO).

Muestra la estructura poblacional difusa + el DRIFT entre campañas (los centroides anuales se
separan), que es el fundamento del split temporal fijo. Reusa src.eda.eda (pca_embed, year_centroids,
drift_auc). PCA descriptiva ajustada sobre dev (2020 no entra => sin leakage del holdout).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.getcwd())
from src.eda.eda import pca_embed, year_centroids, drift_auc

OUTDIR = "experiments/figs/report"
os.makedirs(OUTDIR, exist_ok=True)
DEV_YEARS = [2017, 2018, 2019]

# --- Cargar genotipos dev + año por linea -----------------------------------
geno = pd.read_csv("data/Geno_data.csv")
pheno = pd.read_csv("data/Pheno_data.csv")
gid2year = pheno.drop_duplicates("GID").set_index("GID")["Year"]
geno = geno.copy()
geno["Year"] = geno["GID"].map(gid2year)
dev = geno[geno["Year"].isin(DEV_YEARS)].reset_index(drop=True)
snp_cols = [c for c in geno.columns if c not in ("GID", "Year")]
X = dev[snp_cols].to_numpy(dtype=np.float64)
years = dev["Year"].to_numpy(dtype=int)
print(f"dev: {X.shape[0]} lineas x {X.shape[1]} SNPs  |  por año: "
      f"{ {int(y): int((years==y).sum()) for y in DEV_YEARS} }")

# --- PCA 3D + centroides + drift AUC ----------------------------------------
Z, evr = pca_embed(X, n_components=3)
cen = year_centroids(Z, years, n_pcs=3)
drift = drift_auc(X, years, pairs=[(2017, 2018), (2018, 2019), (2017, 2019)], n_pcs=20)
print("varianza explicada PC1-3:", np.round(evr[:3], 4))
print(drift.to_string(index=False))

COL = {2017: "#1b6ca8", 2018: "#e07a2f", 2019: "#2a9d5c"}
fig = plt.figure(figsize=(9.5, 7.2))
ax = fig.add_subplot(111, projection="3d")
for y in DEV_YEARS:
    m = years == y
    ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=8, alpha=0.35, color=COL[y],
               label=f"{y}  (n={int(m.sum())})", depthshade=True)
# centroides
for _, r in cen.iterrows():
    y = int(r["Year"])
    ax.scatter(r["PC1"], r["PC2"], r["PC3"], s=260, marker="D", color=COL[y],
               edgecolor="black", linewidth=1.6, zorder=10)
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", labelpad=6)
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", labelpad=6)
ax.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)", labelpad=6)
ax.set_title("Estructura poblacional (PCA de 4 612 SNPs) por campaña — dev 2017-2019\n"
             "rombos = centroide anual; su separación = drift genético entre cohortes",
             fontsize=11)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.view_init(elev=18, azim=-60)
# caja de texto con el drift AUC (separabilidad genética entre años)
txt = "Drift (CV-AUC logística sobre 20 PCs):\n" + "\n".join(
    f"  {r['par_de_años']}:  {r['cv_auc']:.2f}" for _, r in drift.iterrows())
txt += "\n(≈0.5 = años indistinguibles · ≈1 = cohortes distintas)"
ax.text2D(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5,
          bbox=dict(boxstyle="round", fc="#fffbe6", ec="#caa", alpha=0.95), va="bottom")
fig.tight_layout()
out = os.path.join(OUTDIR, "fig1_eda_pca3d.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"GUARDADO: {out}")
cen.to_csv(os.path.join(OUTDIR, "fig1_centroids.csv"), index=False)
drift.to_csv(os.path.join(OUTDIR, "fig1_drift_auc.csv"), index=False)
