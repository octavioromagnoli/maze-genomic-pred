"""PCA 3D interactiva del genotipo (SNPs) del DEV set, coloreada por campaña.

Proyecta las líneas de 2017-2019 sobre sus 3 primeras componentes principales y
las grafica en un scatter 3D de plotly: HTML autocontenido que se abre en el
navegador y se puede rotar/reorientar con el mouse (orbitar, pan, zoom). Cada
punto es una línea (GID) y el color es el año (2017/2018/2019).

IMPORTANTE — esto NO es preprocesamiento de modelado. Es exploración de la
estructura poblacional del DEV set completo, así que el PCA se ajusta sobre las
3042 líneas de 2017-2019 A PROPÓSITO. La regla anti-leakage (regla #1 de
CLAUDE.md: fit solo en train) gobierna el pipeline predictivo, no esta figura.
El año 2020 (test) queda excluido adrede para no mirarlo.

Uso:
    python scripts/analysis/pca3d_dev_years.py
    python scripts/analysis/pca3d_dev_years.py --scale --out mi_fig.html
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset

# Okabe-Ito: distinguibles y seguros para daltonismo.
YEAR_COLORS = {"2017": "#0072B2", "2018": "#E69F00", "2019": "#009E73"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geno", default="data/Geno_data.csv", help="CSV de genotipos (GID + SNPs).")
    p.add_argument("--pheno", default="data/Pheno_data.csv", help="CSV de fenotipos (GID, Year, ...).")
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019], help="Campañas del DEV a graficar.")
    p.add_argument("--out", default="experiments/figs/pca3d_dev_2017_2019.html", help="Ruta del HTML de salida.")
    p.add_argument("--scale", action="store_true",
                   help="Estandarizar SNPs (media 0, var 1) antes del PCA. Por defecto solo se centra.")
    p.add_argument("--no-show", action="store_true", help="No abrir el navegador; solo escribir el HTML.")
    return p.parse_args()


def build_line_matrix(geno: str, pheno: str, years: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X, gid, year) con UNA fila por línea del DEV (años disjuntos)."""
    ds = load_dataset(geno, pheno)
    # Una fila por GID; en este dataset cada GID se fenotipa en un solo año.
    sub = ds.pheno[ds.pheno["Year"].isin(years)][["GID", "Year"]].drop_duplicates("GID")
    row_of_gid = {g: i for i, g in enumerate(ds.gid_index)}
    keep = sub["GID"].map(row_of_gid)
    missing = int(keep.isna().sum())
    if missing:
        print(f"[aviso] {missing} GID de pheno sin genotipo en {geno}; se descartan.")
    sub = sub[keep.notna()]
    rows = keep[keep.notna()].astype(int).to_numpy()
    return ds.G[rows], sub["GID"].to_numpy(), sub["Year"].to_numpy(dtype=int)


def main() -> None:
    args = parse_args()
    X, gid, year = build_line_matrix(args.geno, args.pheno, args.years)
    print(f"[info] {X.shape[0]} líneas × {X.shape[1]} SNPs sobre años {sorted(set(year.tolist()))}")

    if args.scale:
        # Descartar SNPs de varianza nula (evita dividir por 0 al estandarizar).
        nz = X.std(axis=0) > 0
        X = StandardScaler().fit_transform(X[:, nz].astype(float))
        print(f"[info] estandarizado; {int((~nz).sum())} SNPs de varianza nula descartados")

    pca = PCA(n_components=3, random_state=0)  # PCA centra las columnas automáticamente
    scores = pca.fit_transform(X.astype(float))
    var = pca.explained_variance_ratio_ * 100
    print(f"[info] varianza explicada: PC1={var[0]:.2f}%  PC2={var[1]:.2f}%  PC3={var[2]:.2f}%  (Σ={var.sum():.2f}%)")

    df = pd.DataFrame({
        "PC1": scores[:, 0], "PC2": scores[:, 1], "PC3": scores[:, 2],
        "GID": gid, "Campaña": year.astype(str),
    }).sort_values("Campaña")

    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3", color="Campaña",
        color_discrete_map=YEAR_COLORS,
        hover_data={"GID": True, "PC1": ":.2f", "PC2": ":.2f", "PC3": ":.2f"},
        title=f"PCA 3D del genotipo (SNPs) — DEV {sorted(set(year.tolist()))}, coloreado por campaña",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.75, line=dict(width=0)))
    fig.update_layout(
        legend=dict(
            title="Campaña",
            itemsizing="constant",
            # Arranca adentro del gráfico (arriba-izquierda) con fondo semitransparente
            # para que se lea sobre la escena 3D. En el HTML se puede arrastrar a gusto.
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1,
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)",
        ),
    )

    # editable de la legend: en el HTML se puede arrastrar y soltar donde uno quiera
    # (sin habilitar edición de textos, para no cambiarlos por accidente).
    config = {"edits": {"legendPosition": True}, "displaylogo": False}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.write_html(args.out, include_plotlyjs=True, full_html=True, config=config)
    print(f"[ok] HTML interactivo escrito en {args.out}")
    if not args.no_show:
        fig.show(config=config)


if __name__ == "__main__":
    main()
