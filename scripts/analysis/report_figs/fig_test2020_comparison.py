"""FIGURA: comparativa de modelos en el holdout 2020 con IC95 real.

Reconstruye el caso cold-start del anexo de spike-in:
train 2017-2019 -> test 2020, sin líneas 2020 agregadas al train. A diferencia
del CSV `fig6_spike_2020.csv`, que solo guarda Pearson agregado por corrida, esta
ruta vuelve a evaluar los modelos y calcula IC95 por bootstrap cluster-linea sobre
las predicciones del holdout 2020.
"""
from __future__ import annotations

import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.eval.nb_eval import make_evaluators, make_multitask_evaluator
from src.utils.config import load_config


N_BOOT = 1000
ALPHA = 0.05
TRAIN_YEARS = [2017, 2018, 2019]
TEST_YEAR = 2020
TARGET_HEAD = "GY_drought"

FIGDIR = "experiments/figs/report"
PUBLISH_DIR = "figuras_informe"
for _dir in (FIGDIR, PUBLISH_DIR):
    os.makedirs(_dir, exist_ok=True)

MODELS = [
    ("GBLUP", "single", "configs/exp_gblup_2017_2018_to_2019_drought.yaml"),
    ("MoE hard", "multitask", "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml"),
    ("Ensemble", "multitask", "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml"),
    ("MoE PLS", "multitask", "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml"),
    ("Contrastivo", "multitask", "configs/exp_moe_contrastive_6head_2017_2018_to_2019.yaml"),
]

COL = {
    "GBLUP": "#888888",
    "MoE hard": "#2a9d8f",
    "MoE soft": "#8ab17d",
    "Ensemble": "#457b9d",
    "MoE GxE": "#e9c46a",
    "MoE PLS": "#e76f51",
    "Contrastivo": "#9d4edd",
}


def test2020_config(path: str) -> dict:
    cfg = copy.deepcopy(load_config(path))
    cfg["cv"] = {
        "scheme": "fixed_temporal",
        "val_year": TEST_YEAR,
        "train_years": TRAIN_YEARS,
        "holdout_test_year": None,
        "enforce_disjoint_lines": True,
    }
    cfg.setdefault("bootstrap", {})
    cfg["bootstrap"].update(enabled=True, n_boot=N_BOOT, alpha=ALPHA)
    cfg["artifacts"] = {"save_model": False}
    return cfg


def _multitask_head(res: dict, head: str) -> tuple[float, float, float, int]:
    rows = [
        r for r in res["per_fold"]
        if r.get("trait") == head and r.get("role") == "scoring"
    ]
    if not rows:
        raise RuntimeError(f"No encontre la cabeza {head!r} en el resultado multitask.")
    r = rows[-1]
    return (
        float(r["pearson_r"]),
        float(r.get("pearson_r_lo", np.nan)),
        float(r.get("pearson_r_hi", np.nan)),
        int(r.get("n_val", 0)),
    )


def _annex_points() -> dict[str, float]:
    csv = os.path.join(PUBLISH_DIR, "fig6_spike_2020.csv")
    if not os.path.exists(csv):
        csv = os.path.join(FIGDIR, "fig6_spike_2020.csv")
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    cold = df[df["spike"] == 0.0]
    return dict(zip(cold["model"], cold["pearson"]))


evaluate_model, _, _, _ = make_evaluators(("pearson_r",), n_boot=N_BOOT, alpha=ALPHA)
eval_multitask = make_multitask_evaluator(("pearson_r",), n_boot=N_BOOT, alpha=ALPHA)
annex = _annex_points()

records: list[dict] = []
for label, kind, cfg_path in MODELS:
    cfg = test2020_config(cfg_path)
    print(f"[eval 2020] {label}", flush=True)
    if kind == "single":
        rows, _ = evaluate_model(label, cfg, n_boot=N_BOOT, alpha=ALPHA)
        r = rows[0]
        point, lo, hi, n = float(r["point"]), float(r["lo"]), float(r["hi"]), int(r["n"])
    else:
        res = eval_multitask(label, cfg, n_boot=N_BOOT, alpha=ALPHA)
        point, lo, hi, n = _multitask_head(res, TARGET_HEAD)

    annex_point = annex.get(label, np.nan)
    delta = point - annex_point if np.isfinite(annex_point) else np.nan
    records.append({
        "model": label,
        "env": "drought",
        "point": point,
        "lo": lo,
        "hi": hi,
        "n": n,
        "annex_point": annex_point,
        "delta_vs_annex": delta,
        "train_years": "-".join(str(y) for y in TRAIN_YEARS),
        "test_year": TEST_YEAR,
        "source_config": cfg_path,
    })
    if np.isfinite(annex_point):
        print(
            f"  r={point:+.6f}  IC95[{lo:+.6f}, {hi:+.6f}]  "
            f"n={n}  delta_anexo={delta:+.2e}",
            flush=True,
        )
    else:
        print(f"  r={point:+.6f}  IC95[{lo:+.6f}, {hi:+.6f}]  n={n}", flush=True)

df = pd.DataFrame(records)
df = df.sort_values("point").reset_index(drop=True)
print("\n=== TABLA Pearson test 2020 (punto [IC95%]) ===")
for _, r in df.sort_values("point", ascending=False).iterrows():
    print(f"  {r['model']:12s}  {r['point']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  n={int(r['n'])}")

order = df["model"].tolist()
fig, ax = plt.subplots(figsize=(8.8, 4.2))
y = np.arange(len(order))
for yi, (_, r) in zip(y, df.iterrows()):
    m = r["model"]
    xerr = [[r["point"] - r["lo"]], [r["hi"] - r["point"]]]
    ax.errorbar(
        r["point"],
        yi,
        xerr=xerr,
        fmt="o",
        ms=7.5,
        capsize=3,
        color=COL.get(m, "#333"),
        ecolor=COL.get(m, "#333"),
        elinewidth=1.6,
        zorder=3,
    )
    ax.annotate(
        f"{r['point']:.3f}",
        (r["point"], yi),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8.5,
        color="#222",
    )

gb = float(df[df["model"] == "GBLUP"]["point"].iloc[0])
ax.axvline(gb, ls="--", lw=1.2, color="#888888", zorder=1)
ax.axvline(0, ls="-", lw=1.0, color="#b00020", alpha=0.55, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.set_ylim(-0.55, len(order) - 0.05)
ax.set_xlabel("Pearson r (test 2020)")
ax.set_title("GY sequía — holdout 2020 sin spike-in", fontsize=11, fontweight="bold")
ax.grid(axis="x", ls=":", alpha=0.5)
ax.set_axisbelow(True)
xmin = float(df["lo"].min())
xmax = float(df["hi"].max())
pad = max(0.03, (xmax - xmin) * 0.18)
ax.set_xlim(min(xmin - pad, -0.02), max(xmax + pad, 0.02))

fig.suptitle(
    "Comparativa de modelos — Pearson predicho vs. observado, test 2020\n"
    "IC95% bootstrap cluster-línea; línea punteada = GBLUP; línea roja = 0",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))

for outdir in (FIGDIR, PUBLISH_DIR):
    png = os.path.join(outdir, "fig2_model_comparison_2020.png")
    csv = os.path.join(outdir, "fig2_model_comparison_2020.csv")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    df.to_csv(csv, index=False)
    print(f"GUARDADO: {png}")
    print(f"GUARDADO: {csv}")
