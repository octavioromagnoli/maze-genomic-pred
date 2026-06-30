"""Figuras reutilizables del notebook de comparación de modelos (IC por bootstrap).

Centraliza los gráficos para respetar la convención del proyecto (los notebooks solo
importan y llaman funciones de ``src/``) y para que cada sección del notebook pueda
levantarlos con un solo import y correr de forma independiente. Todas las funciones
**devuelven el ``Figure``** (no lo guardan): el notebook decide dónde hacer ``savefig``.

Dos formas de entrada, según lo que tengas a mano:

- ``df`` tidy con columnas ``[model, metric, point, lo, hi]`` (lo que arman
  ``evaluate_model`` / :func:`src.eval.ensemble.bootstrap_table`) ⇒ :func:`plot_ci`,
  :func:`plot_metric_vs_param`.
- ``samples_by_model = {model: {metric: muestras_bootstrap}}`` ⇒ :func:`plot_box` y
  :func:`plot_paired_diff` (la diferencia pareada vs una referencia, que es lo que mide
  significancia: misma seed/grupo en el bootstrap ⇒ el ruido compartido del val se cancela).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _label_kw() -> str:
    """``tick_labels`` (matplotlib >= 3.9) vs ``labels`` (versiones viejas) para boxplot."""
    return "tick_labels" if tuple(map(int, mpl.__version__.split(".")[:2])) >= (3, 9) else "labels"


def _metrics_of(df, metrics: Sequence[str] | None) -> list[str]:
    """Métricas a graficar: las pasadas o las del df, preservando el orden de inserción."""
    return list(df["metric"].drop_duplicates()) if metrics is None else list(metrics)


def plot_ci(df, metrics: Sequence[str] | None = None, *,
            title: str = "Estimación puntual ± IC95% (bootstrap)", color: str = "tab:blue"):
    """Un subplot por métrica; cada modelo como punto con bigotes ``[lo, hi]`` (IC95%)."""
    metrics = _metrics_of(df, metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sub = df[df.metric == m].reset_index(drop=True)
        x = np.arange(len(sub))
        point = sub["point"].to_numpy(float)
        lo, hi = sub["lo"].to_numpy(float), sub["hi"].to_numpy(float)
        # yerr no-negativo (el punto puede caer fuera del IC en distribuciones sesgadas).
        yerr = np.vstack([np.maximum(point - lo, 0), np.maximum(hi - point, 0)])
        ax.errorbar(x, point, yerr=yerr, fmt="o", capsize=5, lw=1.5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=25, ha="right")
        ax.set_title(m)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_box(samples_by_model: Mapping[str, Mapping[str, np.ndarray]], metric: str, *,
             whis: tuple[float, float] = (2.5, 97.5), title: str | None = None):
    """Boxplot de la distribución bootstrap de ``metric``. ``whis=(2.5, 97.5)`` ⇒ bigotes = IC95%."""
    models = [k for k, s in samples_by_model.items() if np.isfinite(s[metric]).any()]
    data = [np.asarray(s[metric])[np.isfinite(s[metric])]
            for s in samples_by_model.values() if np.isfinite(s[metric]).any()]
    if not data:
        print(f"(sin muestras finitas para {metric})")
        return None
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 2, 4))
    ax.boxplot(data, whis=whis, showfliers=False, **{_label_kw(): models})
    ax.set_title(title or f"{metric} — distribución bootstrap (bigotes = IC95%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return fig


def plot_paired_diff(samples_by_model: Mapping[str, Mapping[str, np.ndarray]],
                     ref: str | None = None, metrics: Sequence[str] | None = None, *,
                     title: str | None = None):
    """Boxplot de la DIFERENCIA pareada ``(modelo − referencia)`` por métrica.

    Cada caja = ``muestras_modelo − muestras_ref`` sobre el MISMO resample (misma
    seed/grupo en el bootstrap) ⇒ el ruido compartido de qué líneas caen en val se
    cancela. Si la caja/bigotes (IC95%) NO cruzan el 0 (rojo) la diferencia es
    significativa. Signo: ``pearson_r``/``r2``/``matching20`` > 0 ⇒ el modelo le gana;
    ``rmse`` < 0 ⇒ le gana (menos error). ``ref`` por defecto = primer modelo del dict.
    """
    names = list(samples_by_model)
    ref = names[0] if ref is None else ref
    others = [n for n in names if n != ref]
    metrics = list(next(iter(samples_by_model.values()))) if metrics is None else list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        data = []
        for name in others:
            sx = np.asarray(samples_by_model[name][m], dtype=float)
            sr = np.asarray(samples_by_model[ref][m], dtype=float)
            n = min(len(sx), len(sr))
            d = (sx[:n] - sr[:n])
            data.append(d[np.isfinite(d)])
        ax.boxplot(data, whis=(2.5, 97.5), showfliers=False)
        ax.set_xticks(range(1, len(others) + 1))
        ax.set_xticklabels(others, rotation=25, ha="right")
        ax.axhline(0, color="red", ls="--", lw=1)          # 'sin diferencia'
        ax.set_title(f"{m}: modelo − referencia")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title or f"Diferencia pareada vs '{ref}' — caja que NO cruza 0 (rojo) ⇒ significativa",
                 y=1.03)
    fig.tight_layout()
    return fig


def plot_metric_vs_param(df, param_by_model: Mapping[str, float],
                         metrics: Sequence[str] | None = None, *,
                         ref_lines: Mapping[str, Mapping[str, float]] | None = None,
                         xlabel: str = "parámetro", title: str | None = None,
                         color: str = "tab:blue"):
    """Una línea por métrica vs un parámetro continuo (p.ej. ``mse_weight``) con banda IC95%.

    ``df``: tidy ``[model, metric, point, lo, hi]``. ``param_by_model``: ``{model: x}`` para
    los modelos que van sobre la curva (los que no estén acá se ignoran ⇒ permite dejar
    afuera baselines como GBLUP). ``ref_lines``: ``{metric: {label: valor}}`` ⇒ líneas
    horizontales de referencia por subplot (p.ej. el baseline GBLUP y el loss puro).
    """
    metrics = _metrics_of(df, metrics)
    ref_lines = ref_lines or {}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sub = df[df.metric == m]
        pts = [(param_by_model[r.model], r.point, r.lo, r.hi)
               for r in sub.itertuples() if r.model in param_by_model]
        pts.sort()
        if pts:
            xs, ys, los, his = (np.array(v, dtype=float) for v in zip(*pts))
            ax.fill_between(xs, los, his, color=color, alpha=0.15)
            ax.plot(xs, ys, "o-", color=color, lw=1.8)
        for label, val in ref_lines.get(m, {}).items():
            ax.axhline(val, ls="--", lw=1.2, alpha=0.85, label=label)
        ax.set_title(m)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if ref_lines.get(m):
            ax.legend(fontsize=8)
    fig.suptitle(title or f"Métrica vs {xlabel} (banda = IC95%)", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_allele_freq(maf_real: np.ndarray, maf_synth: np.ndarray, *,
                     title: str | None = None, color: str = "tab:blue"):
    """Scatter de frecuencia alélica por SNP real vs sintética (calidad de generación).

    Cada punto es un SNP: ``x`` = frecuencia del alelo 1 en los datos reales, ``y`` = en
    los sintéticos. La diagonal (rojo) es el ideal: cuanto más pegada, mejor reproduce la
    generación la estructura marginal de los marcadores. El título reporta la correlación.
    """
    maf_real = np.asarray(maf_real, dtype=float).ravel()
    maf_synth = np.asarray(maf_synth, dtype=float).ravel()
    finite = np.isfinite(maf_real) & np.isfinite(maf_synth)
    corr = float(np.corrcoef(maf_real[finite], maf_synth[finite])[0, 1]) if finite.sum() > 1 else float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(maf_real[finite], maf_synth[finite], s=6, alpha=0.25, color=color)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("frecuencia alélica real")
    ax.set_ylabel("frecuencia alélica sintética")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"MAF real vs sintética (r = {corr:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dist_overlay(real: np.ndarray, synth: np.ndarray, *, label: str = "valor",
                      bins: int = 40, title: str | None = None):
    """Histogramas superpuestos (densidad) de una magnitud real vs sintética.

    Para comparar, p.ej., la distribución de GY real contra la de los labels sintéticos,
    o la accuracy de reconstrucción. Si la sintética cubre el soporte de la real sin
    concentrarse en un modo, la generación es razonable.
    """
    real = np.asarray(real, dtype=float).ravel()
    synth = np.asarray(synth, dtype=float).ravel()
    real = real[np.isfinite(real)]
    synth = synth[np.isfinite(synth)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(real, bins=bins, density=True, alpha=0.5, label="real", color="tab:blue")
    ax.hist(synth, bins=bins, density=True, alpha=0.5, label="sintético", color="tab:orange")
    ax.set_xlabel(label)
    ax.set_ylabel("densidad")
    ax.set_title(title or f"Distribución de {label}: real vs sintético")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
