"""Data augmentation con el C-β-VAE: generación de pares (X, y) sintéticos + experimento.

Centraliza la lógica del augmentation para que el notebook solo importe y llame (la
convención del proyecto). Dos estrategias de generación (ambas vía la rama generativa
del `CBetaVAERegressor` ya entrenado SOLO con el train — regla 1):

- **posterior-jitter** (`generate_posterior`): encodea líneas reales del train, perturba
  el latente y decodea SNPs cercanos al manifold real, **conservando el label real**.
  Augmentation defendible: interpola alrededor de puntos etiquetados reales.
- **prior-sampling** (`generate_prior`): muestrea ``z~N(0,I)`` y decodea genotipos
  novedosos; el label sale de la cabeza de regresión (auto-generado).

El experimento `evaluate_augmentation` reentrena un modelo downstream **configurable**
(cualquiera con interfaz `fit`/`predict`) con train + sintéticos y lo compara contra el
mismo modelo entrenado solo con el train real, midiendo sobre el val **real** (intocado).
Las filas sintéticas reciben `line_id` centinela que nunca aparece en val.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from src.eval.metrics import (
    bootstrap_ci,
    coincidence_index,
    pearson_r,
    r2,
    regression_metrics,
    rmse,
)
from src.utils.config import load_config, set_nested

# Métricas reportadas (mismas que el resto del proyecto; matching20 = top-20%).
_METRICS: dict[str, Callable] = {
    "pearson_r": pearson_r,
    "rmse": rmse,
    "r2": r2,
    "matching20": coincidence_index,
}


# --- Generación ------------------------------------------------------------


def synthetic_pairs(
    model: Any,
    *,
    strategy: str,
    n: int,
    X_ref: np.ndarray | None = None,
    y_ref: np.ndarray | None = None,
    jitter: float = 1.0,
    temperature: float = 1.0,
    binarize: str = "bernoulli",
    seed: int = 575,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera ``n`` pares (X_synth, y_synth) con la estrategia pedida.

    - ``strategy="prior"``: ``model.generate_prior(n, temperature, binarize)``.
    - ``strategy="posterior"``: remuestrea ``n`` líneas reales (con reemplazo, seed fija)
      de ``X_ref``/``y_ref`` y las jitterea en el latente (``model.generate_posterior``,
      conservando el label real).
    """
    strategy = str(strategy).lower()
    if strategy == "prior":
        return model.generate_prior(n, temperature=temperature, binarize=binarize, seed=seed)
    if strategy == "posterior":
        if X_ref is None or y_ref is None:
            raise ValueError("strategy='posterior' necesita X_ref e y_ref (el train real).")
        X_ref = np.asarray(X_ref, dtype=np.float32)
        y_ref = np.asarray(y_ref, dtype=float).ravel()
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, X_ref.shape[0], int(n))  # remuestreo con reemplazo
        return model.generate_posterior(
            X_ref[idx], y_ref[idx], jitter=jitter, binarize=binarize, seed=seed
        )
    raise ValueError(f"strategy desconocida: {strategy!r}. Use 'posterior' o 'prior'.")


def build_augmented_train(
    X_real: np.ndarray,
    y_real: np.ndarray,
    line_id_real: np.ndarray,
    year_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatena train real + sintético. Las filas sintéticas reciben:

    - ``line_id`` centinela único (``"__synth_i"``) que **nunca** aparece en el val real
      (respeta la regla 2: ninguna línea cruza train/val).
    - ``year`` = último año del train real (para que un `temporal_inner_split` downstream
      siga teniendo sentido si el modelo usa early stopping).
    """
    n_syn = int(X_synth.shape[0])
    syn_line = np.array([f"__synth_{i}" for i in range(n_syn)], dtype=object)
    last_year = int(np.max(np.asarray(year_real, dtype=int)))
    syn_year = np.full(n_syn, last_year, dtype=int)
    X_aug = np.concatenate([np.asarray(X_real), np.asarray(X_synth)], axis=0)
    y_aug = np.concatenate([np.asarray(y_real, dtype=float).ravel(), np.asarray(y_synth, dtype=float).ravel()])
    line_aug = np.concatenate([np.asarray(line_id_real, dtype=object), syn_line])
    year_aug = np.concatenate([np.asarray(year_real, dtype=int), syn_year])
    return X_aug, y_aug, line_aug, year_aug


# --- Calidad de generación (para la celda de inspección) -------------------


def generation_report(
    model: Any,
    X_real: np.ndarray,
    *,
    strategy: str = "prior",
    n: int | None = None,
    y_real: np.ndarray | None = None,
    seed: int = 575,
    **gen_kwargs: Any,
) -> dict[str, Any]:
    """Métricas/arrays para inspeccionar cómo son los datos generados.

    Devuelve un dict con:
    - ``recon_acc``: accuracy de reconstrucción (bit a bit) sobre ``X_real``.
    - ``maf_real`` / ``maf_synth``: frecuencia del alelo 1 por SNP (real vs sintética).
    - ``maf_corr``: correlación entre ambas (1.0 = la generación respeta las frecuencias).
    - ``y_real`` / ``y_synth``: distribuciones del target.
    - ``latent_mu_std``: desvío de ``mu`` por dimensión latente (≈0 ⇒ dimensión colapsada).
    - ``active_dims``: # de dimensiones latentes activas (std > 0.1).
    """
    X_real = np.asarray(X_real, dtype=np.float32)
    n = int(n) if n is not None else X_real.shape[0]

    # Reconstrucción (probabilidades) y accuracy bit a bit.
    probs = model.reconstruct(X_real)
    recon_acc = float(((probs > 0.5).astype(np.float32) == X_real).mean())

    X_synth, y_synth = synthetic_pairs(
        model, strategy=strategy, n=n, X_ref=X_real, y_ref=y_real, seed=seed, **gen_kwargs
    )
    maf_real = X_real.mean(axis=0)
    maf_synth = np.asarray(X_synth, dtype=np.float32).mean(axis=0)
    finite = np.isfinite(maf_real) & np.isfinite(maf_synth)
    maf_corr = float(np.corrcoef(maf_real[finite], maf_synth[finite])[0, 1]) if finite.sum() > 1 else float("nan")

    mu = model.encode(X_real)
    latent_mu_std = mu.std(axis=0)

    return {
        "strategy": strategy,
        "n_synth": int(X_synth.shape[0]),
        "recon_acc": recon_acc,
        "maf_real": maf_real,
        "maf_synth": maf_synth,
        "maf_corr": maf_corr,
        "y_real": (np.asarray(y_real, dtype=float).ravel() if y_real is not None else None),
        "y_synth": np.asarray(y_synth, dtype=float).ravel(),
        "latent_mu_std": latent_mu_std,
        "active_dims": int((latent_mu_std > 0.1).sum()),
        "X_synth": np.asarray(X_synth, dtype=np.float32),
    }


# --- Downstream desde un sweep ---------------------------------------------


def downstream_from_sweep(
    base_config: str | dict[str, Any],
    *,
    sweep_id: str | None = None,
    rank: int = 1,
    metric: str | None = None,
    goal: str | None = None,
    overrides: dict[str, Any] | None = None,
    build_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[], Any]:
    """Arma un ``downstream_builder`` con los HP del mejor modelo de un sweep.

    Para `evaluate_augmentation`: busca el run en posición ``rank`` del sweep (1 = mejor),
    le pega esos HP al ``base_config`` y devuelve un callable sin args que instancia un
    modelo **fresco** (re-leyendo el config en cada llamada, así el baseline y cada
    variante entrenan desde cero). El sweep se lee UNA vez acá; si no se pasa ``sweep_id``
    (o falla la lectura) se usan los HP base del config (con aviso).

    Args:
        base_config: ruta o dict del config del modelo downstream (debe matchear el
            ``model.name`` del sweep). Su bloque ``cv``/``data`` es irrelevante:
            `evaluate_augmentation` entrena el downstream sobre los arrays aumentados
            que le pasa, no por el cv del config.
        sweep_id: ``"entity/project/sweep_id"``. None ⇒ HP base.
        rank: 1 = mejor del sweep, 2 = segundo, etc. (reusa `top_runs_from_sweep`).
        metric / goal: métrica/objetivo del sweep (default: los del config del sweep).
        overrides: HP extra (notación punto) aplicados DESPUÉS de los del sweep.
        build_fn: cómo instanciar el modelo (default `build_model` del registry).

    Returns:
        ``() -> modelo`` listo para pasar como ``downstream_builder``.
    """
    sweep_hp: dict[str, Any] = {}
    if sweep_id:
        try:
            from src.training.sweep import top_runs_from_sweep

            top = top_runs_from_sweep(sweep_id, k=int(rank), metric=metric, goal=goal)
            if len(top) < rank:
                print(f"[augment][warn] sweep {sweep_id}: solo {len(top)} runs (<{rank}); uso el #{len(top)}")
            sweep_hp = top[min(int(rank), len(top)) - 1][0]
            print(f"[augment] downstream con HP del sweep {sweep_id} (rank {rank}, {len(sweep_hp)} HP)")
        except Exception as e:  # sin wandb / sweep inexistente / sin métrica
            print(f"[augment][warn] no pude leer el sweep {sweep_id} ({e}); uso HP base del config")
    merged = {**sweep_hp, **(overrides or {})}

    if build_fn is None:
        from src.models.registry import build_model

        build_fn = build_model

    def _builder() -> Any:
        cfg = load_config(base_config) if isinstance(base_config, str) else copy.deepcopy(base_config)
        for dotted, value in merged.items():
            set_nested(cfg, dotted, value)
        return build_fn(cfg)

    return _builder


# --- Experimento de augmentation -------------------------------------------


def _eval_variant(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    line_id_va: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Calcula métricas puntuales + IC bootstrap (cluster por línea) de una variante."""
    m = regression_metrics(y_true, y_pred)
    m["matching20"] = coincidence_index(y_true, y_pred, 0.2)
    ci = bootstrap_ci(y_true, y_pred, _METRICS, group=line_id_va, n_boot=n_boot, alpha=alpha, seed=seed)
    rows, samples = [], {}
    for metric in _METRICS:
        rows.append({
            "variant": name,
            "model": name,  # alias 'model' para reusar plot_ci/plot_box (esperan esa columna)
            "metric": metric,
            "point": float(m.get(metric, np.nan)),
            "lo": float(ci[metric]["lo"]),
            "hi": float(ci[metric]["hi"]),
            "n": int(np.asarray(y_true).size),
        })
        samples[metric] = np.asarray(ci[metric]["samples"], dtype=float)
    return rows, samples


def evaluate_augmentation(
    *,
    vae_model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    line_id_tr: np.ndarray,
    year_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    line_id_va: np.ndarray,
    downstream_builder: Callable[[], Any],
    strategies: tuple[str, ...] = ("posterior", "prior"),
    n_synth: int | None = None,
    jitter: float = 1.0,
    temperature: float = 1.0,
    binarize: str = "bernoulli",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 575,
) -> dict[str, Any]:
    """Compara un modelo downstream entrenado con train real vs train + sintéticos.

    Args:
        vae_model: `CBetaVAERegressor` YA entrenado solo con el train real (regla 1).
        X_tr/y_tr/line_id_tr/year_tr: train real (2017+2018).
        X_va/y_va/line_id_va: val real (2019), intocado por la generación.
        downstream_builder: callable sin args que devuelve un modelo FRESCO con interfaz
            `fit`/`predict` (p.ej. ``lambda: build_model(load_config('configs/exp_mlp...'))``).
        strategies: estrategias de generación a probar (cada una = una variante).
        n_synth: # de sintéticos a generar (default = tamaño del train real).

    Devuelve ``{"rows": [...], "samples_by_variant": {...}, "n_synth": int}``: filas tidy
    ``{variant, model, metric, point, lo, hi, n}`` (listas para `plot_ci`/`plot_box`) y las
    muestras bootstrap por variante (para el test pareado vs la referencia).
    """
    n_synth = int(n_synth) if n_synth else int(np.asarray(X_tr).shape[0])
    val_lines = set(np.unique(np.asarray(line_id_va, dtype=object)).tolist())
    rows: list[dict[str, Any]] = []
    samples_by_variant: dict[str, dict[str, np.ndarray]] = {}

    # --- Referencia: downstream con SOLO el train real ----------------------
    base = downstream_builder()
    base.set_fit_context(year=np.asarray(year_tr, dtype=int), line_id=np.asarray(line_id_tr, dtype=object))
    base.fit(np.asarray(X_tr), np.asarray(y_tr, dtype=float).ravel())
    ref_name = "original (solo real)"
    r, s = _eval_variant(ref_name, y_va, base.predict(np.asarray(X_va)), line_id_va, n_boot=n_boot, alpha=alpha, seed=seed)
    rows += r
    samples_by_variant[ref_name] = s

    # --- Cada estrategia: downstream con train real + sintéticos -------------
    for strat in strategies:
        X_syn, y_syn = synthetic_pairs(
            vae_model, strategy=strat, n=n_synth, X_ref=X_tr, y_ref=y_tr,
            jitter=jitter, temperature=temperature, binarize=binarize, seed=seed,
        )
        X_aug, y_aug, line_aug, year_aug = build_augmented_train(X_tr, y_tr, line_id_tr, year_tr, X_syn, y_syn)
        # Guard anti-leakage (regla 2): ninguna línea sintética puede estar en el val real.
        syn_only = set(np.unique(line_aug).tolist()) - set(np.unique(np.asarray(line_id_tr, dtype=object)).tolist())
        if syn_only & val_lines:
            raise AssertionError("Una línea sintética colisiona con el val real (leakage).")

        dmodel = downstream_builder()
        dmodel.set_fit_context(year=year_aug, line_id=line_aug)
        dmodel.fit(X_aug, y_aug)
        name = f"+sintetico ({strat}, n={n_synth})"
        r, s = _eval_variant(name, y_va, dmodel.predict(np.asarray(X_va)), line_id_va, n_boot=n_boot, alpha=alpha, seed=seed)
        rows += r
        samples_by_variant[name] = s

    return {"rows": rows, "samples_by_variant": samples_by_variant, "n_synth": n_synth}
