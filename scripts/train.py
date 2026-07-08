"""Único entrypoint de entrenamiento.

Uso:
    python scripts/train.py --config configs/exp_dummy.yaml

Levanta el config, fija la seed, inicializa wandb (offline por defecto, no
requiere login), corre el split temporal definido en el config (por defecto un
único train/val fijo) y loguea las métricas + resumen. El año de test final
(2020) queda fuera del dev.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Permite `import src...` corriendo desde cualquier cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.training.save_model import save_model_artifact
from src.training.run import run_evaluation
from src.training.run_multitask import run_multitask_evaluation
from src.utils.config import load_config, set_nested
from src.utils.seed import set_seed


def apply_sweep_overrides(config: dict, wandb_config: dict) -> dict:
    """Aplica los HP barridos por un sweep sobre el config anidado.

    wandb deja los parámetros del trial como claves planas con punto
    (``"preprocess.pca.n_components"``) en ``run.config``; las llevamos al dict
    anidado que consume `run_evaluation`. En una corrida normal no hay claves con
    punto, así que esto es no-op. Devuelve los overrides aplicados (para log).
    """
    overrides = {k: v for k, v in wandb_config.items() if "." in k}
    for dotted, value in overrides.items():
        set_nested(config, dotted, value)
    return overrides


def make_wandb_logger(run):
    """Callback que loguea solo escalares numéricos a wandb (evita strings)."""

    def _log(d: dict) -> None:
        numeric = {
            k: v for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric:
            run.log(numeric)

    return _log


def print_fold_table(per_fold: list[dict]) -> None:
    if not per_fold:
        print("  (sin folds)")
        return
    cols = ["fold", "role", "val_year", "n_train", "n_val", "pearson_r", "rmse", "r2"]
    header = "  " + "  ".join(f"{c:>10}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in per_fold:
        row = []
        for c in cols:
            v = f.get(c, "")
            row.append(f"{v:>10.4f}" if isinstance(v, float) else f"{str(v):>10}")
        print("  " + "  ".join(row))


def print_ci(per_fold: list[dict]) -> None:
    """Muestra los IC por bootstrap si las métricas los traen (claves _lo/_hi)."""
    metrics = ("pearson_r", "rmse", "r2")
    rows = [f for f in per_fold if any(f"{m}_lo" in f for m in metrics)]
    if not rows:
        return
    print("[train] IC por bootstrap:")
    for f in rows:
        print(f"  fold {f.get('fold')} (val {f.get('val_year')}):")
        for m in metrics:
            if f"{m}_lo" in f:
                point = f.get(m, float("nan"))
                print(f"    {m:>10}: {point:.4f}  IC [{f[f'{m}_lo']:.4f}, {f[f'{m}_hi']:.4f}]")


def print_multitask_fold_table(per_fold: list[dict]) -> None:
    """Tabla de folds multitask: una fila por (fold, trait)."""
    if not per_fold:
        print("  (sin folds)")
        return
    cols = ["fold", "trait", "role", "val_year", "n_train", "n_val", "pearson_r", "rmse", "r2", "matching20"]
    header = "  " + "  ".join(f"{c:>10}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in per_fold:
        row = []
        for c in cols:
            v = f.get(c, "")
            row.append(f"{v:>10.4f}" if isinstance(v, float) else f"{str(v):>10}")
        print("  " + "  ".join(row))


def log_multitask_results(run, results: dict, config: dict, seed: int, save_model: bool) -> None:
    """Logging trait-aware del flujo multitask (aislado de la ruta single-task)."""
    import wandb

    targets = results["targets"]
    per_fold = results["per_fold"]
    print(
        f"[train] MULTITASK targets={targets} env={results['env']} "
        f"n_obs={results['n_obs']} n_lines={results['n_lines']} "
        f"(test final {results['holdout_test_year']} excluido del dev)"
    )
    print_multitask_fold_table(per_fold)

    # Resumen por trait + headline GY -> run.summary (pearson_r_GY lo lee el sweep).
    print("[train] resumen (val) por trait:")
    for k, v in results["summary"].items():
        print(f"    {k}: {v}")
        run.summary[k] = v

    # IC con nombres limpios por trait (split fijo => un fold por trait).
    n_folds = len({f.get("fold") for f in per_fold})
    if n_folds == 1:
        for f in per_fold:
            trait = f.get("trait")
            for met in ("pearson_r", "rmse", "r2", "matching20"):
                if f"{met}_lo" in f:
                    run.summary[f"{trait}/{met}_lo"] = f[f"{met}_lo"]
                    run.summary[f"{trait}/{met}_hi"] = f[f"{met}_hi"]

    # Tabla de folds como artefacto visual (unión de claves por si algún trait no
    # trae IC, p.ej. <2 válidos).
    if per_fold:
        cols = list(dict.fromkeys(k for f in per_fold for k in f))
        table = wandb.Table(columns=cols)
        for f in per_fold:
            table.add_data(*[f.get(c) for c in cols])
        run.log({"folds": table})

    # Histogramas de bootstrap por (trait, métrica).
    for entry in results.get("bootstrap", []):
        prefix = f"boot/{entry.get('trait')}"
        hist = {}
        for name, samples in entry.items():
            if name in ("fold", "trait"):
                continue
            s = np.asarray(samples, dtype=float)
            s = s[np.isfinite(s)]
            if s.size:
                hist[f"{prefix}/{name}"] = wandb.Histogram(s)
        if hist:
            run.log(hist)

    # Learning curves por época (train_loss + val_pearson_<trait>).
    curves = results.get("histories", [])
    for entry in curves:
        hist_rows = entry.get("history") or []
        if not hist_rows:
            continue
        prefix = "curve" if len(curves) == 1 else f"curve/fold{entry['fold']}"
        step_metric = f"{prefix}/epoch"
        run.define_metric(step_metric)
        run.define_metric(f"{prefix}/*", step_metric=step_metric)
        for rec in hist_rows:
            payload = {
                f"{prefix}/{k}": v for k, v in rec.items()
                if k != "epoch" and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if payload:
                payload[step_metric] = int(rec.get("epoch", 0))
                run.log(payload)

    # Modelo entrenado como Artifact (si el config lo pide).
    if save_model and results.get("models"):
        fold0 = results["models"][0]
        val_metrics = {
            f"{f.get('trait')}/pearson_r": f["pearson_r"]
            for f in per_fold
            if f.get("fold") == fold0["fold"] and "pearson_r" in f
        }
        fitted_params = fold0.get("fitted_params") or {}
        if fitted_params:
            print(f"[train] estimaciones del modelo: {fitted_params}")
            for k, v in fitted_params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    run.summary[k] = v
        art_cfg = config.get("artifacts", {})
        art_name = art_cfg.get("name") or (config.get("experiment", {}).get("name") or "model").replace("_", "-")
        art_dir = save_model_artifact(
            fold0["model"],
            name=art_name,
            model_name=config.get("model", {}).get("name"),
            train_years=fold0["train_years"],
            target="+".join(targets),
            env=results["env"],
            seed=seed,
            val_metrics=val_metrics,
            fitted_params=fitted_params,
            run=run,
            aliases=tuple(art_cfg.get("aliases", ["latest"])),
        )
        print(f"[train] modelo guardado en {art_dir} (+ wandb Artifact '{art_name}')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento + CV con logging a wandb.")
    parser.add_argument("--config", required=True, help="Ruta al YAML del experimento.")
    args = parser.parse_args()

    config = load_config(args.config)

    wandb_cfg = config.get("wandb", {})
    # WANDB_MODE del entorno tiene precedencia sobre el config: permite correr el
    # agente de un sweep en online (WANDB_MODE=online wandb agent ...) sin editar
    # el YAML base, que puede quedar en offline para las corridas normales.
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "offline")
    os.environ.setdefault("WANDB_MODE", mode)

    import wandb  # import tardío: el harness no depende de wandb para testear run_evaluation

    run = wandb.init(
        project=wandb_cfg.get("project", "maze-genomic-pred"),
        entity=wandb_cfg.get("entity"),
        name=config.get("experiment", {}).get("name"),
        tags=wandb_cfg.get("tags"),
        mode=mode,
        config=config,
    )

    # Si esto corre bajo un sweep, wandb ya inyectó los HP del trial en run.config
    # como claves planas con punto: las aplicamos al config anidado ANTES de fijar
    # la seed y de modelar. En una corrida normal no hay overrides y es no-op.
    overrides = apply_sweep_overrides(config, dict(run.config))
    if overrides:
        print(f"[train] sweep overrides aplicados: {overrides}")

    seed = int(config.get("seed", 575))
    set_seed(seed)

    art_cfg = config.get("artifacts", {})
    save_model = bool(art_cfg.get("save_model", False))

    print(f"[train] experimento={config.get('experiment', {}).get('name')} seed={seed} wandb_mode={mode}")

    # Dispatch multitask: una red (o un ensemble de redes), varias cabezas (métricas por
    # trait). La ruta single-task de abajo queda intacta. El logging multitask vive en su
    # propio helper. `ensemble_multitask` envuelve `mlp_multitask` y usa el mismo pipeline.
    if config.get("model", {}).get("name") in (
        "mlp_multitask", "ensemble_multitask", "moe_multitask",
        "moe_gblup_residual_multitask", "moe_pls_multitask",
        "moe_gxe_multitask", "moe_soft_multitask",
    ):
        results = run_multitask_evaluation(
            config, log_fn=make_wandb_logger(run), return_models=save_model
        )
        log_multitask_results(run, results, config, seed, save_model)
        run.finish()
        print("[train] corrida finalizada. Si mode=offline, sincronizá con: wandb sync")
        return

    results = run_evaluation(config, log_fn=make_wandb_logger(run), return_models=save_model)

    # Resumen a stdout + wandb.summary.
    print(
        f"[train] target={results['target']} env={results['env']} "
        f"n_obs={results['n_obs']} n_lines={results['n_lines']} "
        f"(test final {results['holdout_test_year']} excluido del dev)"
    )
    print_fold_table(results["per_fold"])
    print_ci(results["per_fold"])
    print("[train] resumen (val):")
    for k, v in results["summary"].items():
        print(f"    {k}: {v}")
        run.summary[k] = v

    # IC con nombres limpios en run.summary (split fijo => un único fold).
    if len(results["per_fold"]) == 1:
        f0 = results["per_fold"][0]
        for m in ("pearson_r", "rmse", "r2", "matching20"):
            if f"{m}_lo" in f0:
                run.summary[f"{m}_lo"] = f0[f"{m}_lo"]
                run.summary[f"{m}_hi"] = f0[f"{m}_hi"]

    # Tabla de folds como artefacto visual de la corrida.
    if results["per_fold"]:
        table = wandb.Table(columns=list(results["per_fold"][0].keys()))
        for f in results["per_fold"]:
            table.add_data(*f.values())
        run.log({"folds": table})

    # Histogramas de la distribución bootstrap por métrica (si se calculó).
    boot = results.get("bootstrap", [])
    for entry in boot:
        prefix = "boot" if len(boot) == 1 else f"boot/fold{entry['fold']}"
        hist = {}
        for name, samples in entry.items():
            if name == "fold":
                continue
            s = np.asarray(samples, dtype=float)
            s = s[np.isfinite(s)]
            if s.size:
                hist[f"{prefix}/{name}"] = wandb.Histogram(s)
        if hist:
            run.log(hist)

    # Learning curves por época (si el modelo las expone, p.ej. el MLP guarda
    # history_). Se loguean con un eje x propio (define_metric) para que el panel use
    # la época y no choque con el step global del run. En el régimen del sweep (1 año
    # de train, sin inner-val) solo hay train_loss; con early stopping (>=2 años) van
    # además val_loss y val_pearson por época.
    curves = results.get("histories", [])
    for entry in curves:
        hist_rows = entry.get("history") or []
        if not hist_rows:
            continue
        prefix = "curve" if len(curves) == 1 else f"curve/fold{entry['fold']}"
        step_metric = f"{prefix}/epoch"
        run.define_metric(step_metric)
        run.define_metric(f"{prefix}/*", step_metric=step_metric)
        for rec in hist_rows:
            payload = {
                f"{prefix}/{k}": v for k, v in rec.items()
                if k != "epoch" and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if payload:
                payload[step_metric] = int(rec.get("epoch", 0))
                run.log(payload)

    # Modelo entrenado como Artifact (si el config lo pide). Con el split fijo se
    # guarda el del único fold de validación; train_years queda en metadata para el
    # chequeo anti-leakage al reusarlo (src/training/save_model.py).
    if save_model and results.get("models"):
        fold0 = results["models"][0]
        f0 = results["per_fold"][0]
        val_metrics = {k: f0[k] for k in ("pearson_r", "rmse", "r2") if k in f0}
        fitted_params = fold0.get("fitted_params") or {}
        if fitted_params:
            print(f"[train] estimaciones del modelo: {fitted_params}")
            # También a run.summary (escalares) para inspeccionar en wandb sin bajar el modelo.
            for k, v in fitted_params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    run.summary[k] = v
        art_name = art_cfg.get("name") or (config.get("experiment", {}).get("name") or "model").replace("_", "-")
        art_dir = save_model_artifact(
            fold0["model"],
            name=art_name,
            model_name=config.get("model", {}).get("name"),
            train_years=fold0["train_years"],
            target=results["target"],
            env=results["env"],
            seed=seed,
            val_metrics=val_metrics,
            fitted_params=fitted_params,
            run=run,
            aliases=tuple(art_cfg.get("aliases", ["latest"])),
        )
        print(f"[train] modelo guardado en {art_dir} (+ wandb Artifact '{art_name}')")

    run.finish()
    print("[train] corrida finalizada. Si mode=offline, sincronizá con: wandb sync")


if __name__ == "__main__":
    main()
