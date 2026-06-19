"""Entrypoint del autoencoder de SNPs (compresor no supervisado).

Uso:
    python scripts/train_ae.py --config configs/exp_ae_2017_to_2018.yaml

Entrena el AE (BCE de reconstrucción, regla 4), loguea las *learning curves*
(train/val BCE y accuracy de bits por época) a wandb, deja la métrica del sweep
(``val_bce_best``) en ``run.summary`` y —si el config lo pide— guarda el AE
entrenado como Artifact (checkpoint local + wandb) para reusarlo en el downstream.

Es el gemelo no supervisado de `scripts/train.py`: este tunea el COMPRESOR (por
reconstrucción); `train.py` evalúa el PREDICTOR (GBLUP/XGBoost) sobre el embedding.
El sweep bayesiano (`configs/sweep_ae.yaml`) corre ESTE programa, un sweep por
dimensión latente.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Permite `import src...` corriendo desde cualquier cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.training.run_ae import run_ae_training
from src.training.save_model import save_ae_artifact
from src.utils.config import load_config, set_nested
from src.utils.seed import set_seed


def apply_sweep_overrides(config: dict, wandb_config: dict) -> dict:
    """Lleva los HP barridos por el sweep (claves planas con punto, p.ej.
    ``preprocess.autoencoder.lr``) al config anidado. No-op en corridas normales."""
    overrides = {k: v for k, v in wandb_config.items() if "." in k}
    for dotted, value in overrides.items():
        set_nested(config, dotted, value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del autoencoder de SNPs.")
    parser.add_argument("--config", required=True, help="Ruta al YAML del AE.")
    args = parser.parse_args()

    config = load_config(args.config)
    wandb_cfg = config.get("wandb", {})
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "offline")
    os.environ.setdefault("WANDB_MODE", mode)

    import wandb

    run = wandb.init(
        project=wandb_cfg.get("project", "maze-genomic-pred"),
        entity=wandb_cfg.get("entity"),
        name=config.get("experiment", {}).get("name"),
        tags=wandb_cfg.get("tags"),
        mode=mode,
        config=config,
    )

    overrides = apply_sweep_overrides(config, dict(run.config))
    if overrides:
        print(f"[train_ae] sweep overrides aplicados: {overrides}")

    seed = int(config.get("seed", 575))
    set_seed(seed)

    art_cfg = config.get("artifacts", {})
    save_model = bool(art_cfg.get("save_model", False))

    ae_cfg = (config.get("preprocess", {}) or {}).get("autoencoder", {})
    latent_dim = int(ae_cfg.get("latent_dim", 0))
    print(f"[train_ae] experimento={config.get('experiment', {}).get('name')} "
          f"latent_dim={latent_dim} seed={seed} wandb_mode={mode}")

    # Learning curves: una llamada por época, eje x = epoch. Además del valor
    # instantáneo, logueo el running-best de val BCE para que la poda temprana del
    # sweep (hyperband) tenga una métrica que seguir época a época.
    _running = {"best": float("inf")}

    def log_fn(rec: dict) -> None:
        epoch = int(rec.get("epoch", 0))
        vb = rec.get("ae/val_bce")
        if vb is not None:
            _running["best"] = min(_running["best"], float(vb))
        payload = {k: v for k, v in rec.items() if k != "epoch"}
        payload["val_bce_best"] = _running["best"]
        run.log(payload, step=epoch)

    results = run_ae_training(config, log_fn=log_fn, return_reducer=save_model)
    fitted = results["fitted"]

    print(f"[train_ae] target={results['target']} env={results['env']} "
          f"train_years={results['train_years']} val_scheme={results['val_scheme']} "
          f"val_year={results['val_year']} n_train={results['n_train']} n_val={results['n_val']}")
    print("[train_ae] reconstrucción:")
    for k, v in fitted.items():
        print(f"    {k}: {v}")

    # Métrica del sweep + diagnóstico en run.summary (nombres limpios, sin el
    # prefijo ae_ para que el controller del sweep lea `val_bce_best`).
    summary_map = {
        "val_bce_best": fitted.get("ae_val_bce_best"),
        "val_recon_acc_best": fitted.get("ae_val_recon_acc_best"),
        "epoch_best": fitted.get("ae_epoch_best"),
        "epochs_run": fitted.get("ae_epochs_run"),
        "n_params": fitted.get("ae_n_params"),
        "train_seconds": fitted.get("ae_train_seconds"),
        "latent_dim": fitted.get("ae_latent_dim"),
        "input_dim": fitted.get("ae_input_dim"),
    }
    for k, v in summary_map.items():
        if v is not None:
            run.summary[k] = v
    run.summary["stopped_by"] = fitted.get("ae_stopped_by")

    if save_model and results.get("reducer") is not None:
        art_name = art_cfg.get("name") or f"ae-snp-d{latent_dim}"
        base = save_ae_artifact(
            results["reducer"],
            name=art_name,
            train_years=results["train_years"],
            target=results["target"],
            env=results["env"],
            seed=seed,
            metrics=fitted,
            extra_meta={"val_scheme": results["val_scheme"], "val_year": results["val_year"],
                        "wandb_run_id": run.id},
            run=run,
            aliases=tuple(art_cfg.get("aliases", ["latest"])),
        )
        print(f"[train_ae] AE guardado en {base} (+ wandb Artifact '{art_name}')")

    run.finish()
    print("[train_ae] corrida finalizada. Si mode=offline, sincronizá con: wandb sync")


if __name__ == "__main__":
    main()
