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

from src.training.run import run_evaluation
from src.utils.config import load_config
from src.utils.seed import set_seed


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
    cols = ["fold", "val_year", "n_train", "n_val", "pearson_r", "rmse", "r2"]
    header = "  " + "  ".join(f"{c:>10}" for c in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in per_fold:
        row = []
        for c in cols:
            v = f.get(c, "")
            row.append(f"{v:>10.4f}" if isinstance(v, float) else f"{str(v):>10}")
        print("  " + "  ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento + CV con logging a wandb.")
    parser.add_argument("--config", required=True, help="Ruta al YAML del experimento.")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    wandb_cfg = config.get("wandb", {})
    mode = wandb_cfg.get("mode", "offline")
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

    print(f"[train] experimento={config.get('experiment', {}).get('name')} seed={seed} wandb_mode={mode}")
    results = run_evaluation(config, log_fn=make_wandb_logger(run))

    # Resumen a stdout + wandb.summary.
    print(
        f"[train] target={results['target']} env={results['env']} "
        f"n_obs={results['n_obs']} n_lines={results['n_lines']} "
        f"(test final {results['holdout_test_year']} excluido del dev)"
    )
    print_fold_table(results["per_fold"])
    print("[train] resumen (val):")
    for k, v in results["summary"].items():
        print(f"    {k}: {v}")
        run.summary[k] = v

    # Tabla de folds como artefacto visual de la corrida.
    if results["per_fold"]:
        table = wandb.Table(columns=list(results["per_fold"][0].keys()))
        for f in results["per_fold"]:
            table.add_data(*f.values())
        run.log({"folds": table})

    run.finish()
    print("[train] corrida finalizada. Si mode=offline, sincronizá con: wandb sync")


if __name__ == "__main__":
    main()
