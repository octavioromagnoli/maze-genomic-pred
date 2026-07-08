from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.training.run import run_evaluation
from src.training.run_multitask import run_multitask_evaluation
from src.utils.config import load_config
from src.utils.seed import set_seed

REPO = pathlib.Path(__file__).resolve().parents[1]
PREDS_DIR = REPO / "preds"
PHENO_CSV = REPO / "data" / "Pheno_data.csv"
TRAIN_YEARS = (2018, 2019)

_MULTITASK_MODELS = {
    "mlp_multitask", "ensemble_multitask", "moe_multitask",
    "moe_gblup_residual_multitask", "moe_pls_multitask",
    "moe_gxe_multitask", "moe_soft_multitask",
}


MODELS: dict[str, dict] = {
    "GBLUP": {
        "optimal": "configs/exp_gblup_2018_2019_to_2020.yaml",
        "drought": "configs/exp_gblup_2018_2019_to_2020_drought.yaml",
    },
    "RedMultitarea": {
        "both": "configs/exp_ensemble_6head_GY_2018_2019_to_2020.yaml",
    },
    "MoE_hard": {
        "both": "configs/exp_moe_hard_6head_2018_2019_to_2020.yaml",
    },
    "MoE_soft": {
        "both": "configs/exp_moe_soft_multienv_2018_2019_to_2020.yaml",
    },
    "MoE_GxE": {
        "both": "configs/exp_moe_gxe_6head_2018_2019_to_2020.yaml",
    },
    "MoE_PLS": {
        "both": "configs/exp_moe_pls_6head_2018_2019_to_2020.yaml",
    },
}

VAL_YEAR = 2020


def _train_gy_stats() -> dict[str, tuple[float, float]]:
    """(media, desvío) de GY por ambiente sobre el TRAIN (2018+2019). Solo train => sin leakage.

    Sirve para reescalar a t/ha las predicciones de los modelos con `standardize_target`:
    el ensemble multitask combina a sus miembros en z-score (re-estandariza cada red con
    sus stats de train y promedia), así que su salida NO está en unidades de GY. La media/
    desvío de GY del train mapean esos z-scores de vuelta a t/ha con una transformación afín
    monótona (Pearson idéntico), consistente con la escala de GBLUP.
    """
    p = pd.read_csv(PHENO_CSV)
    tr = p[p["Year"].isin(TRAIN_YEARS)]
    return {
        "GY_Optimal": (float(tr["GY_Optimal"].mean()), float(tr["GY_Optimal"].std())),
        "GY_Drought": (float(tr["GY_Drought"].mean()), float(tr["GY_Drought"].std())),
    }


_GY_STATS = _train_gy_stats()


def _load(config_path: str) -> dict:
    cfg = load_config(str(REPO / config_path))
    cfg.setdefault("bootstrap", {})["enabled"] = False
    return cfg


def _scoring_fold(preds: list[dict], trait: str | None = None) -> dict:
    """Devuelve la entrada de predicción del fold de scoring de 2020.

    ``trait=None`` => ruta single-task (una entrada por fold). Con ``trait`` set
    filtra la cabeza correspondiente de la ruta multitask.
    """
    cands = [
        p for p in preds
        if p.get("role") == "scoring"
        and str(p.get("val_year")) == str(VAL_YEAR)
        and (trait is None or p.get("trait") == trait)
    ]
    if len(cands) != 1:
        raise RuntimeError(
            f"Esperaba 1 fold de scoring 2020 (trait={trait}), encontré {len(cands)}."
        )
    return cands[0]


def _pearson(p: dict) -> float:
    yt, yp = p["y_true"], p["y_pred"]
    if len(yt) < 2:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def _env_df(p: dict, col: str, *, rescale: bool = False) -> pd.DataFrame:
    """DataFrame GID + columna de predicción a partir de una entrada de predicción.

    ``rescale=True`` mapea la salida (z-score del ensemble multitask) a t/ha con la
    media/desvío de GY del train: ``gy = mean + z*std`` (afín ⇒ Pearson idéntico).
    """
    y = np.asarray(p["y_pred"], dtype=float)
    if rescale:
        mean, std = _GY_STATS[col]
        y = mean + y * std
    return pd.DataFrame({"GID": np.asarray(p["line_id"]).astype(str), col: y})


def predict_model(model_name: str, spec: dict[str, str]) -> pd.DataFrame:
    """Corre la(s) config(s) del modelo y devuelve GID + GY_Optimal + GY_Drought (2020)."""
    frames: dict[str, pd.DataFrame] = {}

    if set(spec) == {"optimal", "drought"}:  # single-task: una corrida por ambiente
        for env, cfg_path in spec.items():
            cfg = _load(cfg_path)
            set_seed(int(cfg.get("seed", 575)))
            res = run_evaluation(cfg)
            p = _scoring_fold(res["predictions"])
            col = "GY_Optimal" if env == "optimal" else "GY_Drought"
            frames[col] = _env_df(p, col)
            print(f"    [{model_name}] {env:>8}: n={len(p['y_pred'])}  pearson(2020)={_pearson(p):.4f}")
    else:  # multitask: una corrida, cabezas GY_optimal / GY_drought
        cfg = _load(spec["both"])
        set_seed(int(cfg.get("seed", 575)))
        res = run_multitask_evaluation(cfg)
        # El ensemble multitask combina a sus miembros en z-score ⇒ reescalar a t/ha (afín,
        # Pearson idéntico) para que el CSV quede en unidades de GY como GBLUP.
        for head, col in (("GY_optimal", "GY_Optimal"), ("GY_drought", "GY_Drought")):
            p = _scoring_fold(res["predictions"], trait=head)
            frames[col] = _env_df(p, col, rescale=True)
            print(f"    [{model_name}] {head:>10}: n={len(p['y_pred'])}  pearson(2020)={_pearson(p):.4f}")

    out = frames["GY_Optimal"].merge(frames["GY_Drought"], on="GID", how="outer")
    out = out.sort_values("GID").reset_index(drop=True)
    return out[["GID", "GY_Optimal", "GY_Drought"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los CSV de predicciones (2020, train 18+19).")
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS),
                        help="Subconjunto de modelos a correr (default: todos).")
    args = parser.parse_args()

    PREDS_DIR.mkdir(exist_ok=True)
    for model_name in args.models:
        print(f"[preds] === {model_name} ===")
        df = predict_model(model_name, MODELS[model_name])
        n_nan = int(df[["GY_Optimal", "GY_Drought"]].isna().sum().sum())
        out_path = PREDS_DIR / f"Cardelli_Romagnoli_{model_name}_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"[preds] {model_name}: {len(df)} filas, NaN={n_nan} -> {out_path.relative_to(REPO)}")

    print("[preds] listo.")


if __name__ == "__main__":
    main()
