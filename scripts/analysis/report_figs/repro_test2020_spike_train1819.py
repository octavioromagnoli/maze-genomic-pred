"""Cierre 2020 con spike-in, TRAIN = 2 años más próximos (2018+2019) — para el informe.

Reproduce el rescate cold-start con la ventana de train que pidió el usuario (2018+2019, no todo el
dev), para los 6 modelos del informe (GBLUP + Red multitarea/Ensemble + MoE hard/soft/GxE/PLS), en
AMBOS ambientes (GY_drought y GY_optimal), scoreando las dos cabezas de un solo fit para los MoE.

spike ∈ {0, 5%, 10%}, reps {1,42,2024}. Guarda CSV incremental. Reusa src.eval.spike/noise (mismos
configs del benchmark val 2019; spike_closure/este loop sobreescriben train_years).
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.eval.metrics import pearson_r
from src.eval.noise import _build_arrays, _fit, _predict_heads, _pearson_head
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed

TRAIN_YEARS = (2018, 2019)
TEST_YEAR = 2020
SPIKES = (0.0, 0.05, 0.10)
REPS = (1, 42, 2024)
WANT = ("GY_drought", "GY_optimal")
OUT = "experiments/figs/report/test2020_spike_2018_2019.csv"
os.makedirs("experiments/figs/report", exist_ok=True)

MULTI = {
    "MoE hard": "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft": "configs/exp_moe_soft_multienv_confirm_2019.yaml",
    "Ensemble": "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE":  "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS":  "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
}
GBLUP = {  # single-task: un config por ambiente
    "GY_drought": "configs/exp_gblup_2017_2018_to_2019_drought.yaml",
    "GY_optimal": "configs/exp_gblup_2017_2018_to_2019.yaml",
}

all_rows: list[dict] = []


def _save():
    pd.DataFrame(all_rows).to_csv(OUT, index=False)


def spike_multi(label, cfg_path):
    """Un fit por (spike, rep); scorea todas las cabezas de WANT que el modelo expone."""
    set_seed(575)
    cfg = load_config(cfg_path)
    ds = load_dataset(cfg["data"]["geno_csv"], cfg["data"]["pheno_csv"])
    arrays, heads, multitask = _build_arrays(cfg, ds)
    year = np.asarray(arrays.year, dtype=int)
    X = np.asarray(arrays.X, dtype=float)
    dev = np.isin(year, list(TRAIN_YEARS))
    test = year == TEST_YEAR
    test_idx = np.flatnonzero(test)
    want = [h for h in WANT if h in heads]
    print(f"\n=== {label} === heads={heads} n_dev={int(dev.sum())} n_test={test_idx.size}", flush=True)
    for s in SPIKES:
        reps_use = (REPS[0],) if s <= 0 else REPS
        n_spike = int(round(float(s) * test_idx.size))
        for rep in reps_use:
            rng = np.random.default_rng(int(rep))
            spiked = rng.permutation(test_idx)[:n_spike]
            sm = np.zeros(year.size, dtype=bool)
            sm[spiked] = True
            held = test & ~sm
            tr = dev | sm
            model = build_model(cfg)
            _fit(model, X[tr], arrays, tr, heads, multitask)
            preds = _predict_heads(model, X[held], heads, multitask, want)
            for h, yp in preds.items():
                p, n = _pearson_head(arrays, held, h, heads, multitask, yp)
                all_rows.append({"model": label, "head": h, "spike": float(s), "rep": int(rep),
                                 "pearson": p, "n_held": n, "n_spike": int(n_spike)})
                print(f"  {label:10s} {h:11s} spike={s:.2f} rep{rep:<5d} r={p:+.3f} (held={n})", flush=True)
            _save()


def spike_gblup(head, cfg_path):
    """GBLUP single-task: un fit por (spike, rep) para la cabeza `head`."""
    set_seed(575)
    cfg = load_config(cfg_path)
    ds = load_dataset(cfg["data"]["geno_csv"], cfg["data"]["pheno_csv"])
    arrays, heads, multitask = _build_arrays(cfg, ds)   # multitask=False
    year = np.asarray(arrays.year, dtype=int)
    X = np.asarray(arrays.X, dtype=float)
    dev = np.isin(year, list(TRAIN_YEARS))
    test = year == TEST_YEAR
    test_idx = np.flatnonzero(test)
    print(f"\n=== GBLUP [{head}] === n_dev={int(dev.sum())} n_test={test_idx.size}", flush=True)
    for s in SPIKES:
        reps_use = (REPS[0],) if s <= 0 else REPS
        n_spike = int(round(float(s) * test_idx.size))
        for rep in reps_use:
            rng = np.random.default_rng(int(rep))
            spiked = rng.permutation(test_idx)[:n_spike]
            sm = np.zeros(year.size, dtype=bool)
            sm[spiked] = True
            held = test & ~sm
            tr = dev | sm
            model = build_model(cfg)
            _fit(model, X[tr], arrays, tr, heads, multitask)
            yp = _predict_heads(model, X[held], heads, multitask, [heads[0]])[heads[0]]
            yt = np.asarray(arrays.y)[held]
            p = float(pearson_r(yt, yp))
            all_rows.append({"model": "GBLUP", "head": head, "spike": float(s), "rep": int(rep),
                             "pearson": p, "n_held": int(np.size(yt)), "n_spike": int(n_spike)})
            print(f"  GBLUP      {head:11s} spike={s:.2f} rep{rep:<5d} r={p:+.3f} (held={np.size(yt)})", flush=True)
            _save()


# GBLUP primero (rápido → parcial temprano), después los MoE pesados
for head, cfg in GBLUP.items():
    spike_gblup(head, cfg)
for label, cfg in MULTI.items():
    spike_multi(label, cfg)

_save()
print(f"\nGUARDADO {OUT}", flush=True)
piv = pd.DataFrame(all_rows).groupby(["head", "model", "spike"])["pearson"].mean().round(3)
print(piv.to_string(), flush=True)
print("SPIKE_2018_2019_DONE", flush=True)
