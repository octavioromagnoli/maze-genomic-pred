"""Chequeo decisivo: ¿la inestabilidad a la seed del MoE hard es por `gating.n_init=1`?

El GMM del gating se inicializa una sola vez (n_init=1) ⇒ una mala inicializacion segun la seed da una
mala particion ⇒ malos expertos. Aca corremos el MoE hard con `gating.n_init=20` (GMM casi determinista
por best-of-20) sobre las MISMAS 5 seeds y comparamos la dispersion de GY_drought contra n_init=1.

Si con n_init=20 se estabiliza ALTO (~0.40) ⇒ el gating SI ayuda pero necesita GMM bien inicializado (el
0.407 es alcanzable de forma robusta). Si se estabiliza BAJO (~0.29) ⇒ el gating no aporta sobre el
ensemble plano y el 0.407 era ruido.
"""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np
from src.eval.metrics import pearson_r
from src.eval.nb_eval import make_multitask_evaluator
from src.utils.config import load_config

OUT = Path(__file__).resolve().parent
REPORT_CV = {"scheme": "fixed_temporal", "val_year": 2019, "train_years": [2017, 2018],
             "holdout_test_year": 2020, "enforce_disjoint_lines": True}
BASE = "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml"
SEEDS = [575, 1, 42, 2024, 7]
HEADS = [f"{t}_{e}" for t in ["GY", "PH", "AD"] for e in ["optimal", "drought"]]


def reseed(cfg, s, n_init):
    cfg = copy.deepcopy(cfg)
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("seed", "random_state") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    o[k] = int(s)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(cfg)
    cfg["seed"] = int(s)
    cfg["cv"] = dict(REPORT_CV)
    cfg["artifacts"] = {"save_model": False}
    cfg["model"]["params"]["gating"]["n_init"] = int(n_init)   # <-- la unica palanca que cambia
    return cfg


def main():
    ev = make_multitask_evaluator(["pearson_r"], n_boot=1, alpha=0.05)
    base = load_config(BASE)
    rows = []
    for n_init in (1, 20):
        for s in SEEDS:
            res = ev(f"hard n_init{n_init} seed{s}", reseed(base, s, n_init))
            for head in HEADS:
                pr = [p for p in res["predictions"] if p["trait"] == head and p["role"] == "scoring"]
                if pr:
                    rows.append({"n_init": n_init, "seed": s, "pheno": head,
                                 "pearson_r": float(pearson_r(pr[-1]["y_true"], pr[-1]["y_pred"]))})
            gyd = next(r["pearson_r"] for r in rows if r["n_init"] == n_init and r["seed"] == s and r["pheno"] == "GY_drought")
            print(f"  n_init={n_init:2d} seed{s}: GY_drought={gyd:.4f}", flush=True)
    (OUT / "ninit_check.json").write_text(json.dumps(rows, indent=2, default=float))

    print("\n=== GY_drought: dispersion por n_init (5 seeds) ===")
    for n_init in (1, 20):
        v = np.array([r["pearson_r"] for r in rows if r["n_init"] == n_init and r["pheno"] == "GY_drought"])
        print(f"  n_init={n_init:2d}:  media={v.mean():.4f}  sd={v.std(ddof=1):.4f}  rango=[{v.min():.3f}, {v.max():.3f}]")


if __name__ == "__main__":
    main()
