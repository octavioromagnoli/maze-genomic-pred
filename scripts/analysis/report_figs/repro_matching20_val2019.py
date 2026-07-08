"""Reproduce pearson_r + matching20 (val 2019) de los 6 modelos del informe, ambos ambientes.
Lee del per_fold cacheado (matching20 se computa siempre) => no re-fittea.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from src.eval.nb_eval import make_evaluators, make_multitask_evaluator

MULTI = {
    "MoE hard":       "configs/exp_moe_hard_6head_2017_2018_to_2019.yaml",
    "MoE soft":       "configs/exp_moe_soft_multienv_confirm_2019.yaml",
    "Red multitarea": "configs/exp_ensemble_6head_GY_2017_2018_to_2019.yaml",
    "MoE GxE":        "configs/exp_moe_gxe_6head_2017_2018_to_2019.yaml",
    "MoE PLS":        "configs/exp_moe_pls_6head_2017_2018_to_2019.yaml",
}
GBLUP = {"drought": "configs/exp_gblup_2017_2018_to_2019_drought.yaml",
         "optimal": "configs/exp_gblup_2017_2018_to_2019.yaml"}

evaluate_model, _, _, _ = make_evaluators(("pearson_r", "matching20"), n_boot=1000, alpha=0.05)
eval_multitask = make_multitask_evaluator(("pearson_r", "matching20"), n_boot=1000, alpha=0.05)


def mt(res, head):
    r = [f for f in res["per_fold"] if f.get("trait") == head and f.get("role") == "scoring"][-1]
    return float(r["pearson_r"]), float(r["matching20"])


rows = {}
for label, cfg in MULTI.items():
    res = eval_multitask(label, cfg)
    for env, head in (("drought", "GY_drought"), ("optimal", "GY_optimal")):
        rows[(label, env)] = mt(res, head)
for env, cfg in GBLUP.items():
    r, _samples = evaluate_model("GBLUP", cfg)
    d = {x["metric"]: x["point"] for x in r}
    rows[("GBLUP", env)] = (d["pearson_r"], d["matching20"])

print("\n=== Pearson / matching20 (val 2019) ===")
print(f"{'modelo':16s} {'pearson_dro':>11s} {'m20_dro':>8s} {'pearson_opt':>12s} {'m20_opt':>8s}")
for label in ["MoE hard", "MoE soft", "Red multitarea", "GBLUP", "MoE PLS", "MoE GxE"]:
    pd_, md = rows[(label, "drought")]
    po, mo = rows[(label, "optimal")]
    print(f"{label:16s} {pd_:11.3f} {md:8.3f} {po:12.3f} {mo:8.3f}")
