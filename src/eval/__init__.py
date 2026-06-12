"""Evaluación centralizada: métricas y splits antileakage.

Ningún modelo reimplementa estas piezas. Las métricas viven en `metrics.py` y
los splitters (temporal + agrupado por línea) en `splits.py`.
"""
from src.eval.metrics import ( 
    pearson_r,
    r2,
    regression_metrics,
    rmse,
    summarize_folds,
)
from src.eval.splits import (  
    LeakageError,
    assert_no_line_leakage,
    final_test_split,
    grouped_by_line_folds,
    temporal_forward_chaining_folds,
    temporal_grouped_folds,
)
