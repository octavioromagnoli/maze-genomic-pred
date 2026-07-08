"""Evaluación centralizada: métricas y splits antileakage.

Ningún modelo reimplementa estas piezas. Las métricas viven en metrics.py y
el split temporal fijo + el guard antileakage en splits.py.
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
    fixed_temporal_split,
)
