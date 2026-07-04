"""Tests de validacion del solver SSI Python (`src/eval/ssi_solver.py`).

Validan el coordinate descent contra propiedades exactas del problema convexo. Corren SIN pytest:

    python tests/test_ssi_solver.py

(y tambien los descubre pytest si esta instalado). Los tres checks obligatorios:
  1. lambda=0 recupera el GBLUP no-sparse b = (K_TS + lambda0 I)^-1 K_0i (sistema lineal directo).
  2. al subir lambda, el support (nro de no-ceros) decrece monotono.
  3. la loss de la Eq. 3 no aumenta durante el coordinate descent.
  4. la prediccion usa fenotipos crudos del training: u_i = sum_j b_ij y_j.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.eval.ssi_solver import (  # noqa: E402
    gblup_weights,
    lambda_grid,
    predict_with_training_phenotypes,
    ssi_objective,
    ssi_solve_batch,
    ssi_solve_path,
)


def _toy_problem(n=40, q=25, m=5, lambda0=0.5, seed=0):
    """K PSD (= Z Z'/q con Z random), P = K + lambda0 I (PD), y C = K_0i de m individuos."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, q))
    K = (Z @ Z.T) / q                         # PSD por construccion
    P = K + lambda0 * np.eye(n)               # PD (lambda0 > 0)
    Zp = rng.standard_normal((m, q))
    C = (Zp @ Z.T).T / q                      # (n, m): columna = similitud train-vs-individuo
    return K, P, C


def test_lambda0_matches_gblup():
    """lambda=0 debe coincidir con la solucion GBLUP directa (P b = C)."""
    _, P, C = _toy_problem()
    B_cd, _ = ssi_solve_batch(P, C, lam=0.0, max_iter=20000, tol=1e-12)
    B_direct = gblup_weights(P, C)
    err = float(np.max(np.abs(B_cd - B_direct)))
    assert err < 1e-5, f"lambda=0 no coincide con GBLUP directo: max|dif|={err:.2e}"
    return err


def test_support_monotone():
    """El support ||b||_0 debe ser NO decreciente a medida que lambda decrece."""
    _, P, C = _toy_problem()
    lambdas = lambda_grid(C, n_lambda=40, lambda_min_ratio=0.005)  # decreciente
    B_path = ssi_solve_path(P, C, lambdas, max_iter=5000, tol=1e-9)
    support = (np.abs(B_path) > 1e-8).sum(axis=(1, 2))  # total no-ceros por lambda
    diffs = np.diff(support)                            # lambda decrece => support debe subir
    assert (diffs >= -1e-9).all(), f"support no monotono al bajar lambda: diffs={diffs}"
    # sanity: en el lambda mas grande casi todo es cero; en el mas chico hay mas soporte.
    assert support[0] <= support[-1], "el support total no crecio al bajar lambda"
    return int(support[0]), int(support[-1])


def test_objective_nonincreasing():
    """La loss de la Eq. 3 no debe aumentar durante el coordinate descent (convergencia)."""
    _, P, C = _toy_problem()
    lam = 0.05 * float(np.max(np.abs(C)))
    _, _, trace = ssi_solve_batch(P, C, lam=lam, max_iter=500, tol=1e-10, return_trace=True)
    trace = np.asarray(trace)
    rises = np.diff(trace)
    assert (rises <= 1e-8).all(), f"la loss aumento en algun barrido: max subida={rises.max():.2e}"
    # coherencia: el objetivo total del trace = suma de ssi_objective por individuo en la solucion.
    B, _ = ssi_solve_batch(P, C, lam=lam, max_iter=500, tol=1e-10)
    total = sum(ssi_objective(B[:, i], P, C[:, i], lam) for i in range(C.shape[1]))
    assert abs(total - trace[-1]) < 1e-6, "trace del objetivo inconsistente con ssi_objective"
    return float(trace[0]), float(trace[-1])


def test_prediction_uses_raw_training_phenotypes():
    """La prediccion del SSI debe ser B' y_TS, no B' (y_TS - mean)."""
    B = np.array([
        [1.0, 0.5],
        [0.0, 0.5],
        [0.0, 0.0],
    ])
    y = np.array([10.0, 20.0, 30.0])
    pred = predict_with_training_phenotypes(B, y)
    assert np.allclose(pred, np.array([10.0, 15.0]))
    centered = B.T @ (y - y.mean())
    assert not np.allclose(pred, centered), "la prediccion no debe centrar y_TS implicitamente"
    return tuple(float(v) for v in pred)


def _main() -> int:
    checks = [
        ("lambda=0 == GBLUP directo", test_lambda0_matches_gblup),
        ("support monotono en lambda", test_support_monotone),
        ("loss no aumenta en el CD", test_objective_nonincreasing),
        ("prediccion usa y_TS crudo", test_prediction_uses_raw_training_phenotypes),
    ]
    ok = True
    for name, fn in checks:
        try:
            info = fn()
            print(f"  PASS  {name}  {info if info is not None else ''}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
    print("OK" if ok else "FALLARON TESTS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
