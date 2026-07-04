"""Solver del Sparse Selection Index (SSI, Lopez-Cruz et al. 2021, Eq. 3).

Para cada individuo i del prediction set se resuelve el problema convexo

    min_b  (1/2) b'(K_TS + lambda0 I) b  -  K_0i' b  +  lambda ||b||_1              (Eq. 3)

y se predice ``u_i = sum_j b_ij y_j`` (suma ponderada de fenotipos crudos del TRAINING),
que es la regla de prediccion del SSI.

Es un lasso con termino cuadratico GENERICO ``b'P b`` (P = K_TS + lambda0 I), NO ortonormal: se
resuelve por **coordinate descent** con soft-thresholding, usando el elemento diagonal ``P_jj``
como paso. Warm starts sobre la grilla de lambda. Todo vectorizado sobre los n_PS individuos (que
comparten P; solo cambia el vector lineal ``c = K_0i``), asi el loop Python es solo sobre las n_TS
coordenadas y cada update opera sobre todas las columnas a la vez.

Este modulo es PURO (algebra); la frontera anti-leakage la garantiza el runner (`run_ssi.py`):
K se arma solo con marcadores/embeddings y la prediccion usa solo fenotipos del TS.

Validacion (ver `tests/test_ssi_solver.py`):
- lambda=0 debe coincidir con el GBLUP no-sparse ``b = (K_TS + lambda0 I)^-1 K_0i`` (sistema lineal directo).
- al subir lambda el support (nro de coeficientes no-cero) decrece monotono.
- la loss Eq.3 no aumenta durante el coordinate descent (convergencia).
"""
from __future__ import annotations

import warnings

import numpy as np

from src.eval.metrics import pearson_r


def soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Operador de soft-thresholding ``sign(z) * max(|z| - lam, 0)`` (elementwise)."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def ssi_objective(b: np.ndarray, P: np.ndarray, c: np.ndarray, lam: float) -> float:
    """Objetivo de la Eq. 3 para UN individuo: ``0.5 b'P b - c'b + lambda ||b||_1``."""
    b = np.asarray(b, dtype=float)
    return float(0.5 * b @ (P @ b) - c @ b + lam * np.sum(np.abs(b)))


def ssi_solve_batch(
    P: np.ndarray,
    C: np.ndarray,
    lam: float,
    *,
    B_init: np.ndarray | None = None,
    PB_init: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    return_trace: bool = False,
):
    """Coordinate descent para la Eq. 3 a lambda FIJO, vectorizado sobre m individuos.

    Args:
        P: ``(n, n)`` = ``K_TS + lambda0 I`` (simetrica, PD por el ``lambda0 I``). Compartida por
           todos los individuos del batch.
        C: ``(n, m)`` — cada columna es el vector lineal ``c = K_0i`` de un individuo del PS.
        lam: penalidad L1.
        B_init / PB_init: warm start (solucion previa ``B`` y su producto ``P @ B``); si ``None``,
           arranca en cero.
        return_trace: si True, ademas devuelve la lista de valores del objetivo TOTAL (sumado sobre
           los m individuos) tras cada barrido — para el test de descenso monotono.

    Returns:
        ``(B, PB)`` con ``B`` ``(n, m)`` la solucion y ``PB = P @ B`` (para encadenar warm starts).
        Si ``return_trace``, devuelve ``(B, PB, trace)``.
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    d = np.diag(P).copy()  # pasos por coordenada (K_TS_jj + lambda0 > 0)
    B = np.zeros((n, m)) if B_init is None else np.array(B_init, dtype=float)
    PB = (P @ B) if PB_init is None else np.array(PB_init, dtype=float)

    trace: list[float] = []
    for _ in range(int(max_iter)):
        max_delta = 0.0
        for j in range(n):
            # rho_j = P_jj b_j + c_j - (P b)_j  (gradiente parcial "sin el propio j")
            rho = d[j] * B[j] + C[j] - PB[j]
            new = soft_threshold(rho, lam) / d[j]
            delta = new - B[j]
            nz = delta != 0.0
            if nz.any():
                PB[:, nz] += np.outer(P[:, j], delta[nz])  # update rank-1 solo en columnas que cambian
                B[j] = new
                md = float(np.max(np.abs(delta[nz])))
                if md > max_delta:
                    max_delta = md
        if return_trace:
            # objetivo total = sum_i [0.5 b_i'P b_i - c_i'b_i + lam||b_i||_1]; usa PB (= P B).
            quad = 0.5 * np.sum(B * PB)
            lin = float(np.sum(C * B))
            l1 = lam * float(np.sum(np.abs(B)))
            trace.append(float(quad - lin + l1))
        if max_delta < tol:
            break
    return (B, PB, trace) if return_trace else (B, PB)


def lambda_grid(C: np.ndarray, *, n_lambda: int = 50, lambda_min_ratio: float = 0.01) -> np.ndarray:
    """Grilla geometrica DECRECIENTE de lambda derivada de los datos.

    ``lambda_max = max|C|``: por encima de ese valor la solucion de todo individuo es b=0 (en b=0 el
    gradiente es -c_j, y el soft-threshold da 0 sii lambda >= |c_j|). Se baja hasta
    ``lambda_min_ratio * lambda_max``. Devuelve el vector ordenado de mayor a menor (para warm starts).
    """
    lam_max = float(np.max(np.abs(C)))
    if lam_max <= 0:
        return np.zeros(1)
    return np.logspace(np.log10(lam_max), np.log10(lam_max * lambda_min_ratio), int(n_lambda))


def ssi_solve_path(
    P: np.ndarray,
    C: np.ndarray,
    lambdas: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> np.ndarray:
    """Resuelve la Eq. 3 sobre una grilla DECRECIENTE de lambda con warm starts.

    Devuelve ``B_path`` ``(len(lambdas), n, m)``: la solucion para cada lambda. Cada lambda usa como
    init la solucion (y su ``P@B``) del lambda anterior (mayor) — clave para eficiencia.
    """
    P = np.asarray(P, dtype=float)
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    lambdas = np.asarray(lambdas, dtype=float)
    B_path = np.empty((lambdas.shape[0], n, m))
    B = np.zeros((n, m))
    PB = np.zeros((n, m))
    for t, lam in enumerate(lambdas):
        B, PB = ssi_solve_batch(P, C, float(lam), B_init=B, PB_init=PB, max_iter=max_iter, tol=tol)
        B_path[t] = B
    return B_path


def predict_with_training_phenotypes(B: np.ndarray, y_TS: np.ndarray) -> np.ndarray:
    """Prediccion SSI ``u_i = sum_j b_ij y_j`` con fenotipos crudos del training.

    ``B`` tiene shape ``(n_TS, n_PS)``: cada columna contiene los pesos ``b_ij`` para
    un individuo del prediction set. No se centra ``y_TS`` aca porque el SSI, a diferencia
    del GBLUP no-sparse centrado, define la prediccion como suma ponderada de fenotipos.
    """
    return np.asarray(B, dtype=float).T @ np.asarray(y_TS, dtype=float)


def select_lambda_cv(
    K_TS: np.ndarray,
    y_TS: np.ndarray,
    lambda0: float,
    lambdas: np.ndarray,
    *,
    n_folds: int = 10,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[float, np.ndarray]:
    """Elige lambda por K-fold CV DENTRO del training (un lambda para todo el TS).

    Por cada fold: el resto del TS es el "training interno" y el fold held-out es un mini-prediction
    set; se resuelve el path SSI y se mide accuracy = correlacion(observado, predicho) en el fold.
    Se promedia sobre folds y se elige el lambda de mayor accuracy media (protocolo del paper).
    ``lambda0`` (del TS completo, via REML) se reusa en los folds (plug-in; el CV solo tunea la
    sparsity ``lambda``). Devuelve ``(lambda_opt, acc_media_por_lambda)``.
    """
    K_TS = np.asarray(K_TS, dtype=float)
    y_TS = np.asarray(y_TS, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    n = K_TS.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)

    acc = np.zeros((n_folds, lambdas.shape[0]))
    for f, val in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), val, assume_unique=False)
        if val.size < 2 or tr.size < 2:
            acc[f] = np.nan
            continue
        P = K_TS[np.ix_(tr, tr)] + lambda0 * np.eye(tr.size)
        C = K_TS[np.ix_(tr, val)]                 # columnas = c de cada held-out
        y_inner = y_TS[tr]
        B_path = ssi_solve_path(P, C, lambdas, max_iter=max_iter, tol=tol)
        for t in range(lambdas.shape[0]):
            u = predict_with_training_phenotypes(B_path[t], y_inner)  # (n_val,)
            acc[f, t] = pearson_r(y_TS[val], u)   # NaN si u constante; se ignora abajo
    with warnings.catch_warnings():                # una columna toda-NaN (u constante en todos los
        warnings.simplefilter("ignore", RuntimeWarning)  # folds) => nanmean avisa; el argmax la ignora
        mean_acc = np.nanmean(acc, axis=0)
    if not np.isfinite(mean_acc).any():            # degenerado: fallback al lambda menos sparse (~GBLUP)
        return float(lambdas[-1]), mean_acc
    best = int(np.nanargmax(mean_acc))
    return float(lambdas[best]), mean_acc


def gblup_weights(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pesos GBLUP no-sparse ``b = (K_TS + lambda0 I)^-1 K_0i`` (lambda=0 de la Eq. 3).

    Resuelve el sistema lineal directo ``P B = C`` (sin invertir). Es la referencia contra la que se
    valida el solver en ``lambda=0`` (el paper: lambda=0 recupera el GBLUP no-sparse).
    """
    return np.linalg.solve(np.asarray(P, dtype=float), np.asarray(C, dtype=float))
