"""Kernels de similitud para el SSI: G aditiva, Gaussiano, y variantes sobre embeddings.

Todos los kernels se construyen a partir de FEATURES (marcadores o embeddings), nunca de fenotipos
⇒ computarlos sobre TODOS los individuos (train + test) no introduce leakage de fenotipo (los SNPs
no llevan info del rinde 2020). El runner los subsetea a bloques TS/PS.

Variantes (config `ssi.kernels`):
- ``G`` (aditiva)         : reusa el GRM del GBLUP, ``G = Z Z' / p`` con Z = marcadores estandarizados.
- ``gaussian_snp``        : ``exp(-theta d^2)`` sobre distancias en el espacio de los SNPs (KSSI).
- ``gram_emb``            : ``Z Z' / q`` con Z = embedding MTL estandarizado por columna (paralelo a G).
- ``gaussian_emb``        : ``exp(-theta d^2)`` sobre distancias en el espacio del embedding.

El termino ``lambda0 I`` del solver ayuda al condicionamiento, pero el solver asume convexidad ⇒ se
chequea que cada K sea PSD (autovalores >= -tol) y se avisa si no.
"""
from __future__ import annotations

import numpy as np

# Reusa el GRM del GBLUP (misma G que el baseline del compañero) — sin reescribir logica.
from src.models.gblup import genomic_relationship, standardize_train


def gram_kernel(F: np.ndarray, *, standardize: bool = True) -> np.ndarray:
    """Kernel lineal (Gram) ``Z Z' / q``. Con ``standardize`` z-scorea las columnas primero.

    Con ``F`` = marcadores reproduce la G aditiva del paper (``G = ZZ'/p``); con ``F`` = embedding MTL
    es el ``K_learned`` Gram (``q`` = dim del embedding). Reusa `standardize_train`/`genomic_relationship`.
    """
    Z = standardize_train(np.asarray(F, dtype=float))[0] if standardize else np.asarray(F, dtype=float)
    return genomic_relationship(Z, Z, Z.shape[1])


def squared_distances(F: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclidianas al cuadrado ``d^2_ij = ||f_i - f_j||^2`` (n x n, >= 0)."""
    F = np.asarray(F, dtype=float)
    sq = np.einsum("ij,ij->i", F, F)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (F @ F.T)
    return np.maximum(D2, 0.0)  # limpia negativos numericos


def theta_from_quantiles(
    F: np.ndarray, quantiles=(0.05, 0.5, 0.95), *, standardize: bool = False
) -> list[float]:
    """Bandwidths ``theta = 1 / (2 d^2_q)`` con ``d^2_q`` los cuantiles de las distancias (K1/K2/K3).

    Replica el protocolo del paper: tres anchos ligados a los cuantiles de la distribucion de
    distancias en el espacio de features (5%/50%/95% por defecto).
    """
    Z = standardize_train(F)[0] if standardize else np.asarray(F, dtype=float)
    D2 = squared_distances(Z)
    iu = np.triu_indices_from(D2, k=1)               # solo pares distintos
    dq = np.quantile(D2[iu], list(quantiles))
    return [float(1.0 / (2.0 * q)) if q > 0 else 0.0 for q in dq]


def gaussian_kernel(F: np.ndarray, theta: float, *, standardize: bool = False) -> np.ndarray:
    """Kernel Gaussiano ``K_ij = exp(-theta d^2_ij)`` sobre distancias en el espacio de ``F``."""
    Z = standardize_train(F)[0] if standardize else np.asarray(F, dtype=float)
    return np.exp(-float(theta) * squared_distances(Z))


def min_eigenvalue(K: np.ndarray) -> float:
    """Autovalor minimo de ``K`` (simetrizado); < 0 => no PSD."""
    Ks = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    return float(np.linalg.eigvalsh(Ks)[0])


def check_psd(K: np.ndarray, *, tol: float = 1e-8, name: str = "K") -> bool:
    """Devuelve True si ``K`` es PSD (min eig >= -tol). Si no, imprime un aviso (no lanza)."""
    lo = min_eigenvalue(K)
    if lo < -abs(tol):
        print(f"[ssi][WARN] kernel '{name}' NO es PSD: min_eig={lo:.3e} (< -{tol:.0e}). "
              "El solver asume convexidad; el termino lambda0*I puede no alcanzar.")
        return False
    return True


def build_kernels(
    markers: np.ndarray,
    embedding: np.ndarray | None,
    spec: list[dict],
) -> dict[str, np.ndarray]:
    """Construye los kernels pedidos sobre TODOS los individuos y valida PSD.

    ``spec`` es la lista `ssi.kernels` del config; cada item ``{"kind": ..., ...}``:
      - ``{"kind": "G"}``                                        -> Gram de marcadores (aditiva).
      - ``{"kind": "gaussian_snp", "quantiles": [...]}``         -> Gaussiano sobre SNPs (una K por theta).
      - ``{"kind": "gram_emb"}``                                 -> Gram del embedding MTL.
      - ``{"kind": "gaussian_emb", "quantiles": [...]}``         -> Gaussiano sobre el embedding (una K por theta).

    Devuelve ``{nombre: K}`` (n_all x n_all). Los Gaussianos generan un nombre por bandwidth
    (``gaussian_snp_k1/k2/k3``). Chequea PSD de cada uno.
    """
    markers = np.asarray(markers, dtype=float)
    out: dict[str, np.ndarray] = {}
    for item in spec:
        kind = str(item["kind"])
        if kind == "G":
            out["G"] = gram_kernel(markers, standardize=True)
        elif kind == "gram_emb":
            if embedding is None:
                raise ValueError("kernel 'gram_emb' pide embedding pero no se paso ninguno.")
            out["gram_emb"] = gram_kernel(embedding, standardize=True)
        elif kind in ("gaussian_snp", "gaussian_emb"):
            F = markers if kind == "gaussian_snp" else embedding
            if F is None:
                raise ValueError(f"kernel '{kind}' pide embedding pero no se paso ninguno.")
            quant = tuple(item.get("quantiles", (0.05, 0.5, 0.95)))
            std = bool(item.get("standardize", kind == "gaussian_emb"))
            thetas = theta_from_quantiles(F, quant, standardize=std)
            for tag, th in zip(("k1", "k2", "k3"), thetas):
                out[f"{kind}_{tag}"] = gaussian_kernel(F, th, standardize=std)
        else:
            raise ValueError(f"kernel desconocido: {kind!r}.")
    for name, K in out.items():
        check_psd(K, name=name)
    return out
