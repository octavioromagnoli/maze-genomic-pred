"""Proyector lineal contrastivo (metric learning superficial) para el ruteo del MoE.

En vez de un autoencoder que reconstruye los SNPs (BCE, píxel por píxel), se entrena un **proyector
lineal de una sola capa** ``f(x) = z·W`` con **triplet loss** para que el espacio latente respete el
**parentesco genómico**: junta a las líneas emparentadas (hermanas) y aleja a las de familias
distintas. Las etiquetas de "familia" no vienen dadas ⇒ se derivan de la matriz de parentesco
``G = Z Z' / p`` (train-only): para cada línea (anchor), los positivos son sus ``n_pos`` líneas de
MAYOR parentesco y los negativos, líneas de bajo parentesco. El GMM del MoE clusteriza ``f(X)``.

Guarda SOLO la matriz lineal ``W_`` (numpy) + la estandarización train-only, así ``transform`` es
``(X - mean)/std @ W`` (sin torch en inferencia ni en el artifact; joblib-serializable). Anti-leakage:
``G``, ``W`` y la estandarización se ajustan SOLO con el train; test se proyecta con ``W`` (solo ``X``).

Interfaz de paso de preprocesamiento (``fit``/``transform``/``fit_transform``/``get_fitted_params``)
para encadenarse en el gating del MoE, igual que `PCAReducer`/`PLSReducer`.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ContrastiveProjector:
    """Proyector lineal entrenado con triplet loss sobre el parentesco genómico G (train-only).

    Args:
        latent_dim: dimensión del espacio latente de ruteo (default 20).
        n_pos: nº de vecinos de MAYOR parentesco por línea que cuentan como positivos (default 5).
        n_neg: nº de candidatos de BAJO parentesco por línea para muestrear negativos (default 20).
        margin: margen del triplet loss ``max(0, d(a,p)² − d(a,n)² + margin)`` (default 1.0).
        epochs, lr, random_state: entrenamiento del proyector.
    """

    def __init__(
        self,
        latent_dim: int = 20,
        *,
        n_pos: int = 5,
        n_neg: int = 20,
        margin: float = 1.0,
        epochs: int = 150,
        lr: float = 1e-2,
        random_state: int = 575,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.n_pos = int(n_pos)
        self.n_neg = int(n_neg)
        self.margin = float(margin)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: Any = None) -> "ContrastiveProjector":
        import torch

        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        # Estandarización train-only (para G y para la entrada del proyector).
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0.0, 1.0, std)
        Z = (X - self.mean_) / self.std_

        # Parentesco genómico y minería de positivos/negativos por línea.
        G = (Z @ Z.T) / p
        G_ns = G.copy()
        np.fill_diagonal(G_ns, -np.inf)                       # excluye self de los positivos
        npos = min(self.n_pos, n - 1)
        nneg = min(self.n_neg, n - 1)
        pos_idx = np.argpartition(-G_ns, npos, axis=1)[:, :npos]   # (n, n_pos) mayor parentesco
        neg_idx = np.argpartition(G_ns, nneg, axis=1)[:, :nneg]    # (n, n_neg) menor parentesco

        # Proyector lineal f(z) = z·W, entrenado con triplet loss.
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        Zt = torch.as_tensor(Z, dtype=torch.float32)
        W = torch.zeros(p, self.latent_dim, dtype=torch.float32)
        torch.nn.init.xavier_uniform_(W)
        W.requires_grad_(True)
        optim = torch.optim.Adam([W], lr=self.lr)
        anchors = np.arange(n)
        history: list[float] = []
        for _ in range(self.epochs):
            # Un positivo y un negativo por anchor (muestreados de sus listas por parentesco).
            pj = pos_idx[anchors, rng.integers(0, npos, size=n)]
            nj = neg_idx[anchors, rng.integers(0, nneg, size=n)]
            optim.zero_grad()
            E = Zt @ W                                        # (n, latent) embeddings de TODAS las líneas
            ea, ep, en = E[anchors], E[pj], E[nj]
            d_ap = ((ea - ep) ** 2).sum(1)
            d_an = ((ea - en) ** 2).sum(1)
            loss = torch.clamp(d_ap - d_an + self.margin, min=0.0).mean()
            loss.backward()
            optim.step()
            history.append(float(loss.item()))
        self.W_ = W.detach().numpy().astype(np.float64)       # (p, latent), numpy => joblib-safe
        self.final_loss_ = float(history[-1]) if history else float("nan")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "W_"):
            raise RuntimeError("ContrastiveProjector.transform llamado antes de fit.")
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        return Z @ self.W_

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "W_"):
            return {}
        return {
            "contrastive_latent_dim": int(self.latent_dim),
            "contrastive_margin": float(self.margin),
            "contrastive_final_loss": float(self.final_loss_),
        }
