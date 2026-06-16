"""Preprocesamiento train-only de SNPs.

`SNPPreprocessor` aprende todo (qué columnas conservar por MAF, y stats de
estandarización) en `fit(X_train)` y lo aplica con `transform`. Se compone con
cualquier modelo vía `src.models.registry.PreprocessedModel`, así no se reescribe
el preprocesamiento por cada modelo: un modelo nuevo solo declara un bloque
`preprocess:` en su YAML.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SNPPreprocessor:
    """Filtro de MAF (+ estandarización opcional), ajustado solo con el train.

    Args:
        maf_threshold: conserva SNPs con ``MAF >= maf_threshold`` (calculado en
            train). ``0`` = no filtra.
        standardize: si ``True``, centra y escala por-SNP (z-score) con la media
            y el desvío del train.
    """

    def __init__(self, maf_threshold: float = 0.0, standardize: bool = False, eps: float = 1e-8) -> None:
        self.maf_threshold = float(maf_threshold)
        self.standardize = bool(standardize)
        self.eps = float(eps)

    def fit(self, X: np.ndarray, y: Any = None) -> "SNPPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        freq = X.mean(axis=0)                      # frec. del alelo '1' en train
        maf = np.minimum(freq, 1.0 - freq)
        self.keep_ = maf >= self.maf_threshold     # máscara de columnas (aprendida en train)
        if not self.keep_.any():
            raise ValueError(f"maf_threshold={self.maf_threshold} dejó 0 SNPs.")
        if self.standardize:
            Xk = X[:, self.keep_]
            self.mean_ = Xk.mean(axis=0)
            self.std_ = Xk.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "keep_"):
            raise RuntimeError("SNPPreprocessor.transform llamado antes de fit.")
        Xk = np.asarray(X, dtype=np.float64)[:, self.keep_]
        if self.standardize:
            Xk = (Xk - self.mean_) / self.std_
        return Xk

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def n_kept(self) -> int | None:
        return int(self.keep_.sum()) if hasattr(self, "keep_") else None


def build_preprocessor(cfg: dict[str, Any] | None) -> SNPPreprocessor | None:
    """Construye un `SNPPreprocessor` desde el bloque `preprocess` del config.

    Devuelve ``None`` si `cfg` es vacío/``None`` (sin preprocesamiento).
    """
    if not cfg:
        return None
    return SNPPreprocessor(
        maf_threshold=cfg.get("maf_threshold", 0.0),
        standardize=cfg.get("standardize", False),
    )
