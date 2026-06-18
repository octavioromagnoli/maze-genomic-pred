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

    def get_fitted_params(self) -> dict[str, Any]:
        return {"n_kept": self.n_kept} if hasattr(self, "keep_") else {}


class PCAReducer:
    """Reducción de dimensionalidad por PCA, ajustada SOLO con train (regla 1).

    Envuelve `sklearn.decomposition.PCA`. `fit` aprende los componentes con el
    train del fold; `transform` proyecta cualquier set (val/test) sobre esos
    mismos componentes. sklearn centra internamente con la media de train; para
    z-score por-SNP antes del PCA (recomendado para los lineales), anteponer
    `SNPPreprocessor(standardize=True)` en la cadena.

    La dimensión latente (`n_components`) es el HP a barrer para elegirla por
    desempeño downstream en vez de por % de varianza explicada.

    Args:
        n_components: componentes a retener. Se clampea a ``min(n_samples,
            n_features)`` del train (no se pueden pedir más).
        whiten: si ``True``, escala cada componente a varianza unitaria.
        svd_solver: solver de sklearn. ``"full"`` (default) es determinista;
            ``"randomized"`` es más rápido en p grande pero usa ``random_state``.
        random_state: semilla del solver randomized (reproducibilidad).
    """

    def __init__(
        self,
        n_components: int,
        *,
        whiten: bool = False,
        svd_solver: str = "full",
        random_state: int | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.svd_solver = str(svd_solver)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None) -> "PCAReducer":
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=np.float64)
        # No se pueden pedir más componentes que el rango de la matriz de train.
        k = min(self.n_components, min(X.shape))
        self.n_components_ = int(k)
        self.pca_ = PCA(
            n_components=self.n_components_,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pca_"):
            raise RuntimeError("PCAReducer.transform llamado antes de fit.")
        return self.pca_.transform(np.asarray(X, dtype=np.float64))

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_fitted_params(self) -> dict[str, Any]:
        """n_components efectivo + varianza acumulada (logueado a wandb por fold)."""
        if not hasattr(self, "pca_"):
            return {}
        evr = self.pca_.explained_variance_ratio_
        return {
            "pca_n_components": self.n_components_,
            "pca_cum_var_explained": float(np.sum(evr)),
        }


class PreprocChain:
    """Cadena de pasos de preprocesamiento, todos train-only.

    `fit_transform` ajusta y aplica en orden sobre el train; `transform` aplica
    (sin re-ajustar) sobre val/test. Expone la interfaz que espera
    `PreprocessedModel` (`fit_transform`/`transform`) y mergea los
    `get_fitted_params` de cada paso que los exponga (p.ej. n_kept del filtro MAF
    y n_components/varianza del PCA → se logean por fold a wandb).
    """

    def __init__(self, steps: list) -> None:
        self.steps = list(steps)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        for step in self.steps:
            X = step.transform(X)
        return X

    def get_fitted_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for step in self.steps:
            gfp = getattr(step, "get_fitted_params", None)
            if callable(gfp):
                out.update(gfp())
        return out


def build_preprocessor(cfg: dict[str, Any] | None):
    """Construye el preprocesamiento train-only desde el bloque `preprocess`.

    Arma una cadena en orden: (1) filtro MAF + estandarización por-SNP
    (`SNPPreprocessor`) si se pide; (2) PCA (`PCAReducer`) si hay sub-bloque
    `pca`. Devuelve el único paso si hay uno solo, un `PreprocChain` si hay
    varios, o ``None`` si no hay nada que hacer. Mantiene compatibilidad con los
    configs que solo declaran `maf_threshold`/`standardize`.
    """
    if not cfg:
        return None

    steps: list = []
    if cfg.get("maf_threshold", 0.0) or cfg.get("standardize", False):
        steps.append(
            SNPPreprocessor(
                maf_threshold=cfg.get("maf_threshold", 0.0),
                standardize=cfg.get("standardize", False),
            )
        )

    pca_cfg = cfg.get("pca")
    if pca_cfg:
        steps.append(
            PCAReducer(
                n_components=pca_cfg["n_components"],
                whiten=pca_cfg.get("whiten", False),
                svd_solver=pca_cfg.get("svd_solver", "full"),
                random_state=pca_cfg.get("random_state"),
            )
        )

    if not steps:
        return None
    return steps[0] if len(steps) == 1 else PreprocChain(steps)
