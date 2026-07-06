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


def ld_prune_mask(
    X: np.ndarray, *, window_size: int = 50, r2_threshold: float = 0.5
) -> np.ndarray:
    """Poda por desequilibrio de ligamiento (LD) sobre los SNPs de TRAIN (vectorizada en NumPy).

    Recorre los SNPs en orden (proxy de posición genómica = orden de columnas) con una **ventana
    deslizante** de ``window_size`` marcadores y descarta el SNP que tenga ``r² > r2_threshold`` con
    ALGÚN SNP ya conservado dentro de la ventana (greedy indep-pairwise, estilo PLINK
    ``--indep-pairwise``). Devuelve una **máscara booleana** ``(p,)`` (True = conservado).

    ``window_size <= 0`` ⇒ modo **global** (ventana = panel entero): cada SNP se compara contra TODOS
    los conservados. Útil cuando las columnas NO están ordenadas por posición genómica (p.ej. IDs de
    marcador arbitrarios), donde una ventana chica no captura el LD porque los SNPs co-heredados no son
    columnas vecinas. Más caro (O(p·conservados·n)) pero capta la redundancia sin importar el orden.

    Eficiencia: se estandarizan las columnas una sola vez (``z = (x-media)/desvío`` con stats de
    train), de modo que la correlación de Pearson es el producto interno normalizado ``(z_j·z_k)/n``;
    cada SNP se compara contra las ≤``window_size`` columnas conservadas de la ventana con un solo
    producto matriz-vector. Los SNPs constantes (desvío 0) quedan en 0 ⇒ correlación 0 ⇒ se conservan.
    Se computa SOLO con train; la máscara se guarda y se aplica idéntica a val/test (`LDPruner`).

    Alternativa equivalente con dependencia extra: ``skallel.locate_unlinked`` de scikit-allel.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if p == 0:
        return np.zeros(0, dtype=bool)
    w = int(window_size) if int(window_size) > 0 else p  # <=0 => global (ventana = panel entero)
    thr = float(r2_threshold)

    # Estandarización train-only por columna: corr(a,b) = (z_a·z_b)/n con z de media 0 y desvío 1.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    nz = std > 0
    Z = np.zeros_like(X)
    Z[:, nz] = (X[:, nz] - mean[nz]) / std[nz]  # columnas constantes -> 0 (corr 0, siempre conservadas)

    keep = np.zeros(p, dtype=bool)
    win: list[int] = []  # índices conservados dentro de la ventana (posiciones >= j - w + 1)
    for j in range(p):
        # Saca de la ventana los conservados que quedaron a >= w posiciones de distancia.
        while win and j - win[0] >= w:
            win.pop(0)
        if win:
            r = (Z[:, win].T @ Z[:, j]) / n      # (len(win),) correlaciones contra los conservados
            if np.any(r * r > thr):
                continue                         # redundante en la ventana -> se descarta
        keep[j] = True
        win.append(j)
    return keep


class LDPruner:
    """Paso de preprocesamiento train-only que poda SNPs por LD (ver `ld_prune_mask`).

    `fit(X_train)` aprende la máscara de SNPs conservados; `transform` la aplica (a train o
    val/test) seleccionando exactamente las mismas columnas ⇒ las dimensiones coinciden al proyectar
    después en el PCA. Pensado para anteponerse a `SNPPreprocessor`/`PCAReducer` en la cadena de
    gating del MoE (poda antes del PCA), sin tocar los SNPs que ven los expertos.

    Args:
        window_size: tamaño de la ventana deslizante en nº de SNPs (default 50).
        r2_threshold: umbral de ``r²`` para considerar redundante un par (default 0.5).
    """

    def __init__(self, window_size: int = 50, r2_threshold: float = 0.5) -> None:
        self.window_size = int(window_size)
        self.r2_threshold = float(r2_threshold)

    def fit(self, X: np.ndarray, y: Any = None) -> "LDPruner":
        self.keep_ = ld_prune_mask(
            X, window_size=self.window_size, r2_threshold=self.r2_threshold
        )
        if not self.keep_.any():
            raise ValueError("LD pruning dejó 0 SNPs; revisá window_size/r2_threshold.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "keep_"):
            raise RuntimeError("LDPruner.transform llamado antes de fit.")
        return np.asarray(X, dtype=np.float64)[:, self.keep_]

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def n_kept(self) -> int | None:
        return int(self.keep_.sum()) if hasattr(self, "keep_") else None

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "keep_"):
            return {}
        total = int(self.keep_.size)
        kept = int(self.keep_.sum())
        return {
            "ld_n_kept": kept,
            "ld_n_total": total,
            "ld_frac_kept": float(kept / total) if total else 0.0,
            "ld_window_size": self.window_size,
            "ld_r2_threshold": self.r2_threshold,
        }


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


class PLSReducer:
    """Reducción de dimensionalidad **supervisada** por PLS, ajustada SOLO con train (regla 1).

    A diferencia del PCA (no supervisado, solo mira ``X``), PLS busca los componentes que maximizan la
    **covarianza entre los SNPs ``X`` y el fenotipo ``y``** ⇒ un espacio latente lineal orientado a
    separar las líneas por potencial fenotípico (p.ej. rendimiento), no solo por la mayor varianza
    genómica. `fit(X, y)` aprende las direcciones con el train (``y`` puede ser 1-D o multi-target
    ``(n, T)``); `transform(X)` proyecta cualquier set (val/test) SOLO desde ``X`` (sin fenotipos de
    test). Pensado para reemplazar al `PCAReducer` en la cadena de gating del MoE.

    Args:
        n_components: componentes latentes a retener (se clampea a ``min(n_samples, n_features)``).
        scale: estandariza ``X``/``Y`` internamente (default True; recomendado por los distintos
            rangos de los traits en el caso multi-target).
    """

    def __init__(self, n_components: int, *, scale: bool = True) -> None:
        self.n_components = int(n_components)
        self.scale = bool(scale)

    def fit(self, X: np.ndarray, y: Any = None) -> "PLSReducer":
        from sklearn.cross_decomposition import PLSRegression

        X = np.asarray(X, dtype=np.float64)
        if y is None:
            raise ValueError("PLSReducer.fit necesita `y` (supervisión); es una reducción supervisada.")
        Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        k = min(self.n_components, min(X.shape))
        self.n_components_ = int(max(1, k))
        self.pls_ = PLSRegression(n_components=self.n_components_, scale=self.scale)
        self.pls_.fit(X, Y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pls_"):
            raise RuntimeError("PLSReducer.transform llamado antes de fit.")
        return self.pls_.transform(np.asarray(X, dtype=np.float64))  # proyección desde X (train components)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_fitted_params(self) -> dict[str, Any]:
        return {"pls_n_components": self.n_components_} if hasattr(self, "pls_") else {}


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

    def set_fit_context(self, **context: Any) -> None:
        """Propaga el contexto del fold (year/line_id) a los pasos que lo acepten
        (p.ej. el autoencoder, que lo usa para su held-out por línea sin leakage)."""
        for step in self.steps:
            setter = getattr(step, "set_fit_context", None)
            if callable(setter):
                setter(**context)

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
    (`SNPPreprocessor`) si se pide; (2) un reductor de dimensionalidad si hay
    sub-bloque `pca` (`PCAReducer`) **o** `autoencoder` (`AutoencoderReducer`).
    Devuelve el único paso si hay uno solo, un `PreprocChain` si hay varios, o
    ``None`` si no hay nada que hacer. Mantiene compatibilidad con los configs que
    solo declaran `maf_threshold`/`standardize`.

    `pca` y `autoencoder` son mutuamente excluyentes (dos compresores no se
    encadenan). El autoencoder exige SNPs binarios (regla 4: BCE), así que NO se
    le antepone `standardize` (el z-score lo hace después GBLUP internamente); su
    filtro MAF va dentro del propio reducer (`autoencoder.maf_threshold`).
    """
    if not cfg:
        return None

    pca_cfg = cfg.get("pca")
    ae_cfg = cfg.get("autoencoder")
    if pca_cfg and ae_cfg:
        raise ValueError("preprocess: 'pca' y 'autoencoder' son excluyentes (dos compresores).")

    # --- Autoencoder: compresor autocontenido (su MAF va adentro) -----------
    if ae_cfg:
        if cfg.get("standardize", False):
            raise ValueError(
                "No uses standardize antes del autoencoder: rompe la binariedad de "
                "los SNPs (regla 4, BCE). El z-score va después (lo hace GBLUP)."
            )
        from src.models.autoencoder import AutoencoderReducer  # import diferido (torch)

        ae_params = dict(ae_cfg)
        # MAF declarado a nivel `preprocess` se delega al reducer si no lo trae él.
        if cfg.get("maf_threshold") and "maf_threshold" not in ae_params:
            ae_params["maf_threshold"] = cfg["maf_threshold"]
        return AutoencoderReducer(ae_params)

    steps: list = []
    if cfg.get("maf_threshold", 0.0) or cfg.get("standardize", False):
        steps.append(
            SNPPreprocessor(
                maf_threshold=cfg.get("maf_threshold", 0.0),
                standardize=cfg.get("standardize", False),
            )
        )

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
