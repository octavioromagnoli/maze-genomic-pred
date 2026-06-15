"""GBLUP — Genomic Best Linear Unbiased Prediction.

Predice valores de cría genómicos (``u_hat_PS``) resolviendo la ecuación (2) del
paper (Lopez-Cruz et. al., el paper base) (forma "hat matrix"):

    u_hat_PS = G_PS,TS (G_TS + lambda_0 I)^-1 (y_TS - mu)

donde ``G = Z Z' / p`` es la matriz de parentesco genómico construida a partir de
los marcadores centrados y escalados ``Z``, y ``lambda_0 = sigma_E^2 / sigma_u^2``
es el ratio de varianzas (regularización).

Engancha en la interfaz común (`fit`/`predict`) y en `run_evaluation`:
- `fit(X_tr, y_tr)` recibe SOLO el train del fold ⇒ todo el preprocesamiento
  (centrado/escalado de marcadores, G, selección de lambda_0) es train-only
  (regla 1, sin leakage).
- `predict(X_va)` reescala el val con los estadísticos de train y arma el bloque
  de covarianza `G_PS,TS` contra el train.

Dos métodos para obtener ``lambda_0`` (`params.lambda_method`):
- ``"cv"``   — Enfoque A: lambda_0 por validación cruzada **sin leakage** del propio
  predictor GBLUP. La CV es **temporal** por defecto (inner-train años previos →
  inner-val último año del train, p.ej. 2017→2018), respetando la regla 3. En cada
  fold se re-estandarizan los marcadores y se re-centra ``y`` usando solo el
  inner-train (regla 1) y se elige el ``lambda_0`` que minimiza el SSE sobre el
  inner-val. Configurable a K-fold con `params.internal_cv: kfold`.
- ``"reml"`` — Enfoque B: estimación REML de los componentes de varianza
  (`sigma_u^2`, `sigma_E^2`) vía eigen-descomposición de `G_TS` + optimizador
  numérico (equivalente conceptual a las medias a posteriori de BGLR/MCMC).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import BaseModel

# Rejilla por defecto para el Enfoque A (escala logarítmica densa).
_DEFAULT_ALPHA_GRID = np.logspace(-4, 4, 100)


# --- Bloques modulares ------------------------------------------------------

def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centra y escala columnas con estadísticos de TRAIN. Devuelve (Z, mean, std).

    Z = (X - mean) / std. Las columnas de varianza nula (SNP constante en train)
    quedan en cero tras centrar; se les fija std=1 para no dividir por cero.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)  # ddof=0, como StandardScaler
    std_safe = np.where(std == 0.0, 1.0, std)
    Z = (X - mean) / std_safe
    return Z, mean, std_safe


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Aplica a un set nuevo (val/test) el centrado/escalado ajustado en train."""
    return (np.asarray(X, dtype=float) - mean) / std


def genomic_relationship(Z_rows: np.ndarray, Z_cols: np.ndarray, p: int) -> np.ndarray:
    """Bloque de la matriz genómica: ``Z_rows Z_cols' / p``.

    Con ``Z_rows == Z_cols`` da `G_TS` (train×train); con ``Z_rows`` del prediction
    set da `G_PS,TS` (prueba×train).
    """
    return (Z_rows @ Z_cols.T) / p


def internal_cv(years, *, scheme: str, cv_folds: int):
    """Construye el `cv` para RidgeCV (selección interna de lambda_0).

    - ``scheme="temporal"`` (default): UN fold temporal dentro del train — entrena
      con los años anteriores al último y valida con el último (p.ej. con train
      2017–2018 ⇒ entrena 2017, valida 2018). Respeta la regla temporal del
      proyecto también en la selección interna y es barato (1 fold). Si hay menos
      de 2 años distintos no se puede partir por año ⇒ cae a K-fold.
    - ``scheme="kfold"``: ``KFold(cv_folds)`` clásico sobre el orden de filas
      (TS-first); úsese solo si se acepta romper el criterio temporal interno.

    Devuelve un iterable de splits ``[(train_idx, val_idx)]`` (temporal) o el
    entero ``cv_folds`` (K-fold), que es lo que RidgeCV espera en `cv`.
    """
    if scheme == "temporal" and years is not None:
        years = np.asarray(years)
        uniq = sorted(int(y) for y in np.unique(years))
        if len(uniq) >= 2:
            vy = uniq[-1]
            tr = np.flatnonzero(years < vy)
            va = np.flatnonzero(years == vy)
            return [(tr, va)]
    return int(cv_folds)


def lambda_via_cv(
    X: np.ndarray,
    y: np.ndarray,
    years,
    *,
    p: int,
    alpha_grid: np.ndarray,
    scheme: str,
    cv_folds: int,
) -> tuple[float, str]:
    """Enfoque A: lambda_0 por CV **sin leakage** sobre el predictor GBLUP real.

    Recibe ``X`` e ``y`` **crudos** (sin estandarizar ni centrar). En cada fold
    recomputa la estandarización de marcadores (media/desvío) y el centrado de
    ``y`` usando **solo el inner-train**, arma el GBLUP (mismo predictor que el
    modelo final) para cada ``alpha`` del grid y mide SSE sobre el inner-val.
    Devuelve ``(lambda_0, etiqueta_del_esquema)``.

    Por qué así y no `RidgeCV` sobre una matriz pre-estandarizada: si se estandariza
    y se centra ``y`` con TODO el train antes de partir 2017→2018, las stats de
    2018 entran en el ajuste interno (leakage, viola la regla 1) y sesgan el
    ``lambda_0`` elegido. Estandarizar/centrar por fold lo evita. ``lambda_0`` queda
    en la misma escala que el modelo final (ambos usan `G = ZZ'/p`), sin conversión.

    Eficiencia: una eigendescomposición de ``G_inner-train`` por fold; cada ``alpha``
    es luego un par de productos matriz-vector (`(G_it+aI)^-1 = V diag(1/(d+a)) V'`).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    alpha_grid = np.asarray(alpha_grid, dtype=float)

    cv = internal_cv(years, scheme=scheme, cv_folds=cv_folds)
    if isinstance(cv, list):
        splits, label = cv, "temporal-1fold"
    else:
        from sklearn.model_selection import KFold

        splits = list(KFold(n_splits=int(cv)).split(np.arange(n)))
        label = f"kfold-{int(cv)}"

    sse = np.zeros(alpha_grid.shape[0])
    for tr, va in splits:
        # Preprocesamiento ajustado SOLO con el inner-train (sin leakage).
        Z_it, mean_it, std_it = standardize_train(X[tr])
        Z_iv = apply_standardize(X[va], mean_it, std_it)
        mu = float(y[tr].mean())
        yc = y[tr] - mu
        G_it = genomic_relationship(Z_it, Z_it, p)   # inner-train × inner-train
        G_vt = genomic_relationship(Z_iv, Z_it, p)   # inner-val × inner-train
        d, V = np.linalg.eigh(G_it)
        d = np.clip(d, 0.0, None)
        Vt_yc = V.T @ yc
        GvV = G_vt @ V
        y_va = y[va]
        for j, a in enumerate(alpha_grid):
            c = Vt_yc / (d + a)            # = V' (G_it + a I)^-1 yc
            pred = mu + GvV @ c            # mu + G_vt (G_it + a I)^-1 yc
            sse[j] += float(np.sum((y_va - pred) ** 2))

    return float(alpha_grid[int(np.argmin(sse))]), label


def lambda_via_reml(
    G_TS: np.ndarray,
    y: np.ndarray,
    *,
    log10_bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[float, float, float]:
    """Enfoque B: lambda_0 por REML vía eigen-descomposición de G_TS.

    Modelo mixto ``y = 1·mu + g + e``, ``g ~ N(0, sigma_u^2 G_TS)``,
    ``e ~ N(0, sigma_E^2 I)``. Con ``G_TS = V D V'`` se rota a un problema
    diagonal y se maximiza la verosimilitud restringida (REML) respecto al ratio
    ``lambda = sigma_E^2 / sigma_u^2``, perfilando ``mu`` (GLS) y ``sigma_u^2``.

    Devuelve ``(lambda_0, sigma_u2, sigma_E2)``.
    """
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    # G_TS es simétrica ⇒ eigh (eigenvalores reales). Clip de negativos numéricos.
    d, V = np.linalg.eigh(G_TS)
    d = np.clip(d, 0.0, None)
    xi = V.T @ y          # fenotipos rotados
    w = V.T @ np.ones(n)  # diseño del intercepto rotado

    def neg_restricted_loglik(log10_lambda: float) -> float:
        lam = 10.0 ** log10_lambda
        v = d + lam  # autovalores de (G_TS + lambda I)
        inv_v = 1.0 / v
        # mu por GLS en el espacio rotado.
        a = float(np.sum(w * w * inv_v))
        mu = float(np.sum(w * xi * inv_v) / a)
        r = xi - mu * w
        # sigma_u^2 perfilado (REML: n - q fijos, q=1 intercepto).
        s2 = float(np.sum(r * r * inv_v) / (n - 1))
        # -2 logL_R(lambda), sin las constantes que no dependen de lambda.
        return float(np.sum(np.log(v)) + np.log(a) + (n - 1) * np.log(s2))

    res = minimize_scalar(
        neg_restricted_loglik, bounds=log10_bounds, method="bounded"
    )
    lam = 10.0 ** float(res.x)
    # Recupera los componentes de varianza en el óptimo.
    v = d + lam
    inv_v = 1.0 / v
    a = float(np.sum(w * w * inv_v))
    mu = float(np.sum(w * xi * inv_v) / a)
    r = xi - mu * w
    sigma_u2 = float(np.sum(r * r * inv_v) / (n - 1))
    sigma_e2 = lam * sigma_u2
    return lam, sigma_u2, sigma_e2


# --- Modelo -----------------------------------------------------------------

class GBLUP(BaseModel):
    """GBLUP con lambda_0 por validación cruzada (Ridge) o por REML.

    `params` (sub-dict `model.params` del config):
        lambda_method : "cv" | "reml"   (default "cv")
        internal_cv   : "temporal" | "kfold"  (solo cv; default "temporal")
        alpha_grid    : grid de lambda_0 a probar en cv (default logspace(-4,4,100))
        cv_folds      : pliegues si internal_cv="kfold" (default 10)
        lambda_0      : float opcional; si está, fuerza ese ratio y saltea A/B.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBLUP":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("GBLUP.fit recibió y vacío.")

        # 1) Centrado/escalado de marcadores con estadísticos de TRAIN (regla 1).
        Z_tr, self.col_mean_, self.col_std_ = standardize_train(X)
        self.p_: int = Z_tr.shape[1]

        # 2) Matriz genómica de train G_TS = Z Z' / p.
        G_TS = genomic_relationship(Z_tr, Z_tr, self.p_)

        # 3) Centrado del fenotipo por la media de train (convención BLUP).
        self.y_mean_: float = float(y.mean())
        y_c = y - self.y_mean_

        # 4) lambda_0 por el método elegido.
        forced = self.params.get("lambda_0")
        if forced is not None:
            self.lambda_0_ = float(forced)
            self.lambda_method_ = "fixed"
        else:
            method = str(self.params.get("lambda_method", "cv")).lower()
            self.lambda_method_ = method
            if method == "cv":
                # CV sobre X/y CRUDOS: re-estandariza y re-centra por fold (sin leakage).
                grid = np.asarray(self.params.get("alpha_grid", _DEFAULT_ALPHA_GRID), dtype=float)
                scheme = str(self.params.get("internal_cv", "temporal")).lower()
                self.lambda_0_, self.internal_cv_ = lambda_via_cv(
                    X, y,
                    self._fit_context.get("year"),
                    p=self.p_,
                    alpha_grid=grid,
                    scheme=scheme,
                    cv_folds=int(self.params.get("cv_folds", 10)),
                )
            elif method == "reml":
                self.lambda_0_, self.sigma_u2_, self.sigma_e2_ = lambda_via_reml(G_TS, y)
            else:
                raise ValueError(
                    f"lambda_method desconocido: {method!r}. Use 'cv' o 'reml' "
                    "(o pase params.lambda_0 para fijarlo)."
                )

        # 5) Resuelve el sistema lineal estable (sin invertir): (G_TS + l I) c = y_c.
        #    c son los coeficientes duales de la fórmula hat-matrix eq. (2).
        n_TS = G_TS.shape[0]
        A = G_TS + self.lambda_0_ * np.identity(n_TS)
        c = np.linalg.solve(A, y_c)

        # 6) Representación rrBLUP del MISMO predictor: efectos de marcador
        #    b = Z_tr' c / p. Como u_hat_PS = G_PS,TS c = (Z_PS Z_tr'/p) c = Z_PS b,
        #    predecir es un solo producto Z_PS·b y solo hay que guardar el vector b
        #    (largo p), no toda la matriz de train Z_tr (n x p). Numéricamente
        #    idéntico a la hat-matrix (verificado a precisión de máquina).
        self.marker_effects_ = (Z_tr.T @ c) / self.p_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "marker_effects_"):
            raise RuntimeError("GBLUP.predict llamado antes de fit.")
        # Reescala el prediction set con los estadísticos de train y proyecta sobre
        # los efectos de marcador (equivalente a G_PS,TS (G_TS+lambda I)^-1 y_c).
        Z_ps = apply_standardize(X, self.col_mean_, self.col_std_)
        u_hat = Z_ps @ self.marker_effects_  # valores de cría (desvíos)
        return self.y_mean_ + u_hat          # vuelve a la escala del fenotipo

    def get_fitted_params(self) -> dict[str, Any]:
        """Hiperparámetros recuperados en el ajuste (para logging/bitácora)."""
        out: dict[str, Any] = {
            "lambda_0": getattr(self, "lambda_0_", None),
            "lambda_method": getattr(self, "lambda_method_", None),
        }
        if hasattr(self, "internal_cv_"):
            out["internal_cv"] = self.internal_cv_
        if hasattr(self, "sigma_u2_"):
            out["sigma_u2"] = self.sigma_u2_
            out["sigma_e2"] = self.sigma_e2_
        return out
