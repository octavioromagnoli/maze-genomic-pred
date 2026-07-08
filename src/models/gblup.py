"""GBLUP (Genomic Best Linear Unbiased Prediction).

Predice valores genómicos resolviendo

    u_hat = G_ps,ts (G_ts + lambda I)^-1 (y - mu)

con G = ZZ'/p la matriz de parentesco sobre los marcadores centrados y
escalados, y lambda = sigma_e^2 / sigma_u^2 el ratio de varianzas. El escalado
de marcadores, G y la selección de lambda se ajustan con el train del fold.

lambda se elige por CV temporal (params.lambda_method="cv", el default) o por
REML de los componentes de varianza ("reml"). También puede fijarse a mano con
params.lambda_0.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import BaseModel

_DEFAULT_ALPHA_GRID = np.logspace(-4, 4, 100)



def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centra y escala columnas con la media/desvío del train. Devuelve (Z, mean, std).

    Los SNP constantes en train (desvío nulo) quedan en cero; se les fija std=1
    para no dividir por cero.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    Z = (X - mean) / std_safe
    return Z, mean, std_safe


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Aplica a un set nuevo (val/test) el centrado/escalado ajustado en train."""
    return (np.asarray(X, dtype=float) - mean) / std


def genomic_relationship(Z_rows: np.ndarray, Z_cols: np.ndarray, p: int) -> np.ndarray:
    """Bloque de la matriz genómica Z_rows Z_cols' / p.

    Con Z_rows == Z_cols da G train×train; con las filas del set a predecir da el
    bloque prueba×train.
    """
    return (Z_rows @ Z_cols.T) / p


def internal_cv(years, *, scheme: str, cv_folds: int):
    """Split interno para elegir lambda_0.

    Con scheme="temporal" (default) usa un único fold: entrena con los años
    anteriores al último del train y valida con el último (2017-2018 -> entrena
    2017, valida 2018). Con menos de 2 años cae a K-fold. Con scheme="kfold" usa
    KFold(cv_folds) sobre el orden de filas.

    Devuelve la lista de splits [(train_idx, val_idx)] o el entero cv_folds.
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
    """lambda_0 por CV sobre el predictor GBLUP real. Devuelve (lambda_0, etiqueta).

    Recibe X e y crudos. En cada fold recomputa la estandarización de marcadores
    y el centrado de y con el inner-train, arma el GBLUP para cada alpha del grid
    y mide SSE sobre el inner-val. Estandarizar por fold (en vez de una sola vez
    sobre todo el train) evita que las stats del inner-val entren en el ajuste.

    Cada fold cuesta una eigendescomposición de G_inner-train; cada alpha es luego
    un par de productos matriz-vector via (G+aI)^-1 = V diag(1/(d+a)) V'.
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
        Z_it, mean_it, std_it = standardize_train(X[tr])
        Z_iv = apply_standardize(X[va], mean_it, std_it)
        mu = float(y[tr].mean())
        yc = y[tr] - mu
        G_it = genomic_relationship(Z_it, Z_it, p)
        G_vt = genomic_relationship(Z_iv, Z_it, p)
        d, V = np.linalg.eigh(G_it)
        d = np.clip(d, 0.0, None)
        Vt_yc = V.T @ yc
        GvV = G_vt @ V
        y_va = y[va]
        for j, a in enumerate(alpha_grid):
            c = Vt_yc / (d + a)
            pred = mu + GvV @ c
            sse[j] += float(np.sum((y_va - pred) ** 2))

    return float(alpha_grid[int(np.argmin(sse))]), label


def lambda_via_reml(
    G_TS: np.ndarray,
    y: np.ndarray,
    *,
    log10_bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[float, float, float]:
    """lambda_0 por REML. Devuelve (lambda_0, sigma_u2, sigma_e2).

    Para el modelo mixto y = 1*mu + g + e con g ~ N(0, sigma_u2 G) y
    e ~ N(0, sigma_e2 I), rota a un problema diagonal via G = V D V' y maximiza la
    verosimilitud restringida respecto al ratio lambda = sigma_e2 / sigma_u2,
    perfilando mu (GLS) y sigma_u2.
    """
    from scipy.optimize import minimize_scalar

    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    d, V = np.linalg.eigh(G_TS)
    d = np.clip(d, 0.0, None)
    xi = V.T @ y
    w = V.T @ np.ones(n)

    def neg_restricted_loglik(log10_lambda: float) -> float:
        lam = 10.0 ** log10_lambda
        v = d + lam
        inv_v = 1.0 / v
        a = float(np.sum(w * w * inv_v))
        mu = float(np.sum(w * xi * inv_v) / a)
        r = xi - mu * w
        s2 = float(np.sum(r * r * inv_v) / (n - 1))
        return float(np.sum(np.log(v)) + np.log(a) + (n - 1) * np.log(s2))

    res = minimize_scalar(
        neg_restricted_loglik, bounds=log10_bounds, method="bounded"
    )
    lam = 10.0 ** float(res.x)
    v = d + lam
    inv_v = 1.0 / v
    a = float(np.sum(w * w * inv_v))
    mu = float(np.sum(w * xi * inv_v) / a)
    r = xi - mu * w
    sigma_u2 = float(np.sum(r * r * inv_v) / (n - 1))
    sigma_e2 = lam * sigma_u2
    return lam, sigma_u2, sigma_e2



class GBLUP(BaseModel):
    """GBLUP con lambda_0 por validación cruzada (Ridge) o por REML.

    params (sub-dict model.params del config):
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

        Z_tr, self.col_mean_, self.col_std_ = standardize_train(X)
        self.p_: int = Z_tr.shape[1]

        G_TS = genomic_relationship(Z_tr, Z_tr, self.p_)

        self.y_mean_: float = float(y.mean())
        y_c = y - self.y_mean_

        forced = self.params.get("lambda_0")
        if forced is not None:
            self.lambda_0_ = float(forced)
            self.lambda_method_ = "fixed"
        else:
            method = str(self.params.get("lambda_method", "cv")).lower()
            self.lambda_method_ = method
            if method == "cv":
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

        n_TS = G_TS.shape[0]
        A = G_TS + self.lambda_0_ * np.identity(n_TS)
        c = np.linalg.solve(A, y_c)

        self.marker_effects_ = (Z_tr.T @ c) / self.p_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "marker_effects_"):
            raise RuntimeError("GBLUP.predict llamado antes de fit.")
        Z_ps = apply_standardize(X, self.col_mean_, self.col_std_)
        u_hat = Z_ps @ self.marker_effects_
        return self.y_mean_ + u_hat

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
