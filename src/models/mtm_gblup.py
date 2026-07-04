"""Multi-trait GBLUP (MTM) — GBLUP multivariado que comparte señal genética entre traits.

Extiende `GBLUP` (single-trait, `src/models/gblup.py`) al caso **multi-trait**: un solo
modelo mixto para GY+PH+AD con una **matriz de covarianza genética** ``G0`` (T×T) y una
**residual** ``R0`` (T×T) estimadas por REML. La correlación genética entre traits deja que
los datos de un trait auxiliar (p.ej. PH, altura de planta) **afinen la estimación de los
efectos genéticos de la portada (GY)** — *borrowing* pleiotrópico. Es el análogo aditivo y
principista del MLP multitask: NO apela a no-linealidad (respeta el techo aditivo de GY en
líneas endocriadas), sino a la covarianza genética entre traits.

Encaja en la MISMA interfaz multitask que `mlp_multitask`/`ensemble_multitask`
(``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``) y en `run_multitask_evaluation`. La
portada (GY) sale como una columna de la predicción; las cabezas auxiliares se predicen igual
(cada una recibe sus métricas por trait).

Decisiones alineadas con las reglas del proyecto:
- **Sin leakage (regla 1).** ``G0``, ``R0``, los efectos de marcador y el centrado/escalado de
  SNPs se estiman SOLO con el train del fold; ``predict`` reescala el val con los estadísticos
  de train (idéntico a `GBLUP`). Determinista: sin seeds ni sorteo (a diferencia de la red).
- **Val solo-genotipo (comparación justa con GBLUP).** Las cabezas auxiliares NO se observan en
  val; el *borrowing* ocurre en la ESTIMACIÓN (train), no usando fenotipos auxiliares del val.

Método (REML por EM sobre la eigen-descomposición de ``G_train``): con ``G = V D V'`` el modelo
``Y = 1μ' + U + E`` (``U ~ N(0, G0 ⊗ G)``, ``E ~ N(0, R0 ⊗ I)``) se rota a ``Ỹ = V'Y`` cuyas
filas son T-variadas **independientes** con ``Ũ_i ~ N(0, d_i G0)``, ``Ẽ_i ~ N(0, R0)``. El EM
alterna: E-step (posterior de ``Ũ_i``) y M-step (``G0``, ``R0`` como medias de los momentos).
Predicción: BLUP de los valores de cría ``Û`` (back-transform) → efectos de marcador
``B = Zᵀ G⁻¹ Û / p`` (representación rrBLUP), y ``GEBV_val = Z_val B + μ`` (un solo matmul,
como `GBLUP`; no se guarda la matriz de train).

**Requiere datos completos por fila** (todos los traits medidos): en `optimal` se cumple
(GY=PH=AD por observación). Si ``Y_mask`` trae faltantes, se levanta un error claro (la
eigen-rotación ``V'Y`` no admite huecos; usar el MLP multitask enmascarado para ese caso).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.gblup import apply_standardize, genomic_relationship, standardize_train


def _em_mtm(
    Y: np.ndarray, G: np.ndarray, *, n_iter: int, tol: float, ridge: float
) -> dict[str, Any]:
    """EM-REML multivariado sobre la eigen-descomposición de ``G``.

    ``Y`` ``(n, T)`` crudo (centrado internamente vía μ por GLS). Devuelve ``G0``, ``R0``,
    ``mu``, los valores de cría de train ``Uhat`` ``(n, T)`` y la eigen-base ``(V, d)`` de ``G``.
    """
    n, t = Y.shape
    d, V = np.linalg.eigh(G)
    d = np.clip(d, 1e-8, None)
    Yt = V.T @ Y              # (n, T) en eigen-espacio
    w = V.T @ np.ones(n)      # intercepto rotado

    # Init: reparte la covarianza fenotípica mitad genética / mitad residual.
    C = np.cov(Y.T) if t > 1 else np.array([[float(np.var(Y))]])
    G0 = 0.5 * C + ridge * np.eye(t)
    R0 = 0.5 * C + ridge * np.eye(t)
    mu = Y.mean(axis=0).astype(float)

    prev = None
    it = 0
    for it in range(n_iter):
        # μ por GLS en eigen-espacio: Ỹ_i ~ N(w_i μ, d_i G0 + R0).
        Vinv = np.empty((n, t, t))
        A = np.zeros((t, t))
        b = np.zeros(t)
        for i in range(n):
            Vi = d[i] * G0 + R0
            Vii = np.linalg.inv(Vi)
            Vinv[i] = Vii
            A += (w[i] ** 2) * Vii
            b += w[i] * (Vii @ Yt[i])
        mu = np.linalg.solve(A, b)

        # E-step + acumuladores del M-step.
        S_u = np.zeros((t, t))
        S_e = np.zeros((t, t))
        for i in range(n):
            Vii = Vinv[i]
            r = Yt[i] - w[i] * mu
            dG = d[i] * G0
            Uhat_i = dG @ Vii @ r                    # posterior mean de Ũ_i
            Ucov_i = dG - dG @ Vii @ dG              # posterior cov de Ũ_i
            S_u += (np.outer(Uhat_i, Uhat_i) + Ucov_i) / d[i]
            Ehat_i = r - Uhat_i
            S_e += np.outer(Ehat_i, Ehat_i) + Ucov_i
        G0 = 0.5 * (S_u / n + (S_u / n).T)
        R0 = 0.5 * (S_e / n + (S_e / n).T)

        if prev is not None:
            delta = np.abs(G0 - prev[0]).max() + np.abs(R0 - prev[1]).max()
            if delta < tol:
                break
        prev = (G0.copy(), R0.copy())

    # BLUP de los valores de cría de train (back-transform del posterior).
    Uhat_eigen = np.empty((n, t))
    for i in range(n):
        Vi = d[i] * G0 + R0
        Uhat_eigen[i] = (d[i] * G0) @ np.linalg.inv(Vi) @ (Yt[i] - w[i] * mu)
    Uhat = V @ Uhat_eigen
    return {"G0": G0, "R0": R0, "mu": mu, "Uhat": Uhat, "V": V, "d": d, "n_iter": it + 1}


class MTMGBLUPRegressor(BaseModel):
    """GBLUP multi-trait (borrowing pleiotrópico) con ``G0``/``R0`` por REML-EM.

    `params` (sub-dict ``model.params`` del config):
        n_iter : máx. iteraciones EM (default 300).
        tol    : tolerancia de convergencia sobre ``G0``/``R0`` (default 1e-7).
        ridge  : regularización diagonal del init de ``G0``/``R0`` (estabilidad; default 1e-3).

    `fit(X, Y, Y_mask)` recibe ``Y`` ``(n, T)`` (portada = primer trait por convención del
    loader, GY) e ``Y_mask`` ``(n, T)`` bool; requiere filas completas. Los nombres de trait
    llegan por `set_fit_context(target_names=...)`. `predict(X)` devuelve ``(n, T)``.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MTMGBLUPRegressor":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y_mask is not None and not np.asarray(Y_mask, dtype=bool).all():
            raise ValueError(
                "MTMGBLUPRegressor requiere datos completos por fila (todos los traits "
                "medidos); Y_mask trae faltantes. Usá mlp_multitask (pérdida enmascarada) "
                "para ese caso, o filtrá a filas completas."
            )
        self.n_targets_ = int(Y.shape[1])
        names = self._fit_context.get("target_names")
        self.target_names_ = (
            [str(m) for m in names] if names is not None else [f"t{j}" for j in range(self.n_targets_)]
        )

        # Estandarización de marcadores con estadísticos de TRAIN (regla 1).
        Z, self.col_mean_, self.col_std_ = standardize_train(X)
        self.p_ = int(Z.shape[1])
        G = genomic_relationship(Z, Z, self.p_)

        fit = _em_mtm(
            Y, G,
            n_iter=int(self.params.get("n_iter", 300)),
            tol=float(self.params.get("tol", 1e-7)),
            ridge=float(self.params.get("ridge", 1e-3)),
        )
        self.G0_, self.R0_, self.mu_ = fit["G0"], fit["R0"], fit["mu"]
        self.n_iter_run_ = fit["n_iter"]

        # Efectos de marcador (rrBLUP): B = Zᵀ G⁻¹ Û / p, con G⁻¹ Û vía la eigen-base.
        V, dd, Uhat = fit["V"], fit["d"], fit["Uhat"]
        Ginv_U = V @ ((V.T @ Uhat) / dd[:, None])       # (n, T)
        self.marker_effects_ = (Z.T @ Ginv_U) / self.p_  # (p, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "marker_effects_"):
            raise RuntimeError("MTMGBLUPRegressor.predict llamado antes de fit.")
        Z = apply_standardize(X, self.col_mean_, self.col_std_)
        return self.mu_ + Z @ self.marker_effects_       # (n, T)

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "G0_"):
            return {}
        out: dict[str, Any] = {"n_iter_run": int(getattr(self, "n_iter_run_", 0))}
        diag = np.sqrt(np.clip(np.diag(self.G0_), 1e-12, None))
        rdiag = np.clip(np.diag(self.R0_), 1e-12, None)
        for a in range(self.n_targets_):
            na = self.target_names_[a]
            out[f"h2_{na}"] = float(self.G0_[a, a] / (self.G0_[a, a] + rdiag[a]))
            for b in range(a + 1, self.n_targets_):
                nb = self.target_names_[b]
                out[f"gcorr_{na}_{nb}"] = float(self.G0_[a, b] / (diag[a] * diag[b]))
        return out

    # --- Persistencia: liviana (efectos de marcador + escalas), sin la matriz de train ---

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "marker_effects_": self.marker_effects_,
                "mu_": self.mu_,
                "col_mean_": self.col_mean_,
                "col_std_": self.col_std_,
                "G0_": self.G0_,
                "R0_": self.R0_,
                "n_targets_": self.n_targets_,
                "target_names_": list(self.target_names_),
                "n_iter_run_": getattr(self, "n_iter_run_", None),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MTMGBLUPRegressor":
        import joblib

        d = joblib.load(Path(path))
        obj = cls(d.get("config"))
        obj.marker_effects_ = d["marker_effects_"]
        obj.mu_ = d["mu_"]
        obj.col_mean_ = d["col_mean_"]
        obj.col_std_ = d["col_std_"]
        obj.G0_ = d["G0_"]
        obj.R0_ = d["R0_"]
        obj.n_targets_ = int(d["n_targets_"])
        obj.target_names_ = list(d["target_names_"])
        obj.n_iter_run_ = d.get("n_iter_run_")
        return obj
