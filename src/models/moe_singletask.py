"""Mixture of experts single-task con gating PCA+GMM (ruteo hard top-1).

Versión single-task del moe_multitask: mismo gating (PCA + GMM, hard top-1,
min_cluster_size y fallback global) pero cada experto es una red single-task (un
MLPRegressor que predice un fenotipo) en vez de una red multitarea con tronco
compartido. Corre por el harness single-task (run_evaluation), elegido por
data.target y data.env. Sirve para el A/B de si el sharing multitarea aporta sobre
modelar un ambiente con expertos independientes.

Cada experto es un _SingleTaskEnsemble (n_models MLPRegressor con seeds distintas,
promedio z-scoreado por modelo, mismo criterio que EnsembleMultitaskRegressor). PCA y
GMM se ajustan con el train del fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocess import PCAReducer, PreprocChain, SNPPreprocessor
from src.models.base import BaseModel
from src.models.mlp import MLPRegressor

_GATING_DEFAULTS: dict[str, Any] = {
    "pca_n_components": 50,
    "standardize": True,
    "whiten": True,
    "k": 3,
    "covariance_type": "full",
    "reg_covar": 1e-6,
    "n_init": 1,
    "min_cluster_size": 150,
    "global_fallback": True,
    "random_state": 575,
}


class _SingleTaskEnsemble(BaseModel):
    """Ensemble de n_models MLPRegressor single-task (promedio z-scoreado por modelo).

    params: inner (params del MLPRegressor), n_models (default 5), seed_step (default
    1000). Cada modelo se entrena con seed random_state + i*seed_step; sus predicciones
    se estandarizan con los estadísticos de train de cada modelo antes de promediar.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.inner_params: dict[str, Any] = dict(self.params.get("inner", {}) or {})
        self.n_models: int = int(self.params.get("n_models", 5))
        self.seed_step: int = int(self.params.get("seed_step", 1000))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SingleTaskEnsemble":
        ctx = self._fit_context
        base_seed = int(self.inner_params.get("random_state", 575))
        self.models_: list[MLPRegressor] = []
        self.scalers_: list[tuple[float, float]] = []
        for i in range(self.n_models):
            p = dict(self.inner_params)
            p["random_state"] = base_seed + i * self.seed_step
            m = MLPRegressor({"params": p})
            m.set_fit_context(year=ctx.get("year"), line_id=ctx.get("line_id"))
            m.fit(X, y)
            ptr = np.asarray(m.predict(X), dtype=float).ravel()
            mu = float(ptr.mean())
            sd = float(ptr.std())
            sd = sd if sd > 1e-12 else 1.0
            self.models_.append(m)
            self.scalers_.append((mu, sd))
        self.history_ = getattr(self.models_[0], "history_", [])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "models_", None):
            raise RuntimeError("_SingleTaskEnsemble.predict llamado antes de fit.")
        acc: np.ndarray | None = None
        for m, (mu, sd) in zip(self.models_, self.scalers_):
            z = (np.asarray(m.predict(X), dtype=float).ravel() - mu) / sd
            acc = z if acc is None else acc + z
        return acc / len(self.models_)

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "models_", None):
            return {}
        first = self.models_[0]
        return {
            "ensemble": 1,
            "n_models": int(self.n_models),
            "n_params": int(getattr(first, "n_params_", 0) or 0) * self.n_models,
        }

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        for i, m in enumerate(self.models_):
            fname = f"submodel_{i}.pt"
            m.save(path.parent / fname)
            files.append(fname)
        joblib.dump(
            {"config": self.config, "n_models": self.n_models, "seed_step": self.seed_step,
             "files": files, "scalers": self.scalers_},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "_SingleTaskEnsemble":
        import joblib

        path = Path(path)
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.n_models = int(d["n_models"])
        obj.seed_step = int(d["seed_step"])
        obj.models_ = [MLPRegressor.load(path.parent / f) for f in d["files"]]
        obj.scalers_ = list(d["scalers"])
        obj.history_ = []
        return obj


class MixtureOfExpertsSingleTaskRegressor(BaseModel):
    """Mixture of experts SINGLE-TASK con gating PCA+GMM y ruteo hard top-1.

    params (sub-dict model.params del config):
        gating : dict con los defaults de _GATING_DEFAULTS (PCA + GMM + salvaguardas p>>n).
        expert : dict de params de _SingleTaskEnsemble (n_models, seed_step e inner con
                 la arquitectura/loss del MLPRegressor). Es lo que comparten todos los expertos.

    Interfaz single-task: fit(X, y) / predict(X) -> (n,). Se elige por data.target/data.env.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.gating_params: dict[str, Any] = {**_GATING_DEFAULTS, **dict(self.params.get("gating", {}) or {})}
        self.expert_params: dict[str, Any] = dict(self.params.get("expert", {}) or {})


    def _build_gating(self) -> Any:
        g = self.gating_params
        steps: list = []
        if bool(g["standardize"]):
            steps.append(SNPPreprocessor(standardize=True))
        steps.append(
            PCAReducer(
                n_components=int(g["pca_n_components"]),
                whiten=bool(g["whiten"]),
                svd_solver="full",
                random_state=int(g["random_state"]),
            )
        )
        return steps[0] if len(steps) == 1 else PreprocChain(steps)

    def _make_expert(self) -> _SingleTaskEnsemble:
        return _SingleTaskEnsemble({"params": dict(self.expert_params)})


    def fit(self, X: np.ndarray, y: np.ndarray) -> "MixtureOfExpertsSingleTaskRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("MixtureOfExpertsSingleTaskRegressor.fit recibió y vacío.")

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k

        self.pre_ = self._build_gating()
        Z = self.pre_.fit_transform(X)
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(Z)
        labels = self.gmm_.predict(Z)

        year = ctx.get("year")
        line_id = ctx.get("line_id")
        min_size = int(g["min_cluster_size"])

        self.experts_: dict[int, _SingleTaskEnsemble] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            self.cluster_sizes_[c] = n_c
            if n_c < min_size:
                continue
            expert = self._make_expert()
            expert.set_fit_context(
                year=None if year is None else np.asarray(year)[mask],
                line_id=None if line_id is None else np.asarray(line_id)[mask],
            )
            expert.fit(X[mask], y[mask])
            self.experts_[c] = expert

        self.fallback_: _SingleTaskEnsemble | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < k):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id)
            fb.fit(X, y)
            self.fallback_ = fb

        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("MixtureOfExpertsSingleTaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)
        Z = self.pre_.transform(X)
        clusters = self.gmm_.predict(Z)
        out = np.empty(X.shape[0], dtype=float)
        for c in np.unique(clusters):
            rows = np.flatnonzero(clusters == c)
            expert = self.experts_.get(int(c), self.fallback_)
            if expert is None:
                raise RuntimeError(
                    f"cluster {int(c)} sin experto ni fallback; activá gating.global_fallback."
                )
            out[rows] = np.asarray(expert.predict(X[rows]), dtype=float).ravel()
        return out

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "gmm_"):
            return {}
        params: dict[str, Any] = {
            "moe_singletask": 1,
            "k": int(self.k_),
            "n_experts": int(len(self.experts_)),
            "global_fallback_used": int(self.fallback_ is not None),
            "min_cluster_size": int(self.gating_params["min_cluster_size"]),
        }
        for c, s in self.cluster_sizes_.items():
            params[f"cluster_size_{c}"] = int(s)
        gfp = getattr(self.pre_, "get_fitted_params", None)
        if callable(gfp):
            params.update(gfp())
        n_params = sum(int(e.get_fitted_params().get("n_params", 0) or 0) for e in self.experts_.values())
        if self.fallback_ is not None:
            n_params += int(self.fallback_.get_fitted_params().get("n_params", 0) or 0)
        params["n_params"] = int(n_params)
        return params


    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        base = path.parent
        base.mkdir(parents=True, exist_ok=True)
        expert_files: dict[int, str] = {}
        for c, e in self.experts_.items():
            sub = base / f"expert_{c}"
            sub.mkdir(parents=True, exist_ok=True)
            e.save(sub / "ensemble.joblib")
            expert_files[c] = f"expert_{c}/ensemble.joblib"
        fallback_file: str | None = None
        if self.fallback_ is not None:
            sub = base / "fallback"
            sub.mkdir(parents=True, exist_ok=True)
            self.fallback_.save(sub / "ensemble.joblib")
            fallback_file = "fallback/ensemble.joblib"
        joblib.dump(
            {
                "config": self.config,
                "gating_params": self.gating_params,
                "expert_params": self.expert_params,
                "pre_": self.pre_,
                "gmm_": self.gmm_,
                "k_": self.k_,
                "cluster_sizes_": self.cluster_sizes_,
                "expert_files": expert_files,
                "fallback_file": fallback_file,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MixtureOfExpertsSingleTaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.gating_params = d["gating_params"]
        obj.expert_params = d["expert_params"]
        obj.pre_ = d["pre_"]
        obj.gmm_ = d["gmm_"]
        obj.k_ = int(d["k_"])
        obj.cluster_sizes_ = {int(c): int(s) for c, s in d["cluster_sizes_"].items()}
        obj.experts_ = {int(c): _SingleTaskEnsemble.load(base / f) for c, f in d["expert_files"].items()}
        obj.fallback_ = (
            _SingleTaskEnsemble.load(base / d["fallback_file"]) if d["fallback_file"] else None
        )
        obj.history_ = []
        return obj
