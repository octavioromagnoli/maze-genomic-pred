"""MoE con ruteo guiado por ambiente (GxE): PCA(SNPs) concatenado con el ambiente.

Igual que moe_multitask pero la compuerta no decide solo con el PCA del genoma:
concatena las componentes del PCA con un one-hot del ambiente (óptimo/sequía, escalado
por env_weight) y ajusta el GMM sobre ese espacio ampliado. Así el ruteo puede
bifurcar por ambiente: una misma línea puede ir a un experto para su predicción en
óptimo y a otro para sequía.

El formato es multi-ambiente (una fila por línea, columnas trait×ambiente; col_env
marca el ambiente de cada columna, 0=óptimo, 1=sequía) y el ruteo es dual:
- Entrenamiento: se arma un punto por (línea, ambiente) = [PCA(línea) || onehot*env_weight];
  el GMM clusteriza los 2n puntos y cada experto entrena con las líneas de su cluster,
  con la máscara de Y restringida a las columnas del ambiente ruteado.
- Inferencia: para cada ambiente e se rutea la línea con su punto correspondiente y sus
  columnas de ese ambiente se predicen con el experto del cluster resultante.

Los expertos ven los SNPs completos; PCA y GMM se ajustan con el train. Es una subclase
de MixtureOfExpertsMultitaskRegressor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.ensemble_multitask import EnsembleMultitaskRegressor
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor


class MoEGxEMultitaskRegressor(MixtureOfExpertsMultitaskRegressor):
    """MoE multi-ambiente con gating guiado por ambiente (PCA(SNPs) || ambiente) + ruteo dual.

    En params.gating agrega env_weight (default 5.0): cuánto pesa el one-hot de ambiente
    frente a las componentes del PCA whitened en el espacio de ruteo.
    """

    def _env_weight(self) -> float:
        return float(self.gating_params.get("env_weight", 5.0))

    @staticmethod
    def _cols_env_from_names(target_names: list[str]) -> np.ndarray:
        """Ambiente de cada columna: 0 = óptimo, 1 = sequía (por el sufijo del nombre de cabeza)."""
        return np.array([0 if "optimal" in str(n).lower() else 1 for n in target_names], dtype=int)

    def _augment(self, P: np.ndarray, e: int) -> np.ndarray:
        """Concatena las componentes del PCA con el one-hot del ambiente e escalado por env_weight."""
        oh = np.zeros((P.shape[0], len(self.envs_)), dtype=float)
        oh[:, self.envs_.index(int(e))] = self._env_weight()
        return np.concatenate([np.asarray(P, dtype=float), oh], axis=1)

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MoEGxEMultitaskRegressor":
        from sklearn.mixture import GaussianMixture

        ctx = self._fit_context
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y_mask = np.ones_like(Y, dtype=bool) if Y_mask is None else np.asarray(Y_mask, dtype=bool)
        self.n_targets_ = int(Y.shape[1])
        n = X.shape[0]

        target_names = ctx.get("target_names") or [f"t{j}" for j in range(self.n_targets_)]
        self.col_env_ = self._cols_env_from_names(target_names)
        self.envs_ = sorted(set(int(e) for e in self.col_env_))

        g = self.gating_params
        k = int(g["k"])
        self.k_ = k
        year = ctx.get("year")
        line_id = ctx.get("line_id")
        min_size = int(g["min_cluster_size"])

        self.pre_ = self._build_gating()
        P = self.pre_.fit_transform(X)
        A = np.concatenate([self._augment(P, e) for e in self.envs_], axis=0)
        pt_line = np.tile(np.arange(n), len(self.envs_))
        pt_env = np.repeat(self.envs_, n)
        self.gmm_ = GaussianMixture(
            n_components=k,
            covariance_type=str(g["covariance_type"]),
            reg_covar=float(g["reg_covar"]),
            n_init=int(g["n_init"]),
            random_state=int(g["random_state"]),
        )
        self.gmm_.fit(A)
        labels = self.gmm_.predict(A)

        self.experts_: dict[int, EnsembleMultitaskRegressor] = {}
        self.cluster_sizes_: dict[int, int] = {}
        for c in range(k):
            pts = np.flatnonzero(labels == c)
            lines_c = np.unique(pt_line[pts])
            self.cluster_sizes_[c] = int(lines_c.size)
            if lines_c.size < min_size:
                continue
            idx_map = {int(ln): i for i, ln in enumerate(lines_c)}
            mask_c = np.zeros((lines_c.size, self.n_targets_), dtype=bool)
            for i in pts:
                ln = int(pt_line[i]); e = int(pt_env[i])
                cols = np.flatnonzero(self.col_env_ == e)
                mask_c[idx_map[ln], cols] = Y_mask[ln, cols]
            expert = self._make_expert()
            expert.set_fit_context(
                year=None if year is None else np.asarray(year)[lines_c],
                line_id=None if line_id is None else np.asarray(line_id)[lines_c],
                target_names=target_names,
            )
            expert.fit(X[lines_c], Y[lines_c], mask_c)
            self.experts_[c] = expert

        self.fallback_: EnsembleMultitaskRegressor | None = None
        if bool(g["global_fallback"]) or (len(self.experts_) < k):
            fb = self._make_expert()
            fb.set_fit_context(year=year, line_id=line_id, target_names=target_names)
            fb.fit(X, Y, Y_mask)
            self.fallback_ = fb

        first = next(iter(self.experts_.values()), None) or self.fallback_
        self.history_ = list(getattr(first, "history_", []) or []) if first is not None else []
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("MoEGxEMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)
        P = self.pre_.transform(X)
        out = np.zeros((X.shape[0], self.n_targets_), dtype=float)
        cache: dict[int, np.ndarray] = {}
        for e in self.envs_:
            clusters = self.gmm_.predict(self._augment(P, e))
            cols_e = np.flatnonzero(self.col_env_ == e)
            for c in np.unique(clusters):
                rows = np.flatnonzero(clusters == c)
                expert = self.experts_.get(int(c), self.fallback_)
                if expert is None:
                    raise RuntimeError(f"cluster {int(c)} sin experto ni fallback; activá global_fallback.")
                key = id(expert)
                if key not in cache:
                    cache[key] = self._as_2d(expert.predict(X))
                out[np.ix_(rows, cols_e)] = cache[key][np.ix_(rows, cols_e)]
        return out

    def get_fitted_params(self) -> dict[str, Any]:
        out = dict(super().get_fitted_params())
        if out:
            out["moe_gxe"] = 1
            out["env_weight"] = self._env_weight()
        return out


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
                "n_targets_": self.n_targets_,
                "col_env_": self.col_env_,
                "envs_": self.envs_,
                "cluster_sizes_": self.cluster_sizes_,
                "expert_files": expert_files,
                "fallback_file": fallback_file,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoEGxEMultitaskRegressor":
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
        obj.n_targets_ = int(d["n_targets_"])
        obj.col_env_ = np.asarray(d["col_env_"], dtype=int)
        obj.envs_ = [int(e) for e in d["envs_"]]
        obj.cluster_sizes_ = {int(c): int(s) for c, s in d["cluster_sizes_"].items()}
        obj.experts_ = {int(c): EnsembleMultitaskRegressor.load(base / f) for c, f in d["expert_files"].items()}
        obj.fallback_ = EnsembleMultitaskRegressor.load(base / d["fallback_file"]) if d["fallback_file"] else None
        obj.history_ = []
        return obj
