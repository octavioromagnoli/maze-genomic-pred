"""MoE multitask con ENSEMBLE SOBRE PARTICIONES: promedia K gatings GMM independientes.

Motivación (blindaje 2026-07-06, ver `docs/robustez_benchmark_6pheno.md`): el MoE hard de una sola
partición (`moe_multitask`) es **frágil a la semilla** — su GY_drought 2019 varía 0.14–0.41 según el
`random_state` del GMM, porque el ruteo *hard top-1* sobre p≫n con clusters chicos es de alta varianza, y
el ensemble de seeds del experto **no** promedia esa fuente de varianza (solo la de las redes, condicionada
a UNA partición). El sweep original "buscando robustez" no la cubría.

Este modelo ataca esa fuente: entrena **K MoE hard** con `gating.random_state` distintos (⇒ K particiones
GMM distintas, cada una con sus expertos) y **promedia las predicciones** (z-scoreadas por columna con stats
de train, misma lógica anti-dominancia-de-escala que `EnsembleMultitaskRegressor`). Reduce la varianza del
ruteo ~1/√K. Pregunta empírica: ¿converge a un valor ALTO robusto (el gating sí aporta) o a ~0.29 = nivel del
ensemble sin gating (el gating no aporta señal, solo costo)?

Se enchufa en `run_multitask_evaluation` igual que `moe_multitask` (interfaz `fit(X, Y, Y_mask)` /
`predict(X) -> (n, T)`); reusa `MixtureOfExpertsMultitaskRegressor` por partición sin reimplementar el gating.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseModel
from src.models.moe_multitask import MixtureOfExpertsMultitaskRegressor

_DEFAULTS = {
    "n_partitions": 5,          # K particiones GMM a promediar (la palanca de robustez)
    "partition_seed_step": 101, # gating.random_state_k = base + k*step (semillas GMM bien separadas)
}


class MoEPartitionEnsembleMultitaskRegressor(BaseModel):
    """Promedio de ``n_partitions`` `MixtureOfExpertsMultitaskRegressor` con gatings GMM distintos.

    `params` (sub-dict ``model.params``):
        n_partitions        : K, cantidad de particiones GMM a ensamblar (default 5).
        partition_seed_step : separación entre semillas de gating (default 101).
        gating, expert      : idénticos a `moe_multitask`; se comparten entre las K particiones salvo
                              `gating.random_state`, que se corre por partición.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.n_partitions: int = int(self.params.get("n_partitions", _DEFAULTS["n_partitions"]))
        self.partition_seed_step: int = int(
            self.params.get("partition_seed_step", _DEFAULTS["partition_seed_step"])
        )
        self._base_gating_seed: int = int((self.params.get("gating", {}) or {}).get("random_state", 575))

    # --- Construcción de cada partición ------------------------------------

    def _make_partition(self, k: int) -> MixtureOfExpertsMultitaskRegressor:
        p = copy.deepcopy(self.params)
        p.pop("n_partitions", None)
        p.pop("partition_seed_step", None)
        gating = dict(p.get("gating", {}) or {})
        gating["random_state"] = self._base_gating_seed + k * self.partition_seed_step
        p["gating"] = gating
        return MixtureOfExpertsMultitaskRegressor({"params": p})

    @staticmethod
    def _as_2d(a: Any) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return a.reshape(-1, 1) if a.ndim == 1 else a

    # --- Interfaz BaseModel -------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray, Y_mask: np.ndarray | None = None) -> "MoEPartitionEnsembleMultitaskRegressor":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_targets_ = int(Y.shape[1])

        self.partitions_: list[MixtureOfExpertsMultitaskRegressor] = []
        self.scalers_: list[tuple[np.ndarray, np.ndarray]] = []  # (mu, sd) por columna, de las preds de train
        for k in range(self.n_partitions):
            moe = self._make_partition(k)
            moe.set_fit_context(**self._fit_context)   # year/line_id/target_names/eval_* al MoE interno
            moe.fit(X, Y, Y_mask)
            ptr = self._as_2d(moe.predict(X))           # preds de TRAIN => stats para z-score (sin tocar val)
            mu = ptr.mean(axis=0, keepdims=True)
            sd = ptr.std(axis=0, keepdims=True)
            sd[sd < 1e-12] = 1.0
            self.partitions_.append(moe)
            self.scalers_.append((mu, sd))

        # Learning curve representativa (primera partición) para que run.py loguee algo.
        self.history_ = list(getattr(self.partitions_[0], "history_", []) or [])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "partitions_", None):
            raise RuntimeError("MoEPartitionEnsembleMultitaskRegressor.predict llamado antes de fit.")
        X = np.asarray(X, dtype=np.float64)
        acc = np.zeros((X.shape[0], self.n_targets_), dtype=float)
        for moe, (mu, sd) in zip(self.partitions_, self.scalers_):
            acc += (self._as_2d(moe.predict(X)) - mu) / sd   # z-score por partición antes de promediar
        return acc / len(self.partitions_)

    def get_fitted_params(self) -> dict[str, Any]:
        if not getattr(self, "partitions_", None):
            return {}
        n_experts = [len(m.experts_) for m in self.partitions_]
        params: dict[str, Any] = {
            "moe_partens": 1,
            "n_partitions": int(self.n_partitions),
            "k": int(self.partitions_[0].k_),
            "n_experts_mean": float(np.mean(n_experts)),
            "n_targets": int(self.n_targets_),
            "n_params": int(sum(int(m.get_fitted_params().get("n_params", 0) or 0) for m in self.partitions_)),
        }
        return params

    # --- Persistencia (cada partición en su subdir) -------------------------

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        base = path.parent
        base.mkdir(parents=True, exist_ok=True)
        part_files: list[str] = []
        for k, m in enumerate(self.partitions_):
            sub = base / f"partition_{k}"
            sub.mkdir(parents=True, exist_ok=True)
            m.save(sub / "moe.joblib")
            part_files.append(f"partition_{k}/moe.joblib")
        joblib.dump(
            {
                "config": self.config,
                "n_partitions": self.n_partitions,
                "partition_seed_step": self.partition_seed_step,
                "n_targets_": self.n_targets_,
                "scalers_": self.scalers_,
                "part_files": part_files,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoEPartitionEnsembleMultitaskRegressor":
        import joblib

        path = Path(path)
        base = path.parent
        d = joblib.load(path)
        obj = cls(d.get("config"))
        obj.n_partitions = int(d["n_partitions"])
        obj.partition_seed_step = int(d["partition_seed_step"])
        obj.n_targets_ = int(d["n_targets_"])
        obj.scalers_ = list(d["scalers_"])
        obj.partitions_ = [MixtureOfExpertsMultitaskRegressor.load(base / f) for f in d["part_files"]]
        obj.history_ = []
        return obj
