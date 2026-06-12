"""Carga directa de los CSV de datos. Sin validaciones ni contrato.

Se asume que los CSV están bien formados y siempre disponibles en `data/`:

- `Geno_data.csv`: columna `GID` + 4612 columnas SNP binarias (0/1).
- `Pheno_data.csv`: formato *wide* — `GID`, `Year` y una columna por
  trait×ambiente: `GY_Optimal`, `PH_Optimal`, `AD_Optimal`, `GY_Drought`,
  `PH_Drought`, `AD_Drought` (celdas vacías = faltante).

`to_modeling_arrays` joinea genotipo↔fenotipo por `GID` y arma los arrays que
consumen los splits (`src/eval/splits.py`) y el loop (`src/training/run.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGETS = ("GY", "PH", "AD")
ENVIRONMENTS = ("optimal", "drought")


@dataclass
class ModelingArrays:
    """Arrays alineados 1:1 por observación."""

    X: np.ndarray        # (n_obs, n_snps) genotipo por observación
    y: np.ndarray        # (n_obs,) target
    year: np.ndarray     # (n_obs,) año (split temporal)
    line_id: np.ndarray  # (n_obs,) GID (agrupar por línea)
    env: np.ndarray      # (n_obs,) ambiente


@dataclass
class Dataset:
    """Genotipos por línea (GID) + fenotipos wide, tal cual vienen del CSV."""

    G: np.ndarray            # (L, n_snps)
    gid_index: np.ndarray    # (L,) GID de cada fila de G
    pheno: pd.DataFrame      # wide (GID, Year, <TRAIT>_<Optimal|Drought>...)

    @property
    def n_lines(self) -> int:
        return self.G.shape[0]

    def _block(self, target: str, env: str):
        """Arrays para un (target, ambiente): selecciona la columna wide y joinea."""
        col = f"{target}_{env.title()}"  # p.ej. GY_Optimal / GY_Drought
        sub = self.pheno[["GID", "Year", col]].dropna(subset=[col])
        row_of_gid = {g: i for i, g in enumerate(self.gid_index)}
        rows = sub["GID"].map(row_of_gid)
        sub = sub[rows.notna()]
        rows = rows[rows.notna()].astype(int).to_numpy()
        return (
            self.G[rows],
            sub[col].to_numpy(dtype=float),
            sub["Year"].to_numpy(dtype=int),
            sub["GID"].to_numpy(),
            np.full(len(sub), env, dtype=object),
        )

    def to_modeling_arrays(self, target: str = "GY", env: str | None = None) -> ModelingArrays:
        """Arma los arrays de modelado.

        `env` en {'optimal','drought'} usa ese ambiente; `None` apila ambos como
        observaciones separadas (el GID puede aparecer en los dos).
        """
        envs = [env] if env else list(ENVIRONMENTS)
        blocks = [self._block(target, e) for e in envs]
        return ModelingArrays(
            X=np.concatenate([b[0] for b in blocks]),
            y=np.concatenate([b[1] for b in blocks]),
            year=np.concatenate([b[2] for b in blocks]),
            line_id=np.concatenate([b[3] for b in blocks]),
            env=np.concatenate([b[4] for b in blocks]),
        )


def load_dataset(geno_csv: str, pheno_csv: str) -> Dataset:
    """Lee los dos CSV y devuelve el `Dataset`"""
    geno = pd.read_csv(geno_csv)
    gid_index = geno["GID"].astype(str).to_numpy()
    G = geno.drop(columns=["GID"]).to_numpy(dtype=np.int8)

    pheno = pd.read_csv(pheno_csv)
    pheno["GID"] = pheno["GID"].astype(str)

    return Dataset(G=G, gid_index=gid_index, pheno=pheno)
