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
    """Arrays alineados 1:1 por observación.

    Los campos multitask ``Y``/``Y_mask`` son opcionales (default ``None``): solo
    los puebla `to_modeling_arrays_multitask`. La ruta single-task (`y` 1D) no los usa.
    """

    X: np.ndarray        # (n_obs, n_snps) genotipo por observación
    y: np.ndarray        # (n_obs,) target
    year: np.ndarray     # (n_obs,) año (split temporal)
    line_id: np.ndarray  # (n_obs,) GID (agrupar por línea)
    env: np.ndarray      # (n_obs,) ambiente
    Y: np.ndarray | None = None       # (n_obs, n_targets) multitask, faltantes -> 0
    Y_mask: np.ndarray | None = None  # (n_obs, n_targets) bool: True = dato real


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

    def _block_multitask(self, targets: list[str], env: str):
        """Arrays multitask para un ambiente: varias columnas wide + máscara + join.

        Espeja `_block`, pero conserva una fila por (GID, Year) si **al menos un**
        trait fue medido (no exige todos): los faltantes se enmascaran. `Y_mask`
        marca True donde había dato real (antes de rellenar los NaN con 0).
        """
        cols = [f"{t}_{env.title()}" for t in targets]  # p.ej. [GY_Optimal, PH_Optimal, AD_Optimal]
        sub = self.pheno[["GID", "Year", *cols]]
        sub = sub[sub[cols].notna().any(axis=1)]  # fila viva si hay >=1 trait medido
        row_of_gid = {g: i for i, g in enumerate(self.gid_index)}
        rows = sub["GID"].map(row_of_gid)
        sub = sub[rows.notna()]
        rows = rows[rows.notna()].astype(int).to_numpy()
        Y_raw = sub[cols].to_numpy(dtype=float)       # NaN donde el trait no se midió
        Y_mask = ~np.isnan(Y_raw)                     # True = dato real (antes del fillna)
        Y = np.nan_to_num(Y_raw, nan=0.0)             # faltantes -> 0 (evita NaN en tensores)
        return (
            self.G[rows],
            Y,
            Y_mask,
            sub["Year"].to_numpy(dtype=int),
            sub["GID"].to_numpy(),
            np.full(len(sub), env, dtype=object),
        )

    def to_modeling_arrays_multitask(
        self, targets: list[str], env: str | None = None
    ) -> ModelingArrays:
        """Arma los arrays de modelado multitask (varios traits a la vez).

        `Y` queda con shape ``(n_obs, len(targets))`` en el orden de `targets`, con
        los faltantes rellenados con 0; `Y_mask` (misma forma, bool) marca los datos
        reales. `env` se comporta igual que en `to_modeling_arrays`: un ambiente, o
        ``None`` apila ambos como observaciones separadas. `y` se setea al primer
        target (compat incidental con código que espera el 1D).
        """
        envs = [env] if env else list(ENVIRONMENTS)
        blocks = [self._block_multitask(targets, e) for e in envs]
        Y = np.concatenate([b[1] for b in blocks])
        return ModelingArrays(
            X=np.concatenate([b[0] for b in blocks]),
            y=Y[:, 0],
            year=np.concatenate([b[3] for b in blocks]),
            line_id=np.concatenate([b[4] for b in blocks]),
            env=np.concatenate([b[5] for b in blocks]),
            Y=Y,
            Y_mask=np.concatenate([b[2] for b in blocks]),
        )


    def to_modeling_arrays_multienv(
        self, target: str = "GY", envs: list[str] | tuple[str, ...] = ENVIRONMENTS
    ) -> ModelingArrays:
        """Arma arrays multitask **multi-AMBIENTE**: un trait, varios ambientes como cabezas.

        A diferencia de `to_modeling_arrays_multitask` (varios traits en UN ambiente,
        o ambos ambientes apilados como observaciones separadas), acá hay UNA fila por
        ``(GID, Year)`` y las columnas de ``Y`` son el MISMO trait en cada ambiente
        (p.ej. ``[GY_Optimal, GY_Drought]``). Así una sola red comparte el tronco
        genómico entre ambientes y una cabeza por ambiente especializa: el tronco se
        estima con los datos de TODOS los ambientes (≈2x filas) y las cabezas no se
        promedian entre ambientes débilmente correlacionados.

        Conserva la fila si **al menos un** ambiente fue medido (los faltantes se
        enmascaran). `Y` queda ``(n_obs, len(envs))`` en el orden de ``envs`` con los
        faltantes en 0; `Y_mask` (bool, misma forma) marca los datos reales. `y` se
        setea a la primera columna (compat con código que espera el 1D). Los nombres de
        cabeza para alinear `loss_weights` son ``[f"{target}_{env}" for env in envs]``.
        """
        envs = list(envs)
        cols = [f"{target}_{e.title()}" for e in envs]  # p.ej. [GY_Optimal, GY_Drought]
        sub = self.pheno[["GID", "Year", *cols]]
        sub = sub[sub[cols].notna().any(axis=1)]  # fila viva si hay >=1 ambiente medido
        row_of_gid = {g: i for i, g in enumerate(self.gid_index)}
        rows = sub["GID"].map(row_of_gid)
        sub = sub[rows.notna()]
        rows = rows[rows.notna()].astype(int).to_numpy()
        Y_raw = sub[cols].to_numpy(dtype=float)       # NaN donde ese ambiente no se midió
        Y_mask = ~np.isnan(Y_raw)                     # True = dato real (antes del fillna)
        Y = np.nan_to_num(Y_raw, nan=0.0)             # faltantes -> 0 (evita NaN en tensores)
        return ModelingArrays(
            X=self.G[rows],
            y=Y[:, 0],
            year=sub["Year"].to_numpy(dtype=int),
            line_id=sub["GID"].to_numpy(),
            env=np.full(len(sub), "multienv", dtype=object),
            Y=Y,
            Y_mask=Y_mask,
        )


    def to_modeling_arrays_multi_trait_env(
        self,
        targets: list[str] | tuple[str, ...] = TARGETS,
        envs: list[str] | tuple[str, ...] = ENVIRONMENTS,
    ) -> ModelingArrays:
        """Arma arrays multitask con TODAS las combinaciones trait×ambiente como cabezas.

        Generaliza `to_modeling_arrays_multienv` (un trait, varios ambientes) al caso de varios
        traits Y varios ambientes: las columnas de ``Y`` son cada ``(trait, ambiente)``, en orden
        trait-mayor ``[f"{t}_{e}" for t in targets for e in envs]`` (p.ej. con
        ``targets=[GY,PH,AD]``/``envs=[optimal,drought]`` ⇒ 6 cabezas: GY_optimal, GY_drought,
        PH_optimal, PH_drought, AD_optimal, AD_drought). Hay UNA fila por ``(GID, Year)`` y el tronco
        genómico se comparte entre las 6 tareas; cada cabeza especializa.

        Conserva la fila si **al menos una** de las columnas fue medida (los faltantes se enmascaran).
        `Y` queda ``(n_obs, len(targets)*len(envs))`` con los faltantes en 0; `Y_mask` (bool, misma
        forma) marca los datos reales. `y` se setea a la primera columna (compat con el 1D). Los
        nombres de cabeza para alinear `loss_weights`/métricas son ``[f"{t}_{e}" ...]``.
        """
        targets = list(targets)
        envs = list(envs)
        cols = [f"{t}_{e.title()}" for t in targets for e in envs]  # wide: GY_Optimal, GY_Drought, ...
        sub = self.pheno[["GID", "Year", *cols]]
        sub = sub[sub[cols].notna().any(axis=1)]  # fila viva si hay >=1 trait×ambiente medido
        row_of_gid = {g: i for i, g in enumerate(self.gid_index)}
        rows = sub["GID"].map(row_of_gid)
        sub = sub[rows.notna()]
        rows = rows[rows.notna()].astype(int).to_numpy()
        Y_raw = sub[cols].to_numpy(dtype=float)       # NaN donde esa combinación no se midió
        Y_mask = ~np.isnan(Y_raw)                     # True = dato real (antes del fillna)
        Y = np.nan_to_num(Y_raw, nan=0.0)             # faltantes -> 0 (evita NaN en tensores)
        return ModelingArrays(
            X=self.G[rows],
            y=Y[:, 0],
            year=sub["Year"].to_numpy(dtype=int),
            line_id=sub["GID"].to_numpy(),
            env=np.full(len(sub), "multi_trait_env", dtype=object),
            Y=Y,
            Y_mask=Y_mask,
        )


def load_dataset(geno_csv: str, pheno_csv: str) -> Dataset:
    """Lee los dos CSV y devuelve el `Dataset`"""
    geno = pd.read_csv(geno_csv)
    gid_index = geno["GID"].astype(str).to_numpy()
    G = geno.drop(columns=["GID"]).to_numpy(dtype=np.int8)

    pheno = pd.read_csv(pheno_csv)
    pheno["GID"] = pheno["GID"].astype(str)

    return Dataset(G=G, gid_index=gid_index, pheno=pheno)
