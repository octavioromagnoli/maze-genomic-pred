"""Protocolo experimental del paper (Lopez-Cruz et al. 2021, Tabla 1) para el SSI.

Trabaja sobre indices de un `ModelingArrays` de UN trait×ambiente (p.ej. GY drought): usa ``year``
y ``line_id`` para armar las particiones del ciclo objetivo (2020) y las composiciones de training.

Idea:
- El ciclo objetivo (2020) se parte 100 veces al azar en un **prediction set** (85% ≈ 413 individuos,
  fenotipos HELD OUT) y un pool **disponible** (15% ≈ 73, cuyos fenotipos SI pueden entrar al training).
- El **training set** se compone segun la Tabla 1: ciclos previos (single: 2017/2018/2019; acumulados:
  2018+2019, 2017–2019) MAS un 0/5/10/15% del ciclo 2020 (esas lineas salen del pool disponible).

La frontera de leakage: los fenotipos del prediction set (2020) nunca entran al training ni a la
estimacion de lambda0; los kernels se arman aparte solo con marcadores/embeddings.
"""
from __future__ import annotations

import numpy as np

# Composiciones de training de la Tabla 1 (single cycles + acumulados).
TABLE1_COMPOSITIONS: list[dict] = [
    {"name": "2017", "years": [2017]},
    {"name": "2018", "years": [2018]},
    {"name": "2019", "years": [2019]},
    {"name": "2018-2019", "years": [2018, 2019]},
    {"name": "2017-2019", "years": [2017, 2018, 2019]},
]
PCT_2020 = (0, 5, 10, 15)  # porcentaje del CICLO 2020 agregado al training (sacado del pool disponible)


def partition_target_cycle(
    year: np.ndarray, *, target_year: int = 2020, frac_predict: float = 0.85, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Parte el ciclo objetivo en (disponible, prediction set).

    Devuelve ``(idx_available, idx_predict)`` — indices ENTEROS en los arrays. ``idx_predict`` es el
    85% (held out); ``idx_available`` el 15% restante (candidato a training). Reproducible por ``seed``.
    """
    year = np.asarray(year)
    idx_cycle = np.flatnonzero(year == int(target_year))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx_cycle)
    n_pred = int(round(frac_predict * idx_cycle.size))
    return np.sort(perm[n_pred:]), np.sort(perm[:n_pred])  # (available, predict)


def compose_training(
    year: np.ndarray,
    idx_available: np.ndarray,
    *,
    cycle_years: list[int],
    pct_2020: int,
    target_year: int = 2020,
    seed: int = 0,
) -> np.ndarray:
    """Arma los indices de training: los ``cycle_years`` completos + ``pct_2020``% del ciclo objetivo.

    El ``pct`` es porcentaje del CICLO 2020 completo (protocolo del paper: "0/5/10/15% de 2020"),
    no del pool. Las lineas se sacan del pool ``idx_available`` (el 15% de 2020 fuera del prediction
    set), capando ``k`` a su tamaño: con ``frac_predict=0.85`` el pool ES el 15% de 2020, asi que
    pct=15 usa el pool entero (~73) y pct=5 ~24 lineas. Reproducible por ``seed`` (que combina la
    particion y la composicion aguas arriba).
    """
    year = np.asarray(year)
    base = np.flatnonzero(np.isin(year, [int(y) for y in cycle_years]))
    idx_available = np.asarray(idx_available, dtype=int)
    n_cycle = int(np.sum(year == int(target_year)))            # tamaño del ciclo objetivo (p.ej. 486)
    k = min(int(round((int(pct_2020) / 100.0) * n_cycle)), idx_available.size)  # % de 2020, cap al pool
    if k > 0:
        rng = np.random.default_rng(seed)
        add = rng.permutation(idx_available)[:k]
    else:
        add = np.empty(0, dtype=int)
    return np.sort(np.unique(np.concatenate([base, add]).astype(int)))


def iter_settings(compositions=None, pcts=None):
    """Itera las (composicion, pct) de la Tabla 1: ``(name, cycle_years, pct)``."""
    comps = TABLE1_COMPOSITIONS if compositions is None else compositions
    pcts = PCT_2020 if pcts is None else pcts
    for comp in comps:
        for pct in pcts:
            label = comp["name"] if pct == 0 else f"{comp['name']}+{pct}%2020"
            yield label, list(comp["years"]), int(pct)
