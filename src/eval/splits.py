"""Splits temporales y agrupados por línea.

"""
from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    """Se levanta cuando una línea aparece en train y val a la vez."""


def _as_int_idx(x) -> np.ndarray:
    return np.asarray(x, dtype=int).ravel()


def temporal_forward_chaining_folds(
    year,
    *,
    holdout_test_year: int | None = None,
    min_train_years: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining con ventana expansiva sobre años ordenados.

    Para años ordenados ``[y0, y1, ..., yK]`` (excluyendo `holdout_test_year`),
    genera folds::

        train = {y0..y_{k-1}},  val = {y_k}    para k = min_train_years .. K

    No hay aleatoriedad: el orden es cronológico. Devuelve ``[]`` si no hay años
    suficientes para formar al menos un fold.
    """
    year = np.asarray(year)
    uniq = sorted(int(y) for y in np.unique(year))
    if holdout_test_year is not None:
        uniq = [y for y in uniq if y != int(holdout_test_year)]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(int(min_train_years), len(uniq)):
        train_years = set(uniq[:k])
        val_year = uniq[k]
        train_idx = np.flatnonzero(np.isin(year, list(train_years)))
        val_idx = np.flatnonzero(year == val_year)
        folds.append((train_idx, val_idx))
    return folds


def enforce_disjoint_lines(
    train_idx,
    val_idx,
    line_id,
    *,
    drop_from: str = "val",
) -> tuple[np.ndarray, np.ndarray]:
    """Hace disjuntas las líneas entre train y val descartando las que cruzan.

    `drop_from='val'` (default) descarta de val las líneas ya vistas en train
    → escenario realista de predecir **líneas nuevas** (estilo CV1 de la
    literatura de genomic selection). `drop_from='train'` hace lo simétrico.
    """
    train_idx, val_idx = _as_int_idx(train_idx), _as_int_idx(val_idx)
    line_id = np.asarray(line_id)
    train_lines = set(line_id[train_idx].tolist())
    val_lines = set(line_id[val_idx].tolist())

    if drop_from == "val":
        keep = ~np.isin(line_id[val_idx], list(train_lines)) if train_lines else np.ones(len(val_idx), bool)
        val_idx = val_idx[keep]
    elif drop_from == "train":
        keep = ~np.isin(line_id[train_idx], list(val_lines)) if val_lines else np.ones(len(train_idx), bool)
        train_idx = train_idx[keep]
    else:
        raise ValueError("drop_from debe ser 'val' o 'train'.")
    return train_idx, val_idx


def temporal_grouped_folds(
    year,
    line_id,
    *,
    holdout_test_year: int | None = None,
    min_train_years: int = 1,
    enforce: str = "val",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split del proyecto: forward-chaining temporal + líneas disjuntas.

    Compone `temporal_forward_chaining_folds` con `enforce_disjoint_lines`, de
    modo que cada fold respeta a la vez la regla temporal y la de agrupar por
    línea. Es el splitter que usa `scripts/train.py`.
    """
    folds = temporal_forward_chaining_folds(
        year, holdout_test_year=holdout_test_year, min_train_years=min_train_years
    )
    out = []
    for tr, va in folds:
        tr, va = enforce_disjoint_lines(tr, va, line_id, drop_from=enforce)
        out.append((tr, va))
    return out


def grouped_by_line_folds(
    line_id,
    n_splits: int,
    *,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold agrupado por línea, determinístico (sin partir líneas).

    No es aleatorio: las líneas únicas se ordenan y se reparten round-robin a los
    folds. `seed` opcional permuta el orden de líneas de forma reproducible (se
    registra en el config); por defecto es puramente cronológico/lexicográfico.

    Útil como baseline agrupado (p. ej. para comparar contra el split temporal),
    NO reemplaza la validación temporal de la regla 3.
    """
    line_id = np.asarray(line_id)
    uniq = np.array(sorted(set(line_id.tolist())))
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        uniq = uniq[rng.permutation(len(uniq))]
    if n_splits < 2 or n_splits > len(uniq):
        raise ValueError(f"n_splits={n_splits} inválido para {len(uniq)} líneas únicas.")

    fold_of_line = {lid: i % n_splits for i, lid in enumerate(uniq)}
    assign = np.array([fold_of_line[l] for l in line_id])

    folds = []
    for f in range(n_splits):
        val_idx = np.flatnonzero(assign == f)
        train_idx = np.flatnonzero(assign != f)
        folds.append((train_idx, val_idx))
    return folds


def final_test_split(year, test_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Separa el año de test final (intocable) del resto (dev)."""
    year = np.asarray(year)
    test_idx = np.flatnonzero(year == int(test_year))
    dev_idx = np.flatnonzero(year != int(test_year))
    return dev_idx, test_idx


def assert_no_line_leakage(train_idx, val_idx, line_id) -> None:
    """Guard duro: lanza `LeakageError` si una línea está en train y val.

    Se llama en cada fold dentro del loop de entrenamiento. Es la red de
    seguridad que vuelve imposible saltarse la regla 2, sin importar qué
    splitter se haya usado.
    """
    train_idx, val_idx = _as_int_idx(train_idx), _as_int_idx(val_idx)
    line_id = np.asarray(line_id)
    overlap = set(line_id[train_idx].tolist()) & set(line_id[val_idx].tolist())
    if overlap:
        ejemplo = list(overlap)[:5]
        raise LeakageError(
            f"Leakage por línea: {len(overlap)} line_id aparecen en train y val "
            f"(p. ej. {ejemplo}). Revisá el splitter / enforce_disjoint_lines."
        )
