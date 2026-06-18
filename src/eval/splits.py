"""Splits temporales con líneas disjuntas + guard antileakage.
"""
from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    """Se levanta cuando una línea aparece en train y val a la vez."""


def _as_int_idx(x) -> np.ndarray:
    return np.asarray(x, dtype=int).ravel()


def enforce_disjoint_lines(
    train_idx,
    val_idx,
    line_id,
    *,
    drop_from: str = "val",
) -> tuple[np.ndarray, np.ndarray]:
    """Hace disjuntas las líneas entre train y val descartando las que cruzan.
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


def fixed_temporal_split(
    year,
    line_id,
    *,
    val_year: int | None = None,
    holdout_test_year: int | None = None,
    train_years: list[int] | None = None,
    enforce: str = "val",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split temporal

    ``train`` = todos los años de dev anteriores a ``val_year``; ``val`` = el
    año ``val_year``. El ``holdout_test_year`` se excluye de ambos (test final
    intocable). Si ``val_year`` es ``None`` se usa el año más reciente del dev.

    ``train_years`` (opcional) fija explícitamente los años de train, en vez de
    "todos los anteriores a ``val_year``". Sirve para experimentos puntuales
    (p.ej. entrenar solo 2018 → predecir 2020). Se intersecta con los años
    disponibles y se restringe a años ESTRICTAMENTE anteriores a ``val_year``
    (nunca se mete un año >= val en train; no hay leak temporal hacia el train).

    Aplica `enforce_disjoint_lines` para respetar la regla de líneas disjuntas:
    con ``enforce='val'`` el set de validación queda solo con líneas NO vistas
    en train.

    Devuelve una lista con un solo fold ``[(train_idx, val_idx)]`` para encajar
    sin casos especiales en el loop de `run_evaluation`. Devuelve ``[]`` si no
    hay al menos un año de train anterior a ``val_year``.
    """
    year = np.asarray(year)
    uniq = sorted(int(y) for y in np.unique(year))
    if holdout_test_year is not None:
        uniq = [y for y in uniq if y != int(holdout_test_year)]

    vy = uniq[-1] if val_year is None else int(val_year)
    if vy not in uniq:
        return []
    if train_years is not None:
        requested = {int(y) for y in train_years}
        sel_years = [y for y in uniq if y < vy and y in requested]
    else:
        sel_years = [y for y in uniq if y < vy]
    if not sel_years:
        return []

    train_idx = np.flatnonzero(np.isin(year, sel_years))
    val_idx = np.flatnonzero(year == vy)
    tr, va = enforce_disjoint_lines(train_idx, val_idx, line_id, drop_from=enforce)
    return [(tr, va)]


def expanding_temporal_split(
    year,
    line_id,
    *,
    val_years: list[int],
    holdout_test_year: int | None = None,
    enforce: str = "val",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación temporal *expanding window* (forward chaining).

    Por cada año en ``val_years`` arma un fold cuyo train son todos los años de
    dev ESTRICTAMENTE anteriores (excluyendo ``holdout_test_year``): p.ej. con
    ``val_years=[2018, 2019]`` da [train 2017 -> val 2018, train 2017+2018 -> val
    2019]. Reusa `fixed_temporal_split` (líneas disjuntas + exclusión del holdout
    por fold), así no se reimplementa el split. Los años sin train anterior se
    saltean (fold vacío). Devuelve la lista concatenada ``[(train_idx, val_idx)]``.
    """
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for vy in val_years:
        folds += fixed_temporal_split(
            year, line_id, val_year=int(vy), holdout_test_year=holdout_test_year, enforce=enforce
        )
    return folds


def final_test_split(year, test_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Separa el año de test final (intocable) del resto (dev)."""
    year = np.asarray(year)
    test_idx = np.flatnonzero(year == int(test_year))
    dev_idx = np.flatnonzero(year != int(test_year))
    return dev_idx, test_idx


def assert_no_line_leakage(train_idx, val_idx, line_id) -> None:
    """lanza `LeakageError` si una línea está en train y val.
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
