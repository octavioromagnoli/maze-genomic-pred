"""Carga de datos: lee los CSV de `data/` tal cual, sin validaciones."""
from src.data.loader import (  # noqa: F401
    ENVIRONMENTS,
    TARGETS,
    Dataset,
    ModelingArrays,
    load_dataset,
)
