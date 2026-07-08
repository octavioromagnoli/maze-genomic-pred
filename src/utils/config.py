"""Carga de configs YAML.

Los hiperparámetros se cambian en el YAML, NUNCA editando código. El config se
carga como dict plano (serializable, va tal cual a wandb.config) y se accede
con helpers para defaults anidados.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Lee un YAML de experimento y lo devuelve como dict anidado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"El config {path} no es un mapping YAML válido.")
    return cfg


def cfg_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Acceso anidado con notación 'a.b.c' y default si falta la clave."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Setea cfg['a']['b']['c'] = value para dotted_key='a.b.c'.

    Crea los dicts intermedios que falten. Inversa de cfg_get. Lo usa
    scripts/train.py para aplicar los overrides de un sweep de wandb (que
    llegan como claves planas con punto en run.config) sobre el config anidado.
    """
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
