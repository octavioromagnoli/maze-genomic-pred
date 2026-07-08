"""Seeding reproducible y centralizado."""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las seeds de los RNG relevantes de forma determinística.

    Cubre random, numpy y, si está disponible, torch (CPU y CUDA). No
    devuelve nada: es un efecto de lado intencional al inicio de cada corrida.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def make_rng(seed: int) -> np.random.Generator:
    """Devuelve un Generator de numpy con seed explícita.

    Preferido frente al RNG global cuando se necesita aleatoriedad local y
    reproducible (p. ej. generación de datos sintéticos).
    """
    return np.random.default_rng(int(seed))
