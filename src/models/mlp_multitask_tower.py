"""MLP multitask con TORRES por cabeza (tower networks): tronco compartido + una sub-red por salida.

Extiende `MLPMultitaskRegressor` (tronco compartido + un único ``Linear(h, T)`` que predice todas las
cabezas de forma plana) dándole a **cada salida su propia "torre"**: ``Linear(h, tower_hidden) → act →
Linear(tower_hidden, 1)``. El tronco modela la no-linealidad compartida (estructura familiar); las
torres especializan cada trait/ambiente para que **los gradientes del estrés (sequía) no se mezclen con
los del riego óptimo** en una capa de salida común. Para GY multi-ambiente (cabezas GY_optimal,
GY_drought) una torre por cabeza = una torre por ambiente, exactamente la idea de "tower networks" por
ambiente.

Solo overridea el armado de red (`_build_net` → `_TowerMLPNet`) y agrega ``tower_hidden_dim`` al cfg;
hereda estandarización train-only, loss Pearson enmascarada, `fit`/`predict`/`save`/`load` (que usan
`_build_net` polimórfico + ``cfg_``). Es subclase de `MLPMultitaskRegressor`, así sirve de ``inner`` de
`EnsembleMultitaskRegressor` (``inner_name: mlp_multitask_tower``) ⇒ se usa como experto del MoE.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.mlp import _ACTIVATIONS, _MLPNet
from src.models.mlp_multitask import MLPMultitaskRegressor


class _TowerMLPNet(nn.Module):
    """Tronco compartido (`_MLPNet` sin su Linear final) + una torre por salida.

    ``forward(x)`` devuelve ``(n, out_dim)`` concatenando la salida de cada torre. Cada torre es
    ``Linear(h, tower_hidden) → act → Linear(tower_hidden, 1)`` (o ``Linear(h, 1)`` si
    ``tower_hidden_dim<=0``). ``h = hidden_dims[-1]``. Los buffers y_mean/y_std se registran desde `fit`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        *,
        dropout: float,
        input_dropout: float,
        activation: str,
        batchnorm: bool,
        out_dim: int,
        tower_hidden_dim: int,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("_TowerMLPNet requiere hidden_dims no vacío (el tronco termina en hidden_dims[-1]).")
        # Tronco = `_MLPNet` SIN su Linear final (mismo bloque que la MTL/soft-MoE). Embedding h = hidden_dims[-1].
        base = _MLPNet(
            in_dim, list(hidden_dims), dropout=dropout, input_dropout=input_dropout,
            activation=activation, batchnorm=batchnorm, out_dim=1,
        )
        self.trunk = nn.Sequential(*list(base.net.children())[:-1])
        h = int(hidden_dims[-1])
        self.out_dim = int(out_dim)
        act_cls = _ACTIVATIONS[str(activation).lower()]
        th = int(tower_hidden_dim)

        def _tower() -> nn.Module:
            if th > 0:
                return nn.Sequential(nn.Linear(h, th), act_cls(), nn.Linear(th, 1))
            return nn.Linear(h, 1)

        self.towers = nn.ModuleList([_tower() for _ in range(self.out_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hh = self.trunk(x)
        return torch.cat([t(hh) for t in self.towers], dim=1)  # (n, out_dim)


class MLPMultitaskTowerRegressor(MLPMultitaskRegressor):
    """MLP multitask con torres por cabeza. `params`: además de los de `mlp_multitask`,
    ``tower_hidden_dim`` (default 32; ancho de la capa oculta de cada torre; <=0 => torre lineal)."""

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = super()._resolve_cfg()
        cfg["tower_hidden_dim"] = int(self.params.get("tower_hidden_dim", 32))
        return cfg

    def _build_net(self, in_dim: int, out_dim: int, cfg: dict[str, Any]) -> _TowerMLPNet:
        return _TowerMLPNet(
            in_dim,
            cfg["hidden_dims"],
            dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"],
            activation=cfg["activation"],
            batchnorm=cfg["batchnorm"],
            out_dim=out_dim,
            tower_hidden_dim=cfg["tower_hidden_dim"],
        )

    def get_fitted_params(self) -> dict[str, Any]:
        out = super().get_fitted_params()
        if out:
            out["tower_hidden_dim"] = int(self.cfg_["tower_hidden_dim"])
            out["moe_tower"] = 1
        return out
