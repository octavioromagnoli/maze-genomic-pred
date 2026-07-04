"""Soft Mixture of Experts multitask: backbone compartido + K cabezas con gate blando.

Camino alternativo al `moe_multitask` (gating PCA+GMM con ruteo *hard top-1* que **parte el
training** por cluster y deja cada expert con hambre de datos). Acá el aprendizaje del MoE-GMM
previo se aplica al revés:

1. **Backbone compartido (ve TODOS los datos).** Reusa el tronco de la MTL (`_MLPNet` sin su
   `Linear` final): produce un embedding ``h`` por línea, dim = ``hidden_dims[-1]``. Nada se
   particiona → se preserva el beneficio multitarea y no hay hambre de datos.
2. **K cabezas de experto sobre h.** Cada experto es una cabeza multitarea (``Linear(h, n_targets)``,
   opcionalmente con una capa oculta) → mantiene la estructura multitask DENTRO de cada experto.
3. **Gate blando.** Una red chica ``gate(h) → softmax(logits / τ)`` sobre K expertos. La
   predicción es la **mezcla** ``Σ_k g_k(h) · expert_k(h)`` (no un argmax duro).

Anti-colapso (lección del MoE-GMM): dos mecanismos para que la mezcla sea real y no degenere a
hard top-1, ambos configurables —

- **Temperatura τ** en el softmax del gate (``gate_tau``; >1 = mezcla más blanda).
- **Load-balancing loss estilo Switch** (Shazeer 2017 / Fedus 2021): ``L_lb = K · Σ_k f_k · P_k``
  con ``P_k`` = peso blando medio del experto k por batch (con gradiente) y ``f_k`` = fracción de
  muestras cuyo argmax es k (detached). Vale 1 con balance perfecto y crece hasta K con colapso.
  Se suma a la pérdida con coeficiente ``lb_coef``.

Se loguea POR ÉPOCA (vía ``history_`` → ``curve/*`` en wandb): entropía media del gate
(``gate_entropy``, y normalizada ``gate_entropy_norm`` ∈ [0,1]), carga por experto
(``gate_load_k``, ``gate_load_max``/``min``) y la ``lb_loss``, para verificar que la mezcla no
degenera a hard.

Implementación: **subclase de `MLPMultitaskRegressor`** (reusa estandarización train-only,
buffers ``y_mean``/``y_std``, `fit`, `predict`, `save`/`load`, monitor-val por época); overridea
solo el armado de red, `_train` (loss con LB + diagnósticos del gate) y los defaults/params.
Se enchufa en `run_multitask_evaluation` igual que `mlp_multitask` (interfaz
``fit(X, Y, Y_mask)`` / ``predict(X) -> (n, T)``); el dispatch de `scripts/train.py` lo incluye
por nombre. Entrenamiento end-to-end, misma loss multitarea y mismos traits que la MTL.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.eval.metrics import pearson_r
from src.models.mlp import _ACTIVATIONS, _MLPNet
from src.models.mlp_multitask import MLPMultitaskRegressor, masked_weighted_pearson_loss

# Defaults: arquitectura del tronco (idéntica a la MTL) + bloque soft-MoE. `_resolve_cfg`
# filtra `model.params` por estas claves, así que sirven de contrato del config.
_DEFAULTS: dict[str, Any] = {
    # --- tronco compartido (mismo bloque que MLPMultitaskRegressor) ---
    "hidden_dims": [128],
    "dropout": 0.5,
    "input_dropout": 0.2,
    "activation": "relu",
    "batchnorm": False,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "loss": "pearson",         # "pearson" (1 - r por trait) | "pearson_mse"
    "mse_weight": 0.3,
    "patience": None,          # sin early stopping (debe quedar null, igual que la MTL v1)
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",
    "loss_weights": {},        # {trait: w}; default 1.0 para el trait ausente del dict
    # --- soft mixture of experts ---
    "n_experts": 4,            # K expertos que el gate mezcla (probar 2, 4, 8)
    "gate_tau": 1.0,           # temperatura del softmax del gate (>1 = mezcla más blanda)
    "lb_coef": 0.01,           # coef de la load-balancing loss (Switch: K·Σ f_k·P_k)
    "gate_hidden_dim": None,   # None => gate lineal Linear(h, K); int => 1 capa oculta
    "expert_hidden_dim": None, # None => cada experto es Linear(h, n_targets); int => 1 capa oculta
}


def switch_load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Load-balancing loss estilo Switch/Shazeer: ``K · Σ_k f_k · P_k``.

    ``gate_weights`` ``(B, K)`` son los pesos softmax del gate. ``P_k = mean_batch g_k`` es el
    peso blando medio del experto k (diferenciable, lleva el gradiente); ``f_k`` es la fracción de
    muestras del batch cuyo argmax es k (detached, sin gradiente, como en la formulación original).
    Vale 1 con balance perfecto (``f_k = P_k = 1/K``) y crece hasta K con colapso total a un
    experto, de modo que minimizarla empuja la mezcla hacia el balance.
    """
    B, K = gate_weights.shape
    P = gate_weights.mean(dim=0)                                    # (K,) con gradiente
    hard = torch.argmax(gate_weights, dim=1)                        # (B,) sin gradiente
    f = torch.bincount(hard, minlength=K).to(gate_weights.dtype) / B  # (K,) sin gradiente
    return K * torch.sum(f * P)


class _SoftMoENet(nn.Module):
    """Backbone compartido (tronco `_MLPNet`) + K cabezas de experto + gate blando.

    ``forward(x)`` devuelve la mezcla ``Σ_k g_k(h) · expert_k(h)`` con shape ``(n, n_targets)``;
    ``forward(x, return_gate=True)`` devuelve además los pesos del gate ``g`` ``(n, K)`` para la
    load-balancing loss y los diagnósticos. Los mu/sigma del target se registran como buffers
    desde `fit` (los agrega el `fit` heredado), fuera de este módulo.
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
        n_targets: int,
        n_experts: int,
        tau: float,
        gate_hidden_dim: int | None = None,
        expert_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError(
                "_SoftMoENet requiere hidden_dims no vacío: el embedding compartido h es hidden_dims[-1]."
            )
        self.n_targets = int(n_targets)
        self.n_experts = int(n_experts)
        self.tau = float(tau)

        # Tronco compartido = `_MLPNet` SIN su Linear final (reusa EXACTAMENTE el bloque de la MTL:
        # [Dropout entrada] -> bloques (Linear[, BN], act, Dropout)). El embedding h tiene
        # dim = hidden_dims[-1]. El Linear(hidden_dims[-1], 1) que arma _MLPNet se descarta (no se
        # registra en ningún submódulo), así el state_dict no arrastra parámetros muertos.
        base = _MLPNet(
            in_dim,
            list(hidden_dims),
            dropout=dropout,
            input_dropout=input_dropout,
            activation=activation,
            batchnorm=batchnorm,
            out_dim=1,
        )
        self.backbone = nn.Sequential(*list(base.net.children())[:-1])
        self.h_dim = int(hidden_dims[-1])
        act_cls = _ACTIVATIONS[str(activation).lower()]

        def _make_head(out_dim: int) -> nn.Module:
            if expert_hidden_dim:
                return nn.Sequential(
                    nn.Linear(self.h_dim, int(expert_hidden_dim)),
                    act_cls(),
                    nn.Linear(int(expert_hidden_dim), out_dim),
                )
            return nn.Linear(self.h_dim, out_dim)

        # K cabezas multitarea (cada una predice los n_targets sobre el MISMO h).
        self.experts = nn.ModuleList([_make_head(self.n_targets) for _ in range(self.n_experts)])

        # Gate blando: h -> logits sobre K expertos.
        if gate_hidden_dim:
            self.gate = nn.Sequential(
                nn.Linear(self.h_dim, int(gate_hidden_dim)),
                act_cls(),
                nn.Linear(int(gate_hidden_dim), self.n_experts),
            )
        else:
            self.gate = nn.Linear(self.h_dim, self.n_experts)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def gate_weights(self, h: torch.Tensor) -> torch.Tensor:
        """Pesos blandos del gate: softmax(logits / τ) sobre los K expertos."""
        return torch.softmax(self.gate(h) / self.tau, dim=-1)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        h = self.backbone(x)
        g = self.gate_weights(h)                                  # (B, K)
        experts = torch.stack([e(h) for e in self.experts], dim=1)  # (B, K, T)
        pred = (g.unsqueeze(-1) * experts).sum(dim=1)             # (B, T)  mezcla blanda
        if return_gate:
            return pred, g
        return pred


class SoftMoEMultitaskRegressor(MLPMultitaskRegressor):
    """Soft MoE multitask: tronco compartido + K cabezas de experto mezcladas por un gate blando.

    Subclase de `MLPMultitaskRegressor`: hereda la estandarización train-only (regla 1), los
    buffers ``y_mean``/``y_std``, `fit`/`predict`, `save`/`load` (reconstruyen la red vía el
    `_build_net` polimórfico + ``cfg_``) y el monitor-val por época. Overridea el armado de red,
    los defaults/params y el loop de entrenamiento (agrega la load-balancing loss y los
    diagnósticos del gate).

    `params` (sub-dict ``model.params`` del config) — además de los de `mlp_multitask`:
        n_experts         : K expertos que el gate mezcla (default 4).
        gate_tau          : temperatura del softmax del gate (default 1.0; >1 = más blando).
        lb_coef           : coeficiente de la load-balancing loss (default 0.01).
        gate_hidden_dim   : None => gate lineal; int => una capa oculta en el gate.
        expert_hidden_dim : None => cada experto Linear(h, T); int => una capa oculta por experto.
    """

    # --- Resolución de hiperparámetros (extiende la MTL con el bloque soft-MoE) ---

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["hidden_dims"] = [int(h) for h in cfg["hidden_dims"]]
        if not cfg["hidden_dims"]:
            raise ValueError(
                "moe_soft_multitask requiere hidden_dims no vacío (el embedding compartido es hidden_dims[-1])."
            )
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["loss"] = str(cfg["loss"]).lower()
        cfg["mse_weight"] = float(cfg["mse_weight"])
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["patience"] = None if cfg["patience"] is None else int(cfg["patience"])
        cfg["loss_weights"] = dict(cfg["loss_weights"] or {})
        # --- soft-MoE ---
        cfg["n_experts"] = int(cfg["n_experts"])
        if cfg["n_experts"] < 1:
            raise ValueError("n_experts debe ser >= 1.")
        cfg["gate_tau"] = float(cfg["gate_tau"])
        if cfg["gate_tau"] <= 0:
            raise ValueError("gate_tau debe ser > 0.")
        cfg["lb_coef"] = float(cfg["lb_coef"])
        cfg["gate_hidden_dim"] = None if not cfg["gate_hidden_dim"] else int(cfg["gate_hidden_dim"])
        cfg["expert_hidden_dim"] = None if not cfg["expert_hidden_dim"] else int(cfg["expert_hidden_dim"])
        return cfg

    # --- Construcción de red (reemplaza el tronco+cabeza única de la MTL) ---

    def _build_net(self, in_dim: int, out_dim: int, cfg: dict[str, Any]) -> _SoftMoENet:
        return _SoftMoENet(
            in_dim,
            cfg["hidden_dims"],
            dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"],
            activation=cfg["activation"],
            batchnorm=cfg["batchnorm"],
            n_targets=out_dim,
            n_experts=cfg["n_experts"],
            tau=cfg["gate_tau"],
            gate_hidden_dim=cfg["gate_hidden_dim"],
            expert_hidden_dim=cfg["expert_hidden_dim"],
        )

    # --- Diagnósticos del gate (para el log por época) ---

    def _gate_diagnostics(self, g: torch.Tensor) -> dict[str, float]:
        """Entropía media del gate y carga por experto sobre un batch de pesos ``g`` ``(N, K)``.

        ``gate_entropy`` en nats; ``gate_entropy_norm`` = entropía / log(K) ∈ [0, 1] (1 = mezcla
        uniforme). ``gate_load_k`` = peso blando medio del experto k (la "fracción de carga");
        ``gate_load_max``/``min`` resumen el (des)balance. Todos numéricos ⇒ se loguean por época.
        """
        K = int(g.shape[1])
        eps = 1e-9
        entropy = float((-(g * (g + eps).log()).sum(dim=1).mean()).item())
        load = g.mean(dim=0)                                        # (K,)
        out: dict[str, float] = {
            "gate_entropy": entropy,
            "gate_entropy_norm": entropy / float(np.log(K)) if K > 1 else 0.0,
            "gate_load_max": float(load.max().item()),
            "gate_load_min": float(load.min().item()),
        }
        for k in range(K):
            out[f"gate_load_{k}"] = float(load[k].item())
        return out

    # --- Loop de entrenamiento (loss multitarea + LB + diagnósticos del gate) ---

    def _train(self, Xs, Ys, M, *, cfg, device) -> list[dict[str, float]]:
        net = self.net_
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=device)
        Yt = torch.as_tensor(Ys, dtype=torch.float32, device=device)
        Mt = torch.as_tensor(M, dtype=torch.bool, device=device)
        w = torch.as_tensor(self.loss_weights_, dtype=torch.float32, device=device)
        mse_w = cfg["mse_weight"] if cfg["loss"] == "pearson_mse" else 0.0
        lb_coef = float(cfg["lb_coef"])

        n = Xt.shape[0]
        bs = cfg["batch_size"]
        full_batch = bs <= 0 or bs >= n
        mon = self._monitor_val(device)

        history: list[dict[str, float]] = []
        for epoch in range(int(cfg["max_epochs"])):
            net.train()
            if full_batch:
                batches = [torch.arange(n, device=device)]
            else:
                perm = torch.randperm(n, device=device)
                batches = list(torch.split(perm, bs))
            run_loss = 0.0
            run_lb = 0.0
            seen = 0
            for idx in batches:
                optim.zero_grad()
                pred, g = net(Xt[idx], return_gate=True)          # (B, T), (B, K)
                task_loss = masked_weighted_pearson_loss(pred, Yt[idx], Mt[idx], w, mse_weight=mse_w)
                # LB solo con >=2 muestras (con 1 el argmax/P son triviales); si no, término 0.
                lb = switch_load_balance_loss(g) if g.shape[0] >= 2 else pred.new_zeros(())
                loss = task_loss + lb_coef * lb
                if loss.requires_grad:   # algún trait tuvo >=2 válidos (o lb_coef>0)
                    loss.backward()
                    optim.step()
                run_loss += float(task_loss.item()) * idx.numel()
                run_lb += float(lb.item()) * idx.numel()
                seen += int(idx.numel())
            rec: dict[str, float] = {
                "epoch": epoch,
                "train_loss": run_loss / max(seen, 1),
                "lb_loss": run_lb / max(seen, 1),
            }
            # Diagnósticos del gate sobre TODO el train (eval => sin dropout, determinista).
            net.eval()
            with torch.no_grad():
                g_full = net(Xt, return_gate=True)[1]              # (N, K)
                rec.update(self._gate_diagnostics(g_full))
            # Curva de val externa del fold por época (Pearson por trait; sin selección => sin leakage).
            if mon is not None:
                with torch.no_grad():
                    Xv, Yv_np, Mv_np = mon
                    pv = self._forward2d(Xv).detach().cpu().numpy()
                for j, name in enumerate(self.target_names_):
                    colv = Mv_np[:, j]
                    if colv.sum() >= 2:
                        rec[f"val_pearson_{name}"] = pearson_r(Yv_np[colv, j], pv[colv, j])
            history.append(rec)
        return history

    # --- Inspección de la mezcla (diagnóstico / notebooks) ---

    def gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Pesos blandos del gate ``(n, K)`` para ``X`` crudo (la mezcla por muestra).

        Aplica el z-score train-only de features y devuelve el softmax del gate. Utilidad de
        diagnóstico (verificar que la mezcla no colapsa a un experto); no interviene en `predict`.
        """
        if not hasattr(self, "net_"):
            raise RuntimeError("gate_weights llamado antes de fit.")
        device = next(self.net_.parameters()).device
        Xs = (np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_std_
        self.net_.eval()
        with torch.no_grad():
            h = self.net_.embed(torch.as_tensor(Xs, dtype=torch.float32, device=device))
            g = self.net_.gate_weights(h)
        return g.detach().cpu().numpy().astype(float)

    # --- Estimaciones para el log (numéricas van a run.summary) ---

    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "net_"):
            return {}
        last = self.history_[-1] if getattr(self, "history_", None) else {}
        params: dict[str, Any] = {
            "moe_soft": 1,
            "n_experts": int(self.cfg_["n_experts"]),
            "gate_tau": float(self.cfg_["gate_tau"]),
            "lb_coef": float(self.cfg_["lb_coef"]),
            "n_params": getattr(self, "n_params_", None),
            "n_targets": getattr(self, "n_targets_", None),
            "hidden_dims": str(self.cfg_["hidden_dims"]),   # no-numérico => metadata
            "loss_weights": str(
                {n: float(wv) for n, wv in zip(self.target_names_, self.loss_weights_)}
            ),
            "target_names": str(self.target_names_),
        }
        # Estado final del gate (última época) => run.summary, para chequear que no colapsó.
        for key in ("gate_entropy", "gate_entropy_norm", "gate_load_max", "gate_load_min"):
            if key in last:
                params[f"final_{key}"] = float(last[key])
        return params
