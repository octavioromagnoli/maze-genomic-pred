"""Soft Mixture of Experts multitask: backbone compartido + K cabezas con gate blando.

Alternativa al moe_multitask (gating PCA+GMM con ruteo hard top-1, que parte el
training por cluster y deja cada experto con poca muestra). Acá nada se particiona:
un backbone compartido (el tronco de la red multitask sin su Linear final) produce un
embedding h por línea, K cabezas de experto multitarea predicen sobre h, y un gate
blando gate(h) = softmax(logits / tau) las mezcla como sum_k g_k(h) * expert_k(h),
sin argmax duro. Así se preserva el beneficio multitarea.

Para que la mezcla no degenere a hard hay dos mecanismos configurables: la temperatura
tau del softmax (gate_tau; >1 = más blando) y una load-balancing loss estilo Switch
(Shazeer 2017 / Fedus 2021), L_lb = K * sum_k f_k * P_k, con P_k el peso blando medio
del experto k (con gradiente) y f_k la fracción de muestras cuyo argmax es k
(detached); vale 1 con balance perfecto y crece hasta K con colapso, y se suma con
coeficiente lb_coef. Por época se loguean la entropía del gate, la carga por experto y
la lb_loss para verificar que la mezcla no colapse.

Es una subclase de MLPMultitaskRegressor: hereda la estandarización, los buffers del
target, fit/predict/save/load y el monitor-val, y overridea el armado de red, el loop
_train (loss con LB + diagnósticos del gate) y los defaults.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.eval.metrics import pearson_r
from src.models.mlp import _ACTIVATIONS, _MLPNet
from src.models.mlp_multitask import MLPMultitaskRegressor, masked_weighted_pearson_loss

_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [128],
    "dropout": 0.5,
    "input_dropout": 0.2,
    "activation": "relu",
    "batchnorm": False,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "loss": "pearson",
    "mse_weight": 0.3,
    "patience": None,
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",
    "loss_weights": {},
    "n_experts": 4,
    "gate_tau": 1.0,
    "lb_coef": 0.01,
    "gate_hidden_dim": None,
    "expert_hidden_dim": None,
}


def switch_load_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Load-balancing loss estilo Switch/Shazeer: K * sum_k f_k * P_k.

    gate_weights (B, K) son los pesos softmax del gate. P_k = media por batch de g_k es
    el peso blando medio del experto k (con gradiente); f_k es la fracción de muestras
    cuyo argmax es k (detached). Vale 1 con balance perfecto (f_k = P_k = 1/K) y crece
    hasta K con colapso a un experto, así que minimizarla empuja hacia el balance.
    """
    B, K = gate_weights.shape
    P = gate_weights.mean(dim=0)
    hard = torch.argmax(gate_weights, dim=1)
    f = torch.bincount(hard, minlength=K).to(gate_weights.dtype) / B
    return K * torch.sum(f * P)


class _SoftMoENet(nn.Module):
    """Backbone compartido (tronco _MLPNet) + K cabezas de experto + gate blando.

    forward(x) devuelve la mezcla sum_k g_k(h) * expert_k(h) con shape (n, n_targets);
    forward(x, return_gate=True) devuelve además los pesos del gate g (n, K) para la
    load-balancing loss y los diagnósticos. Los mu/sigma del target los registra como
    buffers el fit heredado, fuera de este módulo.
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

        self.experts = nn.ModuleList([_make_head(self.n_targets) for _ in range(self.n_experts)])

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
        g = self.gate_weights(h)
        experts = torch.stack([e(h) for e in self.experts], dim=1)
        pred = (g.unsqueeze(-1) * experts).sum(dim=1)
        if return_gate:
            return pred, g
        return pred


class SoftMoEMultitaskRegressor(MLPMultitaskRegressor):
    """Soft MoE multitask: tronco compartido + K cabezas de experto mezcladas por un gate blando.

    Subclase de MLPMultitaskRegressor: hereda la estandarización, los buffers del target,
    fit/predict/save/load y el monitor-val, y overridea el armado de red, los defaults y
    el loop de entrenamiento (load-balancing loss + diagnósticos del gate).

    params (además de los de mlp_multitask):
        n_experts         : K expertos que el gate mezcla (default 4).
        gate_tau          : temperatura del softmax del gate (default 1.0; >1 = más blando).
        lb_coef           : coeficiente de la load-balancing loss (default 0.01).
        gate_hidden_dim   : None => gate lineal; int => una capa oculta en el gate.
        expert_hidden_dim : None => cada experto Linear(h, T); int => una capa oculta por experto.
    """


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


    def _gate_diagnostics(self, g: torch.Tensor) -> dict[str, float]:
        """Entropía media del gate y carga por experto sobre un batch de pesos g (N, K).

        gate_entropy en nats; gate_entropy_norm = entropía / log(K) en [0, 1] (1 = mezcla
        uniforme). gate_load_k es el peso blando medio del experto k; gate_load_max/min
        resumen el (des)balance. Se loguean por época.
        """
        K = int(g.shape[1])
        eps = 1e-9
        entropy = float((-(g * (g + eps).log()).sum(dim=1).mean()).item())
        load = g.mean(dim=0)
        out: dict[str, float] = {
            "gate_entropy": entropy,
            "gate_entropy_norm": entropy / float(np.log(K)) if K > 1 else 0.0,
            "gate_load_max": float(load.max().item()),
            "gate_load_min": float(load.min().item()),
        }
        for k in range(K):
            out[f"gate_load_{k}"] = float(load[k].item())
        return out


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
                pred, g = net(Xt[idx], return_gate=True)
                task_loss = masked_weighted_pearson_loss(pred, Yt[idx], Mt[idx], w, mse_weight=mse_w)
                lb = switch_load_balance_loss(g) if g.shape[0] >= 2 else pred.new_zeros(())
                loss = task_loss + lb_coef * lb
                if loss.requires_grad:
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
            net.eval()
            with torch.no_grad():
                g_full = net(Xt, return_gate=True)[1]
                rec.update(self._gate_diagnostics(g_full))
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


    def gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Pesos blandos del gate (n, K) para X crudo (la mezcla por muestra).

        Aplica el z-score de features y devuelve el softmax del gate. Es utilidad de
        diagnóstico (ver que la mezcla no colapse); no interviene en predict.
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
            "hidden_dims": str(self.cfg_["hidden_dims"]),
            "loss_weights": str(
                {n: float(wv) for n, wv in zip(self.target_names_, self.loss_weights_)}
            ),
            "target_names": str(self.target_names_),
        }
        for key in ("gate_entropy", "gate_entropy_norm", "gate_load_max", "gate_load_min"):
            if key in last:
                params[f"final_{key}"] = float(last[key])
        return params
