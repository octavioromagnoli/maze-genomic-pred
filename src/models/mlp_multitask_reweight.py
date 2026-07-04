"""MLP multitask con reponderación de muestras por meta-learning (Learning to Reweight Examples).

Ren et al. 2018 (*Learning to Reweight Examples for Robust Deep Learning*): en vez de fijar la
importancia de cada muestra por parentesco/heurística, se **aprende online un peso por muestra**
que optimiza la pérdida en un meta-set. Cada paso de train:

1. Pérdida por-muestra ``L_i`` del batch de train con los params actuales θ.
2. Paso virtual ponderado ``θ̂(ε) = θ − η·∇_θ Σ_i ε_i L_i`` con ``ε=0`` (diferenciable, 2º orden).
3. Meta-pérdida en el **meta-set** con ``θ̂(ε)``; su gradiente respecto de ε dice qué muestras, al
   subirles el peso, bajan la meta-pérdida: ``w_i = relu(−∂L_meta/∂ε_i)``, normalizado por batch.
4. Update real de θ con la train loss reponderada por ``w`` (detached).

**Sin leakage (reglas 1–3).** El meta-set es un slice **held-out del TRAIN**, disjunto por línea
(regla 2): el val del fold (2018) NUNCA entra al fit. La reponderación es un dispositivo interno de
entrenamiento; el split de evaluación (train→val→holdout 2020) queda intacto y temporal (regla 3).

**Loss por-muestra = MSE enmascarada ponderada por trait.** L2RW necesita pérdidas por-muestra y la
Pearson es un estadístico de batch (no descomponible), así que este camino entrena con
``L_i = Σ_j w_trait_j · m_ij · (ŷ_ij − y_ij)²`` sobre el target estandarizado. La **Pearson sigue
siendo la métrica de portada** (regla 5), la calcula `run_multitask_evaluation` al evaluar.

Subclase de `MLPMultitaskRegressor`: misma arquitectura (tronco `_MLPNet` + cabeza multitask),
estandarización train-only, `predict`/`save`/`load` heredados. Overridea solo `_resolve_cfg`
(HP del meta-learning) y `_train` (loop L2RW). Se enchufa en `run_multitask_evaluation` por nombre.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.func import functional_call

from src.eval.metrics import pearson_r
from src.models.mlp_multitask import MLPMultitaskRegressor

# Defaults: los del MLP multitask + bloque de meta-learning. `_resolve_cfg` filtra por estas claves.
_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [128],
    "dropout": 0.5,
    "input_dropout": 0.2,
    "activation": "relu",
    "batchnorm": False,        # L2RW hace forwards funcionales: mantener BN off (sin running stats)
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": None,
    "standardize_inputs": True,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",
    "loss_weights": {},        # peso por trait (multiplica la MSE por-muestra de cada columna)
    # --- meta-learning (Learning to Reweight Examples) ---
    "meta_frac": 0.2,          # fracción de LÍNEAS del train reservadas como meta-set (held-out)
    "meta_lr": None,           # η del paso virtual; None => usa `lr`
}


def _masked_persample_mse(pred, target, mask, trait_w):
    """Pérdida MSE por-muestra ``(B,)``: Σ_j w_trait_j · m_ij · (pred_ij − target_ij)² (solo válidos)."""
    se = (pred - target) ** 2 * mask.to(pred.dtype)     # (B,T), 0 donde está enmascarado
    return (se * trait_w).sum(dim=1)                     # (B,)


class MLPMultitaskReweightRegressor(MLPMultitaskRegressor):
    """MLP multitask con pesos por-muestra aprendidos online (L2RW).

    `params` (sub-dict ``model.params``) — además de los de `mlp_multitask`:
        meta_frac : fracción de líneas del train reservadas como meta-set held-out (default 0.2).
        meta_lr   : tasa del paso virtual del meta-gradiente (default None => usa ``lr``).
    """

    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["hidden_dims"] = [int(h) for h in cfg["hidden_dims"]]
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["patience"] = None if cfg["patience"] is None else int(cfg["patience"])
        cfg["loss_weights"] = dict(cfg["loss_weights"] or {})
        cfg["loss"] = "mse"          # este camino entrena con MSE por-muestra por construcción
        cfg["mse_weight"] = 0.0
        cfg["meta_frac"] = float(cfg["meta_frac"])
        if not (0.0 < cfg["meta_frac"] < 1.0):
            raise ValueError("meta_frac debe estar en (0, 1).")
        cfg["meta_lr"] = None if cfg["meta_lr"] is None else float(cfg["meta_lr"])
        return cfg

    # --- Split train/meta held-out, DISJUNTO POR LÍNEA (regla 2; interno al train => sin leakage) ---

    def _split_main_meta(self, n: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(cfg["random_state"])
        line_id = self._fit_context.get("line_id")
        if line_id is None:
            idx = rng.permutation(n)
            n_meta = max(1, int(round(cfg["meta_frac"] * n)))
            meta = np.zeros(n, dtype=bool)
            meta[idx[:n_meta]] = True
            return ~meta, meta
        line_id = np.asarray(line_id)
        uniq = np.unique(line_id)
        rng.shuffle(uniq)
        n_meta_lines = max(1, min(len(uniq) - 1, int(round(cfg["meta_frac"] * len(uniq)))))
        meta_lines = set(uniq[:n_meta_lines].tolist())
        meta = np.array([l in meta_lines for l in line_id], dtype=bool)
        return ~meta, meta

    # --- Loop de entrenamiento con reponderación L2RW ---

    def _train(self, Xs, Ys, M, *, cfg, device) -> list[dict[str, float]]:
        net = self.net_
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        meta_lr = cfg["lr"] if cfg["meta_lr"] is None else cfg["meta_lr"]
        w_trait = torch.as_tensor(self.loss_weights_, dtype=torch.float32, device=device)

        main_mask, meta_mask = self._split_main_meta(Xs.shape[0], cfg)
        self.n_meta_ = int(meta_mask.sum())

        def _tens(a, dt):
            return torch.as_tensor(a, dtype=dt, device=device)

        Xt, Yt, Mt = _tens(Xs, torch.float32), _tens(Ys, torch.float32), _tens(M, torch.bool)
        Xm = Xt[_tens(meta_mask, torch.bool)]
        Ym = Yt[_tens(meta_mask, torch.bool)]
        Mm = Mt[_tens(meta_mask, torch.bool)]
        main_idx_all = np.flatnonzero(main_mask)

        bs = cfg["batch_size"]
        full_batch = bs <= 0 or bs >= len(main_idx_all)
        mon = self._monitor_val(device)

        history: list[dict[str, float]] = []
        for epoch in range(int(cfg["max_epochs"])):
            net.train()
            order = np.random.permutation(main_idx_all)
            batches = [order] if full_batch else np.array_split(order, max(1, len(order) // bs))
            run_loss = 0.0
            seen = 0
            ess_frac_acc = 0.0
            zero_frac_acc = 0.0
            n_batches = 0
            for bidx in batches:
                if len(bidx) < 2:
                    continue
                bi = _tens(bidx, torch.long)
                xb, yb, mb = Xt[bi], Yt[bi], Mt[bi]
                B = xb.shape[0]

                # 1) meta-gradiente respecto de ε (pesos por-muestra, init 0) vía paso virtual.
                params = {n: p for n, p in net.named_parameters()}
                pred_v = functional_call(net, params, (xb,))
                L = _masked_persample_mse(pred_v, yb, mb, w_trait)          # (B,)
                eps = torch.zeros(B, device=device, requires_grad=True)
                grads = torch.autograd.grad((eps * L).sum(), tuple(params.values()), create_graph=True)
                theta_hat = {n: p - meta_lr * g for (n, p), g in zip(params.items(), grads)}
                pred_m = functional_call(net, theta_hat, (Xm,))
                meta_loss = _masked_persample_mse(pred_m, Ym, Mm, w_trait).mean()
                grad_eps = torch.autograd.grad(meta_loss, eps)[0]          # (B,)

                # 2) pesos = relu(-meta-grad) normalizados por batch (suman 1; uniforme si todo 0).
                w = torch.relu(-grad_eps)
                s = w.sum()
                w = w / s if float(s) > 0 else torch.full_like(w, 1.0 / B)

                # 3) update real con la train loss reponderada (w detached).
                optim.zero_grad()
                L_real = _masked_persample_mse(net(xb), yb, mb, w_trait)   # (B,)
                loss = (w.detach() * L_real).sum()
                loss.backward()
                optim.step()

                run_loss += float(loss.item()) * B
                seen += B
                with torch.no_grad():
                    ess_frac_acc += float((1.0 / (B * (w ** 2).sum())).item())  # ESS/B ∈ (0,1]
                    zero_frac_acc += float((w <= 0).to(torch.float32).mean().item())
                n_batches += 1

            rec: dict[str, float] = {
                "epoch": epoch,
                "train_loss": run_loss / max(seen, 1),
                "w_ess_frac": ess_frac_acc / max(n_batches, 1),   # 1 = uniforme, bajo = concentrado
                "w_zero_frac": zero_frac_acc / max(n_batches, 1),  # fracción de muestras down-weighted a 0
            }
            if mon is not None:
                net.eval()
                with torch.no_grad():
                    Xv, Yv_np, Mv_np = mon
                    pv = self._forward2d(Xv).detach().cpu().numpy()
                for j, name in enumerate(self.target_names_):
                    colv = Mv_np[:, j]
                    if colv.sum() >= 2:
                        rec[f"val_pearson_{name}"] = pearson_r(Yv_np[colv, j], pv[colv, j])
            history.append(rec)
        return history

    def get_fitted_params(self) -> dict[str, Any]:
        base = super().get_fitted_params()
        if not base:
            return base
        last = self.history_[-1] if getattr(self, "history_", None) else {}
        base.update({
            "reweight": 1,
            "meta_frac": float(self.cfg_["meta_frac"]),
            "n_meta": int(getattr(self, "n_meta_", 0)),
        })
        for k in ("w_ess_frac", "w_zero_frac"):
            if k in last:
                base[f"final_{k}"] = float(last[k])
        return base
