"""Smoke test del soft MoE multitask (`src/models/moe_soft_multitask.py`).

Verifica lo CRITICO del diseno (leccion del MoE-GMM): que el gate blando NO colapsa a hard top-1.
Corre SIN pytest (rapido, datos sinteticos, pocas epocas):

    python tests/test_moe_soft_multitask.py

(y tambien lo descubre pytest si esta instalado). Checks:
  1. switch_load_balance_loss = 1 con balance perfecto y = K con colapso total (propiedad exacta).
  2. Temperatura: subir tau sube la entropia del softmax del gate (semantica del knob anti-colapso).
  3. Forma de predict (n, T) y simplex del gate (pesos en [0,1], suman 1).
  4. El gate NO colapsa: entropia > 0 en TODAS las epocas y ningun experto con carga ~0 al final.
  5. Round-trip save/load: reconstruye la arquitectura (backbone + K expertos + gate) y predice igual.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.moe_soft_multitask import (  # noqa: E402
    SoftMoEMultitaskRegressor,
    switch_load_balance_loss,
)


def _toy_multitask(n=240, p=60, T=3, missing_frac=0.1, seed=0):
    """X gaussiano, Y = X @ W + ruido por trait (senal real para la loss Pearson).

    Introduce faltantes en las columnas auxiliares (PH, AD) para ejercitar la mascara multitask.
    Devuelve ``(X, Y, Y_mask, target_names)``.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    W = rng.standard_normal((p, T))
    Y = X @ W + 0.5 * rng.standard_normal((n, T))
    M = np.ones((n, T), dtype=bool)
    for j in range(1, T):  # deja GY (col 0) completo; enmascara parte de PH/AD
        drop = rng.random(n) < missing_frac
        M[drop, j] = False
        Y[drop, j] = 0.0
    return X, Y, M, ["GY", "PH", "AD"][:T]


def _fit(params, seed=0, n=240, **kw):
    X, Y, M, names = _toy_multitask(n=n, seed=seed, **kw)
    base = {
        "hidden_dims": [32],
        "dropout": 0.3,
        "input_dropout": 0.1,
        "activation": "relu",
        "batchnorm": False,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 15,
        "loss": "pearson",
        "loss_weights": {"GY": 1.0, "PH": 0.5, "AD": 0.5},
        "patience": None,
        "standardize_inputs": True,
        "standardize_target": True,
        "random_state": 575,
        "device": "cpu",
        "n_experts": 4,
        "gate_tau": 1.0,
        "lb_coef": 0.01,
        "gate_hidden_dim": None,
        "expert_hidden_dim": None,
    }
    base.update(params)
    model = SoftMoEMultitaskRegressor({"params": base})
    model.set_fit_context(target_names=names)
    model.fit(X, Y, M)
    return model, X, Y, M, names


def test_load_balance_loss_exact():
    """L = 1 con balance perfecto; L = K con colapso total a un experto."""
    for K in (2, 4, 8):
        balanced = torch.full((16, K), 1.0 / K)
        collapsed = torch.zeros((16, K))
        collapsed[:, 0] = 1.0
        lb_bal = float(switch_load_balance_loss(balanced).item())
        lb_col = float(switch_load_balance_loss(collapsed).item())
        assert abs(lb_bal - 1.0) < 1e-5, f"K={K}: balance deberia dar 1.0, dio {lb_bal}"
        assert abs(lb_col - K) < 1e-5, f"K={K}: colapso deberia dar {K}, dio {lb_col}"
    print("  [ok] load-balance loss: balance=1, colapso=K")


def test_temperature_increases_entropy():
    """Subir la temperatura tau aplana el softmax => mas entropia (propiedad exacta, sin training)."""
    rng = np.random.default_rng(1)
    logits = torch.as_tensor(rng.standard_normal((100, 4)), dtype=torch.float32)

    def _entropy(tau):
        g = torch.softmax(logits / tau, dim=-1)
        return float((-(g * (g + 1e-9).log()).sum(dim=1).mean()).item())

    e_lo, e_hi = _entropy(0.5), _entropy(3.0)
    assert e_hi > e_lo, f"tau alto deberia dar mas entropia: tau=3 -> {e_hi}, tau=0.5 -> {e_lo}"
    print(f"  [ok] temperatura: entropia(tau=3.0)={e_hi:.3f} > entropia(tau=0.5)={e_lo:.3f}")


def test_predict_shape_and_gate_simplex():
    """predict -> (n, T); los pesos del gate viven en el simplex (>=0, suman 1)."""
    model, X, Y, M, names = _fit({})
    yhat = model.predict(X)
    assert yhat.shape == (X.shape[0], len(names)), f"predict dio {yhat.shape}, esperado {(X.shape[0], len(names))}"
    g = model.gate_weights(X)
    assert g.shape == (X.shape[0], 4), f"gate_weights dio {g.shape}"
    assert np.all(g >= -1e-6) and np.all(g <= 1 + 1e-6), "pesos del gate fuera de [0,1]"
    assert np.allclose(g.sum(axis=1), 1.0, atol=1e-5), "los pesos del gate no suman 1"
    print(f"  [ok] predict {yhat.shape} y gate en el simplex (suma 1)")


def test_gate_does_not_collapse():
    """CRITICO: entropia > 0 en TODAS las epocas y ningun experto con carga ~0 al final."""
    model, X, Y, M, names = _fit({"n_experts": 4})
    hist = model.history_
    assert hist, "history_ vacio"
    # (a) entropia estrictamente positiva en cada epoca (la mezcla nunca es one-hot).
    ents = [rec["gate_entropy"] for rec in hist]
    assert all(e > 0.0 for e in ents), f"alguna epoca con entropia <= 0: {ents}"
    # (b) al final, ningun experto muere: carga minima por encima de un umbral chico.
    last = hist[-1]
    loads = [last[f"gate_load_{k}"] for k in range(4)]
    assert min(loads) > 0.05, f"un experto quedo con carga ~0 (colapso): {loads}"
    # (c) la mezcla es realmente blanda: entropia normalizada lejos de 0.
    assert last["gate_entropy_norm"] > 0.5, f"entropia normalizada baja (mezcla casi hard): {last['gate_entropy_norm']}"
    # (d) la lb_loss se registra y es finita.
    assert np.isfinite(last["lb_loss"]), f"lb_loss no finita: {last['lb_loss']}"
    print(
        f"  [ok] gate no colapsa: entropia_norm final={last['gate_entropy_norm']:.3f}, "
        f"carga por experto={[round(x, 3) for x in loads]}"
    )


def test_save_load_roundtrip():
    """save/load reconstruye la arquitectura soft-MoE y predice identico (heredado de mlp_multitask)."""
    model, X, Y, M, names = _fit({})
    y0 = model.predict(X)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "moe_soft.pt"
        model.save(path)
        loaded = SoftMoEMultitaskRegressor.load(path)
    y1 = loaded.predict(X)
    assert np.allclose(y0, y1, atol=1e-5), "predict difiere tras save/load"
    print("  [ok] round-trip save/load: predict identico")


def main():
    torch.manual_seed(0)
    tests = [
        test_load_balance_loss_exact,
        test_temperature_increases_entropy,
        test_predict_shape_and_gate_simplex,
        test_gate_does_not_collapse,
        test_save_load_roundtrip,
    ]
    print(f"[test_moe_soft_multitask] corriendo {len(tests)} checks...")
    for t in tests:
        t()
    print("[test_moe_soft_multitask] OK — todos los checks pasaron.")


if __name__ == "__main__":
    main()
