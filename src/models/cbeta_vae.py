"""C-beta-VAE conjunto: regresión de GY con reconstrucción de SNPs como tarea auxiliar.

Una sola red con encoder compartido que alimenta dos ramas. La rama VAE produce los
parámetros de una gaussiana latente (mu, logvar); con el reparametrization trick
z = mu + exp(0.5*logvar)*eps muestrea z y un decoder lo reconstruye a logits de SNPs.
La reconstrucción usa BCEWithLogitsLoss (SNPs binarios) y la regularización es la KL
contra la prior N(0, I), ponderada por beta (beta>1 fuerza un latente más regularizado).
La cabeza de regresión mapea z (en train) o mu (en inferencia) a la predicción de GY
con pearson_loss (1 - r).

Es un multitask con tarea auxiliar = reconstrucción: el encoder compartido aprende un
latente útil para predecir el fenotipo y para reconstruir genotipos. La pérdida total
es L = pearson(y_hat, y) + lambda_aux * (BCE(decoder, X) + beta*KL).

Reusa piezas del autoencoder (_mlp/_activation) y de la red de regresión (pearson_loss,
temporal_inner_split). Estandariza el target train-only dentro de fit pero no X: la
entrada y el target de la BCE quedan binarios, así que no usa el bloque preprocess:.
Con patience activa early stopping monitoreando el Pearson de regresión.

generate_prior (muestrea de la prior, label de la cabeza de regresión) y
generate_posterior (perturba el latente de líneas reales conservando el label real)
producen pares (X, y) sintéticos para data augmentation. save/load por torch con
arquitectura + state_dict + scalers, igual que el MLP.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.metrics import pearson_r
from src.models.autoencoder import _activation, _mlp
from src.models.base import BaseModel
from src.models.mlp import pearson_loss, temporal_inner_split

_DEFAULTS: dict[str, Any] = {
    "latent_dim": 64,
    "encoder_hidden": [1024, 256],
    "reg_hidden": [64],
    "activation": "relu",
    "dropout": 0.0,
    "reg_dropout": 0.2,
    "batchnorm": False,
    "lambda_aux": 1.0,
    "beta": 1.0,
    "kl_anneal_epochs": 0,
    "free_bits": 0.0,
    "denoising_p": 0.0,
    "reg_loss": "pearson",
    "mse_weight": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": None,
    "val_metric": "pearson",
    "refit_full_train": True,
    "min_delta": 0.0,
    "standardize_target": True,
    "random_state": 575,
    "device": "cpu",
}

_BINARIZE = ("bernoulli", "threshold", "prob")


class _CBetaVAENet(nn.Module):
    """Encoder compartido + decoder (VAE) + cabeza de regresión.

    El encoder emite 2*latent_dim (concat de mu y logvar). forward devuelve
    (logits, y_hat, mu, logvar, z). Con sample=True (train) z se muestrea por
    reparametrización; con sample=False (inferencia) z = mu.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: list[int],
        reg_hidden: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        reg_dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        enc_hidden = [int(h) for h in (encoder_hidden or [])]
        self.encoder = _mlp([self.input_dim, *enc_hidden, 2 * self.latent_dim], activation, dropout, batchnorm)
        self.decoder = _mlp([self.latent_dim, *list(reversed(enc_hidden)), self.input_dim], activation, dropout, batchnorm)
        reg_h = [int(h) for h in (reg_hidden or [])]
        self.reg_head = _mlp([self.latent_dim, *reg_h, 1], activation, reg_dropout, batchnorm)
        self._activation_name = activation
        _ = _activation

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, *, sample: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        logits = self.decoder(z)
        yhat = self.reg_head(z).squeeze(-1)
        return logits, yhat, mu, logvar, z


class CBetaVAERegressor(BaseModel):
    """C-β-VAE conjunto: regresión de GY con reconstrucción de SNPs como tarea auxiliar.

    params (sub-dict model.params del config) — ver _DEFAULTS:
        latent_dim, encoder_hidden, reg_hidden, activation, dropout, reg_dropout,
        batchnorm                : arquitectura.
        lambda_aux (λ), beta (β), kl_anneal_epochs, free_bits, denoising_p,
        reg_loss, mse_weight     : pérdida combinada.
        lr, weight_decay, batch_size, max_epochs : optimización.
        patience, val_metric, refit_full_train, min_delta : early stopping temporal.
        standardize_target, random_state, device : escala / runtime.
    """


    def _resolve_cfg(self) -> dict[str, Any]:
        cfg = {**_DEFAULTS, **{k: v for k, v in self.params.items() if k in _DEFAULTS}}
        cfg["encoder_hidden"] = [int(h) for h in cfg["encoder_hidden"]]
        cfg["reg_hidden"] = [int(h) for h in (cfg["reg_hidden"] or [])]
        cfg["latent_dim"] = int(cfg["latent_dim"])
        cfg["activation"] = str(cfg["activation"]).lower()
        cfg["reg_loss"] = str(cfg["reg_loss"]).lower()
        cfg["val_metric"] = str(cfg["val_metric"]).lower()
        cfg["mse_weight"] = float(cfg["mse_weight"])
        cfg["lambda_aux"] = float(cfg["lambda_aux"])
        cfg["beta"] = float(cfg["beta"])
        cfg["kl_anneal_epochs"] = int(cfg["kl_anneal_epochs"])
        cfg["free_bits"] = float(cfg["free_bits"])
        cfg["denoising_p"] = float(cfg["denoising_p"])
        cfg["batch_size"] = int(cfg["batch_size"])
        cfg["max_epochs"] = int(cfg["max_epochs"])
        cfg["random_state"] = int(cfg["random_state"])
        cfg["patience"] = None if cfg["patience"] is None else int(cfg["patience"])
        return cfg

    def _resolve_device(self, device: str) -> "torch.device":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return torch.device(device)

    @staticmethod
    def _seed(seed: int) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def _check_binary(X: np.ndarray) -> None:
        """Los SNPs deben ser binarios {0,1} para la BCE."""
        u = np.unique(X[: min(len(X), 256)])
        if not np.all(np.isin(u, (0.0, 1.0))):
            raise ValueError(
                "CBetaVAERegressor espera SNPs binarios {0,1} (regla 4: BCE de la rama "
                f"de reconstrucción). Encontré valores {u[:6]}... ¿pusiste un standardize "
                "ANTES del modelo? Este modelo NO usa el bloque preprocess; estandariza "
                "solo el target adentro de fit."
            )


    def _reg_loss_fn(self, cfg: dict[str, Any]):
        """Pérdida de la cabeza de regresión (sobre el target estandarizado)."""
        if cfg["reg_loss"] == "mse":
            return nn.MSELoss()
        if cfg["reg_loss"] == "pearson":
            return pearson_loss
        if cfg["reg_loss"] == "pearson_mse":
            _mse = nn.MSELoss()
            _w = cfg["mse_weight"]

            def _fn(pred, target):
                return pearson_loss(pred, target) + _w * _mse(pred, target)

            return _fn
        raise ValueError(
            f"reg_loss desconocida: {cfg['reg_loss']!r}. Disponibles: 'pearson', 'pearson_mse', 'mse'."
        )

    @staticmethod
    def _kl(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL(N(mu,σ) || N(0,I)) sumada en el latente, promediada en el batch.

        free_bits pone un piso por dimensión (evita que la KL colapse a 0 y la
        rama generativa ignore el latente).
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=1).mean()


    def _build_net(self, in_dim: int, cfg: dict[str, Any]) -> _CBetaVAENet:
        return _CBetaVAENet(
            input_dim=in_dim,
            latent_dim=cfg["latent_dim"],
            encoder_hidden=cfg["encoder_hidden"],
            reg_hidden=cfg["reg_hidden"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            reg_dropout=cfg["reg_dropout"],
            batchnorm=cfg["batchnorm"],
        )

    def _train(
        self,
        Xb: np.ndarray,
        ys: np.ndarray,
        Xv: np.ndarray | None,
        yv: np.ndarray | None,
        *,
        cfg: dict[str, Any],
        max_epochs: int,
        patience: int | None = None,
        restore_best: bool = False,
        device: "torch.device",
    ):
        """Entrena la red conjunta. Devuelve (net, history, best_epoch, best_val).

        Xb son SNPs binarios (n, p); ys el target estandarizado (n,). Con
        Xv/yv (inner-val binario + target estandarizado) trackea el Pearson de
        regresión por época; con patience corta si no mejora; con restore_best
        deja los pesos de la mejor época.
        """
        net = self._build_net(Xb.shape[1], cfg).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        reg_loss_fn = self._reg_loss_fn(cfg)
        lam, beta = cfg["lambda_aux"], cfg["beta"]
        free_bits, denoise_p = cfg["free_bits"], cfg["denoising_p"]
        anneal = cfg["kl_anneal_epochs"]
        drop_singleton = cfg["batchnorm"] or cfg["reg_loss"] in ("pearson", "pearson_mse")

        Xt = torch.as_tensor(Xb, dtype=torch.float32, device=device)
        yt = torch.as_tensor(ys, dtype=torch.float32, device=device)
        has_val = Xv is not None and yv is not None
        if has_val:
            Xvt = torch.as_tensor(Xv, dtype=torch.float32, device=device)
            yv_np = np.asarray(yv, dtype=np.float64).ravel()

        n = Xt.shape[0]
        bs = cfg["batch_size"]
        full_batch = bs <= 0 or bs >= n

        history: list[dict[str, float]] = []
        best_score = np.inf
        best_epoch = -1
        best_val: dict[str, float] = {}
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0

        for epoch in range(int(max_epochs)):
            net.train()
            beta_eff = beta if anneal <= 0 else beta * min(1.0, (epoch + 1) / anneal)
            if full_batch:
                batches = [torch.arange(n, device=device)]
            else:
                perm = torch.randperm(n, device=device)
                batches = list(torch.split(perm, bs))
                if drop_singleton and batches and batches[-1].numel() == 1:
                    batches = batches[:-1]
            run_loss = run_bce = run_kl = run_reg = 0.0
            seen = 0
            for idx in batches:
                xb = Xt[idx]
                yb = yt[idx]
                xin = xb
                if denoise_p > 0:
                    mask = (torch.rand_like(xb) < denoise_p)
                    xin = xb * (~mask).to(xb.dtype)
                optim.zero_grad()
                logits, yhat, mu, logvar, _ = net(xin, sample=True)
                recon = F.binary_cross_entropy_with_logits(logits, xb)
                kl = self._kl(mu, logvar, free_bits)
                reg = reg_loss_fn(yhat, yb)
                loss = reg + lam * (recon + beta_eff * kl)
                loss.backward()
                optim.step()
                nb = idx.numel()
                run_loss += float(loss.item()) * nb
                run_bce += float(recon.item()) * nb
                run_kl += float(kl.item()) * nb
                run_reg += float(reg.item()) * nb
                seen += int(nb)
            rec: dict[str, float] = {
                "epoch": epoch,
                "train_loss": run_loss / max(seen, 1),
                "train_bce": run_bce / max(seen, 1),
                "train_kl": run_kl / max(seen, 1),
                "train_reg": run_reg / max(seen, 1),
                "beta_eff": beta_eff,
            }

            if has_val:
                net.eval()
                with torch.no_grad():
                    logits_v, yhat_v, mu_v, logvar_v, _ = net(Xvt, sample=False)
                    v_recon = float(F.binary_cross_entropy_with_logits(logits_v, Xvt).item())
                    v_kl = float(self._kl(mu_v, logvar_v, free_bits).item())
                    yv_t = torch.as_tensor(yv, dtype=torch.float32, device=device)
                    v_reg = float(reg_loss_fn(yhat_v, yv_t).item())
                v_pearson = pearson_r(yv_np, yhat_v.detach().cpu().numpy())
                rec["val_bce"] = v_recon
                rec["val_kl"] = v_kl
                rec["val_reg"] = v_reg
                rec["val_pearson"] = v_pearson
                if cfg["val_metric"] == "pearson":
                    score = -v_pearson if np.isfinite(v_pearson) else np.inf
                else:
                    score = v_reg + lam * (v_recon + beta * v_kl)
                if score < best_score - cfg["min_delta"]:
                    best_score = score
                    best_epoch = epoch
                    best_val = {"val_pearson": v_pearson, "val_bce": v_recon, "val_reg": v_reg}
                    best_state = copy.deepcopy(net.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                history.append(rec)
                if patience is not None and no_improve >= patience:
                    break
            else:
                history.append(rec)

        if restore_best and best_state is not None:
            net.load_state_dict(best_state)
        return net, history, best_epoch, best_val


    def _compute_y_scaler(self, y: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
        if cfg["standardize_target"]:
            mu = float(y.mean())
            sd = float(y.std())
            return mu, (sd if sd > 0 else 1.0)
        return 0.0, 1.0


    def fit(self, X: np.ndarray, y: np.ndarray) -> "CBetaVAERegressor":
        cfg = self._resolve_cfg()
        self.cfg_ = cfg
        device = self._resolve_device(cfg["device"])
        X = np.asarray(X, dtype=np.float32)
        self._check_binary(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("CBetaVAERegressor.fit recibió y vacío.")
        self.n_features_in_ = int(X.shape[1])

        split = (
            temporal_inner_split(self._fit_context.get("year"))
            if cfg["patience"] is not None
            else None
        )

        if split is None:
            self._seed(cfg["random_state"])
            self.y_mean_, self.y_std_ = self._compute_y_scaler(y, cfg)
            ys = (y - self.y_mean_) / self.y_std_
            Xv, yv = self._monitor_val()
            net, history, _, _ = self._train(
                X, ys, Xv, yv, cfg=cfg, max_epochs=cfg["max_epochs"], device=device
            )
            self.net_ = net
            self.history_ = history
            self.early_stopping_used_ = False
            self.selected_epochs_ = int(cfg["max_epochs"])
            self.best_epoch_ = None
            self.best_val_ = {}
            self.refit_full_train_ = False
            self._finalize(device)
            return self

        tr, va = split
        ym, ysd = self._compute_y_scaler(y[tr], cfg)
        ytr = (y[tr] - ym) / ysd
        yva = (y[va] - ym) / ysd
        self._seed(cfg["random_state"])
        _, history, best_epoch, best_val = self._train(
            X[tr], ytr, X[va], yva,
            cfg=cfg, max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            restore_best=True, device=device,
        )
        self.history_ = history
        self.early_stopping_used_ = True
        self.best_epoch_ = int(best_epoch) if best_epoch >= 0 else None
        self.best_val_ = best_val
        selected = max(1, (best_epoch + 1) if best_epoch >= 0 else cfg["max_epochs"])
        self.selected_epochs_ = int(selected)
        self.refit_full_train_ = bool(cfg["refit_full_train"])

        if cfg["refit_full_train"]:
            self._seed(cfg["random_state"])
            self.y_mean_, self.y_std_ = self._compute_y_scaler(y, cfg)
            ys = (y - self.y_mean_) / self.y_std_
            net, _, _, _ = self._train(X, ys, None, None, cfg=cfg, max_epochs=selected, device=device)
            self.net_ = net
        else:
            self.y_mean_, self.y_std_ = ym, ysd
            self._seed(cfg["random_state"])
            net, _, _, _ = self._train(X[tr], ytr, None, None, cfg=cfg, max_epochs=selected, device=device)
            self.net_ = net
        self._finalize(device)
        return self

    def _monitor_val(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Val externo del fold (de set_fit_context) para la curva por época.

        Devuelve (Xv_bin, yv_std) o (None, None). Estandariza el target con los scalers
        de train; X queda binaria.
        """
        Xe = self._fit_context.get("eval_X")
        ye = self._fit_context.get("eval_y")
        if Xe is None or ye is None:
            return None, None
        Xe = np.asarray(Xe, dtype=np.float32)
        ye = np.asarray(ye, dtype=np.float64).ravel()
        if Xe.ndim != 2 or Xe.shape[0] == 0 or Xe.shape[1] != self.n_features_in_ or ye.size != Xe.shape[0]:
            return None, None
        return Xe, (ye - self.y_mean_) / self.y_std_

    def _finalize(self, device: "torch.device") -> None:
        self.net_.eval()
        self.device_ = str(device)
        self.n_params_ = int(sum(p.numel() for p in self.net_.parameters()))

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.predict llamado antes de fit.")
        device = next(self.net_.parameters()).device
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        _, yhat, _, _, _ = self.net_(xt, sample=False)
        y = yhat.detach().cpu().numpy().ravel()
        return (y * self.y_std_ + self.y_mean_).astype(float)


    @torch.no_grad()
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Embedding latente (mu, determinista) de cada observación. (n, latent_dim)."""
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.encode llamado antes de fit.")
        device = next(self.net_.parameters()).device
        self.net_.eval()
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        mu, _ = self.net_.encode(xt)
        return mu.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstrucción (probabilidades sigmoide por SNP) desde mu. Para diagnóstico."""
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.reconstruct llamado antes de fit.")
        device = next(self.net_.parameters()).device
        self.net_.eval()
        xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        mu, _ = self.net_.encode(xt)
        logits = self.net_.decode(mu)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    def _binarize(self, probs: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
        if mode == "bernoulli":
            return torch.bernoulli(probs, generator=gen)
        if mode == "threshold":
            return (probs > 0.5).to(probs.dtype)
        if mode == "prob":
            return probs
        raise ValueError(f"binarize={mode!r} no soportado. Use uno de {_BINARIZE}.")

    @torch.no_grad()
    def _decode_to_samples(self, z: torch.Tensor, *, binarize: str, gen: torch.Generator):
        """Decodea z a (X_synth, y_synth) en escala original."""
        logits = self.net_.decode(z)
        probs = torch.sigmoid(logits)
        X_synth = self._binarize(probs, binarize, gen)
        y_std = self.net_.reg_head(z).squeeze(-1)
        y_synth = y_std.detach().cpu().numpy().ravel() * self.y_std_ + self.y_mean_
        return X_synth.detach().cpu().numpy().astype(np.float32), y_synth.astype(float)

    @torch.no_grad()
    def generate_prior(
        self, n: int, *, temperature: float = 1.0, binarize: str = "bernoulli", seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera n muestras desde la prior N(0, I).

        z ~ N(0, I)*temperature, pasa por el decoder a X_synth (binario) y por la cabeza
        de regresión a y_synth (de-estandarizado). El label lo genera la propia red, así
        que no aporta señal nueva más allá de lo que el modelo ya sabe.
        """
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.generate_prior llamado antes de fit.")
        device = next(self.net_.parameters()).device
        self.net_.eval()
        gen = torch.Generator(device="cpu").manual_seed(int(seed) if seed is not None else int(self.cfg_["random_state"]))
        z = torch.randn(int(n), self.net_.latent_dim, generator=gen).to(device) * float(temperature)
        gen_dev = torch.Generator(device=device).manual_seed((int(seed) if seed is not None else 0) + 13)
        return self._decode_to_samples(z, binarize=binarize, gen=gen_dev)

    @torch.no_grad()
    def generate_posterior(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray | None = None,
        *,
        jitter: float = 1.0,
        binarize: str = "bernoulli",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera muestras perturbando el latente de líneas reales (posterior-jitter).

        Encodea X_ref a (mu, sigma), muestrea z = mu + jitter*sigma*eps y decodea a
        X_synth (binario, cercano al manifold real). Con y_ref conserva el label real
        (interpola alrededor de puntos etiquetados); si no, usa la cabeza de regresión.
        """
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.generate_posterior llamado antes de fit.")
        device = next(self.net_.parameters()).device
        self.net_.eval()
        xt = torch.as_tensor(np.asarray(X_ref, dtype=np.float32), device=device)
        mu, logvar = self.net_.encode(xt)
        std = torch.exp(0.5 * logvar)
        g = torch.Generator(device=device).manual_seed(int(seed) if seed is not None else int(self.cfg_["random_state"]) + 1)
        eps = torch.randn(mu.shape, generator=g, device=device)
        z = mu + float(jitter) * std * eps
        gen_dev = torch.Generator(device=device).manual_seed((int(seed) if seed is not None else 0) + 29)
        X_synth, y_self = self._decode_to_samples(z, binarize=binarize, gen=gen_dev)
        if y_ref is not None:
            y_out = np.asarray(y_ref, dtype=float).ravel()
            if y_out.size != X_synth.shape[0]:
                raise ValueError(
                    f"y_ref tiene {y_out.size} labels pero X_ref tiene {X_synth.shape[0]} filas."
                )
            return X_synth, y_out
        return X_synth, y_self


    def get_fitted_params(self) -> dict[str, Any]:
        if not hasattr(self, "net_"):
            return {}
        out: dict[str, Any] = {
            "selected_epochs": getattr(self, "selected_epochs_", None),
            "early_stopping_used": int(getattr(self, "early_stopping_used_", False)),
            "n_params": getattr(self, "n_params_", None),
            "latent_dim": int(self.cfg_["latent_dim"]),
            "lambda_aux": float(self.cfg_["lambda_aux"]),
            "beta": float(self.cfg_["beta"]),
            "encoder_hidden": str(self.cfg_["encoder_hidden"]),
        }
        if getattr(self, "best_epoch_", None) is not None:
            out["best_epoch"] = self.best_epoch_
        if getattr(self, "best_val_", None):
            for k in ("val_pearson", "val_bce", "val_reg"):
                v = self.best_val_.get(k)
                if v is not None and np.isfinite(v):
                    out[f"best_{k}"] = float(v)
        if self.history_:
            out["kl_final"] = float(self.history_[-1].get("train_kl", np.nan))
            out["bce_final"] = float(self.history_[-1].get("train_bce", np.nan))
        return out


    def _arch(self) -> dict[str, Any]:
        keys = ("latent_dim", "encoder_hidden", "reg_hidden", "activation", "dropout", "reg_dropout", "batchnorm")
        return {k: self.cfg_[k] for k in keys}

    def save(self, path: str | Path) -> Path:
        if not hasattr(self, "net_"):
            raise RuntimeError("CBetaVAERegressor.save llamado antes de fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "format": "cbeta_vae_v1",
            "config": self.config,
            "cfg_": self.cfg_,
            "arch": self._arch(),
            "n_features_in_": self.n_features_in_,
            "state_dict": self.net_.state_dict(),
            "y_mean_": float(self.y_mean_),
            "y_std_": float(self.y_std_),
            "selected_epochs_": getattr(self, "selected_epochs_", None),
            "early_stopping_used_": getattr(self, "early_stopping_used_", False),
            "best_epoch_": getattr(self, "best_epoch_", None),
            "best_val_": getattr(self, "best_val_", {}),
            "n_params_": getattr(self, "n_params_", None),
            "device_": getattr(self, "device_", "cpu"),
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CBetaVAERegressor":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(ckpt.get("config"))
        obj.cfg_ = ckpt["cfg_"]
        obj.n_features_in_ = int(ckpt["n_features_in_"])
        obj.y_mean_ = float(ckpt["y_mean_"])
        obj.y_std_ = float(ckpt["y_std_"])
        net = obj._build_net(obj.n_features_in_, obj.cfg_)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        obj.net_ = net
        obj.selected_epochs_ = ckpt.get("selected_epochs_")
        obj.early_stopping_used_ = ckpt.get("early_stopping_used_", False)
        obj.best_epoch_ = ckpt.get("best_epoch_")
        obj.best_val_ = ckpt.get("best_val_", {})
        obj.n_params_ = ckpt.get("n_params_")
        obj.device_ = ckpt.get("device_", "cpu")
        obj.history_ = []
        return obj
