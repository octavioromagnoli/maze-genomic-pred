# Sweeps pendientes de correr

Registro de los sweeps pendientes para la comparación **coherente soft-MoE vs hard-MoE** en el
plano **multienv** (GY, cabezas optimal+drought, un solo modelo, una corrida = ambas cabezas).

**Regla de coherencia:** entre soft y hard lo ÚNICO que cambia es el mecanismo de mezcla (gate
blando de K expertos vs. gating PCA+GMM top-1). Mismo fold de selección (train 2017 → val 2018),
misma métrica (`pearson_r_GY` = cabeza de `cover_env`), mismos rangos de backbone, `loss_weights`
1.0/1.0 no barridos. El número honesto es siempre el **confirm 2019 vs GBLUP**, nunca el máximo del
sweep (winner's curse).

---

## Pendientes

### 🔴 Prioridad ALTA — drought (la cancha donde el MoE le gana a GBLUP)

- [x] **Soft multienv, portada drought** — `configs/sweep_moe_soft_multienv.yaml` — ✅ **HECHO** (120 trials, FINISHED)
  - Sweep id: **`nvphlc1o`** · Ganador **`wx4dyzfw`** (pearson_r_GY drought val2018 = 0.2411; K=8, tau 0.98, gate_hidden 32, hidden [256,128], leaky_relu). Top-5 pegados 0.239–0.241 (landscape plano).
  - **Confirm 2019** (`exp_moe_soft_multienv_confirm_2019.yaml`): **drought 0.291** (>GBLUP 0.241), optimal 0.279 (<GBLUP 0.315). NO se desinfló (subió de 0.24 a 0.29). Hard multienv (~0.369) sigue arriba del soft en drought.

### 🟡 Prioridad BAJA — optimal (por completitud del informe; GBLUP casi imbatible acá, ~0.31)

- [ ] **Hard multienv, portada optimal** — `configs/sweep_moe_multienv_optimal.yaml`
  - Base: `configs/exp_moe_multienv_GY_optimal_2017_to_2018.yaml` · run_cap 120 · caro ((k+1)·n_models redes/trial)
  - Lanzar: `wandb sweep configs/sweep_moe_multienv_optimal.yaml`
  - Ganador: `__________` (run id)

- [ ] **Soft multienv, portada optimal** — `configs/sweep_moe_soft_multienv_optimal.yaml`
  - Base: `configs/exp_moe_soft_multienv_GY_optimal_2017_to_2018.yaml` · run_cap 120 · ~5× más barato que el hard
  - Lanzar: `wandb sweep configs/sweep_moe_soft_multienv_optimal.yaml`
  - Ganador: `__________` (run id)

---

## Ya corridos (referencia)

- [x] **Hard multienv, portada drought** — `configs/sweep_moe_multienv.yaml`
  - Sweep `87qyjrcr`, ganador trial **`u1kgstvr`** (k=4, pca=100, cov=full, min_cluster=250, inner [128] leaky_relu, n_models=5).
  - ⚠️ Sus HP están hoy en `exp_moe_hard_2019_{optimal,drought}.yaml` pero **trasplantados a un confirm trait-multitask** (GY+PH+AD) → ese confirm es incoherente para el plano multienv. Para el A/B limpio, confirmar en setup multienv.

---

## Cómo lanzar un sweep (recordatorio)

```bash
wandb sweep configs/<sweep>.yaml            # imprime el <sweep_id>
# no hay tmux en esta máquina → usar screen:
screen -dmS <nombre> bash -c 'WANDB_MODE=online wandb agent oromagnoli-/maze-genomic-pred/<sweep_id>'
# reattach: screen -r <nombre>   |   matar: screen -S <nombre> -X quit
```

## Después de cada sweep (paso obligatorio)

1. Anotar el `run id` ganador en este archivo.
2. Crear el confirm 2019 con los HP ganadores (reentrena 2017+2018 → val 2019; una corrida
   multienv reporta **ambas** cabezas optimal+drought).
3. Comparar contra GBLUP por ambiente: `exp_gblup_2017_2018_to_2019_drought.yaml` (drought) y
   `exp_gblup_2017_2018_to_2019.yaml` (optimal). **Ese** es el número que va al informe.
4. 2020 sigue sellado hasta el cierre.
