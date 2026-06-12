# maze-genomic-pred

Predicción de rendimiento de grano (Grain Yield) en maíz a partir de ~4.612
marcadores genéticos (SNPs) binarios de ~3.529 líneas. Regresión supervisada en
régimen *p ≫ n*. Datos por campaña 2017–2020 (CIMMYT, Lopez-Cruz et al.).
Targets secundarios: altura de planta (PH) y días a floración (AD). Baseline de
referencia: Ridge / GBLUP.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Unix: source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (Fase 0)

Los datos ya están en `data/` (`Geno_data.csv`, `Pheno_data.csv`).

```bash
# 1) Correr la corrida dummy end-to-end (loguea a wandb en modo offline)
python scripts/train.py --config configs/exp_dummy.yaml

# 2) (opcional) sincronizar la corrida offline con la nube
wandb sync
```

## Estructura

```
src/
  data/      loader de los CSV (sin validaciones)
  models/    interfaz fit/predict + dummy + registry
  training/  loop de CV reutilizable
  eval/      métricas (Pearson/RMSE/R²) + splits antileakage
  utils/     seeding + carga de config
scripts/     train.py (único entrypoint)
configs/     un YAML por experimento
docs/        bitácora (incluye el formato de los datos)
notebooks/   solo exploración; importan de src/
```

`data/`, `experiments/`, `wandb/` y `checkpoints/` están gitignored.
