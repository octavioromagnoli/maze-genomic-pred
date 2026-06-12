# Bitácora

Registro vivo del proyecto. Cada número del informe linkea a su corrida de
wandb; nada copiado a mano.

---

## Datos (formato de los CSV)

Los datos viven en `data/` y se cargan **tal cual**, sin validaciones, vía
[`src/data/loader.py`](../src/data/loader.py). Dos archivos:

### `Geno_data.csv` — genotipos por línea
- Columna `GID` (id de línea) + **4612** columnas SNP binarias (nombres `L9.*`),
  valores `{0, 1}`. Una fila por línea (~3528 líneas).

### `Pheno_data.csv` — fenotipos, formato *wide*
Una fila por `GID`. Columnas:

| columna       | tipo  | nota                                   |
|---------------|-------|----------------------------------------|
| `GID`         | id    | joinea con `Geno_data.csv`             |
| `Year`        | int   | {2017, 2018, 2019, 2020}               |
| `GY_Optimal`  | float | Grain Yield, ambiente óptimo           |
| `PH_Optimal`  | float | plant height, óptimo                   |
| `AD_Optimal`  | float | days to anthesis, óptimo               |
| `GY_Drought`  | float | Grain Yield, sequía                    |
| `PH_Drought`  | float | plant height, sequía                   |
| `AD_Drought`  | float | days to anthesis, sequía               |

Celdas vacías = faltante (se descartan al armar el target). Cada `GID` pertenece
a un único año.

### Matriz de modelado
`Dataset.to_modeling_arrays(target, env)` selecciona la columna `<target>_<Env>`,
descarta faltantes, joinea el genotipo por `GID` y devuelve `ModelingArrays`
(`X`, `y`, `year`, `line_id`, `env`). `env=None` apila óptimo + sequía como
observaciones separadas. Rutas de los CSV vía config (`data.geno_csv`,
`data.pheno_csv`); nada hardcodeado.

---

## Fase 0 — Infraestructura

**Objetivo:** corrida end-to-end verde (dummy → métrica → wandb), antes de
cualquier modelo real.

### Qué quedó armado
- **Scaffold**: `src/{data,models,training,eval,utils}`, `scripts/`, `configs/`,
  `docs/`, `notebooks/`. `data/`, `experiments/`, `wandb/`, `checkpoints/`
  gitignored. Seeds en el config (`seed: 67`), logueadas a wandb.
- **Datos**: loader que lee los dos CSV directo de `data/`, sin validaciones, y
  arma los arrays de modelado joineando genotipo↔fenotipo por `GID`.
- **Harness**:
  - `src/eval/metrics.py`: Pearson r (portada), RMSE, R², robustos a casos
    degenerados; agregación por fold.
  - `src/eval/splits.py`: forward-chaining temporal + split agrupado por línea,
    con guard `assert_no_line_leakage` corrido en cada fold.
  - Modelo dummy `MeanPredictor` (predictor por la media).
  - `scripts/train.py`: único entrypoint (config → seed → CV → wandb).

### Decisiones y notas
- **Pearson del dummy = `nan` por construcción**: predicción constante ⇒ varianza
  cero ⇒ Pearson indefinido. Es el resultado nulo esperado (por eso hace falta un
  modelo con señal); RMSE y R² sí quedan bien definidos. El "verde" de Fase 0 es
  que el pipeline corre y loguea de punta a punta, no que el Pearson sea alto.
- **R² negativo del dummy**: predecir la media de años anteriores sobre un año
  posterior es peor que su propia media ⇒ R² < 0. Refleja el shift temporal entre
  campañas (justamente lo que mide la validación forward-chaining).
- **Folds** (config dummy, target=GY, env=optimal; dev = 2017–2019, 2020 holdout):
  fold 0 → train {2017} (n=901), val 2018 (n=1418);
  fold 1 → train {2017,2018} (n=2319), val 2019 (n=722).
  Como cada `GID` está en un solo año, ninguna línea cruza folds.
- **wandb offline** por defecto (sin login). Sincronizar con `wandb sync`.

### Corrida de referencia
- Config: `configs/exp_dummy.yaml`.
- Run de wandb: _(pegar acá el link/id de la corrida una vez sincronizada)_.
