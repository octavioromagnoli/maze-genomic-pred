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
  gitignored. Seeds en el config (`seed: 575`), logueadas a wandb.
- **Datos**: loader que lee los dos CSV directo de `data/`, sin validaciones, y
  arma los arrays de modelado joineando genotipo↔fenotipo por `GID`.
- **Harness**:
  - `src/eval/metrics.py`: Pearson r (portada), RMSE, R², robustos a casos
    degenerados; agregación por fold + IC por bootstrap (`bootstrap_ci`).
  - `src/eval/splits.py`: split temporal fijo (`fixed_temporal_split`) +
    `enforce_disjoint_lines`, con guard `assert_no_line_leakage` por fold.
  - Modelo dummy `MeanPredictor` (predictor por la media).
  - `scripts/train.py`: único entrypoint (config → seed → split → wandb), vía
    `run_evaluation`.

### Decisiones y notas
- **Pearson del dummy = `nan` por construcción**: predicción constante ⇒ varianza
  cero ⇒ Pearson indefinido. Es el resultado nulo esperado (por eso hace falta un
  modelo con señal); RMSE y R² sí quedan bien definidos. El "verde" de Fase 0 es
  que el pipeline corre y loguea de punta a punta, no que el Pearson sea alto.
- **R² negativo del dummy**: predecir la media de años anteriores sobre un año
  posterior es peor que su propia media ⇒ R² < 0. Refleja el shift temporal entre
  campañas (justamente lo que mide la validación temporal).
- **Split** (config dummy, target=GY, env=optimal; dev = 2017–2019, 2020 holdout):
  un único split fijo → train {2017, 2018} (n=2319), val 2019 (n=722). Como cada
  `GID` está en un solo año, ninguna línea cruza train/val (val 2019 = todo 2019).
- **wandb offline** por defecto (sin login). Sincronizar con `wandb sync`.

### Revisión: split fijo + IC por bootstrap
- **De CV a split fijo.** Con solo 3 años de dev (2017–2019), el forward-chaining
  daba 2 folds y uno entrenaba con un único año → más ruido que señal. Se pasó a
  un **split temporal fijo** (train 2017–2018, val 2019): replica el escenario real
  (entrenar todo el dev → predecir el año siguiente, 2020) y maximiza el train. El
  esquema se elige por config (`cv.scheme: fixed_temporal`, `cv.val_year`).
- **Limpieza del harness.** Se removieron el forward-chaining
  (`temporal_grouped_folds`, `temporal_forward_chaining_folds`) y el K-fold por
  línea (`grouped_by_line_folds`), que ya no se usaban. Quedan el split fijo, el
  guard antileakage y `final_test_split` (para el cierre sobre 2020). `run_cv` →
  `run_evaluation`.
- **IC por bootstrap.** Cada métrica reporta IC 95% por **bootstrap cluster a
  nivel línea** (`group=line_id`) sobre validación, configurable con
  `bootstrap.{enabled,n_boot,alpha}`. Se loguea como series (`fold/*_lo`,
  `fold/*_hi`), `run.summary` (`rmse_lo/hi`, `r2_lo/hi`) e histogramas
  (`boot/rmse`, `boot/r2`). Reproducible (seed derivada por fold). El IC de
  Pearson queda `nan` en el dummy por el mismo motivo que el punto.
- **Números de referencia** (dummy, GY/optimal): RMSE 1.3414, IC95% [1.2796,
  1.4013]; R² −0.7677, IC95% [−0.9389, −0.6356].
- **Notebook de comparación.** `notebooks/comparar_modelos_ci.ipynb` corre
  `run_evaluation` (sin wandb, offline) sobre una lista de configs y grafica
  punto ± IC95% (errorbar) y boxplot con bigotes en los percentiles 2.5/97.5
  (= el IC95%). Reutilizable: sumar un modelo = agregar su config al dict
  `CONFIGS`. Figuras en `experiments/figs/`.

### Corrida de referencia
- Config: `configs/exp_dummy.yaml`.

### Persistencia: caché de resultados + Artifacts de modelo
Centralizado en [`src/training/save_model.py`]

- **Caché de resultados (notebook sin reentrenar).** `run_evaluation` se cachea a
  disco indexado por hash de `(data, cv, model, seed, n_boot, alpha)`. El notebook
  `comparar_modelos_ci.ipynb` lo usa vía `USE_CACHE`: en *miss* evalúa y guarda; en
  *hit* levanta de `outputs/eval_cache` (gitignored) sin reentrenar. Cambiar
  cualquier hiperparámetro del config invalida su entrada sola. Verificado: hit
  reproduce RMSE 1.3414 en ~2 ms vs ~1 s del recómputo.
- **Artifacts de modelo.** `BaseModel` ahora define `save`/`load` (default joblib;
  los modelos torch deben overridear con `state_dict` + arquitectura). Con
  `artifacts.save_model: true` en el config, `scripts/train.py` guarda el modelo del
  fold de validación como copia local (`checkpoints/<name>/`) + wandb Artifact
  (`type=model`), con metadata `{model_name, target, env, train_years, seed,
  val_metrics}`. `run_evaluation(..., return_models=True)` expone los modelos sin
  cambiar la firma por defecto.
- **Guard anti-leakage al reusar** `load_model_artifact(ref,
  expected_train_years=...)` compara los `train_years` del artifact contra los del
  fold donde se lo quiere reusar y levanta `LeakageError` si difieren. Así, reusar un
  encoder/AE para proyectar al espacio latente no puede meter datos de val/test en el
  ajuste. Implicancia para el cierre: el modelo para predecir 2020 se reentrena sobre
  todo el dev (2017–2019) → es otro artifact, con distinto `train_years`.
- **Verificado end-to-end** con el dummy: `train.py` guardó `dummy-mean-gy-optimal`
  (`train_years [2017, 2018]`), recargable con el guard activo; el notebook hace
  cache hit con los mismos números.

---

## Fase 2 — GBLUP (baseline genómico)

**Objetivo:** primer modelo con señal. GBLUP (Genomic BLUP) para predecir los
valores de cría genómicos `û_PS`, con el ratio de varianzas `λ₀ = σ²_E/σ²_u`
obtenido por dos metodologías independientes. Resuelve la ecuación (2) del paper
base (Lopez-Cruz et al.), forma *hat matrix*:

    û_PS = G_PS,TS (G_TS + λ₀ I)⁻¹ (y_TS − μ̂)

### Qué quedó armado
- **Modelo** [`src/models/gblup.py`](../src/models/gblup.py): `GBLUP(BaseModel)`
  enganchado a la interfaz común (`fit`/`predict`) y al harness de Fase 0. Funciones
  modulares: `standardize_train`/`apply_standardize` (centrado+escalado de marcadores),
  `genomic_relationship` (bloque `Z Z'/p`), `lambda_via_cv` (Enfoque A) y
  `lambda_via_reml` (Enfoque B). Registrado como `gblup` en `src/models/registry.py`.
- **Configs**: [`configs/exp_gblup_cv.yaml`](../configs/exp_gblup_cv.yaml) (Enfoque A) y
  [`configs/exp_gblup_reml.yaml`](../configs/exp_gblup_reml.yaml) (Enfoque B). El método
  de `λ₀` se elige por `model.params.lambda_method`, sin tocar código.

### Preprocesamiento y matriz G (train-only, regla 1)
- Marcadores centrados y escalados por columna con estadísticos **de train**:
  `Z = (X − X̄)/sd`. Columnas de varianza nula (SNP constante en train) → `sd=1`
  (quedan en cero tras centrar, no aportan). El mismo escalador se aplica a val/test.
- Matriz genómica `G = Z Z'/p` (p = nº de marcadores). El *slicing* en `G_TS`
  (train×train) y `G_PS,TS` (val×train) sale natural del split temporal fijo de
  Fase 0: el harness entrega `X_tr/y_tr` (train) y `X_va` (val) ya line-disjoint.
- **Fenotipo centrado por μ̂ = media de train** (convención BLUP) y se suma de vuelta
  en la predicción: `ŷ_PS = μ̂ + û_PS`. El paper escribe `y_TS` directo, pero sin el
  centrado `fit_intercept=False` no sería correcto y RMSE/R² quedarían sin sentido.
  Pearson es invariante a este corrimiento; RMSE/R² mejoran y quedan bien definidos.
- Predicción por sistema lineal estable (`np.linalg.solve`, sin invertir).

### Enfoque A — `λ₀` por validación cruzada temporal
- La selección interna replica el criterio temporal del harness: dentro del train
  externo 2017–2018 se usa **inner-train 2017 → inner-val 2018**. Se prueba la
  rejilla `np.logspace(-4, 4, 100)` y se elige el `λ₀` que minimiza el SSE de 2018.
- La CV recibe `X` e `y` crudos. En cada fold recalcula la media/desvío de los SNP
  y la media del fenotipo usando **solo inner-train**, y aplica esos estadísticos a
  inner-val. Esto evita que la información de 2018 intervenga en el ajuste interno.
- Para cada candidato se evalúa directamente el predictor GBLUP:
  `μ_it + G_iv,it (G_it + λ₀I)⁻¹(y_it − μ_it)`. No se usa `RidgeCV(X=G_TS)`,
  porque eso ajustaría ridge ordinaria sobre `G` como features y optimizaría un
  parámetro distinto del `λ₀` de la fórmula *hat matrix*.
- `model.params.internal_cv: kfold` queda disponible como alternativa; el default y
  la corrida de referencia usan `temporal`.

### Enfoque B — `λ₀` por REML (componentes de varianza)
- Equivalente conceptual a las medias a posteriori de BGLR (MCMC), pero por **REML**
  sin entorno MCMC. Modelo mixto `y = 1·μ + g + e`, `g ~ N(0, σ²_u G_TS)`,
  `e ~ N(0, σ²_E I)`.
- **Estrategia**: eigen-descomposición `G_TS = V D V'` (`np.linalg.eigh`, clip de
  autovalores negativos numéricos), rotación de `y` y del intercepto, y optimización
  numérica (`scipy.optimize.minimize_scalar`, búsqueda acotada en `log₁₀λ`) de la
  verosimilitud **restringida** perfilando `μ` (GLS) y `σ²_u`. Devuelve `λ₀`, `σ²_u`,
  `σ²_E` (expuestos vía `get_fitted_params` para la bitácora).

### Persistencia: representación rrBLUP (efectos de marcador)
- Se mantiene **toda la derivación GBLUP** (se construye `G_TS`, se resuelven los
  coeficientes duales `c = (G_TS+λ₀I)⁻¹ y_c` — eq. 2 intacta), pero el predictor se
  **almacena como vector de efectos de marcador** `b = Z_tr' c / p`. Como
  `û_PS = G_PS,TS c = (Z_PS Z_tr'/p) c = Z_PS b`, predecir es un solo producto `Z_PS·b`
  y el modelo **ya no guarda la matriz de train `Z_tr`** (n×p ≈ 85 MB) sino solo `b`
  (largo p ≈ 37 KB). Mismo predictor, artifact ~2000× más chico y `predict` más rápido.
  Verificado: `predict` reproduce la hat-matrix explícita a `2e-15`.

### Verificación
- **Smoke test sintético** (datos chicos con el formato del loader) vía `run_evaluation`:
  ambos enfoques corren end-to-end con split temporal fijo + IC por bootstrap,
  métricas finitas, identidad hat-matrix exacta y escalado train-only.
- **Verificado sobre los datos reales** (`GY/optimal`): train 2017–2018
  (`n=2319`), validación 2019 (`n=722`), con 2020 excluido como holdout final.
  Ambas corridas completaron `scripts/train.py` con wandb offline, bootstrap,
  resumen, copia local y Artifact de modelo.
- **CV temporal:** `λ₀=10000` (borde superior de la grilla), Pearson `0.2304`
  IC95% `[0.1659, 0.2977]`, RMSE `1.3403` IC95% `[1.2784, 1.3993]`,
  R² `−0.7649` IC95% `[−0.9380, −0.6425]`.
- **REML:** `λ₀=0.6850`, `σ²_u=0.7821`, `σ²_E=0.5357`, Pearson `0.3145`
  IC95% `[0.2469, 0.3802]`, RMSE `1.3016` IC95% `[1.2400, 1.3636]`,
  R² `−0.6644` IC95% `[−0.8559, −0.4976]`.
- Los Artifacts `gblup-cv-gy-optimal` y `gblup-reml-gy-optimal` recargan con
  sus parámetros estimados y el guard de `train_years` rechaza su reutilización
  bajo un régimen de entrenamiento incompatible.

### Corridas de referencia
- Configs: `configs/exp_gblup_cv.yaml` (Enfoque A) y `configs/exp_gblup_reml.yaml`
  (Enfoque B). Corridas ejecutadas en modo wandb offline. El cierre sobre 2020
  sigue pendiente y requiere reentrenar sobre todo el dev 2017–2019.
