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

## Fase 1 — EDA

**Objetivo:** caracterizar datos y targets, medir estructura poblacional / drift y
relatedness, y **cerrar dos decisiones**: (a) protocolo de validación, (b) filtros
de preprocesamiento.

- **Notebook:** [`notebooks/eda.ipynb`](../notebooks/eda.ipynb) — solo orquesta y
  grafica.
- **Lógica reutilizable:** [`src/eda/eda.py`](../src/eda/eda.py) — cómputo puro (MAF,
  correlaciones, PCA/drift, IBS), importable y testeable. Todos los números de abajo
  son reproducibles corriendo el notebook (sin wandb: es EDA descriptivo).

### Sellado del test 2020 (anti-leakage)
Todo el EDA se hace **solo sobre dev (2017–2019)**. El año 2020 **no se toca** —ni
features ni targets— hasta la evaluación de cierre, como pide la regla del proyecto
(*"intocable hasta el cierre"*). Mirar el test, aunque sea solo `X`, mete *researcher
degrees of freedom* y le saca su rol de prueba limpia; y **no hace falta**: ninguna
decisión de esta fase depende de 2020 (el drift *entre años de dev* ya alcanza para
fijar la validación temporal). Lo único que se usa de 2020 es metadato de gestión
(existe, 486 líneas, mismas columnas, sin faltantes-relevantes).

### Qué muestran los datos (todo sobre dev)
- **Shapes / integridad.** 3528 líneas × 4612 SNPs = **3042 dev + 486 test sellado**;
  join genotipo↔fenotipo perfecto. Genotipo **binario `{0,1}` y sin faltantes**.
  Fenotipo: 1–2 celdas faltantes por columna (todas en dev, <0.07 %), descartadas por
  bloque (target, env) en el loader; no se imputan targets. Cada `GID` está en **un
  único año** ⇒ ninguna línea cruza años.
- **QC de SNPs (dev).** **0 monomórficos**; **MAF mínima 0.033**. La matriz viene
  curada upstream a `MAF≥0.05` sobre *todas* las campañas; recomputada solo sobre dev,
  **7 SNPs** caen por debajo de 0.05 (`MAF≥0.10` sacaría innecesariamente 148). 
- **Targets (dev).** El estrés hídrico baja fuerte el rendimiento (GY ≈ 7.0 óptimo vs
  ≈ 3.0 sequía) y la altura (PH 246 vs 211). Efecto año marcado en las medias (cada
  campaña es un ambiente distinto). `GY_Optimal` con cola derecha leve: 3 puntos sobre
  la cota IQR (máx 11.7) — plausibles, **se conservan**.
- **¿Multitarea? (dev).** Dentro de un ambiente, **GY↔PH r≈0.64** (óptimo) / 0.31
  (sequía), GY↔AD débil (0.31 / −0.04), PH↔AD moderada en óptimo (0.31) / casi nula en
  sequía (0.10). Señal compartida sobre todo en GY↔PH ⇒ multitarea justificada como
  **regularización por tarea auxiliar** (PH auxiliar de GY), no por redundancia.
- **GxE (clave, dev).** `r(GY_Optimal, GY_Drought) = 0.14`: el rendimiento en óptimo y
  en sequía es casi un trait distinto. PH (0.51) y AD (0.62) más estables entre
  ambientes. Implicancia: **GY se modela por ambiente**; apilar óptimo+sequía mezcla
  dos problemas poco correlacionados.
- **Estructura y drift (dev).** PCA con varianza **muy difusa** (PC1 ~6.3 %, 10 PC
  ~31 %) → la compresión lineal retiene poco: **anticipa el AE no lineal**. Los
  centroides de los 3 años de dev se separan netamente. Drift cuantificado con
  separabilidad genética (CV-AUC de logística sobre PCs): **2017 vs 2018 = 0.95, 2018
  vs 2019 = 0.93, 2017 vs 2019 = 0.98**. Cohortes anuales genéticamente distintas ⇒ un
  CV aleatorio filtraría la estructura de año.
- **Relatedness / duplicados (dev).** **0 clones exactos** (3042 perfiles únicos),
  pero relatedness alta: NN-IBS mediana ≈0.88, 25 % de las líneas con un vecino > 0.90.
  Cruzando años de dev: **59 pares con IBS > 0.99 son cross-año** y **16 líneas (0.5 %)**
  tienen un casi-clon en otro año de dev → parientes a ambos lados de la frontera
  train(17–18)/val(19): inevitable y realista (escenario de despliegue).

### Decisión (a) — Protocolo de validación: **split temporal fijo** (LOCK)
Train **2017–2018**, validación **2019**, test final **2020** (intocable). **No**
forward-chaining. Materializado en
`fixed_temporal_split(val_year=2019, holdout_test_year=2020)` (split a nivel línea:
train 2320, val 722, test 486; la n por target baja apenas al descartar los pocos
fenotipos faltantes). Fundamentos del EDA (**sin mirar 2020**):
- **Drift fuerte entre años de dev** (AUC 0.95–0.98) ⇒ la validación tiene que ser
  temporal; un CV aleatorio mezclaría cohortes y sobreestimaría el desempeño.
- **Forward-chaining daría un fold entrenado solo en 2017** (n≈900), la cohorte más
  atípica del dev (AUC 0.95 vs 2018, 0.98 vs 2019) → estimación ruidosa, pesimista y
  poco representativa, sin comprar robustez con apenas 3 años de dev.
- El split fijo **maximiza el train** y **replica el despliegue**: 2019 (año más
  reciente del dev) es el análogo natural de "predecir la campaña siguiente".
- **Líneas disjuntas garantizadas** (cada `GID` en un año). El ~0.5 % de casi-clones
  cross-año se **acepta como realista** (en mejoramiento las líneas de un año son
  parientes de las del siguiente), y queda documentado.

### Decisión (b) — Filtros de preprocesamiento (LOCK)
- **MAF ≥ 0.05**, **recalculado con el train de cada fold**. Sobre dev descarta 7 SNPs
  (deja 4605); por fold el número puede variar. Sin remoción de monomórficos (no hay).
- **Imputación de genotipo: no requerida** (0 faltantes). Política codificada por si
  una reextracción introdujera NaN: **alelo mayoritario ajustado solo en el train del
  fold** (`eda.impute_major_allele`). **Fenotipo:** las 1–2 celdas faltantes por
  columna se **descartan** por bloque (target, env); no se imputan targets.
- **Escalado:** SNPs binarios → estandarización por-SNP (media/var) **solo para los
  modelos lineales** (Ridge/GBLUP), ajustada en train; el AE consume `{0,1}` crudo con
  salida sigmoide + BCE (regla 4). *(Estar en {0,1} fija el rango, no la escala: un SNP
  binario tiene varianza p(1−p), que entre MAF 0.05 y 0.5 va ~5×; la L2 de Ridge no es
  invariante a escala → sin estandarizar sobre-penaliza los SNPs de baja MAF. Por eso
  es opt-in por modelo.)*

**Implementación (general, train-only).** El preprocesamiento vive una sola vez en
[`src/data/preprocess.py`](../src/data/preprocess.py) (`SNPPreprocessor`: filtro de MAF
+ estandarización opcional, ajustado SOLO con el train) y se compone con **cualquier**
modelo vía `PreprocessedModel` (en [`src/models/registry.py`](../src/models/registry.py)),
sin reescribirlo por modelo. Se activa con el bloque `preprocess:` del config
(`maf_threshold`, `standardize`) y `build_model` lo envuelve automáticamente; viaja
dentro del artifact guardado (reusar no puede aplicar otro preprocesamiento). El hash
del caché incluye `preprocess`. Verificado con el dummy: RMSE 1.3414 sin cambios (ignora
X) y el train 2017–2018 filtra 21 SNPs (el conteo depende del fold).
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

## Fase 4 — XGBoost base

**Objetivo:** primer modelo no lineal sobre los SNPs. Gradient boosting de árboles
(XGBoost) como baseline alternativo a GBLUP, para contrastar si capturar
interacciones/no-linealidades aporta sobre el modelo lineal mixto en régimen p≫n.

### Qué quedó armado
- **Modelo** [`src/models/xgb.py`](../src/models/xgb.py): `XGBoostRegressor(BaseModel)`,
  wrapper fino sobre `xgboost.XGBRegressor` enganchado a la interfaz común
  (`fit`/`predict`) y al harness de Fase 0. Registrado como `xgboost` en
  [`src/models/registry.py`](../src/models/registry.py). No hace preprocesamiento
  propio: con bloque `preprocess:` lo envuelve `PreprocessedModel` (máscara MAF
  train-only, regla 1).
- **Configs**: [`configs/exp_xgboost_2017_2018_to_2019.yaml`](../configs/exp_xgboost_2017_2018_to_2019.yaml)
  (early stopping temporal) y
  [`configs/exp_xgboost_2017_2018_to_2019_2000fixed.yaml`](../configs/exp_xgboost_2017_2018_to_2019_2000fixed.yaml)
  (2000 árboles fijos, sin early stopping). HP por `model.params`, sin tocar código.

### Early stopping temporal + refit sobre todo el train (regla 3)
- Con `params.early_stopping_rounds`, en vez de un holdout aleatorio se usa un
  **inner-split temporal** dentro del train del fold (`temporal_inner_split`):
  inner-train = años previos, inner-val = último año (train 2017+2018 ⇒ ajusta
  2017, monitorea 2018). El eval set vive dentro del train externo, no toca
  val/test (no viola la regla 3).
- XGBoost agrega árboles de a uno y mide RMSE sobre el inner-val ronda a ronda;
  `early_stopping_rounds` es la **paciencia** (frena tras N rondas sin mejora) y se
  queda con `best_iteration`. El wrapper toma `selected = best_iteration + 1`
  (acotado a `[1, n_estimators]`).
- **`refit_full_train`** (default `True`): el modelo de selección se descarta y se
  reentrena uno final desde cero con `selected` árboles sobre **todo** el train del
  fold (2017+2018). Así el número de árboles se valida temporalmente pero el modelo
  final no desperdicia el último año. Con un solo año en train el early stopping se
  desactiva solo y se usan los `n_estimators` fijos.

### Integración con `PreprocessedModel` (correcciones)
- `run_evaluation` setea el `fit_context` (con `year`) sobre el modelo **externo**.
  `PreprocessedModel` ahora **propaga `set_fit_context` al modelo interno**; sin
  esto el `XGBoostRegressor` envuelto recibía `year=None` y el early stopping
  temporal quedaba silenciosamente desactivado (todos los configs usan `preprocess:`).
- `PreprocessedModel.fit` delega en `fit_with_preprocessor` si el modelo interno la
  expone: para la **selección** el preprocesador (máscara MAF) se ajusta solo con el
  inner-train; para el **modelo final refitteado** se ajusta con todo el train
  externo. Refuerza la regla 1 en el paso de selección.
- `PreprocessedModel.get_fitted_params` propaga los del modelo interno (para que
  wandb logee `best_iteration`/`selected_n_estimators` sin bajar el modelo).
- **Entorno**: en local (XGBoost 3.2.0 + torch) `n_jobs=-1` segfaultea por doble
  carga de `libomp`. Default conservador `n_jobs=1`; alternativa al correr vía
  entorno: `OMP_NUM_THREADS=1`.

### Verificación
- **Smoke test sintético** (`build_model` + `set_fit_context`): el early stopping
  engancha con ≥2 años (selecciona árboles, refittea), se desactiva con 1 año, y
  los `fitted_params` se propagan por el wrapper.
- **Control de no-regresión**: con early stopping desactivado se reproduce
  **exactamente** el resultado de 2000/1000 árboles fijos, aislando el efecto de la
  selección.

### Corridas de referencia (`GY/optimal`, train 2017–2018 `n=2319`, val 2019 `n=722`, 2020 holdout)
- **Early stopping (base)** ([run `qssy1kot`](https://wandb.ai/oromagnoli-/maze-genomic-pred/runs/qssy1kot)):
  selecciona **5 árboles** (`best_iteration=4`; el RMSE de inner-val 2018 no mejora
  más allá), refit sobre todo el train. Pearson `0.0966` IC95% `[0.0322, 0.1669]`,
  RMSE `1.3216`, R² `−0.7158`.
- **2000 árboles fijos** ([run `z9wmphtw`](https://wandb.ai/oromagnoli-/maze-genomic-pred/runs/z9wmphtw)):
  Pearson `0.2164` IC95% `[0.1442, 0.2849]`, RMSE `1.3498`, R² `−0.7898`. Duplicar
  los árboles respecto de 1000 no cambia nada (saturado): el techo de esta familia
  de HP por la vía de "más árboles" es ~`0.216`.
- **Lectura**: el early stopping minimiza **RMSE de 2018** (inner-val), objetivo que
  diverge de la métrica de portada (**Pearson de 2019**); frena tempranísimo y
  underfittea para el ranking. Ese techo de árboles fijos (~`0.216`) queda igual
  **por debajo de GBLUP-REML** (`0.3145`): el boosting de árboles no cierra la
  brecha del modelo lineal mixto en este p≫n; los grados de libertad pendientes son
  `max_depth`/`learning_rate`/submuestreo/regularización, no más árboles.
- Runs de referencia sincronizadas a wandb (`qssy1kot`, `z9wmphtw`). El cierre
  sobre 2020 sigue intocable (regla 3).

## Fase 4 — MLP base (red neuronal)

**Objetivo:** segundo modelo no lineal de la fase base. Un perceptrón multicapa
(PyTorch) que toma los SNPs crudos como capa de entrada y predice GY con una
cabeza de regresión lineal, para contrastar una no-linealidad por
representaciones distribuidas (frente a los árboles de XGBoost y al lineal mixto
de GBLUP) en régimen p≫n.

### Qué quedó armado
- **Modelo** [`src/models/mlp.py`](../src/models/mlp.py): `MLPRegressor(BaseModel)`
  enganchado a la interfaz común (`fit`/`predict`) y al harness de Fase 0.
  Arquitectura configurable (`_MLPNet`): [dropout de entrada] → bloques
  `Linear[+BatchNorm]+activación+dropout` → `Linear(1)`. Optimizador Adam con
  weight decay y pérdida **MSE**. Registrado como `mlp` en
  [`src/models/registry.py`](../src/models/registry.py).
- **Config** [`configs/exp_mlp_2017_2018_to_2019.yaml`](../configs/exp_mlp_2017_2018_to_2019.yaml):
  arquitectura, regularización, optimizador y early stopping por `model.params`,
  sin tocar código.

### Preprocesamiento train-only dentro del modelo + persistencia torch
- A diferencia de XGBoost, el MLP **no usa el bloque `preprocess:`/`PreprocessedModel`**:
  hace su estandarización (z-score de features y del target) **dentro de `fit`**,
  ajustada solo con el train del fold (regla 1). El motivo es la persistencia:
  `PreprocessedModel` serializa con joblib (picklearía el `nn.Module`), lo que
  choca con el contrato del proyecto de que los modelos torch overridean
  `save`/`load` con `state_dict` + arquitectura. Así todo el preprocesamiento
  (medias/desvíos de X e y) viaja dentro del checkpoint torch.
- **`save`/`load` override**: el checkpoint guarda arquitectura (`cfg_`),
  `state_dict`, escaladores y diagnóstico; `load_model_artifact` lo reconstruye
  sin reentrenar, con el guard de `train_years` activo.

### Early stopping temporal + refit sobre todo el train (regla 3)
- Con `params.patience` se reserva el **último año del train** como inner-val
  (`temporal_inner_split`: train 2017+2018 ⇒ ajusta 2017, monitorea 2018), mismo
  criterio que XGBoost. Los estadísticos de escala se recomputan con el
  inner-train (sin leakage en la selección).
- La selección elige el número de épocas que **maximiza el Pearson de inner-val**
  (`val_metric: pearson`) y luego, con `refit_full_train: true`, reentrena el
  modelo final desde cero con todo el train (2017+2018) por las épocas elegidas.
- **Por qué Pearson y no MSE en la selección**: con p≫n la red sobreajusta desde
  la primera época, así que el **MSE de inner-val es mínimo en la época 0**
  (predecir la media) y `val_metric: loss` degeneraría a entrenar ~1 época. El
  Pearson —la métrica de portada (regla 5)— sigue mejorando unas épocas y elige
  un punto útil. El default de código queda en `loss` (convención segura,
  no-NaN); el experimento usa `pearson` con esa justificación.

### Decisiones de diseño
- **MSE, no BCE**: la regla 4 (BCE) aplica al autoencoder que reconstruye SNPs
  binarios; acá la salida es un fenotipo continuo.
- **Reproducibilidad**: `random_state` siembra el RNG de torch dentro de `fit`
  (la seed global no fija la inicialización de pesos por fold). Verificado: dos
  ajustes con la misma seed dan predicción idéntica. El historial por época queda
  en `self.history_` (el harness no propaga un hook por época; los escalares de
  diagnóstico van por `get_fitted_params`).

### Verificación
- **Smoke test sintético** (`build_model` + `set_fit_context`): registry, early
  stopping con ≥2 años (selecciona épocas, refittea) y su desactivación con 1
  año, save/load con round-trip exacto de la predicción, y reproducibilidad por
  seed.
- **End-to-end sobre datos reales** vía `scripts/train.py` (wandb offline,
  bootstrap, resumen, copia local + Artifact `mlp-2017-2018-to-2019`); el
  Artifact recarga vía `load_model_artifact` y el guard de `train_years` rechaza
  su reutilización bajo un régimen incompatible.

### Corrida de referencia (`GY/optimal`, train 2017–2018 `n=2319`, val 2019 `n=722`, 2020 holdout)
- Arquitectura `[128]` (1 capa oculta, 590 593 parámetros), `dropout=0.5`,
  `input_dropout=0.3`, `lr=5e-4`, `weight_decay=1e-3`, `batch_size=64`,
  `max_epochs=200`, `patience=25`, selección por Pearson.
- El early stopping eligió **6 épocas** (`best_epoch=5`; Pearson de inner-val
  2018 `0.366`) y refitteó sobre todo el train. Pearson `0.2908` IC95%
  `[0.2268, 0.3585]`, RMSE `1.2640` IC95% `[1.2019, 1.3195]`, R² `−0.5695`
  IC95% `[−0.7395, −0.4114]`. *(Corrida offline; sincronizar con `wandb sync`
  para el link.)*
- **Baseline nulo (predictor de media 2017+2018 sobre 2019)**: Pearson **`nan`**
  (predicción constante ⇒ varianza cero ⇒ Pearson indefinido), RMSE `1.3414`,
  R² `−0.7677`. El MLP le gana claramente en RMSE (`1.2640 < 1.3414`) y aporta
  el ranking que el predictor de media no tiene.
- **Lectura comparada** (mismo split): el MLP da el **mejor RMSE y el R² menos
  negativo** de los modelos base, y un Pearson `0.291` que se ubica por **encima
  de XGBoost** (`0.216`, árboles fijos) y apenas por **debajo de GBLUP-REML**
  (`0.3145`), con ICs solapados (`[0.227, 0.359]` vs `[0.247, 0.380]`). La
  no-linealidad por red no supera al lineal mixto en este p≫n, pero queda a la
  par y mejora la calibración (RMSE/R²). El cierre sobre 2020 sigue intocable
  (regla 3).
