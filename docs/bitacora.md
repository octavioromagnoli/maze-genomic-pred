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

## Fase 3 — Reducción de dimensionalidad (PCA) + GBLUP

**Objetivo:** ver si comprimir los 4612 SNPs con PCA (train-only) antes de GBLUP
mejora la predicción, eligiendo la dimensión latente por **desempeño downstream**
(Pearson en val 2019), no por % de varianza. Es además el **baseline lineal**
contra el AE no lineal (Fase siguiente).

### Infra
- `PCAReducer` ([`src/data/preprocess.py`](../src/data/preprocess.py)): envuelve
  `sklearn.PCA`, ajustado SOLO con el train del fold (regla 1), encadenado después
  de `SNPPreprocessor(standardize=True)` (z-score por-SNP con stats de train) vía
  `PreprocChain`. Se activa con el sub-bloque `preprocess.pca` del config;
  `build_model` lo envuelve en `PreprocessedModel`. Loguea `pca_n_components` y
  `pca_cum_var_explained` por fold.
- Sweep [`configs/sweep_pca_dim.yaml`](../configs/sweep_pca_dim.yaml) (grid sobre
  `preprocess.pca.n_components`).

### Trampa p≫n: el techo de rango (por qué el primer sweep estaba roto)
El primer barrido probó dims altas `[900..1200]` y dio un resultado engañoso:
- En p≫n el nº de componentes con varianza > 0 está acotado por `min(n_train−1, p)`,
  **NO por p**. El fold más chico del expanding (train 2017 → val 2018) tiene
  n_train=901 ⇒ techo ~900 componentes, no 4612. `PCAReducer` clampa a
  `min(n_components, min(n, p))` (`preprocess.py:103`).
- Pedir `n_components ≥ n_train` no reduce nada (`pca_cum_var_explained → 1.0`) y
  mete una componente degenerada de varianza ~0; GBLUP la re-estandariza
  internamente (divide por ~0) ⇒ `G_TS` casi-singular, REML manda `λ₀ → 0` y la
  predicción **explota** (RMSE ~1.3e13, R² ~−1e26 en val 2018). Reproducible para
  todo `n_components ≥ 950` (clampados a 901); a 900 todavía andaba (`λ₀≈680`).
- **GBLUP es ~invariante a PCA sin truncar:** PCA es una rotación ortogonal y
  `G = ZZ'/p` se preserva bajo rotación; recién al truncar fuerte cambia. Por eso a
  dims altas (~900–1200 ≈ sin reducción) el Pearson de val 2019 quedaba clavado en
  ~0.31–0.34 ≈ el GBLUP sin PCA (0.3145): no medía nada.

**Corrección:** barrer dims BAJAS `[16, 32, 64, 128, 256, 512]`, bien por debajo de
901, donde el truncamiento es real; y mover la selección al split FIJO.

### Selección sobre split fijo, no expanding
El expanding metía como fold de scoring el train-2017-only (val 2018), que es (a) la
cohorte más atípica del dev (EDA, AUC 0.95–0.98) ⇒ ruidosa y no representativa (ya
rechazada en la Decisión (a) de Fase 1), y (b) la del techo de rango 901 que rompía
PCA. Su Pearson saltaba sin patrón (0.22→0.38→0.23 según la dim) y, promediado en
`pearson_r_mean`, **falseaba la selección** (coronaba dim=128, que en el fold que
importa —2019— es peor que 512). ⇒ Config nuevo
[`configs/exp_pca_gblup_2017_2018_to_2019.yaml`](../configs/exp_pca_gblup_2017_2018_to_2019.yaml):
split fijo train 2017+2018 → val 2019 (un único fold de scoring), al que ahora
apunta el sweep. Así `pearson_r_mean` = Pearson de val 2019 limpio. 2020 sellado.

### Resultados del barrido (split fijo, GY/optimal, train 2017+2018 → val 2019, n=722)
Sweep [`9phyqmc8`](https://wandb.ai/oromagnoli-/maze-genomic-pred/sweeps/9phyqmc8)
(corrido en expanding; el fold val-2019 = el split fijo); reproducido offline con
seed 575 vía `run_evaluation`.

| n_comp | cum_var | Pearson | RMSE | R² | λ₀ |
|-------:|--------:|--------:|-----:|---:|---:|
| 16  | 0.387 | 0.2161 | 1.2429 | −0.518 | 0.79 |
| 32  | 0.480 | 0.2385 | 1.2440 | −0.520 | 0.70 |
| 64  | 0.578 | 0.2488 | 1.2756 | −0.599 | 0.63 |
| 128 | 0.679 | 0.2454 | 1.3119 | −0.691 | 0.55 |
| 256 | 0.779 | 0.2577 | 1.2879 | −0.630 | 0.46 |
| **512** | 0.876 | **0.2991** | 1.2997 | −0.660 | 0.41 |
| **sin PCA (4612)** | 1.000 | **0.3145** | 1.3016 | −0.664 | 0.69 |

- El Pearson sube ~monótonamente con la dim y el máximo es **no reducir** (0.3145):
  todas las dims de PCA quedan por debajo del baseline ⇒ **la reducción lineal no le
  aporta señal a GBLUP, se la cuesta** (coherente con la rotación-invarianza y con el
  EDA: *"PCA muy difusa → anticipa el AE no lineal"*).
- En RMSE el orden casi se invierte (dims bajas levemente mejores: 16 → 1.2429 vs
  1.3016), pero la métrica de portada es Pearson (regla 5) ⇒ gana el baseline.

### Head-to-head: PCA-512 (mejor) vs sin PCA
Vía [`notebooks/comparar_modelos_ci.ipynb`](../notebooks/comparar_modelos_ci.ipynb)
(offline, n_boot=2000, seed 575). Figuras en `experiments/figs/`.

| modelo | Pearson [IC95%] | RMSE [IC95%] | R² [IC95%] |
|---|---|---|---|
| sin PCA | 0.3145 [0.246, 0.378] | 1.3016 [1.240, 1.363] | −0.664 [−0.874, −0.490] |
| PCA-512 | 0.2991 [0.230, 0.363] | 1.2997 [1.236, 1.361] | −0.660 [−0.874, −0.486] |

- **Bootstrap PAREADO de ΔPearson** (sin PCA − PCA-512, mismo resample de líneas):
  Δ=**0.0154**, IC95% **[−0.007, 0.036]**, P(Δ>0)=0.92 ⇒ **diferencia NO
  significativa**. PCA-512 es estadísticamente indistinguible del baseline:
  comprimir 4612→512 no cuesta nada medible, pero tampoco gana. Para mejorar hay que
  cambiar de **modelo** (capturar no-linealidad), no de embedding lineal.

### Re-barrido a dims altas (950–1100): cuidado con el winner's curse
Un segundo sweep
([`ziu59tim`](https://wandb.ai/oromagnoli-/maze-genomic-pred/sweeps/ziu59tim))
sobre dims altas `[950, 1000, 1010, 1040, 1050, 1100]` (split fijo) parece dar
**mejor** que el baseline: el ganador es **dim 1000 → Pearson 0.3402** vs 0.3145.
Hay que leerlo con cuidado:

| dim | 950 | 1000 | 1010 | 1040 | 1050 | 1100 |
|---|---|---|---|---|---|---|
| Pearson | 0.323 | **0.340** | 0.333 | 0.322 | 0.325 | 0.335 |
| cum_var | 0.951 | 0.957 | 0.958 | 0.960 | 0.961 | 0.966 |

- **No es reducción, es denoising de la cola.** `cum_var(1000)=0.957` ⇒ PCA-1000 ≈
  GBLUP completo menos el ~4 % de varianza más baja. Es la misma **meseta de
  rotación-invarianza** (0.32–0.34) de los dims altos; truncar la cola de baja
  varianza actúa como una regularización extra sobre la λ₀ de GBLUP.
- **Selección sobre val 2019 = post-selección (winner's curse).** El "mejor" (1000)
  le gana a 1010/1040 por ruido (no monotónico; cambiar la dim en 10 mueve el
  Pearson 0.007). En bootstrap de val, 1000 es el máximo en 77 % de las réplicas y
  1100 en 22 % ⇒ "algún dim alto gana" es robusto en val 2019, pero *cuál* es ruido.
- **El test pareado, sin embargo, no es nulo.** ΔPearson (baseline − PCA-1000) sobre
  el MISMO resample de líneas = **−0.0255**, IC95% **[−0.049, −0.0025]**,
  P(baseline>PCA-1000)=0.016 ⇒ en val 2019 PCA-1000 es paired-significativamente
  mejor (a diferencia de PCA-512, que no lo era). El efecto es chico (~0.026) pero
  consistente bajo remuestreo.
- **Lectura honesta:** es una **hipótesis de denoising por truncado de PCA**,
  plausiblemente real en parte, pero **inflada por selección** (se eligió la dim
  maximizando el mismo val 2019 contra el que se testea) y medida sobre el único
  holdout de dev. No se puede bankear desde dev: la adjudicación limpia es **2020
  sellado**. Para el informe: PCA **no "vence" a GBLUP de forma robusta**; la métrica
  seleccionada sobre val sobreestima, y de fondo sigue siendo ≈ GBLUP completo.

### Cómo traer el mejor modelo del sweep (infra)
[`src/training/sweep.py`](../src/training/sweep.py): `best_run_from_sweep` (busca el
trial que optimiza `metric` del sweep en `run.summary`) y `best_config_from_sweep`
(aplica sus overrides dot-notation sobre el config base, mismo merge que
`train.py`). Como GBLUP es determinista, el "mejor modelo" se **re-fittea** del
config + seed (bit-a-bit igual a la corrida del sweep); no hace falta bajar pickle.
`comparar_modelos_ci.ipynb` lo usa para comparar baseline vs el mejor del sweep, con
un **test pareado** de la diferencia (las muestras bootstrap de los dos modelos
comparten seed+grupo ⇒ restarlas es un bootstrap pareado válido). Para modelos
caros/estocásticos (NN/AE) el camino es `artifacts.save_model: true` + bajar el
Artifact con `load_model_artifact` (chequea `train_years`, anti-leakage).

### Notas de auditoría (bootstrap + notebook)
- `bootstrap_ci` (percentil, cluster a nivel línea, `metrics.py`) está **correcto**.
  **Pero** en GY/optimal cada línea aparece una sola vez en val (722 líneas = 722
  obs) ⇒ el cluster bootstrap **degenera a bootstrap de filas** (verificado: K==n).
  El clustering por línea recién muerde con `env=None` (apila optimal+drought, 2
  obs/línea).
- El bootstrap mantiene las predicciones **fijas** y resamplea solo el set de val ⇒
  el IC refleja variabilidad de muestreo de la métrica en val, no del entrenamiento.
  Es la interpretación estándar y correcta para "IC del Pearson de validación".
- Para comparar dos modelos no alcanza con ver si sus IC marginales se solapan: se
  necesita el bootstrap **pareado** de la diferencia (agregado en el head-to-head).
  El notebook toma `per_fold[-1]` (ok con split fijo; sería un footgun si se le pasa
  un config expanding con folds de `report`).

### Corridas de referencia
- Sweep dims bajas → split fijo:
  [`sweep_pca_dim.yaml`](../configs/sweep_pca_dim.yaml) →
  `exp_pca_gblup_2017_2018_to_2019.yaml`
  ([`9phyqmc8`](https://wandb.ai/oromagnoli-/maze-genomic-pred/sweeps/9phyqmc8)). El
  primer sweep `[900..1200]`
  ([`wso0ujai`](https://wandb.ai/oromagnoli-/maze-genomic-pred/sweeps/wso0ujai))
  queda como contraejemplo del techo de rango (**no usar**).
- Baseline sin PCA: GBLUP-REML de Fase 2 (0.3145).

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
  
=======
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
 
## Fase 5 — Autoencoder (compresión no lineal)

El PCA (Fase 3) mostró que la reducción **lineal** no le aporta señal a GBLUP (es
una rotación; el kernel `G=ZZ'/p` ya usa toda la info). La hipótesis del AE es
capturar **no-linealidad**: comprimir 4612 SNPs a un latente chico que un modelo
downstream pueda explotar. El AE no le va a ganar a GBLUP **por sí solo** (GBLUP es
lineal sobre las features); la apuesta es **AE + modelo no lineal** (XGBoost/NN).

### Qué quedó armado
- [`src/models/autoencoder.py`](../src/models/autoencoder.py): `SNPAutoencoder` (MLP
  encoder/decoder simétrico, salida logits + `BCEWithLogitsLoss` = sigmoide+BCE
  fusionadas, regla 4; latente lineal) y `AutoencoderReducer`, que expone la **misma
  interfaz que `PCAReducer`** (`fit`/`transform`/`get_fitted_params`) ⇒ entra en la
  cadena `preprocess:` y lo usa cualquier modelo vía `PreprocessedModel` sin
  reescribir nada. Train-only (regla 1). HP ricos y todos barribles: profundidad/
  ancho del encoder, `latent_dim`, activación, dropout, batchnorm, *tied weights*,
  *denoising* (masking de bits), *sparsity* L1 del latente, optimizador, lr, weight
  decay, batch size, épocas, paciencia, grad clip, scheduler. Early stopping por BCE
  de val, **time budget** (`max_train_seconds`, corte por época) y callback de
  *learning curves* (train/val BCE + accuracy de bits por época).
- Pipeline: `build_preprocessor` reconoce `preprocess.autoencoder` (excluyente con
  `pca`; prohíbe `standardize` antes — rompería la binariedad); `PreprocessedModel`
  y `PreprocChain` propagan el contexto del fold (`line_id`) al reducer para su
  held-out por línea sin leakage.
- Entrenamiento: [`src/training/run_ae.py`](../src/training/run_ae.py) +
  [`scripts/train_ae.py`](../scripts/train_ae.py) (entrypoint **separado** de
  `train.py`). Recuperación: `best_ae_from_sweep` /
  `best_ae_hparams_from_sweep` ([`src/training/ae_select.py`](../src/training/ae_select.py)).
  Persistencia torch: `save_ae_artifact`/`load_ae_artifact`
  ([`src/training/save_model.py`](../src/training/save_model.py)).
- Configs: [`exp_ae_2017_to_2018.yaml`](../configs/exp_ae_2017_to_2018.yaml) (base) +
  [`sweep_ae.yaml`](../configs/sweep_ae.yaml) (bayesiano).

### Decisión de diseño: tunear el compresor por reconstrucción, no por downstream
El AE se selecciona por **BCE de reconstrucción** (no supervisado), NO por el Pearson
downstream. Razón: con un solo inner-val, tunear el AE end-to-end contra el Pearson
le mete tantos grados de libertad al holdout que lo sobreajusta (p≫n, n chico). Así
el embedding de cada dimensión es **único** (no depende del modelo) ⇒ se genera una
vez y lo reusan GBLUP/XGBoost/NN. Por eso el AE NO pasa por `run_evaluation`
(supervisado, por fold): tiene su propio entrypoint. El único entrypoint **supervisado**
sigue siendo `train.py`; `train_ae.py` es el del compresor.

### Régimen temporal y qué persiste (receta vs pesos)
Sweep del AE: **train 2017 → val-reconstrucción BCE 2018** (mismo split 2017→2018 que
GBLUP/XGB usan internamente). 2019 (val externo) y 2020 (sellado) intactos. El AE se
**re-ajusta train-only en cada nivel** (regla 1): 2017 para elegir la dimensión
(GBLUP/XGB 2017→2018); 2017+2018 para reportar en 2019; 2017+2018+2019 al cierre. Lo
que persiste entre niveles es la **receta de HP** (`meta['ae_params']`), no los pesos
(verificado: `train_years` en `n=901` para 2017, `n=1418` para 2018).

### Dónde quedan los AE y cómo se recuperan
Cada trial del sweep guarda su AE entrenado en `checkpoints/ae-snp-d<dim>/`
(`ae.pt` + `meta.json` con HP, métricas y `train_years`) y como wandb Artifact
`ae-snp-d<dim>` (versionado por trial). El mejor de cada dimensión se trae con
`best_ae_from_sweep("entity/project/<sweep_id>")` (cruza `val_bce_best` mínimo con su
Artifact y devuelve el `AutoencoderReducer` listo para `.transform`). Verificado
end-to-end offline (entrena, loguea curvas, guarda y recupera idéntico, guard
anti-leakage activo).

### Plan de barrido (pendiente de correr)
Un sweep bayesiano por dimensión, grilla solapada con la de PCA para comparar
`AE-d` vs `PCA-d`: **32 / 64 / 128 / 256** (bien por debajo del techo de rango 901;
el interés está donde el AE no lineal podría ganarle a PCA de la misma dim, no en
dims altas que ya son *denoising*). Después: selección de dimensión por Pearson de
GBLUP/XGBoost en inner-val 2018, comparación contra baseline en 2019, cierre en 2020.

## Fase 6 — Sweep del MLP + pérdida Pearson + matching20

**Objetivo:** llevar la red de la Fase 4 (MLP base, HP fijos a mano, pérdida MSE) a una
**selección sistemática de HP por sweep**, con la red **optimizando Pearson** (loss =
`1 − r`, no MSE); comparar los mejores contra GBLUP con el mismo protocolo honesto que
XGBoost; y sumar una métrica de **utilidad para selección** (matching20) a todo el
harness. 2020 sigue sellado.

### Qué quedó armado
- **Pérdida Pearson** en [`src/models/mlp.py`](../src/models/mlp.py): `pearson_loss`
  minimiza `1 − r` con la **misma fórmula** (medias, `ddof=0`) que
  [`metrics.py`](../src/eval/metrics.py)`.pearson_r`. Se elige por el HP
  `model.params.loss` (`"mse" | "pearson"`); **default `"mse"`** para no alterar la Fase 4.
  Robustez: descarta batches de tamaño 1 (como con BatchNorm) y un `eps` evita el NaN.
- **Config base** [`exp_mlp_2017_to_2018.yaml`](../configs/exp_mlp_2017_to_2018.yaml)
  (train 2017 → val 2018, `loss: pearson`, `patience: null`, `device: auto`) +
  **sweep bayesiano** [`sweep_mlp.yaml`](../configs/sweep_mlp.yaml) (200 trials,
  maximiza `pearson_r_mean`, barre arquitectura/activación/batchnorm/dropouts/wd/lr/
  batch_size/`max_epochs`).
- **matching20** (coincidence index al top-20%) centralizado en
  [`metrics.py`](../src/eval/metrics.py)`.coincidence_index` e integrado a
  `run_evaluation` (per_fold + IC por bootstrap). La cache key de resultados sube a
  `metrics_v=2` para invalidar lo viejo sin borrar a mano.
- **Learning curves** del MLP: `run_evaluation` devuelve `history_` por fold y
  [`train.py`](../scripts/train.py) lo loguea a wandb con eje x = época (`define_metric`).
- **Recuperación del top-k**: `top_runs_from_sweep`
  ([`src/training/sweep.py`](../src/training/sweep.py)) trae los k mejores trials.

### Pérdida = Pearson (loss = 1 − r): por qué y qué implica
- **Alinea con la portada (regla 5).** Maximizar Pearson ≡ minimizar `1 − r`. En p≫n el
  MSE se minimiza prediciendo ~la media; con la loss Pearson el gradiente apunta directo
  al ranking.
- **La salida queda sin calibrar en escala** (el r es invariante a afines): **RMSE/R² de
  estos MLP NO son informativos** (no compararlos como con los otros modelos); el Pearson
  y el matching20 sí. Para el informe se reporta Pearson.

### Selección honesta + max_epochs como HP
El sweep entrena **2017** y valida en **2018** (sin tocar 2019). Con un **único año de
train** no hay inner-split temporal (`temporal_inner_split` necesita ≥2 años) ⇒ **sin
early stopping**: `max_epochs` es un HP más (mismo criterio que `n_estimators` en
XGBoost), por eso `patience: null`.

### Resultados de selección (val 2018, sweep `kyl04lf5`)
Los 3 mejores trials por Pearson en 2018 (todos `leaky_relu`/`relu`, sin batchnorm):

| # | run | Pearson 2018 | arquitectura | lr | dropout | in_drop | wd | batch | epochs |
|---|-----|------|------|------|------|------|------|------|------|
| 1 | `hhhrp5a0` | **0.418** | `[128, 64]` | 0.0037 | 0.64 | 0.35 | 0.0057 | 32 | 249 |
| 2 | `t03pixd3` | 0.402 | `[2048,1024,512,256]` | 0.0012 | 0.13 | 0.002 | 0.0040 | 64 | 364 |
| 3 | `4lh9j7jg` | 0.389 | `[2048,1024,512,256]` | 0.0009 | 0.03 | 0.45 | 1e-6 | 64 | 310 |

Esos 0.40+ son en 2018 (cohorte ruidosa, set de selección), **no** el veredicto. El
reporte honesto (reentrenar 2017+2018 → val 2019, vs GBLUP) corre en el notebook
[`comparar_modelos_ci.ipynb`](../notebooks/comparar_modelos_ci.ipynb); como con XGBoost
(2018 0.46 → 2019 0.25), esperar que 2019 baje. *(Pendiente de cerrar con los números de
2019; cada uno linkeará a su corrida.)*

### matching20 — coincidencia del top-20% (utilidad de selección)
- **Qué mide**: de las **líneas/genotipos** que están en el mejor 20% por GY observado en
  val, qué proporción el modelo también pone en su mejor 20% por GY predicho
  (`|top_obs ∩ top_pred| / k`, `k = round(0.2·n)`). Es el *coincidence index* estándar de
  selección truncada — sobre líneas, **no** sobre genes/marcadores.
- **Cómo se lee**: una selección al azar acierta ~**0.20** (el piso a batir); 1.0 =
  recupera exacto el top. Asume "más GY = mejor". Tiene su IC por el mismo bootstrap
  cluster a nivel línea, y aparece para **todos** los modelos del notebook (en `METRICS`)
  con la vista de conteo "a cuántas líneas les pegó".

### Por qué no se logueaban las curvas (y qué cambió)
El MLP guardaba `history_` pero el camino supervisado (`run_evaluation`) solo logueaba
métricas **finales por fold** — no propagaba un hook por época (a diferencia de
`train_ae.py`, que sí lo hace para el AE). Por eso en wandb solo quedaba el resultado
final: era esperable, no un bug. Ahora `train.py` loguea la curva por época. **Limitación
del régimen del sweep**: con 1 año de train (2017) no hay inner-val, así que por época
solo existe `train_loss` (la curva de val aparece solo con early stopping, ≥2 años). Los
200 trials ya corridos no tienen curva (no se puede recuperar); aplica a corridas futuras.
En el notebook las learning curves de los 3 MLP finales se grafican desde `history_`.

## Fase 7 — Modelo sparse (selección de variables): el mayor salto en sequía

**Objetivo:** atacar la portada (GY drought) por el ángulo que faltaba: la **arquitectura
genética** del trait. GBLUP es un modelo *infinitesimal* (prior gaussiano de igual varianza
sobre TODOS los marcadores ≡ ridge sobre `G=ZZ'/p`); es óptimo cuando el trait es muy
poligénico, pero **sub-óptimo cuando unos pocos loci de efecto grande dominan**. La hipótesis:
GY bajo sequía tiene arquitectura más rala que en óptimo, así que un estimador de **selección
de variables** (LASSO = MAP de un prior Laplace, el análogo frecuentista de BayesB) debería
ganarle a GBLUP en sequía. Resultó ser, por lejos, el mayor salto del proyecto sobre GBLUP.

### Diagnóstico de arquitectura (train 2017+2018, GWAS single-marker)
Regresión marcador-a-marcador sobre el train, χ² ≈ n·r² por SNP:
- **GY óptimo:** 2751 marcadores significativos (Bonferroni), χ²_max ≈ 636 → señal densa y fuerte,
  repartida en muchísimos loci ⇒ **infinitesimal** ⇒ GBLUP es el techo.
- **GY sequía:** 634 significativos, χ²_max ≈ 114 → mucho menos marcadores con señal y efectos
  más chicos ⇒ arquitectura **más rala** ⇒ hay lugar para selección de variables.

La heredabilidad genómica (REML `σ²_u/(σ²_u+σ²_E)`) es parecida entre ambientes (GY óptimo
0.47–0.60, sequía 0.39–0.67 según año), así que la diferencia no es "cuánta señal hay" sino
**cómo está distribuida** entre marcadores.

### Modelo: `sparse_genomic` (LASSO/ElasticNet baggeado)
[`src/models/sparse_genomic.py`](../src/models/sparse_genomic.py), registrado como
`sparse_genomic`. Respeta la interfaz común y las reglas:
- **Estandarización train-only** de marcadores DENTRO de `fit` (z-score por SNP, igual que
  GBLUP; la L1 no es invariante a escala y los SNP binarios tienen varianza `p(1-p)` que
  varía ~5× entre MAF 0.05 y 0.5). No usa el bloque `preprocess:` (marcadores crudos, como GBLUP).
- **`alpha` por inner-CV TEMPORAL** (regla 3): con train 2017+2018 ⇒ inner 2017→2018; con los 3
  años ⇒ inner 2017+2018→2019. Se elige el `alpha` que maximiza el Pearson de inner-val (portada).
- **Bagging = análogo del ensemble multi-seed de la red.** Un LASSO single-fit es de **alta
  varianza** ante submuestreo del train (en LD, L1 elige arbitrariamente uno de varios marcadores
  correlacionados). Baggear —promediar coeficientes sobre subsamples— reduce esa varianza. Como
  baggear un lineal ≡ promediar coeficientes, `predict` es un solo `Z @ coef_` (misma idea que la
  representación rrBLUP de GBLUP). Persistencia por joblib (numpy puro); round-trip exacto verificado.

### Resultado (portada, GY sequía, train 2017+2018 → val 2019, n=721)
Vía `run_evaluation` (pipeline oficial; verificado idéntico al harness rápido) y delta pareado por
bootstrap cluster a nivel línea:

| modelo | Pearson [IC95%] | Δ vs GBLUP (pareado) |
|---|---|---|
| GBLUP-REML (baseline) | 0.2413 [0.172, 0.309] | — |
| MLP multienv + XGB (SOTA previo) | ~0.333 | +0.092 |
| **LASSO baggeado (sparse)** | **0.3942 [0.331, 0.451]** | **+0.152 [+0.118, +0.184]** |

**+0.15 de Pearson sobre GBLUP (+63 % relativo, 0.24 → 0.39)**, y **+0.06 sobre el mejor modelo
previo**. matching20 = 0.43 (recupera el 43 % del top-20 % de líneas vs 20 % al azar). El alpha
elegido (~0.043) selecciona un núcleo de ~40–100 marcadores estables (69 en ≥50 % de los bags).

### Robustez (lo que lo distingue de los falsos positivos de fases anteriores)
- **Determinista** (no hay lotería de seed como en la red): la seed del bagging mueve el Pearson
  con sd 0.005 (esencialmente fijo). Elimina de raíz la fuente #1 de winner's curse del proyecto.
- **Le gana a GBLUP en el 100 % de 20 submuestreos del train** (85 % de líneas), minΔ = **+0.059**,
  con la misma dispersión que GBLUP (sd 0.030). El bagging es lo que convierte el LASSO single-fit
  (frágil: frac(Δ>0)=0.80, minΔ=−0.145, número full-train inflado en el máximo de su distribución)
  en un predictor robusto.
- **Regla de oro (aparece en un 2º fold de alta potencia):** LOYO train {2017,2019} → val **2018**
  (n=1418) da Δ **+0.062** [+0.021, +0.101]. No es idiosincrasia de 2019.

### Caveats honestos (documentados, no escondidos)
- **Necesita train adecuado (≥2 años).** Con 1 solo año (2017→2018) el LASSO PIERDE (−0.034): la
  selección de marcadores con n=901 es inestable. El salto grande aparece con ≥2 años; para 2020 el
  train son los 3 años de dev ⇒ régimen favorable. Por eso el número honesto de sequía se reporta
  con train 2017+2018 (≥2 años), no con el 2017→2018 single-year.
- **Es específico de GY×sequía.** PH y AD bajo sequía **no** se benefician (LASSO ≈ o < GBLUP),
  así que no es un fenómeno genérico de "sequía = ralo": es la arquitectura de *este* trait×ambiente.
- **En óptimo LASSO PIERDE** (control por `run_evaluation`: 0.139 vs GBLUP 0.315), consistente con
  su arquitectura infinitesimal. **Veredicto general: el mejor modelo depende de la arquitectura del
  trait×ambiente** — GBLUP para óptimo (poligénico), sparse para sequía (más ralo).
- **El blend no aporta (puerta cerrada).** GBLUP + LASSO + XGB en sequía 2019 correlacionan ~0.82
  entre sí; el blend **oracle** (techo post-hoc sobre 2019) da 0.393 = LASSO sola (pesos G=0, L=0.7,
  X=0.3) y el blend **honesto** (pesos elegidos en inner 2017→2018) EMPEORA a 0.323 (ruido de
  selección de pesos, mismo patrón que la selección de α en un solo año). El LASSO baggeado ya captura
  la señal explotable; XGB (no lineal) y GBLUP no suman ni en el techo ⇒ la red multienv (más débil,
  ~0.32, correlación similar) tampoco lo haría. **El modelo de sequía es LASSO baggeado solo** — el más
  simple es el mejor y, encima, determinista.

### Configs y verificación
- Dev/reporte: [`exp_sparse_lasso_GY_drought_2017_2018_to_2019.yaml`](../configs/exp_sparse_lasso_GY_drought_2017_2018_to_2019.yaml)
  (Pearson 0.3942 por `run_evaluation` y por `scripts/train.py` offline).
- Sweep confirmatorio (grid chico l1_ratio×select_by): [`sweep_sparse_lasso_drought.yaml`](../configs/sweep_sparse_lasso_drought.yaml).
- **TEST FINAL 2020** (sellado, correr una sola vez al cierre): [`exp_sparse_lasso_GY_drought_final_2020.yaml`](../configs/exp_sparse_lasso_GY_drought_final_2020.yaml)
  (train 2017+2018+2019 → 2020, n_bags=40, `save_model`). Se compara contra GBLUP-drought
  reentrenado igual. 2020 intocable hasta entonces.
- Verificado: pipeline oficial == diagnóstico (Pearson 0.3942), save/load round-trip exacto
  (max|Δpred|=0), `sparse_genomic` registrado y ruteado por `train.py` (single-task).

### ¿Se puede superar el LASSO con una red? No — el techo es la señal, no el modelo
Exploración exhaustiva de variantes de red y refinamientos (todo multi-seed/bagged, delta pareado,
chequeo de la regla de oro en 2018): **nada supera de forma robusta al bagged-LASSO (0.390)**.
- **Red densa MSE** (hidden [64], 10 seeds): 0.358 < LASSO. **Red con group-L1 en la entrada**
  (selección de marcadores, MSE para que el L1 tenga dientes): cualquier L1 fuerte empeora; máximo = la
  densa 0.358. **LassoNet** (skip lineal + MLP): < LASSO.
- **Probe de no-aditividad** (MLP sobre los top-K marcadores que elige LASSO): K=50 da 0.404 en 2019
  (Δ+0.013, **no significativo P=0.80**) pero **FALLA la regla de oro**: en 2018 Δ−0.044 (P=0.00, peor).
  El "0.404" es ruido de 2019 ⇒ **no hay interacciones/no-aditividad explotables** (coherente con el
  blend oracle GBLUP+LASSO+XGB = LASSO).
- **Sparse autoencoder** (latente BCE + L1, regla 4): latente→LASSO 0.32–0.34 (comprimir pierde la info
  de marcadores); latente+crudos ≈ LASSO. No aporta.
- **MultiTaskLasso** (soporte compartido opt+dro): gana solo en →2019 (0.419) y pierde en →2017/→2018
  (GxE: corr opt-dro 0.14 mete marcadores de optimal inútiles en sequía). No robusto.
- **Relaxed/adaptive LASSO** (debiasar el soporte): EMPEORA (→2018 Δ−0.073): el shrinkage del L1 es
  regularización beneficiosa en p≫n.

**Lectura:** la señal de GY drought es **lineal-aditiva y sparse** ⇒ LASSO (con bagging para bajar la
varianza) está en el techo; una red solo puede ganar si hay no-linealidad, y no la hay. El número
robusto de la portada es **bagged-LASSO 0.390** (el single-fit 0.408 y el MTL 0.419 en 2019 están
inflados/no-robustos). Se prioriza el robusto sobre el punto más alto pero frágil.

### Segunda vuelta (más ideas nuevas): el patrón universal 2019-sí / 2018-no
Barrido adicional pedido para exprimir todavía más el LASSO (attention, ideas transductivas, y otras).
Todo con multi-seed, delta pareado (cluster por línea) y el chequeo de la **regla de oro en 2018**.
Emergió un patrón nítido: **casi todo mejora →2019 (el fold análogo a deploy) y falla →2018**.
- **Warm-start net** (LASSO congelado + MLP residual muy regularizado, γ inicial 0): lo mejor de las
  redes — 2019 Δ**+0.020** (P=1.00) y 2018 Δ=0.000 exacto (nunca daña). Pero el **nested subsample lo
  mata**: perturbando el train al 85%, Δ→+0.0015 con frac>0=**0.58** (moneda). El +0.020 era del
  train puntual. No robusto.
- **Idea transductiva** ("que el modelo atienda a lo parecido al año target", usando la X de 2019/2018
  que sí está disponible): (a) penalizar marcadores que *driftean* train↔target → 2019 +0.0075 pero
  **2018 −0.014**; (b) pesar líneas de train por parecido genómico al target → DAÑA en toda escala. Falla.
- **Estabilidad cruzada entre años** (quedarse con marcadores reproducibles en 2017 *y* 2018): solo
  **6 de 341** son reproducibles → filtrar destruye la señal (la potencia viene de poolear los años).
- **Multi-trait GY+PH+AD** (borrow strength de traits secundarios más heredables): el soporte compartido
  **daña** GY (−0.10 en 2019). Los marcadores de PH/AD no son los de GY.
- **Adaptive LASSO** bien implementado (des-shrinkear los efectos grandes, proxy frecuentista de
  BayesB/SCAD): 2019 +0.015 pero **2018 −0.037** (P=0.00). Junto con relaxed (−0.073) y ElasticNet
  (peor), **cierra la familia de shrinkage adaptativo sin necesidad de MCMC**: el shrinkage uniforme
  del L1 ya es ~óptimo (no hay QTL grandes limpios que debiasar).
- **Random-subspace LASSO** (*Random-LASSO*, Wang 2011): **lo único robusto en 2019**. Cada base-LASSO
  ve un subset aleatorio de marcadores (frac 0.7) → decorrelaciona qué marcador de cada grupo en LD se
  elige y baja la varianza del coef promediado **más** que el bagging por muestras. →2019 Δ**+0.006**
  (P=0.99), reproducido en 4 implementaciones (incl. la clase real: 0.3942→0.4006), y **sobrevive el
  nested subsample: frac>0 = 1.00** (gana en las 12 perturbaciones, min +0.0009) — lo contrario del
  warm-start. En →2018 es **NEUTRO dentro del ruido**: el punto matched (nb60, 5 seeds) da Δ−0.004
  (P=0.08) pero el nested subsample da +0.004 (frac>0=0.67, min −0.012/max +0.019) — el signo es
  inestable. A diferencia de adaptive (−0.037) / drift (−0.014) / probe-MLP (−0.044) que **dañan claro**
  2018, random-subspace **no lo daña**; pero tampoco lo **mejora** claramente ⇒ no pasa el criterio
  estricto de la regla de oro (ganar en AMBOS). Es lo más cerca que estuvo algo de ganar en los dos.
  Se agregó como parámetro **`subspace_frac`** en `sparse_genomic.py` (default **1.0** = bagged-LASSO
  exacto, backward-compatible; `0.7` activa el random-subspace) y se **ADOPTÓ `subspace_frac=0.7`
  (nb=60)** en los configs de dev y de 2020 final — es la única mejora robusta y estable hallada, chica
  (~piso de ruido en absoluto) pero nunca dañina; `1.0` queda como fallback conservador.

### Tercera vuelta (post-revisión): no-linealidad, transfer y LOYO — descartados; lección de método
Empujando aún más (redes/attention ya cubiertos), se probaron hipótesis NUEVAS y ninguna sobrevive:
- **RKHS gaussiano** (kernel ridge = GBLUP no-lineal; ~red de ancho infinito): **daña** (2019 0.31, Δ−0.078).
  Cierra la no-aditividad también en el espacio de PARENTESCO (no solo en SNPs) ⇒ por eso ninguna red ayuda.
- **Transfer desde el ambiente ÓPTIMO** (usar |β| de LASSO-optimal como pesos de penalidad para sequía):
  **daña** (2019 Δ−0.025 a −0.046). Muro GxE (corr 0.14): los marcadores de optimal no son los de sequía.
- **LOYO-alpha** (elegir la regularización por leave-one-year-out): **artefacto, rechazado.** En scratchpad
  aparentaba +0.019 en 2018, pero era doble artefacto: (a) el baseline del scratchpad elegía un alpha malo
  (0.041) que la **clase real ya evita** (temporal elige 0.073, el "bueno" que LOYO supuestamente hallaba);
  (b) con 2 años LOYO es inestable (nested 2019 mean−0.020, frac>0=0.33, min−0.108 catastrófico). Revertido.
  **Lección de método: verificar SIEMPRE por la clase real productizada, no por el scratchpad — un baseline
  débil de scratchpad infla ganancias aparentes** (por eso RSS, que se validó por la clase real, sí se adopta).

**Conclusión reforzada:** tras dos vueltas exhaustivas, **ningún método —red, transductivo,
multi-trait, o refinamiento de shrinkage/ensemble— supera al bagged-LASSO de forma robusta en AMBOS
folds.** El fold →2019 (deploy-análogo) tiene señal que →2018 (interpolación) no comparte, y por eso
tanto método "gana en 2019"; la regla de oro (exigir ambos) los descarta. La **única mejora amplia y
robusta del proyecto sigue siendo LASSO sobre GBLUP (+0.15, +63% en sequía)** — no algo sobre LASSO.
`subspace_frac=0.7` es la única palanca que mejora el fold deploy-análogo de forma robusta (+0.006,
matched nb, nested frac>0=1.00) y es **neutra** (dentro del ruido) en el fold 2018 — nunca lo daña. Se
**adoptó** en los configs (dev y 2020) como el modelo de portada de sequía; `1.0` queda como fallback.
No existe una mejora *ampliamente* significativa sobre el LASSO: estamos en el techo de predictibilidad
genómica (señal aditiva+sparse). La mejora amplia del proyecto es LASSO sobre GBLUP (+63%).

## Fase 8 — Sparsity × no-linealidad con deep learning: por qué no hay techo que romper

**Objetivo:** atacar de frente la pregunta que quedaba abierta —¿se puede superar el bagged-LASSO
(0.400 en drought) combinando **sparsity con no-linealidad** mediante deep learning complejo?—. Las
vueltas previas (Fase 7) ya habían descartado varias redes, pero por el ángulo de "probar modelos". Acá
se ataca por dos ejes NUEVOS y ortogonales, con un **diagnóstico model-agnostic** que explica el
mecanismo, no solo el resultado. Protocolo: la **regla de oro** del proyecto (ganar en AMBOS folds:
`deploy2019` train 2017+2018→2019 y `golden2018` LOYO train 2017+2019→2018), delta pareado y, para
cualquier candidato que asome, chequeo de replicación out-of-sample. 2020 sellado. Baselines del
campeón (bagged random-subspace LASSO, `subspace_frac=0.7`, nb=60): **deploy2019 = 0.4006**,
**golden2018 = 0.2650** (reproducidos exactos).

### Punto de partida conceptual: con SNPs binarios, no-linealidad = epistasis
Los marcadores son binarios `{0,1}` ⇒ una función por-marcador toma 2 valores = un coeficiente lineal
la captura exacta (**no hay dominancia con 2 estados**, y por eso un Neural Additive Model colapsa a
lineal). La ÚNICA no-linealidad genotípica posible es la **epistasis** (interacciones SNP×SNP). El probe
de Fase 7 (MLP sobre los top-50 marcadores marginales) solo miraba interacciones entre marcadores con
efecto marginal fuerte; los árboles de XGBoost hacen splits greedy guiados por lo marginal (fallan la
epistasis pura tipo XOR). Faltaba el detector correcto: **interacciones explícitas + selección sparse**,
que alcanza epistasis entre marcadores de efecto marginal **nulo**.

### Eje 1 — Reluctant Sparse Interaction LASSO (epistasis explícita, screen COMPLETO)
Método (Yu, Bien & Tibshirani 2019, *reluctant interaction selection*): (1) ajustar el campeón aditivo,
residual `r`; (2) **screen de las ~10.6M interacciones pairwise** `Zi·Zj` contra `r`, barato vía
matrices tipo Gram `p×p` (`N=Zᵀdiag(r_c)Z/n`, `var=Q−G²`), rankear por `|corr(Zi·Zj, r)|` y quedarse con
las top-T; (3) bagged-LASSO sobre esas T interacciones prediciendo `r`. Todo train-only (el screen, el
z-score y las features de interacción se congelan en train). A diferencia del probe MLP-top50, este
screen COMPLETO alcanza interacciones de marcadores sin señal marginal.

Resultado: **daña en los dos folds, en todo T**.

| T interacciones | deploy2019 (Δ) | golden2018 (Δ) |
|---:|---|---|
| 1000 | 0.284 (−0.116) | 0.113 (−0.152) |
| 3000 | 0.340 (−0.061) | 0.179 (−0.086) |
| 8000 | 0.340 (−0.061) | 0.195 (−0.071) |

### Diagnóstico decisivo (model-agnostic): las interacciones NO replican out-of-sample
Test independiente de cómo se ajusten: se toman las top-2000 interacciones por `|corr|` con el residual
de **train** y se mide su correlación con el residual de **val** (mismas parejas). Si hubiera epistasis
explotable, las de mayor corr en train deberían correlacionar —con el mismo signo— en val.

- **deploy2019:** `|corr|` train hasta 0.164; en val la fracción de **mismo signo = 0.535** (≈0.5 =
  ruido puro), `rank-corr(|train|,|val|) = −0.01`, la top-1 interacción tiene corr en val de **+0.004**
  (cero). La estructura "más fuerte" en train es ruido.
- **golden2018:** `|corr|` train hasta 0.192; en val **mismo signo = 0.438** (por DEBAJO de 0.5) y
  `corr(cc_train, cc_val)` con signo = **−0.151**: las interacciones top de train **anti-replican** en
  val. Elegir interacciones por train no solo no ayuda, apunta en contra.

Esto es lo que ninguna vuelta anterior había mostrado: **la razón mecánica** por la que toda red falla
—no hay señal epistática que replique entre cohortes; lo que un modelo flexible "encuentra" es ruido de
train—. Es el argumento killer porque no depende de la implementación del modelo.

### Eje 1' — Sparse Interaction Machine neuronal (FM factorizada + gate sparse)
Confirmación desde el lado del modelo flexible. Red = **skip aditivo congelado** (campeón, coef 1) +
cabeza no-lineal sobre el soporte sparse que elige el LASSO: **Factorization Machine** (interacciones
pairwise low-rank `½[(ZV)²−Z²V²]`, O(s·k)) + gate `sigmoid` con L1 por marcador (sparsity dentro del
soporte) + opcional MLP residual (orden superior). `γ` (escala de la cabeza) init 0 ⇒ arranca EXACTO en
el campeón. Loss `1−Pearson` + L1(gate) + L2(V), ensemble multi-seed. Dos regímenes:

- **Free-running (300 épocas):** daña fuerte (deploy 0.15–0.26, golden 0.12–0.19) y `γ` tiene varianza
  ENORME entre seeds (media ~0, sd 0.4–1.0): la cabeza ajusta puro ruido, distinto en cada seed.
- **Honest early-stop temporal** (inner-val = último año del train, con el skip del campeón calculado
  fuera de muestra; se elige la época que maximiza Pearson de inner-val, protocolo del MLP del proyecto):
  sigue **dañando** (deploy 0.33–0.35, Δ−0.05 a −0.12; golden 0.25–0.25, Δ−0.01 a −0.04) y `best_ep`
  es inestable (59±111): los seeds no coinciden en cuándo parar porque no hay señal que hallar.

### Eje 2 — Sparse Mixture-of-Experts (heterogeneidad de líneas, gating no-lineal)
Eje ORTOGONAL: en vez de interacciones de features, modelar **heterogeneidad de arquitectura entre
líneas**. Skip aditivo congelado + K **expertos lineales sparse** (L1) sobre el soporte, ruteados por un
**gate no-lineal** (MLP sobre PCs del genotipo) por línea: `y = y_main + Σ_k gate_k(pc)·(Z_sup·w_k)`. La
no-linealidad está en el ruteo (piecewise-linear en el espacio de genotipo). Expertos warm-start 0,
honest early-stop, multi-seed. Hipótesis: distintas líneas se predicen mejor con distinta combinación de
marcadores.

Resultado: **daña en ambos folds** (K∈{2,4,8}: deploy 0.29–0.34, Δ−0.06 a −0.11; golden 0.24–0.25,
Δ−0.02 a −0.03) y con early-stop honesto `best_ep≈4–11`: el inner-val quiere **quedarse en el campeón**
(W→0). Coherente con la memoria (el oráculo a nivel AÑO daba solo +0.003): no hay headroom en mezclar
subpoblaciones tampoco. (La estructura poblacional del EDA es difusa; el eje real es el año, ya agotado.)

### Eje 3 — No-linealidad en el link de salida (recalibración monótona/spline)
Mecanismo que ningún experimento previo tocó (todos asumían link lineal): el campeón da un score lineal
sparse `s=Zw`; si `E[y|genotipo]=g(s)` con `g` no-lineal monótona, recalibrar `s→g(s)` subiría el Pearson
(invariante a afín, NO a monótona). `g` se aprende HONESTA sobre predicciones out-of-fold (inner temporal
del train). Resultado: **falla la regla de oro y daña fuerte en golden2018**. Polinomio grado-2 da
deploy2019 +0.0036 (ruido, dentro del sd~0.005 del campeón) pero **golden2018 −0.175**; grados 3/5 e
isotónica dañan en ambos (golden2018 hasta −0.42). El coeficiente cuadrático de `g` en el inner-val es
grande (0.16–0.70) pero **espurio**: la curvatura ajustada en un año anti-generaliza al otro. El link
también es lineal.

### Contexto del techo: es una pared de información, no de modelo
El campeón captura `r²=0.160` en deploy2019; con `h²_SNP≈0.45–0.55` (REML, Fase 7) eso es solo el
**~29–36 %** del techo genético in-sample. La brecha restante la fija la **relatedness cross-cohorte + n**
(predecir un año de líneas disjuntas desde otros), NO la no-linealidad faltante: por eso agregar
capacidad no-lineal solo agrega varianza (daña). Es el patrón conocido en predicción genómica —la
accuracy entre cohortes está limitada por parentesco, y la varianza epistática, aunque exista, **no
predice out-of-sample** a través de la estructura de parentesco (la "missing heritability" epistática no
es útil predictivamente).

### Veredicto (honesto) y qué NO se productiviza
Tres ejes independientes (epistasis de features + heterogeneidad de líneas + link de salida), **cinco
métodos nuevos** (reluctant interaction LASSO, FM neuronal free y honest-ES, Sparse MoE, recalibración
no-lineal del link) y **cero** mejora robusta: todos DAÑAN en ambos folds (o solo tocan +0.003 de ruido en
2019 y destrozan 2018), y el diagnóstico de replicación explica por qué. Ninguno se agrega a `src/models` ni a los
configs (norma del proyecto: se adopta solo lo que gana robusto; acá no gana nada). El modelo de portada
de sequía **sigue siendo el bagged random-subspace LASSO (0.400)**; el de óptimo, GBLUP. Para el informe
esto es un resultado **fuerte, no un fracaso**: se descarta la no-linealidad explotable con el detector
correcto (screen completo de epistasis) y un test de replicación out-of-sample, cerrando de raíz el
ángulo "deep learning complejo vencería al lineal". La única mejora amplia del proyecto es, y queda,
**LASSO sobre GBLUP (+63 % en sequía)**; el techo es aditivo+sparse y es informacional.

## Fase 8 — Multi-trait GBLUP: la única mejora ADITIVA y ROBUSTA sobre GBLUP en óptimo

**Objetivo:** revisar si el sweep del MLP multitask ensemble (`23v91t2s`, GY+PH+AD óptimo, train
2017+2018 → val 2019) que "le gana a GBLUP" es real o winner's curse, y si hay una mejora
robusta que rescatar. Resultó que la red es winner's curse, pero el mecanismo que insinúa
(pleiotropía PH↔GY) sí da una mejora **cuando se lo implementa con la herramienta aditiva
correcta**: GBLUP multi-trait.

### El multitask neuronal es winner's curse (empata en 2018)
El top del sweep reporta Pearson GY **0.3439** en val 2019 (vs GBLUP-REML **0.3145**). Re-evaluado
honesto con un ensemble fresco de 8 seeds del mismo config, el edge se encoge y **no replica fuera
del fold de selección**:
- **val 2019 (fold de selección):** 0.3376, Δ+0.023 vs GBLUP, **IC95%[−0.005, +0.052] incluye 0**.
- **val 2018 (out-of-sample honesto, train 2017):** 0.3241 vs GBLUP 0.3234, **Δ+0.0007 (empate plano)**.

Firma clásica de winner's curse (memoria [[redes-no-superan-gblup-optimal]]): gana solo donde se
eligió el HP, empata en el fold independiente. El ensemble mitiga el ruido de *seed*, no el de
selección de HP sobre el set de reporte.

**Matiz (selección honesta + seeds frescas):** no es winner's curse *puro*. De los 14 top-por-2019, la mayoría se cae en 2018 (Δ−0.05 a −0.075) y el super-ensemble ingenuo de los 14 falla 2018 (Δ−0.025), pero **existe una sub-región robusta**: elegir el mejor por 2018 aterriza en `x3yci2pl` (2019 +0.023 / 2018 +0.013) y `nkr46w7r` gana ambos también. Clave mecánica: **los únicos configs robustos pesan PH fuerte** (`loss_weights.PH` 0.80–0.86) — el MISMO mecanismo pleiotrópico que el MTM. Pero es **frágil**: reevaluados con 10 seeds FRESCAS e independientes (base 9999) el edge se encoge (x3yci2pl 2019 +0.016 P=0.97 / 2018 +0.007 P=0.92; nkr46w7r 2019 +0.004 empate / 2018 +0.018 P=1.0) y **ningún IC excluye 0 en ambos folds**. El neuronal es una aproximación ruidosa y seed-dependiente de la señal que el **MTM captura determinista** (rank-corr 2019↔2018 solo +0.48).

### Multi-trait GBLUP (`mtm_gblup`): pleiotropía por la vía aditiva
[`src/models/mtm_gblup.py`](../src/models/mtm_gblup.py), registrado `mtm_gblup`, encaja en el
dispatch multitask (misma interfaz `fit(X,Y,Y_mask)`/`predict(X)→(n,T)` que `mlp_multitask`).
GBLUP multivariado: un modelo mixto para GY+PH+AD con covarianza **genética** `G0` (T×T) y
**residual** `R0` (T×T) estimadas por **REML-EM** sobre la eigen-descomposición de `G=ZZ'/p`
(`Y=1μ'+U+E`, `U~N(0,G0⊗G)`, `E~N(0,R0⊗I)`; la rotación `Ỹ=V'Y` vuelve las filas T-variadas
independientes con `Ũ_i~N(0,d_i·G0)` ⇒ EM barato). Efectos de marcador en representación rrBLUP
`B=ZᵀG⁻¹Û/p` ⇒ `predict` es un solo matmul (como GBLUP). **Determinista** (sin seeds), **aditivo**
(no apela a no-linealidad ⇒ respeta el techo de líneas endocriadas), **val solo-genotipo** (las
cabezas PH/AD NO se observan en val; el *borrowing* ocurre en la estimación, no usando fenotipos
auxiliares del val ⇒ comparación justa con GBLUP). Requiere datos completos por fila (en óptimo
GY=PH=AD por observación; se cumple).

### Control que valida la implementación y aísla el efecto
El MTM restringido a **un solo trait (GY)** reproduce **GBLUP-REML EXACTO** (0.3145 en 2019,
0.3233 en 2018) ⇒ el EM está bien y la ganancia **no** viene de un shrinkage distinto. Toda la
mejora es **borrowing cross-trait**:

| fold | GBLUP-REML | MTM (GY+PH+AD) | Δ | IC95% del Δ (bootstrap cluster-línea) | P(Δ>0) |
|------|-----------|----------------|---|----------------------------------------|--------|
| val 2019 (train 2017+2018) | 0.3145 | **0.3199** | +0.0054 | [−0.0033, +0.0149] | 0.87 |
| val 2018 (train 2017) | 0.3234 | **0.3334** | +0.0101 | **[+0.0069, +0.0138]** | **1.00** |

**Positivo en AMBOS folds** (dirección de la regla de oro), **significativo en 2018** (n=1418, IC
excluye 0). Correlación genética estimada `gcorr(GY,PH)=0.71 (2019) / 0.52 (2018)` (fuerte),
`gcorr(GY,AD)≈0.03–0.41`, `h²_GY≈0.46–0.59`. **PH hace todo el borrowing; AD no aporta**
(GY+PH ≈ GY+PH+AD), consistente con `corr(GY,AD)≈0`. wandb (ambos folds en una corrida,
esquema expanding): [`l5sgbbif`](https://wandb.ai/oromagnoli-/maze-genomic-pred/runs/l5sgbbif).
Configs: [`exp_mtm_gblup_GY_optimal_2017_2018_to_2019.yaml`](../configs/exp_mtm_gblup_GY_optimal_2017_2018_to_2019.yaml)
(deploy 2019) y [`exp_mtm_gblup_GY_optimal_goldenrule.yaml`](../configs/exp_mtm_gblup_GY_optimal_goldenrule.yaml)
(regla de oro).

### Techo trait-assisted: si se observa PH en el set a predecir
Índice de selección (OLS out-of-fold: `GY ~ GEBV_GY + PH_val + AD_val`) que **sí usa el fenotipo PH
del val** — problema de información distinto (GS asistida por trait secundario, estándar en la
literatura; PH/AD se miden antes de la cosecha de GY): 2019 **0.368** (Δ+0.054, IC[+0.043,+0.064]),
2018 **0.446** (Δ+0.123, IC[+0.111,+0.136]), ambos con P(Δ>0)=1.00 y PH aportando casi todo. Es la
mejora **grande y robusta** disponible, pero rompe la interfaz `predict(X)` (necesita Y de val) y no
es apples-to-apples con GBLUP genotype-only ⇒ NO se productiviza como modelo; queda documentada como
el techo si el pipeline de breeding tiene PH del target.

### Veredicto
Se revisa —sin contradecir— la memoria "nada supera a GBLUP en óptimo": lo que no lo supera es la
**no-linealidad** (el multitask neuronal es winner's curse). Pero el **multi-trait GBLUP aditivo** sí
da una mejora **chica, robusta, determinista y principista** (+0.005/+0.010, positiva en 2019 y 2018,
significativa en 2018) por **pleiotropía aditiva** GY↔PH. Es la primera mejora *fair* del proyecto en
óptimo que **no se da vuelta a negativo en 2018**. Coherente con el techo aditivo: la ganancia es
borrowing aditivo entre traits, no varianza no-aditiva. Se adopta a `src/models` + configs (norma:
se productiviza solo lo que gana robusto). Para el modelo final de 2020 (reentrenar 2017+2018+2019),
usar `mtm_gblup` sobre GBLUP single-trait; 2020 sigue sellado.

## Fase 9 — Deep learning vs GBLUP en óptimo: sí existe una red que gana, pero el deep es inerte

**Objetivo:** encontrar una arquitectura de DEEP LEARNING que supere a GBLUP en GY óptimo — "algo que
no probamos". Se testearon 4 arquitecturas genuinamente nuevas (además de descartar una quinta con
argumento). Veredicto: **el componente profundo/no-lineal SIEMPRE agrega ruido**, pero una red que
lleva el aditivo adentro (skip lineal anclado en MTM) **sí le gana a GBLUP en ambos folds**.

### Arquitecturas nuevas probadas (GPU, ensemble, IC pareado cluster-línea, seeds frescas)
Baseline GBLUP-REML 0.3145 (2019) / 0.3234 (2018); MTM (Fase 8) 0.3199 / 0.3334.

1. **Wide&Deep con backbone aditivo APRENDIDO** (`Linear(p,3)` + tronco, todo desde cero, MSE, ES):
   el skip lineal **no alcanza el shrinkage de GBLUP** (0.298 en 2019) y **overfittea catastrófico en
   2018** (0.166, sin inner-split temporal para early-stop). Un net lineal from-scratch no reproduce
   GBLUP (que es ridge con λ por REML), menos aún lo supera.
2. **Backbone MTM anclado (congelado) + corrección DEEP** (skip init EXACTO en los efectos de marcador
   del MTM, cabeza deep init 0, ES por línea, ensemble 15): la corrección profunda **DAÑA** vs el MTM
   puro (2019 0.3182 = −0.002; 2018 0.3278 = −0.006). El deck no-lineal sobre el mejor aditivo resta.
3. **Fine-tune end-to-end** (backbone MTM + deep, todo entrenable): **colapsa** (2019 0.298, 2018 0.242,
   Δ−0.02 / −0.08). Adam desarma la solución MTM y overfittea; confirma que el aditivo hay que
   congelarlo.
4. **6 cabezas = máxima pleiotropía** (GY/PH/AD × optimal/drought sobre la misma línea). Test aditivo
   decisivo y barato: **el MTM de 6 traits NO mejora al de 3** — hasta daña (2019 0.2986 = Δ−0.021 vs
   MTM-3; 2018 empate). Las correlaciones altas de los traits de sequía con GY_opt (AD_dro 0.65,
   PH_dro 0.36 *pooled*) son **artefacto de medias entre años**, no señal genética within-fold
   (corr(GY_opt, GY_dro)=0.004). Si el MTM óptimo no extrae nada de los traits extra, un deck de 6
   cabezas solo overfittearía. Sin headroom.
5. **CNN 1D sobre orden de SNPs** — DESCARTADA con argumento: las columnas SNP (`L9.` + hash hex,
   ordenadas lexicográficamente) **no tienen orden posicional genómico** (no reflejan LD/linkage),
   así que una convolución modelaría adyacencias espurias.

### La red que sí gana: `widedeep_mt` (skip aditivo anclado en MTM + tronco chico)
[`src/models/widedeep_mt.py`](../src/models/widedeep_mt.py), registrado `widedeep_mt`. Con el skip
lineal **congelado en el MTM** (garantiza ≥ GBLUP) + un tronco DEEP chico y muy regularizado
(`[32]`, dropout 0.7), ES honesto (inner temporal 2017→2018 en el fold 2019; holdout por línea en el
fold 2018) y ensemble de 10 seeds, la red **supera a GBLUP en AMBOS folds**: val 2019 **0.3212**
(Δ+0.007) y val 2018 **0.3343** (Δ+0.011, significativo). wandb
[`z5znac1u`](https://wandb.ai/oromagnoli-/maze-genomic-pred/runs/z5znac1u). Config:
[`exp_widedeep_mt_GY_optimal_goldenrule.yaml`](../configs/exp_widedeep_mt_GY_optimal_goldenrule.yaml).

**Advertencia honesta (clave para el informe):** la ganancia es **100% del skip aditivo**; el tronco
profundo aporta ~0 (barrido: cuanto más chico el deep, más converge al MTM; con `[16]` la red ES el
MTM). O sea: **sí existe una red neuronal que le gana a GBLUP, pero solo porque lleva el multi-trait
GBLUP adentro** — no hay mecanismo puramente profundo/no-lineal que lo supere. Es la confirmación más
fuerte del techo aditivo de GY en líneas endocriadas (dominancia estructuralmente nula, epistasis que
no replica): incluso montando deep sobre el mejor aditivo, lo no-lineal es ruido. Cierra el ángulo
"deep learning vencería al lineal" en óptimo. La mejora real y limpia sigue siendo el **MTM aditivo**
(Fase 8); `widedeep_mt` es su versión "deep learning" para quien exija una red.

## Fase 10 — Pleiotropía en sequía: el techo genómico (LASSO) aguanta; solo el fenotipo de PH ayuda

**Objetivo:** superar el techo de sequía (bagged-LASSO, 0.4006 en deploy2019) con el mecanismo que
funcionó en óptimo (pleiotropía GY↔PH, Fase 8). En sequía la correlación within-year GY_dro↔PH_dro es
fuerte (0.55/0.22/0.51 en 2017/2018/2019; GY↔AD ≈ −0.11, inútil), así que había fundamento. Resultado:
**la señal GENÓMICA de PH no rompe el techo; solo su FENOTIPO observado lo supera** (otro régimen).

### El flip por fold y por qué el borrowing genómico falla
LASSO single-trait: **deploy2019 0.4006** (train 2017+2018) / **golden2018 0.1973** (train 2017). En
2019 la selección sparse domina; en 2018 (1 solo año de train) es débil. **Tres métodos de pleiotropía
genómica nuevos, todos DAÑAN en deploy2019** (rompen la sparsity → densos → pierden el borde del LASSO):

| método (fair, genotype-only) | deploy2019 | golden2018 |
|---|---|---|
| LASSO bagged (techo) | **0.4006** | 0.1973 |
| MTM (multi-trait GBLUP denso, GY+PH+AD) | 0.2402 (Δ−0.16) | 0.2192 (Δ+0.02) |
| MT-LASSO bagged (MultiTaskLasso, soporte L2,1 compartido, z-score) | 0.2978 (Δ−0.10, ns≈3500) | 0.2003 (empate) |
| Adaptive-LASSO (penalidad de GY reponderada por soporte de PH) | 0.2716 (Δ−0.13) | 0.1879 (Δ−0.01) |

Mecanismo: forzar a GY a compartir/priorizar el soporte de PH lo vuelve **denso** (MT-LASSO selecciona
~3500 marcadores vs ~20–50 del LASSO sparse), y la ventaja del LASSO en 2019 ES la sparsity. En el fold
débil (2018) el MTM le gana al LASSO por poco (0.219 vs 0.197) — pero 2018 no es donde está el techo, y
ningún método gana en AMBOS. La señal **genómica** de PH es redundante con la de GY (ambas capadas por
el mismo parentesco) ⇒ no agrega. Coherente con el techo informacional de
[[redes-no-superan-lasso-drought]].

### Lo único que supera al LASSO: observar el FENOTIPO de PH (trait-assisted)
Índice `GY_dro ~ LASSO_GY(genotipo) + PH_dro + AD_dro` con PH/AD **observados en las líneas de val**:
**deploy2019 0.549** (Δ+0.149, IC[+0.115,+0.170]) y **golden2018 0.287** (Δ+0.090), ambos P(Δ>0)=1.00.
Grande y robusto en ambos folds. **Pero es sobre todo la correlación fenotípica de PH, no mejor
genómica:** PH_dro observado **solo** (sin genotipo) ya da **0.5113** en 2019 y 0.2191 en 2018 — por
ENCIMA del LASSO genómico (0.4006 / 0.1973). El índice suma a PH un empujón genómico (LASSO) y un poco
de AD (con signo invertido). Es GS asistida por trait secundario (phenomic selection), régimen de
información distinto (rompe `predict(X)`: necesita PH del val), no comparable al LASSO genotype-only.

### Veredicto
El **techo GENÓMICO (genotype-only) de GY sequía es el bagged-LASSO** — reconfirmado desde 3 ángulos de
pleiotropía nuevos (MTM, MT-LASSO, adaptive-LASSO), todos rotos por la misma razón (el borrowing
densifica y mata la sparsity que gana en 2019). La pleiotropía que SÍ funcionó en óptimo (donde el
techo es GBLUP denso) **no transfiere a sequía** (donde el techo es LASSO sparse): meter PH densifica.
Lo único que baja el techo es **observar PH** en las líneas a predecir, y eso refleja la correlación
fenotípica de PH (0.51) más que genómica. **Para el modelo final 2020:** si el pipeline mide PH_dro de
las líneas 2020 antes de necesitar GY, el índice trait-assisted (LASSO+PH) da un salto grande
(0.40→0.55 en el análogo 2019); si no, el bagged-LASSO sigue siendo la portada. Nada nuevo se
productiviza en `src/models` (norma: solo lo que gana robusto y fair; acá el fair no gana y el
assisted es otro problema). La mejora amplia del proyecto sigue siendo **LASSO sobre GBLUP (+63 %)**.

## Fase 11 — No-linealidad en traits secundarios (PH/AD): el deep le gana a GBLUP en AD y PH (deploy), pero no transfiere a GY

**Objetivo:** ver si el deep learning (no-linealidad) mejora la predicción de PH (altura) o AD (días a
floración) en ambos ambientes y —si mejora— si sumar esas predicciones a los modelos lineales rompe los
techos de GY (optimal ~0.31, drought ~0.40). La hipótesis tenía fundamento distinto al de GY: PH y AD son
**más heredables y estables entre ambientes** que GY (`r(opt,dro)` = 0.51 PH / 0.62 AD vs **0.14** GY), así
que si hubiera arquitectura no-lineal explotable debería verse mejor ahí (más SNR); y AD (floración) tiene
genes de efecto grande (Vgt1/Vgt2, ZmCCT), candidatos a efectos umbral que un modelo aditivo promedia.

### Fase A — ¿la no-linealidad predice mejor PH/AD? (gating)
Esquema expanding (val 2018 = train 2017; val 2019 = train 2017+2018), Pearson por fold. GBLUP-REML
(lineal) vs MLP (no-lineal) vs XGBoost (árboles). Números single-seed 575:

| trait | env | GBLUP 2018 / 2019 | MLP 2018 / 2019 | XGB 2018 / 2019 |
|-------|-----|-------------------|-----------------|-----------------|
| PH | optimal | 0.443 / 0.071 | 0.408 / 0.109 | 0.367 / 0.066 |
| PH | drought | 0.340 / −0.048 | 0.357 / −0.096 | 0.303 / −0.013 |
| AD | optimal | 0.463 / **0.298** | 0.485 / **0.312** | 0.370 / 0.062 |
| AD | drought | 0.259 / 0.317 | 0.243 / 0.285 | 0.272 / 0.140 |

- **XGBoost pierde en casi todo** (colapsa en AD-opt 2019: 0.06). Los árboles no suman miles de efectos
  aditivos chicos ⇒ la no-linealidad tipo árbol NO ayuda (mismo diagnóstico que en GY).
- **PH (con ensemble chico): parece que no ayuda; con ensemble grande, SÍ (ver "Segundo ataque").** Ensemble
  de 5 seeds en optimal: 2018 0.422 (< GBLUP 0.443), 2019 0.072 (≈ GBLUP 0.071). El "+0.037" de single-seed en
  2019 era ruido (per-seed −0.018…+0.128). PH→2019 es hiper-inestable a la seed: con **10 seeds** el número se
  estabiliza en 0.118 y SÍ le gana a GBLUP significativamente en el fold deploy (detalle abajo). En drought,
  todo ~0/negativo (el drift lo mata) y ahí el deep no ayuda.
- **AD-drought: el MLP pierde** (2019 0.285 < GBLUP 0.317). No ayuda.
- **AD-optimal: ÚNICO caso donde la red gana robusto.** **Cada seed** predice 2019 mejor que GBLUP
  (per-seed 0.31–0.40, todas > 0.298); el ensemble de 5 seeds da **val2019 0.33–0.36 según el set**
  (config canónico `seeds 575–579` por `train.py`: **0.334, +0.036**; set `575,1,2,3,4`: 0.359, +0.061) y
  **val2018 ≈ GBLUP** (0.46, empate). No es winner's curse (gana en el año deploy y empata out-of-sample,
  el patrón *opuesto* al flip-por-fold). La magnitud oscila (~+0.04…+0.06) pero el signo es estable.

**Ablación lineal** (misma receta MLP, `hidden=[]` lineal vs `[128]` vs `[256,128]`; ensemble 5 seeds),
para separar no-linealidad de "estimador distinto":

| arquitectura | AD-opt 2018 | AD-opt 2019 |
|--------------|-------------|-------------|
| `[]` (lineal) | 0.408 | 0.324 |
| `[128]` (no-lineal) | 0.462 | 0.359 |
| `[256,128]` | 0.434 | 0.267 |

(tabla y bootstrap sobre el ensemble `575,1,2,3,4`). El +0.061 sobre GBLUP en 2019 se descompone en
**+0.026 = estimador** (dropout/early-stop-pearson vs ridge; el MLP-lineal ya da 0.324) y **+0.035 =
no-linealidad** (la capa oculta). Es **shallow**: dos capas overfittean (0.267 < lineal), coherente con
p≫n. Bootstrap **pareado** cluster-por-línea (n=722 en 2019,
1418 en 2018): `[128]−GBLUP` **2019 Δ+0.061 IC[+0.035,+0.086] sig**, 2018 Δ−0.001 ns (empate);
`[128]−lineal` (= efecto puro de la no-linealidad) **2018 Δ+0.053 IC[+0.031,+0.075] sig**, 2019 Δ+0.035
IC[−0.001,+0.070] (borde). La señal no-lineal es **real y significativa** (no un artefacto del estimador),
pero acotada a AD-optimal y a una sola capa. Config: `exp_mlp_ens_AD_optimal_goldenrule.yaml`; baselines
`exp_gblup_2017_2018_to_2019_{PH,AD}_{optimal,drought}.yaml`.

### Fase B — ¿rompe el techo de GY sumando PH/AD no-lineal? (índice trait-assisted genotype-only)
Índice OLS ajustado **out-of-fold** (pesos en 2018 con modelos entrenados en 2017; aplicado a 2019 con
modelos en 2017+2018): `GY_pred = w0 + w1·GEBV_GY + w2·PĤ(X) + w3·AD̂(X)`, con PĤ/AD̂ **predichos**
(genotype-only, fair — NO usa el fenotipo del val, a diferencia del trait-assisted de la Fase 10). Control:
secundario **lineal** (GBLUP) para aislar si aporta la no-linealidad. Δ vs techo con bootstrap pareado:

**GY optimal** (techo GBLUP = 0.3145 en 2019):

| variante | r(2019) | Δ vs techo | IC95%(Δ) |
|----------|---------|-----------|----------|
| base + PĤ(MLP) | 0.334 | +0.019 | [−0.008,+0.045] ns |
| base + AD̂(MLP) | 0.307 | −0.008 | [−0.015,−0.002] **sig NEG** |
| base + PĤ(GBLUP) [ctrl lineal] | 0.337 | +0.022 | [−0.013,+0.056] ns |

**GY drought** (techo bagged-LASSO = 0.4006 en 2019):

| variante | r(2019) | Δ vs techo | IC95%(Δ) |
|----------|---------|-----------|----------|
| base + PĤ(MLP) | 0.109 | −0.292 | **sig NEG** (colapso) |
| base + AD̂(MLP) | 0.401 | −0.000 | [−0.023,+0.025] ns (peso≈0) |
| base + PĤ(GBLUP) [ctrl] | 0.113 | −0.288 | **sig NEG** |

Tres hechos: (1) el secundario **no-lineal (MLP) nunca supera al lineal (GBLUP)** para ayudar a GY —en
optimal PĤ(MLP) +0.019 ≤ PĤ(GBLUP) +0.022— ⇒ lo poco que aporta PH es **pleiotropía aditiva** (la mejora
chica del MTM-GBLUP de la Fase 8, no-significativa acá en un solo fold), NO la no-linealidad. (2) **AD̂**
—el único con no-linealidad real— **daña GY en optimal** (Δ−0.008, IC excluye 0) y es neutro en drought
(peso≈0): confirma `gcorr(GY,AD)≈0`. (3) En drought meter PH **colapsa** el índice (0.40→0.11): la
predicción genómica de PH no transfiere 2018→2019 por GxE, así que el peso que le da el ajuste en 2018 le
inyecta ruido a 2019. **Ningún índice rompe el techo en ningún ambiente.**

### Veredicto
(1) La no-linealidad **casi no** mejora la predicción de los traits secundarios: el único caso robusto y
significativo es **AD en optimal** (una red shallow le gana a GBLUP +0.061 en el año deploy, con ~+0.035
atribuible a la no-linealidad). PH: con ensemble chico parece no mejorar, pero con **10 seeds le gana a GBLUP
en el fold deploy** (ver "Segundo ataque"); AD-drought no mejora. (2) Esa no-linealidad
**NO rompe el techo de GY**: el trait donde existe (AD, floración) es genómicamente ~ortogonal a GY
—agregarlo daña u es neutro—, y el trait relevante para GY (PH) no gana nada por no-linealidad (su proxy
lineal empata al no-lineal, y equivale a la mejora aditiva ya conocida del MTM-GBLUP). Los techos aguantan
(**GY optimal ~0.31, GY drought ~0.40**). Coherente con la raíz biológica del techo aditivo de GY (líneas
endocriadas homocigotas ⇒ dominancia = 0): la no-linealidad de AD es real pero, por su ortogonalidad
genética con GY, **no productivizable** para rendimiento. Reproducible con el harness de scratchpad de la
sesión (`explore_nl.py`, `phase_b.py`, `verify_ad.py`) y los configs citados; corridas en modo offline
(sin sync a wandb).

### Segundo ataque — mejorar PH con deep learning para asistir a GY
GBLUP es pésimo prediciendo PH hacia 2019 (0.071) y PH es el trait más correlacionado con GY (0.64 opt), así
que la hipótesis era: si el deep logra un PĤ mucho mejor, el índice trait-assisted podría mover el techo de
GY. Se atacó PH con **varias alternativas**.

**Diagnóstico — la causa es DOMAIN/CONCEPT-SHIFT, no linealidad.** Predictibilidad de PH_opt val2019 según el
set de train (GBLUP, determinista): train[2017]=**0.184**, train[2018]=0.031, train[2017+2018]=**0.071**.
Entrenar con 2017 solo predice 2019 mucho mejor que el pool: **la cohorte 2018 anti-transfiere**. Y 2017 no es
un año flojo (train[2017]→val2018 = 0.443). El covariate-shift weighting (ponderar el train por afinidad
genómica a 2019) **empeora** (0.118→0.050): el density-ratio pesa MÁS a 2018 (0.67) porque genómicamente 2019
se parece a su año adyacente, pero 2018 es el que anti-transfiere. Es decir, el shift está en **P(Y|X)**
(concept drift), no en P(X) ⇒ ningún criterio basado solo en X (covshift, selección de año por SNPs) lo
arregla. Coherente con [[elegir-anio-durante-entrenamiento]].

**PH_opt→2019 es hiper-inestable a la seed** (per-seed −0.02…+0.20). Un sweep de 3 seeds da números que
oscilan 0.065–0.197 entre configs casi idénticos: puro azar. Solo el ensemble de 10 seeds + bootstrap
pareado cluster-línea es confiable.

**Resultado (ensemble 10 seeds, val2019 = análogo deploy; GBLUP 2018=0.443, 2019=0.071):**

| método deep | val2018 (Δ vs GBLUP) | val2019 (Δ vs GBLUP) |
|-------------|----------------------|----------------------|
| MLP pearson_mse | 0.407 (−0.037 sig) | **0.118 (+0.046 SIG, IC[+0.018,+0.073])** |
| MLP mse | 0.414 (−0.029 sig) | 0.105 (+0.034 sig) |
| MLP multi-ambiente PH_opt+PH_dro | 0.410 (−0.034 sig) | 0.114 (+0.043 sig) |

**Sí se le gana a GBLUP en PH, en el fold deploy.** Las 3 familias (MLP single, multi-ambiente, multitask)
convergen: **+0.043…+0.046 significativo en val2019** (train 2017+2018 → predecir el último año = análogo de
predecir 2020). El MLP es más robusto que el ridge de GBLUP a la transferencia negativa de 2018, por eso
brilla justo en el fold con más datos y shift. Cae en val2018 (train solo-2017, régimen no-deploy). Config
`exp_mlp_ens_PH_optimal_deploy.yaml`. Otras alternativas: XGBoost, blend GBLUP+MLP, [256,128] — ninguna
robustamente mejor. En PH_drought el deep NO ayuda (todo ~0 hacia 2019: el drift lo mata).

**Aporte a GY: un empujón MARGINAL en optimal, nada en drought — no es un quiebre robusto.** Metiendo el PĤ
deep (pearson_mse, genotype-only) en el índice OOF de GY (fit 2018 → aplica 2019): GY-opt = 0.3426, **Δ+0.028
vs GBLUP, IC[+0.004,+0.051] (apenas sig)**; el control lineal PĤ-GBLUP da +0.022 (ns), o sea el deep supera
al lineal por apenas +0.006 (ese es el aporte de la no-linealidad; el resto es pleiotropía aditiva). GY-drought
COLAPSA (0.40→0.10, sig neg) por GxE. Es un solo fold (2019), apenas significativo y seed-dependiente: con 3
sets distintos de seeds del PĤ el Δ va de **+0.022 a +0.028** (uno ns), siguiendo la calidad del PĤ ⇒ NO es un
quiebre robusto del techo, comparado con el MTM-GBLUP (+0.005/+0.010 en AMBOS folds, determinista). El eslabón "PĤ mejor → GY mejor" queda casi cortado porque PĤ→2019 sigue siendo débil (0.118):
su correlación con GY está acotada
por 0.118·gcorr(PH,GY) y GEBV_GY ya captura el aditivo compartido.

### Veredicto del segundo ataque
Se le **gana a GBLUP en PH con deep learning en el año deploy** (+0.046 sig, robusto a 3 métodos y 10 seeds),
por mayor robustez de la red al concept-drift entre cohortes — un resultado positivo y publicable. Pero esa
mejora **no se transfiere a GY**: PĤ es todavía demasiado débil (0.118) para mover el rinde más allá de la
pleiotropía aditiva ya conocida (MTM), y en sequía la GxE lo rompe. Los techos de GY siguen intactos (opt
~0.31, dro ~0.40). Config `exp_mlp_ens_PH_optimal_deploy.yaml`; reproducible con `ph_campaign.py`,
`ph_verify.py`, `covshift.py`, `ph_final.py` (scratchpad); offline (sin sync a wandb).

### Tercer ataque — ¿mejora AMPLIA de GY combinando deep PH/AD + lineal? El límite y la vía phenomic
**Consigna:** mejorar *ampliamente* el techo de GY combinando predicciones deep de PH/AD con los ganadores
lineales, probando modelos distintos para predecir PH/AD lo mejor posible.

**Argumento de imposibilidad (genotype-only).** El aporte de PĤ al índice de GY está acotado por
`r(PĤ, PH_true) · gcorr(PH, GY)`. Y `r(PĤ, PH_true)` hacia 2019 tiene **techo ~0.12–0.18** (concept-drift:
la cohorte 2018 anti-transfiere; ni el oráculo train-2017-only = 0.184 se supera sin ver 2019). Con gcorr≈0.6
⇒ el aporte de PH a GY está capado en ~0.1, y GEBV_GY ya tiene el aditivo compartido. AD, aunque se predice
bien (0.36), es **genéticamente ⊥ a GY** (gcorr≈0). Y GY es aditivo (dominancia = 0) ⇒ sin no-linealidad de
GY para explotar. A priori: **no hay mejora amplia genotype-only.**

**Modelos deep NUEVOS para maximizar PĤ — ninguno rompe ~0.13 en 2019:** AE-embedding no-lineal (dim128, BCE
train-only) + GBLUP (2019 = −0.006, la compresión pierde la señal), AE + MLP (2019 = 0.143, single-seed,
dentro del ruido), multitask PH+GY+AD 15 seeds (2019 = 0.125), y los previos (MLP pearson_mse 0.118,
multi-ambiente 0.114, covshift, XGBoost, [256,128]). Todos convergen a **~0.12–0.13**: es un techo
INFORMACIONAL (concept-drift), no de capacidad del modelo.

**Combinación comprehensiva (índice OOF, fit 2018 → aplica 2019), genotype-only vs PHENOMIC (usa el
fenotipo OBSERVADO del val — otro régimen de información, se mide antes de la cosecha de GY):**

| régimen | GY optimal (techo 0.3145) | GY drought (techo 0.4006) |
|---------|---------------------------|---------------------------|
| genotype-only: +PĤ+AD̂ deep | 0.3418 (+0.027, apenas sig, no robusto; AD̂ no aporta) | ~0.10 (COLAPSA por GxE) |
| **PHENOMIC: +PH_obs** | **0.4265 (+0.112 SIG)** | **0.5771 (+0.177 SIG)** |
| PHENOMIC: +PH_obs+AD_obs | 0.4304 (+0.116 SIG) | 0.5663 (+0.166 SIG) |

(PH observado **solo**, sin genotipo, ya correlaciona con GY_2019: 0.31 óptimo / **0.51 sequía**.)

### Veredicto del tercer ataque
**La mejora AMPLIA existe, pero es PHENOMIC (medir PH), no genotype-only (predecir PH).** Combinar el ganador
lineal (GEBV_GY) con el fenotipo **observado** de PH da GY-opt **0.31→0.43 (+0.11)** y GY-dro **0.40→0.58
(+0.18)** — grande y significativo en ambos. El deep NO puede sustituir la medición: el gap es 100% calidad
de predicción de PH, y PH→2019 está capado en ~0.12–0.18 por concept-drift (probado con AE, multitask,
multi-ambiente, covshift, XGBoost, ensembles grandes). El aporte a GY escala con `r(PĤ,PH)`: PĤ (0.12) da
+0.03 (marginal); PH_obs (1.0) da +0.11–0.18 (amplio). **Recomendación operativa:** para mejorar ampliamente
el modelo final de 2020, el pipeline de breeding debe **MEDIR PH (y PH_dro) de las líneas 2020** antes de
necesitar GY, y usar el índice trait-assisted `GY ~ GEBV_GY + PH_obs`. El deep aporta un PĤ algo mejor que
GBLUP en el deploy (+0.046 en PH), útil si NO se mide PH, pero insuficiente para el salto amplio. Reproducible:
`combine.py`, `phenomic.py`, `new_models.py` (scratchpad). Ver [[deep-mejora-PH-deploy-no-GY]] y
[[mejora-amplia-GY-es-phenomic]].
