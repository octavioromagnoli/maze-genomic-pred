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
