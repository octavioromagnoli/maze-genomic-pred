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
  **7 SNPs** caen por debajo de 0.05 (`MAF≥0.10` sacaría 148). Ilustra por qué el
  filtro se recalcula **train-only** y no una vez global.
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
  salida sigmoide + BCE (regla 4).
