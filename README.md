# maze-genomic-pred

Predicción del rendimiento de grano (grain yield, GY) en maíz a partir del
genotipo. Es un problema de regresión en régimen p >> n: 4612 marcadores SNP
binarios de 3528 líneas del programa CIMMYT (Lopez-Cruz et al.), organizadas por
campaña 2017-2020. Como caracteres secundarios se modelan la altura de planta (PH)
y los días a floración (AD), cada uno en ambiente óptimo y de sequía.

La pregunta del trabajo es si una red Mixture-of-Experts (MoE) multitarea, que
caracteriza la subpoblación genómica de cada línea y rutea la predicción a un
experto especializado, le gana a GBLUP, el modelo lineal mixto estándar de la
selección genómica.

## Resultados principales

En validación temporal (train 2017-2018, val 2019), sobre el rendimiento en sequía
(el ambiente más difícil), el MoE con ruteo duro llega a un Pearson de 0.407 contra
0.241 de GBLUP, una diferencia significativa por bootstrap pareado. En el ambiente
óptimo el orden se invierte: los ruteos supervisados por fenotipo (GxE, PLS) superan
a GBLUP y el MoE duro queda por debajo. No hay un modelo que domine en los dos
ambientes, lo que es coherente con la baja correlación entre ellos (r = 0.14).

Sobre el ciclo nuevo (test 2020) la predicción colapsa para todos los modelos: no
hay solapamiento de líneas con el desarrollo. Fenotipar un 5-10% de las líneas del
ciclo objetivo y sumarlas al train (spike-in) restablece correlaciones de hasta 0.50,
con ventaja para la familia MoE sobre el baseline aditivo.

El detalle completo está en el informe (`informe/Cardelli_Romagnoli_TPFINAL.pdf`) y
los números exactos con sus configuraciones en `informe/RESULTADOS_Y_CONFIGURACIONES.txt`.

## Datos

`data/Geno_data.csv` (GID + 4612 columnas SNP binarias) y `data/Pheno_data.csv`
(GID, Year y una columna por trait x ambiente). La correspondencia genotipo-fenotipo
es completa y cada línea aparece en una única campaña. `data/` está gitignored.

## Entorno

```bash
pip install -r requirements.txt
```

El proyecto se corrió sobre un entorno conda llamado `tpfia`. Las versiones están
pinneadas en `requirements.txt` (numpy, pandas, scikit-learn, xgboost, torch, wandb).

## Uso

Todo experimento se lanza con el mismo entrypoint, eligiendo el modelo y el split por
config:

```bash
PYTHONPATH="$PWD" python scripts/train.py --config configs/exp_moe_hard_6head_2017_2018_to_2019.yaml
```

Los hiperparámetros se cambian en el YAML, nunca editando código. `train.py` loguea a
Weights & Biases en modo offline por defecto (se sincroniza después con `wandb sync`).
Si en macOS aparece un crash de OpenMP, agregar los flags
`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE KMP_INIT_AT_FORK=FALSE`.

Para reproducir las figuras y tablas del informe, los scripts viven en
`scripts/analysis/report_figs/`:

```bash
# comparativa val 2019 (Pearson + IC95%) de los 6 modelos, ambos ambientes
PYTHONPATH="$PWD" python scripts/analysis/report_figs/fig_model_comparison.py

# cierre 2020 con spike-in (train 2018+2019 + fracción de 2020)
PYTHONPATH="$PWD" python scripts/analysis/report_figs/repro_test2020_spike_train1819.py

# robustez al ruido de genotipado
PYTHONPATH="$PWD" python scripts/analysis/report_figs/noise_analysis.py
```

Los resultados de validación quedan cacheados en `outputs/eval_cache/` (indexado por
hash del config), así que volver a correr no reentrena.

## Modelos

Cada modelo respeta la interfaz `fit`/`predict` de `src/models/base.py` y se elige por
`model.name` en el config (`src/models/registry.py`):

- Baselines: `gblup` (referencia de industria), `xgboost`, `mlp`, `mean_predictor`.
- Multitarea: `mlp_multitask`, `ensemble_multitask` (ensemble de seeds; también
  envuelve al soft MoE).
- Mixture-of-Experts: `moe_multitask` (ruteo duro PCA+GMM, el titular), `moe_soft_multitask`
  (ruteo blando), `moe_gxe_multitask` (ruteo por ambiente), `moe_pls_multitask` (ruteo
  supervisado por fenotipo).
- Variantes exploradas: `gblup_residual`, `moe_gblup_residual_multitask`, `moe_singletask`,
  `cbeta_vae`.

## Estructura

```
src/
  data/      loader de los CSV + preprocesamiento train-only (MAF, PCA, PLS, AE)
  eda/       cómputo del análisis exploratorio (drift, parentesco, correlaciones)
  models/    interfaz fit/predict, registry y todos los modelos
  training/  loops de evaluación temporal (run, run_multitask), sweeps, persistencia
  eval/      métricas (Pearson/RMSE/R²), splits antileakage, spike-in, ruido, ensembles
  utils/     seeding y carga de config
scripts/     train.py (entrypoint), train_ae.py, analysis/report_figs/ (figuras)
configs/     un YAML por experimento (~120)
notebooks/   exploración; solo importan y llaman funciones de src/
informe/     informe LaTeX, PDF y registro de resultados
docs/        bitácora del proyecto
```

`data/`, `experiments/`, `wandb/`, `checkpoints/` y `outputs/` están gitignored.

## Evaluación

Las reglas que hacen honesta la comparación:

- El preprocesamiento (escalado, PCA, autoencoder, imputación) se ajusta solo con el
  train de cada fold y se aplica a val/test.
- Los splits agrupan por línea: ninguna línea cruza entre train y test.
- La validación es temporal, nunca aleatoria. Desarrollo 2017-2019 (train 2017-2018,
  val 2019); 2020 es el holdout final. El esquema se elige por config.
- La métrica de portada es la correlación de Pearson predicho-observado por fold, con
  matching20 (coincidencia del 20% superior) como medida de utilidad de selección.
