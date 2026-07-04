# CLAUDE.md

Contexto del proyecto para Claude Code. Leé esto antes de tocar nada.

## Qué es este proyecto

Predicción del rendimiento de grano (Grain Yield) en maíz a partir de ~4.612 marcadores genéticos (SNPs) binarios de ~3.529 líneas. Es **regresión supervisada** en régimen **p ≫ n** (más features que muestras). Datos organizados por campaña 2017–2020 (CIMMYT, dataset de Lopez-Cruz et al.). Targets secundarios: altura de planta (PH) y días a floración (AD), medidos en ambiente óptimo y de sequía. Baseline de referencia: Ridge / GBLUP.

## Reglas no negociables (NUNCA las violes)

Estas invalidan el experimento si se rompen. Si una tarea te pide algo que rompe una de estas, frená y avisá antes de seguir.

1. **Sin leakage de preprocesamiento.** Scaler, PCA, autoencoder e imputación se ajustan SOLO con el train de cada fold y se aplican a val/test. Nunca `.fit()` sobre todo el dataset.
2. **No partir una línea entre train y test.** Agrupar por id de línea en los splits.
3. **Validación temporal, nunca aleatoria.** **Split temporal fijo** (el forward-chaining se descartó: con solo 3 años de dev daba folds de un único año, más ruido que señal). Dev = 2017–2019: train 2017–2018, val 2019; **2020 es holdout final intocable** hasta el cierre. El esquema se elige por config (`cv.scheme: fixed_temporal`, `cv.val_year`, `cv.holdout_test_year`), nunca a mano. Prohibido `KFold` / `train_test_split` aleatorio sobre todas las filas. Para el cierre, reentrenar sobre todo el dev (2017–2019) para predecir 2020.
4. **Autoencoders con BCE, no MSE.** Los SNPs son binarios → salida sigmoide + binary cross-entropy.
5. **Métrica de portada: correlación de Pearson** predicho–observado por fold. RMSE y R² de apoyo.

## Estructura del repo

- `src/{data,models,training,eval,utils}/` — toda la lógica reutilizable vive acá.
- `configs/` — un YAML por experimento. Los hiperparámetros se cambian en el YAML, NUNCA editando código a mano.
- `scripts/` — `train.py` (único entrypoint).
- `notebooks/` — solo exploración; **solo importan y llaman funciones de `src/`**, sin lógica adentro.
- `experiments/`, `data/`, `wandb/`, `checkpoints/` y `outputs/` — gitignored.

## Infraestructura de Fase 0 — engancharse acá, no reimplementar

Fase 0 dejó el harness end-to-end armado (ver `docs/bitacora.md`). Toda implementación nueva se enchufa en estas piezas; no rehacer ni puentear:

- **Datos**: `src/data/loader.py` → `load_dataset(geno_csv, pheno_csv)` y `Dataset.to_modeling_arrays(target, env)` devuelven `ModelingArrays` (`X`, `y`, `year`, `line_id`, `env`). Es la única vía de cargar datos; rutas por config (`data.geno_csv`, `data.pheno_csv`).
- **Modelos**: heredar de `BaseModel` (`src/models/base.py`) — contrato `fit(X, y) -> self` / `predict(X) -> np.ndarray` — y **registrar la clase en `src/models/registry.py`** (`REGISTRY`). El modelo se elige por `model.name` en el YAML; `train.py` instancia vía `build_model(config)`, nunca importando clases a mano.
- **Evaluación**: `src/training/run.py` → `run_evaluation(config, log_fn=None, return_models=False)` orquesta carga → split → fit por fold (modelo fresco por fold, sin fuga de estado) → métricas → IC. NO escribir loops de evaluación ad-hoc; consumir esta función.
- **Splits**: `src/eval/splits.py` → `fixed_temporal_split`, guard `assert_no_line_leakage` por fold, `final_test_split` para el cierre sobre 2020. No reimplementar splits en ningún modelo.
- **Métricas**: `src/eval/metrics.py` → `regression_metrics`, `pearson_r`/`rmse`/`r2`, `bootstrap_ci` (cluster a nivel línea), `summarize_folds`. Centralizadas; no recalcular a mano.
- **Persistencia**: `src/training/save_model.py` centraliza caché de resultados (indexada por hash del config) + Artifacts de modelo + `load_model_artifact(ref, expected_train_years=...)` con guard `LeakageError`. Reusar embeddings/encoders va por acá (chequea `train_years` para no meter val/test en el ajuste).

## Comandos

- Correr un experimento: `python scripts/train.py --config configs/exp_xxx.yaml`
- Lanzar un sweep: `wandb sweep configs/sweep_xxx.yaml`, después `wandb agent <sweep_id>` dentro de tmux (para que sobreviva al cierre de sesión).
- Instalar entorno: `pip install -r requirements.txt` (versiones pinneadas).
- Notebooks: `nbstripout` ya limpia el output antes de cada commit.

## Convenciones

- Seeds (numpy, torch, sklearn) seteadas y guardadas DENTRO del config; nada de aleatoriedad sin registrar.
- Rutas de datos por config o variable de entorno; nada hardcodeado.
- Un modelo nuevo en `src/models/` respeta la interfaz común (`fit`/`predict`), se registra en `registry.py` y hace su preprocesamiento train-only dentro de `fit` (regla 1). El default de `BaseModel.save/load` es joblib; **los modelos torch deben overridear `save`/`load`** para serializar `state_dict` + arquitectura.
- Métricas centralizadas en `src/eval/metrics.py`; splits en `src/eval/splits.py`. No reimplementar splits ni métricas ad-hoc en cada modelo.

## wandb

- `train.py` llama `wandb.init()` logueando el config completo y las métricas por fold (y por época en modelos torch). `wandb.mode` por config; default `offline` (sin login) → sincronizar después con `wandb sync`.
- Con `artifacts.save_model: true`, `train.py` guarda el modelo del fold de validación como copia local (`checkpoints/<name>/`) + wandb Artifact (`type=model`), con metadata `{model_name, target, env, train_years, seed, val_metrics}`. Para usar lo que entrenó el otro, bajá el Artifact en lugar de re-entrenar.
- Los embeddings de PCA/AE/VAE se guardan como Artifacts POR FOLD (recordá: ajustados solo en train). Al reusarlos, `load_model_artifact(..., expected_train_years=...)` valida que no haya leakage.
- Caché de resultados: `run_evaluation` se cachea a disco indexado por hash de `(data, cv, model, seed, n_boot, alpha)`; el notebook de comparación levanta de `outputs/eval_cache` sin reentrenar. Cambiar cualquier hiperparámetro invalida solo su entrada.
- En la bitácora/informe, cada número linkea a su corrida de wandb; nada copiado a mano.

## Git

- `main` siempre funcional. Una rama por feature/experimento (`feat/...`, `exp/...`).
- Merge solo por Pull Request con revisión del otro integrante.
- Tareas en GitHub Issues + Projects.

## Estilo

- Comunicación en español rioplatense; términos técnicos en inglés.
- En el informe (LaTeX): prosa fluida, sin bullet points. Bibliografía base: Bishop & Bishop (2024) y Murphy (2022).