# Recomendación: cómo mejorar AMPLIAMENTE la predicción de GY (modelo final 2020)

**TL;DR.** La única forma de mejorar ampliamente el techo de GY es **phenomic selection**: combinar el
predictor genómico lineal (GEBV_GY) con el **fenotipo OBSERVADO de PH** de las líneas objetivo. No se logra
prediciendo PH/AD con deep learning (genotype-only): eso está capado. Ver Fase 11 de `bitacora.md`.

## El resultado
Índice `GY ~ GEBV_GY + PH_observado` (OOF, fit 2018 → aplica 2019; bootstrap pareado cluster-línea):

| GY | techo genómico | + PH observado | salto |
|----|----------------|----------------|-------|
| optimal | 0.3145 (GBLUP) | **0.4265** | **+0.112** (IC[+0.044, +0.181]) |
| drought | 0.4006 (bagged-LASSO) | **0.5771** | **+0.177** (IC[+0.123, +0.227]) |

PH observado **solo** (sin genotipo) ya correlaciona con GY_2019: 0.31 (opt) / **0.51** (dro). AD observado
no aporta (incluso resta en dro) ⇒ el salto es específico de PH (su correlación genética con GY).

## Por qué el deep genotype-only NO alcanza (probado, no por falta de esfuerzo)
El aporte de un PĤ **predicho** al índice está acotado por `r(PĤ, PH_real) · gcorr(PH, GY)`, y `r(PĤ,PH)`
hacia la campaña siguiente tiene **techo informacional ~0.12–0.18** por **concept-drift** entre cohortes de
año (la cohorte 2018 anti-transfiere; ni el oráculo train-2017-only = 0.184 se supera sin ver 2019). Se
verificó con ~15 modelos deep (MLP arch×loss, multi-ambiente, multitask 15-seed, AE-embedding no-lineal,
covariate-shift, XGBoost, ensembles de 20 seeds, stacker no-lineal + features cross-ambiente): **todos caen
en ~0.12–0.13**. Con PĤ≈0.12 el índice de GY sube solo +0.03 (opt, marginal/no robusto) y **colapsa** en dro.
Además: AD se predice bien (0.36) pero es genéticamente ⊥ a GY (gcorr≈0), y GY es aditivo (líneas
endocriadas ⇒ dominancia=0), sin no-linealidad de GY para explotar.

## Acción concreta para 2020
1. **Medir PH (óptimo y sequía) de las líneas 2020** antes de necesitar GY (PH se mide temprano/barato).
2. Ajustar el índice `GY ~ GEBV_GY + PH_obs` en el DEV (2017–2019) y aplicarlo a 2020.
3. Si por logística NO se mide PH: usar el **PĤ deep** (MLP ensemble ≥10 seeds, pearson_mse,
   `exp_mlp_ens_PH_optimal_deploy.yaml`), que le gana a GBLUP en PH en el año deploy (+0.046), como
   sustituto parcial — pero el salto será chico, no amplio.
