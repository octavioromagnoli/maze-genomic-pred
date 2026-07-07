# Blindaje del benchmark de 6 fenotipos — robustez a la semilla y multiplicidad

**Fecha:** 2026-07-06 · **Fold:** val 2019 (train 2017–2018, líneas disjuntas) · **Métrica:** Pearson r.

Este documento registra el **blindaje** del benchmark de los 6 fenotipos (GY/PH/AD × óptimo/sequía)
contra GBLUP. La pregunta: ¿los números del benchmark (puntos con seed 575) se sostienen para el informe?
La respuesta corta: **no como estaban.** El resultado de portada del proyecto (MoE hard, GY_drought ≈ 0.41)
**no es robusto a la semilla**.

## Qué se testeó y cómo

1. **Robustez a la semilla.** Cada modelo neuronal se reentrena sobre varias semillas re-sembrando `seed`
   y **todos** los `random_state` (incluído el del GMM del gating y el del ensemble de expertos). Core
   (hard/soft/ensemble) con 5 semillas `[575, 1, 42, 2024, 7]`; secundarios (gxe/pls/contr) con 3
   `[575, 1, 42]` por costo de cómputo (gxe es lento y ya pierde en la portada). Se reporta **media ± sd**.
   GBLUP es determinista (REML) ⇒ un solo valor.
2. **Multiplicidad.** Sobre los 36 tests (6 modelos × 6 fenotipos) vs GBLUP del punto seed-575 se calcula un
   p-valor bootstrap pareado de dos colas (mismo resample, cluster a nivel línea) y se aplica
   **Benjamini-Hochberg (FDR 5%)**.
3. **Criterio de robustez de una ventaja:** el Δ vs GBLUP es **sign-estable** si mantiene el signo en TODAS
   las semillas. Una ventaja solo es defendible si el IC pareado (muestreo) excluye 0 **y** el Δ es sign-estable.

Reproducible: runners en `scripts/analysis/` — `benchmark_6pheno.py` (puntos + IC pareado),
`robustez_6pheno.py` (semillas + BH-FDR), `ninit_check.py` (diagnóstico del GMM). Reusan el harness
(`run_multitask_evaluation`, `align_by_line`, `bootstrap_table`, `paired_diff_summary`), todo cacheado en
`outputs/eval_cache`. Correr desde la raíz del repo con el entorno `tpfia` y `PYTHONPATH=.`, p.ej.
`PYTHONPATH=. python scripts/analysis/robustez_6pheno.py` (las salidas CSV/JSON quedan junto al script).
Config nuevo: `configs/exp_moe_soft_6head_2017_2018_to_2019.yaml`.

## Resultado: Pearson r media ± sd sobre las semillas

| Fenotipo | GBLUP | MoE hard | MoE soft | Ensemble (sin gating) | MoE GxE | MoE PLS | Contrastivo |
|---|---|---|---|---|---|---|---|
| GY_optimal | 0.315 | 0.197 ± 0.015 ▼ | 0.288 ± 0.019 ○ | **0.320 ± 0.010** ○ | 0.240 ± 0.126 ○ | 0.298 ± 0.098 ○ | 0.237 ± 0.011 ▼ |
| GY_drought | 0.241 | 0.281 ± 0.111 ○ | 0.287 ± 0.031 ○ | **0.298 ± 0.016 ▲** | 0.220 ± 0.076 ○ | 0.212 ± 0.075 ○ | 0.278 ± 0.101 ○ |
| PH_optimal | 0.071 | 0.158 ± 0.117 ○ | 0.110 ± 0.060 ○ | 0.117 ± 0.018 ▲ | 0.135 ± 0.074 ○ | **0.254 ± 0.071 ▲** | 0.238 ± 0.014 ▲ |
| PH_drought | −0.048 | 0.038 ± 0.076 ▲ | −0.046 ± 0.054 ○ | −0.017 ± 0.012 ▲ | −0.028 ± 0.057 ○ | −0.046 ± 0.016 ○ | **0.082 ± 0.034 ▲** |
| AD_optimal | 0.298 | 0.238 ± 0.084 ○ | 0.308 ± 0.009 ▲ | **0.324 ± 0.011 ▲** | 0.288 ± 0.116 ○ | 0.277 ± 0.062 ○ | 0.263 ± 0.097 ○ |
| AD_drought | **0.317** | 0.152 ± 0.097 ▼ | 0.255 ± 0.025 ▼ | 0.262 ± 0.007 ▼ | 0.254 ± 0.039 ▼ | 0.129 ± 0.013 ▼ | 0.157 ± 0.009 ▼ |

▲ = gana a GBLUP sign-estable · ▼ = pierde vs GBLUP sign-estable · ○ = Δ cambia de signo entre semillas (no
confiable). **Negrita** = mayor media por fenotipo (ganador robusto).

## Hallazgos del blindaje

1. **El “campeón 0.407” no replica.** El MoE hard da 0.407 en GY_drought con la seed 575, pero su media sobre
   5 semillas es **0.281 ± 0.111** (rango 0.142–0.407), la **mayor dispersión de todos**. El 0.407 era la mejor
   de cinco. El gating GMM reparticiona la población según la semilla y arrastra a los expertos (y **no** se
   arregla subiendo `n_init` — ver §Diagnóstico del GMM).

2. **El modelo robusto es el ensemble sin gating.** GY_drought **0.298 ± 0.016** — mayor media y menor sd, y el
   **único** que le gana a GBLUP con signo estable en sequía. También lidera GY_optimal (0.320) y AD_optimal
   (0.324) con sd mínima. **El gating (hard/soft/gxe/pls/contr) no le gana al ensemble plano en la portada:**
   0.298 (ens) ≥ 0.287 (soft) ≥ 0.281 (hard) ≥ 0.278 (contr) ≥ 0.220 (gxe) ≥ 0.212 (pls).

3. **Varios “ganadores” del benchmark eran suerte de semilla (sesgo de selección).** El GxE “gana” GY_optimal
   con 0.385 → media 0.240 ± 0.126. El MoE hard “rescata” PH_drought con 0.137 → media 0.038 ± 0.076. Los picos
   no sobreviven al promedio. Es el mismo mecanismo que ya habíamos visto en [soft MoE val 2018].

4. **Lo que SÍ aguanta.** (a) **GBLUP imbatido y sign-estable en AD_drought** (0.317; el trait aditivo/floración
   es su terreno). (b) **PLS y contrastivo le ganan a GBLUP en PH_optimal** de forma estable (aunque GBLUP ahí
   es ~0.07, techo bajo). (c) **El ensemble > GBLUP en GY_drought** de forma robusta (~0.30 vs 0.24). El aporte
   real del enfoque neuronal multitarea sobre GBLUP en sequía existe, pero es **modesto y lo da el ensemble,
   no el gating**.

## Multiplicidad (BH-FDR 5%, punto seed 575)

De 36 comparaciones vs GBLUP, **26 son significativas sin corregir y 25 sobreviven BH-FDR 5%** (solo cae una
marginal: GxE en GY_optimal, p=0.042 → p_BH=0.058). La corrección por multiplicidad es suave. **Pero** esto
mide la incertidumbre de **muestreo** (líneas) a seed fija; la incertidumbre de **semilla** (las sd de arriba)
es una fuente aparte y mayor. Un ▲ del punto 575 que en la tabla de robustez aparece como ○ **no es defendible**.

## Diagnóstico del GMM (`n_init`)

¿La inestabilidad es del GMM mal inicializado (fixeable con `n_init`) o inherente al modelo? Se comparó
GY_drought del MoE hard con `gating.n_init=1` vs `gating.n_init=20` (GMM best-of-20 likelihood) sobre las
mismas 5 semillas:

| `n_init` | media | sd | rango |
|---|---|---|---|
| 1 | 0.281 | 0.111 | [0.142, 0.407] |
| 20 | 0.222 | 0.107 | [0.046, 0.313] |

**Subir `n_init` NO estabiliza.** La sd queda igual (~0.11) y el pico 0.407 **desaparece** (máx 0.313); la
media incluso baja. El mejor-likelihood del GMM no es único ni predice la utilidad downstream: particiones de
likelihood parecido dan Pearson muy dispares en el val (con seed 42 colapsa a **0.046**). ⇒ **la fragilidad es
inherente al ruteo hard top-1 sobre una población con pocos datos por cluster (p≫n), no un artefacto de
configuración.** No hay fix por hiperparámetro. El 0.407 (n_init=1, seed 575) era una **partición sub-óptima
afortunada** para este val en particular, no una buena. (Runner: `ninit_check.py`.)

## Qué es defendible en el informe

- **NO** reportar 0.407 como el resultado de GY_drought. El número honesto es la **media ± sd** entre semillas.
- El resultado defendible en sequía: **un ensemble multitarea de 6 cabezas supera a GBLUP de forma robusta
  (~0.30 vs 0.24)**; el gating por subpoblación **no aporta señal adicional y sí inestabilidad**.
- **GBLUP sigue siendo el mejor para floración (AD)**, imbatido en AD_drought — resultado sólido.
- Siempre acompañar cada Pearson con su sd entre semillas y su estado de sign-estabilidad. Cualquier corrida
  única (una sola seed) es un punto de una distribución ancha, sobre todo para los modelos con gating GMM.
- Todo esto es **intra-ciclo** (2019). No transfiere a 2020 (ver [cierre 2020 no predecible]).

## Recomendaciones

1. Fijar la práctica de reportar **media ± sd sobre ≥5 semillas** para todo modelo neuronal del informe.
   Cualquier sweep debe seleccionar por un objetivo **promediado entre semillas**, no con una sola seed (el
   sweep original co-seleccionó HP con la seed 575, lo que amplificó el sesgo).
2. **No usar el MoE hard (gating PCA+GMM) como modelo de portada.** Su ventaja no es robusta y **no es fixeable**
   por hiperparámetro (subir `n_init` no estabiliza; ver diagnóstico). El ruteo hard top-1 es inherentemente
   frágil en p≫n con pocos datos por cluster.
3. Usar el **ensemble multitarea de 6 cabezas sin gating** como modelo neuronal de referencia: es el más robusto
   (menor sd) y el mejor en media en la portada, y le gana a GBLUP en GY_drought de forma sign-estable.
4. Reencuadrar el informe: el aporte metodológico defendible **no** es "el gating por subpoblación despega la
   sequía" (refutado), sino "un ensemble multitarea de 6 cabezas mejora modestamente sobre GBLUP en rinde-sequía,
   mientras GBLUP mantiene la ventaja en floración (trait aditivo)" — más el propio hallazgo negativo sobre la
   fragilidad del gating, que es un resultado metodológico valioso por derecho propio.
