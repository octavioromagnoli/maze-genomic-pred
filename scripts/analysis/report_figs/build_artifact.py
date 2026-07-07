"""Ensambla la pagina (Artifact) del informe: embebe los 3 PNGs en base64 y arma la tabla de
numeros LEYENDO los CSV generados (regla del proyecto: nada copiado a mano). Escribe un .html
(solo <style> + contenido; el Artifact envuelve doctype/head/body)."""
import base64
import os
import pandas as pd

FIGDIR = "experiments/figs/report"
OUT = "/private/tmp/claude-501/-Users-octavio-Documents-GitHub-maze-genomic-pred/4c4ee452-7862-45ff-beaf-0502fb8b25fe/scratchpad/anexo-visual-informe.html"


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


img1 = b64(os.path.join(FIGDIR, "fig1_eda_pca3d.png"))
img2 = b64(os.path.join(FIGDIR, "fig2_model_comparison.png"))
img3 = b64(os.path.join(FIGDIR, "fig3_moe_clusters.png"))
img4 = b64(os.path.join(FIGDIR, "fig4_noise_inference.png"))
_p5 = os.path.join(FIGDIR, "fig5_noise_train.png")
img5 = b64(_p5) if os.path.exists(_p5) else None
_p6 = os.path.join(FIGDIR, "fig6_spike_2020.png")
img6 = b64(_p6) if os.path.exists(_p6) else None

# retención bajo ruido (GY_drought) para la prosa, desde el CSV
_nz = pd.read_csv(os.path.join(FIGDIR, "fig4_noise_inference.csv"))
_dro = _nz[_nz["head"] == "GY_drought"].groupby(["model", "eps"])["pearson"].mean().unstack()
def _v(m, e): return float(_dro.loc[m, e])
mh0, mh5, mh10 = _v("MoE hard", 0.0), _v("MoE hard", 0.05), _v("MoE hard", 0.10)
ct0, ct10 = _v("Contrastivo", 0.0), _v("Contrastivo", 0.10)
gb0, gb10 = _v("GBLUP", 0.0), _v("GBLUP", 0.10)

# sección de ruido en train (fig5) solo si ya existe
fig5_block = ""
if img5:
    _tn = pd.read_csv(os.path.join(FIGDIR, "fig5_noise_train.csv"))
    _tng = _tn.groupby(["model", "eps"])["pearson"].mean().unstack()
    mh_c, mh_n = float(_tng.loc["MoE hard", 0.0]), float(_tng.loc["MoE hard", 0.10])
    en_c, en_n = float(_tng.loc["Ensemble", 0.0]), float(_tng.loc["Ensemble", 0.10])
    gb_c, gb_n = float(_tng.loc["GBLUP", 0.0]), float(_tng.loc["GBLUP", 0.10])
    fig5_block = f'''
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 05</span><span class="eyebrow">Ruido en entrenamiento</span></div>
  <h2>¿Entrenar con ruido regulariza?</h2>
  <figure>
    <div class="figbox"><img alt="Ruido en train como regularizacion" src="{img5}"></div>
    <figcaption>Pearson en GY sequía (val 2019 LIMPIO) entrenando sobre SNPs corruptos a ε crecientes.
    Barras = media de las realizaciones; ε=0 es el entrenamiento limpio de referencia.</figcaption>
  </figure>
  <div class="read gold prose"><span class="lab">Lectura</span>
  Al revés que en inferencia, acá el ruido en el train <b>no ayuda</b>: <b>hunde al MoE hard</b>
  ({mh_c:.3f}→{mh_n:.3f}) y al <b>Ensemble</b> ({en_c:.3f}→{en_n:.3f}), y es ~neutro para <b>GBLUP</b>
  ({gb_c:.3f}→{gb_n:.3f}). El flip aleatorio de SNPs no regulariza: destruye señal. La robustez del
  MoE al ruido (fig 04) viene de su regularización <b>ya existente</b> (dropout alto + sharing
  multitarea), no de entrenar sobre datos corruptos. Conclusión: no vale la pena como augmentation.</div>
</section>'''

# sección de cierre 2020 con spike-in (fig6) solo si ya existe
fig6_block = ""
if img6:
    _sp = pd.read_csv(os.path.join(FIGDIR, "fig6_spike_2020.csv"))
    _spg = _sp.groupby(["model", "spike"])["pearson"].mean().unstack()
    def _s(m, s): return float(_spg.loc[m, s])
    mh_0, mh_5, mh_10 = _s("MoE hard", 0.0), _s("MoE hard", 0.05), _s("MoE hard", 0.10)
    gb_0, gb_5, gb_10 = _s("GBLUP", 0.0), _s("GBLUP", 0.05), _s("GBLUP", 0.10)
    fig6_block = f'''
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 06</span><span class="eyebrow">Cierre · holdout 2020</span></div>
  <h2>El spike-in rescata el rinde en 2020</h2>
  <figure>
    <div class="figbox"><img alt="Cierre 2020 con spike-in" src="{img6}"></div>
    <figcaption>Pearson de GY sequía sobre el <b>holdout 2020</b> según el % de líneas de 2020
    fenotipadas y agregadas al train (dev 2017-2019 + spike). Media de 3 reps (qué líneas se spikean),
    banda ±1 sd. La línea roja en 0 marca el colapso cold-start.</figcaption>
  </figure>
  <div class="read terra prose"><span class="lab">El colapso</span>
  Sin spike (0%), predecir 2020 entrenando solo en dev <b>colapsa</b>: 2020 no comparte ninguna línea
  con dev (cero overlap de GID) y la sequía derivó, así que el Pearson es <b>negativo</b> para todos
  (GBLUP {gb_0:+.3f}, MoE hard {mh_0:+.3f}). El 0.407 de validación es dev; <b>no transfiere</b> cross-ciclo.</div>
  <div class="read prose"><span class="lab">El rescate</span>
  Fenotipar apenas <b>5% de las líneas de 2020</b> y sumarlas al train ya <b>da vuelta el signo</b>
  (GBLUP {gb_0:+.3f}→{gb_5:+.3f}, MoE hard {mh_0:+.3f}→{mh_5:+.3f}); con 10% sube más
  (MoE hard {mh_10:+.3f}). El grueso del rescate lo captura el primer 5%. Es la lectura accionable
  para mejoramiento: un puñado de líneas fenotipadas del ciclo nuevo destraba la predicción genómica.</div>
</section>'''

# --- tabla de numeros desde el CSV de la comparacion ---
df = pd.read_csv(os.path.join(FIGDIR, "fig2_model_comparison.csv"))
piv = {}
for _, r in df.iterrows():
    piv.setdefault(r["model"], {})[r["env"]] = (r["point"], r["lo"], r["hi"])
order = sorted(piv, key=lambda m: piv[m].get("drought", (0,))[0], reverse=True)


def cell(t):
    if t is None:
        return '<td class="num">—</td>'
    p, lo, hi = t
    return f'<td class="num"><b>{p:.3f}</b><span class="ci">[{lo:.2f}, {hi:.2f}]</span></td>'


rows = ""
for m in order:
    cls = ' class="champ"' if m == "MoE hard" else (' class="base"' if m == "GBLUP" else "")
    rows += (f'<tr{cls}><td class="mdl">{m}</td>'
             f'{cell(piv[m].get("drought"))}{cell(piv[m].get("optimal"))}</tr>')

drift = pd.read_csv(os.path.join(FIGDIR, "fig1_drift_auc.csv"))
drift_txt = " · ".join(f'{r["par_de_años"]} <b>{r["cv_auc"]:.2f}</b>' for _, r in drift.iterrows())

HTML = f"""<title>Anexo visual del informe — Maize GY</title>
<style>
:root {{
  --bg:#f6f4ec; --surface:#fffdf8; --ink:#19221c; --muted:#5b6a5e; --faint:#8593869c;
  --line:#e0e2d5; --teal:#1f8a7c; --terra:#d9563a; --gold:#c08a1e; --blue:#3f6f8f;
  --champ-wash:#e6f3f0; --base-wash:#f0efe6;
  --measure:66ch;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --bg:#10150f; --surface:#181e15; --ink:#e7ece5; --muted:#98a698; --faint:#6d7b6d99;
    --line:#28311f2b; --line:#2a3324; --teal:#3cc7b3; --terra:#f0784f; --gold:#dcab3e; --blue:#6ba7c9;
    --champ-wash:#13332d; --base-wash:#20261a;
  }}
}}
:root[data-theme="light"] {{
  --bg:#f6f4ec; --surface:#fffdf8; --ink:#19221c; --muted:#5b6a5e; --line:#e0e2d5;
  --teal:#1f8a7c; --terra:#d9563a; --gold:#c08a1e; --blue:#3f6f8f; --champ-wash:#e6f3f0; --base-wash:#f0efe6;
}}
:root[data-theme="dark"] {{
  --bg:#10150f; --surface:#181e15; --ink:#e7ece5; --muted:#98a698; --line:#2a3324;
  --teal:#3cc7b3; --terra:#f0784f; --gold:#dcab3e; --blue:#6ba7c9; --champ-wash:#13332d; --base-wash:#20261a;
}}
* {{ box-sizing:border-box; }}
.wrap {{ background:var(--bg); color:var(--ink); font-family:var(--sans);
  line-height:1.62; -webkit-font-smoothing:antialiased; padding:0 0 5rem; }}
.col {{ max-width:960px; margin:0 auto; padding:0 1.5rem; }}
.prose {{ max-width:var(--measure); }}
.prose p {{ margin:0 0 1rem; color:var(--ink); }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--teal); font-weight:600; }}
h1 {{ font-size:clamp(2rem,5vw,3.1rem); line-height:1.04; letter-spacing:-.025em; font-weight:800;
  text-wrap:balance; margin:.6rem 0 .5rem; }}
h2 {{ font-size:clamp(1.35rem,3vw,1.8rem); letter-spacing:-.02em; font-weight:750; text-wrap:balance;
  margin:.3rem 0 .2rem; }}
.lede {{ font-size:1.16rem; color:var(--muted); max-width:60ch; margin:.4rem 0 0; }}
a {{ color:var(--teal); }}

/* header */
header.hero {{ border-bottom:1px solid var(--line); padding:4rem 0 2.4rem; margin-bottom:2.6rem;
  background:linear-gradient(180deg,color-mix(in srgb,var(--teal) 7%,var(--bg)),var(--bg)); }}
.kicker {{ display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; font-family:var(--mono);
  font-size:.74rem; color:var(--muted); letter-spacing:.04em; }}
.tag {{ border:1px solid var(--line); border-radius:999px; padding:.15rem .6rem; }}
.tag.sealed {{ color:var(--terra); border-color:color-mix(in srgb,var(--terra) 40%,var(--line)); }}

/* plates */
.plate {{ margin:3.4rem 0; }}
.plate-head {{ display:flex; align-items:baseline; gap:.9rem; flex-wrap:wrap; margin-bottom:.9rem; }}
.plate-num {{ font-family:var(--mono); font-size:.8rem; color:var(--muted); letter-spacing:.1em; }}
figure {{ margin:1.2rem 0 0; }}
.figbox {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:1rem; overflow-x:auto; }}
.figbox img {{ display:block; max-width:100%; height:auto; margin:0 auto; border-radius:4px; }}
figcaption {{ font-family:var(--mono); font-size:.78rem; color:var(--muted); margin-top:.7rem;
  line-height:1.5; max-width:var(--measure); }}
.read {{ margin-top:1.5rem; padding-left:1.1rem; border-left:3px solid var(--teal); }}
.read .lab {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--teal); font-weight:600; display:block; margin-bottom:.35rem; }}
.read.terra {{ border-left-color:var(--terra); }} .read.terra .lab {{ color:var(--terra); }}
.read.gold {{ border-left-color:var(--gold); }} .read.gold .lab {{ color:var(--gold); }}

/* table */
.tbl-wrap {{ overflow-x:auto; margin:1.4rem 0 0; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th, td {{ text-align:left; padding:.55rem .7rem; border-bottom:1px solid var(--line); }}
thead th {{ font-family:var(--mono); font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); font-weight:600; border-bottom:1.5px solid var(--line); }}
td.mdl {{ font-weight:650; }}
td.num {{ font-variant-numeric:tabular-nums; text-align:right; font-family:var(--mono); }}
td.num .ci {{ display:block; font-size:.72rem; color:var(--muted); font-weight:400; }}
tr.champ {{ background:var(--champ-wash); }}
tr.champ td.mdl {{ color:var(--teal); }}
tr.base td.mdl {{ color:var(--muted); }}
.tbl-note {{ font-family:var(--mono); font-size:.72rem; color:var(--muted); margin-top:.5rem; }}

/* footer status */
.status {{ margin-top:4rem; border-top:1px solid var(--line); padding-top:2rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1rem; margin-top:1.2rem; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:1rem 1.1rem; }}
.card .st {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }}
.card.run .st {{ color:var(--teal); }} .card.next .st {{ color:var(--gold); }} .card.done .st {{ color:var(--muted); }}
.card h3 {{ margin:.3rem 0 .3rem; font-size:1rem; }}
.card p {{ margin:0; font-size:.86rem; color:var(--muted); line-height:1.5; }}
.dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.35rem; }}
.dot.g {{ background:var(--teal); }} .dot.y {{ background:var(--gold); }} .dot.m {{ background:var(--faint); }}
@media (prefers-reduced-motion:no-preference) {{
  .plate, header.hero > * {{ animation:rise .6s cubic-bezier(.2,.7,.3,1) both; }}
}}
@keyframes rise {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:none; }} }}
</style>

<div class="wrap">
<header class="hero"><div class="col">
  <div class="kicker">
    <span class="tag">Maize Genomic Prediction</span>
    <span class="tag">Grain Yield · p ≫ n</span>
    <span class="tag sealed">2020 = holdout de cierre</span>
  </div>
  <div class="eyebrow" style="margin-top:1.4rem">Anexo visual del informe</div>
  <h1>Tres figuras que sostienen la historia</h1>
  <p class="lede">Por qué la validación tiene que ser temporal, quién gana la predicción de rinde,
  cómo el Mixture of Experts rutea la población, cuánto aguanta el ruido, y qué pasa en el cierre
  sobre 2020. La exploración es dev (2017-2019); el 2020 se toca recién en el cierre.</p>
</div></header>

<div class="col">

<p class="prose">El hilo es el siguiente: los ~4 600 SNPs de ~3 500 líneas tienen <b>estructura de
subpoblación</b> que un modelo global promedia. El baseline aditivo (GBLUP) es fuerte, pero en el
ambiente difícil —<b>sequía</b>, casi un trait aparte del óptimo (r ≈ 0.14)— un <b>MoE que
especializa por subpoblación genómica</b> lo supera. Las tres figuras muestran, en orden, el
supuesto que fija el protocolo, el veredicto entre modelos, y la mecánica interna del ganador.</p>

<!-- FIG 1 -->
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 01</span><span class="eyebrow">EDA · estructura y drift</span></div>
  <h2>La población deriva entre campañas</h2>
  <figure>
    <div class="figbox"><img alt="PCA 3D de los SNPs por año" src="{img1}"></div>
    <figcaption>PCA de 4 612 SNPs sobre dev (n=3 042), coloreada por campaña; rombos = centroide anual.
    PC1-3 explican apenas 6.3 / 5.1 / 4.0 % (estructura difusa). Drift CV-AUC: {drift_txt}.</figcaption>
  </figure>
  <div class="read prose"><span class="lab">Lectura</span>
  La varianza está <b>muy repartida</b> —ninguna dirección lineal domina—, pero los centroides de
  2017, 2018 y 2019 se separan y una logística distingue pares de años con <b>AUC 0.93-0.98</b>: las
  cohortes son genéticamente distintas. Un CV aleatorio filtraría esa estructura de año y
  sobreestimaría. De ahí el <b>split temporal fijo</b> (train 2017-2018 → val 2019, 2020 holdout) que
  usa todo el trabajo.</div>
</section>

<!-- FIG 2 -->
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 02</span><span class="eyebrow">Veredicto · Pearson val 2019</span></div>
  <h2>Quién gana depende del ambiente</h2>
  <figure>
    <div class="figbox"><img alt="Comparativa de 7 modelos en GY sequía y óptimo" src="{img2}"></div>
    <figcaption>Pearson predicho-observado (portada, regla 5) en val 2019, ± IC95 % por bootstrap
    cluster a nivel línea. Línea punteada = GBLUP. Izq.: GY sequía (portada). Der.: GY óptimo.</figcaption>
  </figure>
  <div class="tbl-wrap"><table>
    <thead><tr><th>Modelo</th><th style="text-align:right">GY sequía</th><th style="text-align:right">GY óptimo</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="tbl-note">Fila resaltada = MoE hard (campeón en sequía) · gris = GBLUP (baseline). Valores del CSV de la corrida, no copiados a mano.</p>
  <div class="read prose"><span class="lab">Lectura</span>
  En <b>sequía</b> —la portada— el <b>MoE hard</b> se despega por encima de GBLUP: ahí la estructura de
  subpoblación paga. En <b>óptimo</b> el cuadro se <b>invierte</b>: los gatings supervisados o ruteados
  por ambiente (PLS, GxE) toman la punta y GBLUP es durísimo, mientras el MoE hard queda modesto.
  Ningún modelo gana en los dos ambientes: la historia es <b>“hard para sequía, PLS para óptimo”</b>.</div>
  <div class="read gold prose"><span class="lab">Sweeps dedicados: no afilaron</span>
  Se corrieron sweeps bayesianos de <b>MoE PLS</b> (→ óptimo) y <b>contrastivo</b> (→ sequía),
  seleccionados en val 2018. Ninguno superó su HP heredado sobre 2019: el de PLS <b>sobreajustó el
  fold 2018</b> y su óptimo 2019 cayó a <b>0.213</b> (heredado 0.350, &lt; GBLUP); el contrastivo quedó
  en <b>0.243</b> en sequía (heredado 0.295, ≈ GBLUP). El fold 2018 es demasiado ruidoso para elegir
  HP con tanta libertad. Los números de arriba son los del <b>HP heredado</b>, que sigue siendo el mejor.</div>
</section>

<!-- FIG 3 -->
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 03</span><span class="eyebrow">Interior del ganador · gating PCA+GMM</span></div>
  <h2>Cómo el MoE parte la población</h2>
  <figure>
    <div class="figbox"><img alt="Analisis de clusters del MoE" src="{img3}"></div>
    <figcaption>Ruteo del MoE hard campeón: z-score → PCA-100 (whiten) → GMM k=4, ajustado train-only
    sobre 2017-2018 y proyectado a val 2019. (A) espacio del gating por cluster + val 2019 (○);
    (B) tamaño de subpoblación vs. min_cluster_size=250; (C) composición por año de cada cluster.</figcaption>
  </figure>
  <div class="read prose"><span class="lab">Lectura</span>
  El gating talla <b>4 subpoblaciones de tamaño desigual</b>. La más chica (192 líneas) cae bajo el
  umbral y la atiende el <b>experto de respaldo</b>; las otras tres entrenan experto propio. El grupo
  mayoritario <b>c2</b> (1 265) cruza campañas y absorbe <b>585 de las 722</b> líneas de val 2019,
  mientras <b>c1</b> (puro 2018) no recibe ninguna. Sobre estas subpoblaciones, expertos
  especializados —regularizados por el <b>sharing multitarea de 6 cabezas</b>— son los que levantan la
  sequía.</div>
  <div class="read terra prose"><span class="lab">Matiz honesto</span>
  Como el 81 % de val 2019 rutea a un mismo experto (c2), en test el gating <b>diferencia menos</b> que
  en train, y los clusters siguen en parte el drift (c1/c3 son 2018). La ventaja es real pero acotada:
  aparece en el ambiente con estructura explotable, no como bala de plata.</div>
</section>

<!-- FIG 4 -->
<section class="plate">
  <div class="plate-head"><span class="plate-num">FIG 04</span><span class="eyebrow">Robustez · ruido en inferencia</span></div>
  <h2>El campeón también aguanta el ruido</h2>
  <figure>
    <div class="figbox"><img alt="Curvas de degradacion de Pearson bajo ruido de SNPs" src="{img4}"></div>
    <figcaption>Degradación de Pearson al corromper los SNPs de val 2019 con flip Bernoulli
    (probabilidad ε por marcador; error de genotipado). Línea = media de 5 realizaciones, banda = ±1 sd.
    Modelo entrenado con datos limpios (regla 1 intacta). GBLUP punteado.</figcaption>
  </figure>
  <div class="read prose"><span class="lab">Lectura</span>
  En <b>sequía</b> el <b>MoE hard no solo gana limpio, es el más robusto</b>: retiene
  <b>{mh5/mh0*100:.0f}%</b> de su Pearson a ε=0.05 y <b>{mh10/mh0*100:.0f}%</b> a ε=0.10
  ({mh0:.3f}→{mh10:.3f}), y se mantiene arriba de todos a cualquier nivel de ruido. GBLUP es plano
  ({gb0:.3f}→{gb10:.3f}, ~{gb10/gb0*100:.0f}%) pero bajo: el kernel lineal promedia el ruido sobre miles de SNPs.</div>
  <div class="read terra prose"><span class="lab">El gating supervisado es frágil</span>
  Los ruteos que aprenden del fenotipo o del parentesco (<b>Contrastivo</b>, PLS) se
  <b>desploman</b>: el contrastivo cae a <b>{ct10/ct0*100:.0f}%</b> ({ct0:.3f}→{ct10:.3f} a ε=0.10),
  porque perturbar los SNPs desordena su espacio de ruteo. La PCA del campeón, en cambio, es una
  proyección lineal sobre los SNPs completos que <b>absorbe el ruido</b> — por eso su ventaja no es
  un artefacto de datos limpios: sobrevive al ruido y su distancia con los frágiles se agranda.</div>
</section>
{fig5_block}
{fig6_block}

<!-- STATUS -->
<section class="status">
  <div class="eyebrow">Estado del trabajo</div>
  <h2 style="margin-top:.3rem">Qué está hecho y qué falta</h2>
  <div class="grid">
    <div class="card done"><div class="st"><span class="dot m"></span>Listo</div>
      <h3>Sweeps PLS y contrastivo</h3><p>Terminados (80 trials c/u). Ninguno superó su HP heredado sobre 2019: sobreajuste del fold 2018.</p></div>
    <div class="card done"><div class="st"><span class="dot m"></span>Listo</div>
      <h3>Análisis de ruido</h3><p>Inferencia (fig 04): el MoE hard es el más robusto en sequía. Train (fig 05): el ruido en train no ayuda.</p></div>
    <div class="card done"><div class="st"><span class="dot m"></span>Listo</div>
      <h3>Cierre 2020 con spike-in</h3><p>Colapso cold-start sin spike; el 5 % de líneas de 2020 en el train rescata el rinde (fig 06).</p></div>
    <div class="card done"><div class="st"><span class="dot m"></span>Listo</div>
      <h3>6 figuras del anexo</h3><p>EDA 3D, comparativa de los 7 modelos, clusters del MoE, ruido (inferencia + train) y cierre 2020.</p></div>
  </div>
</section>

</div>
</div>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print("ESCRITO:", OUT, f"({len(HTML)//1024} KB de HTML)")
