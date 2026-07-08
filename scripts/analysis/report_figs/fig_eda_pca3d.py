"""FIGURA EDA: PCA 3D de los SNPs coloreado por año.

Muestra la estructura poblacional difusa + el DRIFT entre campañas (los centroides anuales se
separan), que es el fundamento del split temporal fijo. Reusa src.eda.eda (pca_embed, year_centroids,
drift_auc). PCA descriptiva ajustada sobre dev 2017-2019 y proyeccion de 2020, sin ajustar con el
holdout.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.getcwd())
from src.eda.eda import pca_embed, year_centroids, drift_auc

OUTDIR = "experiments/figs/report"
PUBLISH_DIR = "figuras_informe"
EXPORT_DIRS = [OUTDIR, PUBLISH_DIR]
for _dir in EXPORT_DIRS:
    os.makedirs(_dir, exist_ok=True)
DEV_YEARS = [2017, 2018, 2019]
PLOT_YEARS = [2017, 2018, 2019, 2020]


def out_paths(filename: str) -> list[str]:
    return [os.path.join(d, filename) for d in EXPORT_DIRS]


def write_interactive_html(points: pd.DataFrame, centroids: pd.DataFrame, drift: pd.DataFrame, evr: np.ndarray, path: str) -> None:
    """HTML 3D rotatable sin dependencias externas; abre directo con el navegador."""
    years_visible = [int(y) for y in sorted(points["Year"].unique())]
    year_buttons = "\n      ".join(
        f'<button class="year active" type="button" data-year="{y}">{y}</button>'
        for y in years_visible
    )
    payload_points = [
        {
            "gid": str(r.GID),
            "year": int(r.Year),
            "pc1": round(float(r.PC1), 6),
            "pc2": round(float(r.PC2), 6),
            "pc3": round(float(r.PC3), 6),
        }
        for r in points.itertuples(index=False)
    ]
    payload_centroids = [
        {
            "year": int(r.Year),
            "n": int(r.n),
            "pc1": round(float(r.PC1), 6),
            "pc2": round(float(r.PC2), 6),
            "pc3": round(float(r.PC3), 6),
        }
        for r in centroids.itertuples(index=False)
    ]
    payload_drift = [
        {"pair": str(r.par_de_años), "n": int(r.n), "auc": round(float(r.cv_auc), 3)}
        for r in drift.itertuples(index=False)
    ]
    payload = {
        "points": payload_points,
        "centroids": payload_centroids,
        "drift": payload_drift,
        "evr": [round(float(x), 6) for x in evr[:3]],
        "colors": {str(k): v for k, v in COL.items()},
        "years": years_visible,
        "fitYears": DEV_YEARS,
    }
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCA 3D interactivo - 2017-2020</title>
<style>
:root {{
  --bg:#f7f6f0;
  --panel:#fffdf8;
  --ink:#1a211b;
  --muted:#5e695f;
  --line:#d9dccf;
  --accent:#1f8a7c;
  --shadow:0 10px 28px rgba(25,34,28,.09);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0;
  background:var(--bg);
  color:var(--ink);
}}
.wrap {{
  min-height:100vh;
  display:grid;
  grid-template-rows:auto 1fr;
}}
header {{
  display:flex;
  gap:1rem;
  justify-content:space-between;
  align-items:flex-end;
  padding:1rem clamp(1rem, 3vw, 2rem);
  border-bottom:1px solid var(--line);
  background:rgba(255,253,248,.86);
}}
h1 {{
  margin:.15rem 0 0;
  font-size:clamp(1.15rem, 2vw, 1.65rem);
  line-height:1.15;
  font-weight:750;
  letter-spacing:0;
}}
.meta {{
  margin:0;
  color:var(--muted);
  font-size:.86rem;
}}
.tools {{
  display:flex;
  gap:.5rem;
  align-items:center;
  flex-wrap:wrap;
  justify-content:flex-end;
}}
button {{
  appearance:none;
  border:1px solid var(--line);
  background:var(--panel);
  color:var(--ink);
  min-height:2rem;
  padding:.35rem .7rem;
  border-radius:6px;
  font:inherit;
  font-size:.86rem;
  cursor:pointer;
}}
button.active {{
  border-color:var(--accent);
  color:var(--accent);
  background:#e9f4f1;
}}
main {{
  position:relative;
  min-height:0;
  overflow:hidden;
}}
canvas {{
  display:block;
  width:100%;
  height:100%;
  cursor:grab;
  background:linear-gradient(180deg,#fffdf8 0%,#f2f0e7 100%);
}}
canvas:active {{ cursor:grabbing; }}
.hud {{
  position:absolute;
  left:1rem;
  bottom:1rem;
  max-width:min(34rem, calc(100% - 2rem));
  background:rgba(255,253,248,.92);
  border:1px solid var(--line);
  border-radius:8px;
  box-shadow:var(--shadow);
  padding:.75rem .85rem;
  color:var(--muted);
  font-size:.82rem;
  line-height:1.45;
}}
.hud b {{ color:var(--ink); }}
.tip {{
  position:absolute;
  z-index:5;
  pointer-events:none;
  display:none;
  background:#172019;
  color:#fff;
  border-radius:6px;
  padding:.45rem .55rem;
  font-size:.78rem;
  box-shadow:var(--shadow);
  white-space:nowrap;
}}
.legend {{
  display:flex;
  gap:.8rem;
  flex-wrap:wrap;
  margin-top:.35rem;
}}
.swatch {{
  display:inline-block;
  width:.75rem;
  height:.75rem;
  border-radius:50%;
  margin-right:.28rem;
  vertical-align:-.08rem;
}}
@media (max-width:720px) {{
  header {{ align-items:flex-start; flex-direction:column; }}
  .tools {{ justify-content:flex-start; }}
  .hud {{ font-size:.76rem; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <p class="meta">PCA ajustada con 2017-2019; 2020 proyectado sobre ese espacio</p>
      <h1>Estructura poblacional por campaña</h1>
    </div>
    <div class="tools" aria-label="Controles">
      <button id="reset" type="button" title="Restablecer vista">Reset</button>
      <button id="centroids" class="active" type="button" title="Mostrar u ocultar centroides">Rombos</button>
      {year_buttons}
    </div>
  </header>
  <main>
    <canvas id="plot"></canvas>
    <div id="tip" class="tip"></div>
    <div class="hud" id="hud"></div>
  </main>
</div>
<script id="pca-data" type="application/json">{data_json}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("pca-data").textContent);
  const canvas = document.getElementById("plot");
  const ctx = canvas.getContext("2d");
  const tip = document.getElementById("tip");
  const hud = document.getElementById("hud");
  const state = {{
    rotX: -0.32,
    rotY: -0.9,
    zoom: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    lastX: 0,
    lastY: 0,
    showCentroids: true,
    visible: new Set(data.years.map(String)),
    projected: []
  }};
  const cubeLabels = [
    {{ name: `PC1 (${{(data.evr[0] * 100).toFixed(1)}}%)`, at: [0, -1, -1], color: "#4d6675" }},
    {{ name: `PC2 (${{(data.evr[1] * 100).toFixed(1)}}%)`, at: [-1, 0, -1], color: "#6b7046" }},
    {{ name: `PC3 (${{(data.evr[2] * 100).toFixed(1)}}%)`, at: [-1, -1, 0], color: "#70505c" }}
  ];
  const maxAbs = Math.max(
    ...data.points.flatMap(p => [Math.abs(p.pc1), Math.abs(p.pc2), Math.abs(p.pc3)]),
    ...data.centroids.flatMap(p => [Math.abs(p.pc1), Math.abs(p.pc2), Math.abs(p.pc3)])
  ) || 1;
  const counts = data.points.reduce((acc, p) => {{
    acc[p.year] = (acc[p.year] || 0) + 1;
    return acc;
  }}, {{}});

  function resize() {{
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }}

  function rotatePoint(p) {{
    const cy = Math.cos(state.rotY), sy = Math.sin(state.rotY);
    const cx = Math.cos(state.rotX), sx = Math.sin(state.rotX);
    const x1 = cy * p[0] + sy * p[2];
    const z1 = -sy * p[0] + cy * p[2];
    const y1 = cx * p[1] - sx * z1;
    const z2 = sx * p[1] + cx * z1;
    return [x1, y1, z2];
  }}

  function project(x, y, z) {{
    const rect = canvas.getBoundingClientRect();
    const r = rotatePoint([x, y, z]);
    const perspective = 1 / (1 + (r[2] / (maxAbs * 7)));
    const scale = Math.min(rect.width, rect.height) * 0.36 * state.zoom / maxAbs;
    return {{
      x: rect.width / 2 + state.panX + r[0] * scale * perspective,
      y: rect.height / 2 + state.panY - r[1] * scale * perspective,
      z: r[2],
      perspective
    }};
  }}

  function drawCube() {{
    const L = maxAbs * 1.08;
    const corners = [
      [-L, -L, -L], [ L, -L, -L], [ L,  L, -L], [-L,  L, -L],
      [-L, -L,  L], [ L, -L,  L], [ L,  L,  L], [-L,  L,  L]
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7]
    ].map(edge => {{
      const a = project(...corners[edge[0]]);
      const b = project(...corners[edge[1]]);
      return {{ a, b, z: (a.z + b.z) / 2 }};
    }}).sort((a, b) => a.z - b.z);
    ctx.save();
    ctx.lineWidth = 1.15;
    ctx.strokeStyle = "#899287";
    for (const edge of edges) {{
      ctx.globalAlpha = 0.4 + 0.22 * ((edge.z + L) / (2 * L));
      ctx.beginPath();
      ctx.moveTo(edge.a.x, edge.a.y);
      ctx.lineTo(edge.b.x, edge.b.y);
      ctx.stroke();
    }}
    ctx.globalAlpha = 1;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    cubeLabels.forEach(label => {{
      const p = project(label.at[0] * L, label.at[1] * L, label.at[2] * L);
      ctx.fillStyle = label.color;
      ctx.fillText(label.name, p.x + 7, p.y);
    }});
    ctx.restore();
  }}

  function draw() {{
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    drawCube();
    const projected = [];
    for (const p of data.points) {{
      if (!state.visible.has(String(p.year))) continue;
      const q = project(p.pc1, p.pc2, p.pc3);
      projected.push({{ ...q, kind: "point", raw: p }});
    }}
    if (state.showCentroids) {{
      for (const p of data.centroids) {{
        if (!state.visible.has(String(p.year))) continue;
        const q = project(p.pc1, p.pc2, p.pc3);
        projected.push({{ ...q, kind: "centroid", raw: p }});
      }}
    }}
    projected.sort((a, b) => a.z - b.z);
    state.projected = projected;
    for (const p of projected) {{
      const color = data.colors[String(p.raw.year)];
      if (p.kind === "centroid") {{
        const s = 8.5 * p.perspective;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = color;
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.rect(-s, -s, s * 2, s * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }} else {{
        const radius = Math.max(1.5, 2.3 * p.perspective);
        ctx.fillStyle = color;
        ctx.strokeStyle = "#1d261f";
        ctx.lineWidth = Math.max(0.35, 0.45 * p.perspective);
        ctx.globalAlpha = 0.72;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 0.82;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }}
    }}
  }}

  function updateHud() {{
    const drift = data.drift.map(d => `${{d.pair}} <b>${{d.auc.toFixed(2)}}</b>`).join(" · ");
    const legend = Object.entries(data.colors).map(([year, color]) =>
      `<span><i class="swatch" style="background:${{color}}"></i>${{year}} n=${{counts[year] || 0}}</span>`
    ).join("");
    hud.innerHTML = `<b>Arrastrar</b>: rotar · <b>rueda</b>: zoom · <b>Shift + arrastrar</b>: mover<br>` +
      `PCA fit: <b>${{data.fitYears.join("-")}}</b> · ` +
      `PC1-3: <b>${{(data.evr[0] * 100).toFixed(1)}}%</b> / <b>${{(data.evr[1] * 100).toFixed(1)}}%</b> / <b>${{(data.evr[2] * 100).toFixed(1)}}%</b> · ` +
      `Drift CV-AUC: ${{drift}}<div class="legend">${{legend}}</div>`;
  }}

  function showTip(evt) {{
    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    let best = null;
    let bestD = Infinity;
    for (const p of state.projected) {{
      const dx = p.x - x;
      const dy = p.y - y;
      const d = dx * dx + dy * dy;
      if (d < bestD) {{
        bestD = d;
        best = p;
      }}
    }}
    if (!best || bestD > 100) {{
      tip.style.display = "none";
      return;
    }}
    const r = best.raw;
    tip.style.display = "block";
    tip.style.left = `${{x + 12}}px`;
    tip.style.top = `${{y + 12}}px`;
    tip.innerHTML = best.kind === "centroid"
      ? `Centroide ${{r.year}}<br>n=${{r.n}} · PC=(${{r.pc1.toFixed(2)}}, ${{r.pc2.toFixed(2)}}, ${{r.pc3.toFixed(2)}})`
      : `GID ${{r.gid}} · ${{r.year}}<br>PC=(${{r.pc1.toFixed(2)}}, ${{r.pc2.toFixed(2)}}, ${{r.pc3.toFixed(2)}})`;
  }}

  canvas.addEventListener("pointerdown", evt => {{
    state.dragging = true;
    state.lastX = evt.clientX;
    state.lastY = evt.clientY;
    canvas.setPointerCapture(evt.pointerId);
  }});
  canvas.addEventListener("pointerup", evt => {{
    state.dragging = false;
    try {{ canvas.releasePointerCapture(evt.pointerId); }} catch (err) {{}}
  }});
  canvas.addEventListener("pointerleave", () => {{
    tip.style.display = "none";
  }});
  canvas.addEventListener("pointermove", evt => {{
    if (state.dragging) {{
      const dx = evt.clientX - state.lastX;
      const dy = evt.clientY - state.lastY;
      state.lastX = evt.clientX;
      state.lastY = evt.clientY;
      if (evt.shiftKey) {{
        state.panX += dx;
        state.panY += dy;
      }} else {{
        state.rotY += dx * 0.008;
        state.rotX += dy * 0.008;
        state.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.rotX));
      }}
      draw();
    }} else {{
      showTip(evt);
    }}
  }});
  canvas.addEventListener("wheel", evt => {{
    evt.preventDefault();
    const factor = Math.exp(-evt.deltaY * 0.001);
    state.zoom = Math.max(0.35, Math.min(4.5, state.zoom * factor));
    draw();
  }}, {{ passive: false }});
  document.getElementById("reset").addEventListener("click", () => {{
    state.rotX = -0.32;
    state.rotY = -0.9;
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    draw();
  }});
  document.getElementById("centroids").addEventListener("click", evt => {{
    state.showCentroids = !state.showCentroids;
    evt.currentTarget.classList.toggle("active", state.showCentroids);
    tip.style.display = "none";
    draw();
  }});
  document.querySelectorAll("button.year").forEach(btn => {{
    btn.addEventListener("click", () => {{
      const year = btn.dataset.year;
      if (state.visible.has(year)) {{
        if (state.visible.size === 1) return;
        state.visible.delete(year);
        btn.classList.remove("active");
      }} else {{
        state.visible.add(year);
        btn.classList.add("active");
      }}
      draw();
    }});
  }});
  window.addEventListener("resize", resize);
  updateHud();
  resize();
}})();
</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# --- Cargar genotipos + año por linea ----------------------------------------
geno = pd.read_csv("data/Geno_data.csv")
pheno = pd.read_csv("data/Pheno_data.csv")
gid2year = pheno.drop_duplicates("GID").set_index("GID")["Year"]
geno = geno.copy()
geno["Year"] = geno["GID"].map(gid2year)
plot_df = geno[geno["Year"].isin(PLOT_YEARS)].reset_index(drop=True)
snp_cols = [c for c in geno.columns if c not in ("GID", "Year")]
X = plot_df[snp_cols].to_numpy(dtype=np.float64)
years = plot_df["Year"].to_numpy(dtype=int)
fit_mask = np.isin(years, DEV_YEARS)
print(f"plot: {X.shape[0]} lineas x {X.shape[1]} SNPs  |  por año: "
      f"{ {int(y): int((years==y).sum()) for y in PLOT_YEARS} }")
print(f"PCA fit solo con dev: {DEV_YEARS} ({int(fit_mask.sum())} lineas); 2020 solo proyectado")

# --- PCA 3D + centroides + drift AUC ----------------------------------------
Z, evr = pca_embed(X, n_components=3, fit_mask=fit_mask)
# Convencion visual: invertir los tres ejes del PCA. El signo de cada PC es arbitrario.
Z[:, :3] *= -1
cen = year_centroids(Z, years, n_pcs=3)
drift = drift_auc(X[fit_mask], years[fit_mask], pairs=[(2017, 2018), (2018, 2019), (2017, 2019)], n_pcs=20)
print("varianza explicada PC1-3:", np.round(evr[:3], 4))
print(drift.to_string(index=False))

COL = {2017: "#1b6ca8", 2018: "#e07a2f", 2019: "#2a9d5c", 2020: "#7b4ab2"}
fig = plt.figure(figsize=(9.8, 7.6))
ax = fig.add_subplot(111, projection="3d")
ax.set_proj_type("ortho")  # proyeccion paralela como la figura de referencia (Lopez-Cruz)
for y in PLOT_YEARS:
    m = years == y
    alpha = 0.58 if y in DEV_YEARS else 0.72
    marker = "o" if y in DEV_YEARS else "^"
    ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=8, alpha=alpha, marker=marker, color=COL[y],
               edgecolors="#1d261f", linewidths=0.22,
               label=f"{y}  (n={int(m.sum())})", depthshade=True)
# centroides
for _, r in cen.iterrows():
    y = int(r["Year"])
    ax.scatter(r["PC1"], r["PC2"], r["PC3"], s=260, marker="D", color=COL[y],
               edgecolor="black", linewidth=1.6, zorder=10)

# Cubo de referencia, con escala equivalente en los tres ejes.
mins = Z[:, :3].min(axis=0)
maxs = Z[:, :3].max(axis=0)
center = (mins + maxs) / 2
half = float((maxs - mins).max() * 0.56)
lo = center - half
hi = center + half
ax.set_xlim(lo[0], hi[0])
ax.set_ylim(lo[1], hi[1])
ax.set_zlim(lo[2], hi[2])
ax.set_box_aspect((1, 1, 1))
corners = np.array([
    [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
    [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
])
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
for a, b in edges:
    ax.plot(*zip(corners[a], corners[b]), color="#6f7a70", lw=1.0, alpha=0.45, zorder=0)
ax.text(center[0], lo[1], lo[2], f"PC1 ({evr[0]*100:.1f}%)",
        color="#3f5662", fontsize=9, ha="center", va="top")
ax.text(lo[0], center[1], lo[2], f"PC2 ({evr[1]*100:.1f}%)",
        color="#5f653c", fontsize=9, ha="right", va="center")
ax.text(hi[0], lo[1], center[2], f"PC3 ({evr[2]*100:.1f}%)",
        color="#6a4b58", fontsize=9, ha="right", va="center")
ax.set_axis_off()
ax.set_title("Estructura poblacional por campaña — PCA fit 2017-2019, 2020 proyectado\n"
             "rombos = centroides anuales; triangulos = 2020",
             fontsize=11, pad=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9, bbox_to_anchor=(0.02, 0.98))
ax.view_init(elev=22, azim=60)
# caja de texto con el drift AUC (separabilidad genética entre años)
txt = "Drift (CV-AUC logística sobre 20 PCs):\n" + "\n".join(
    f"  {r['par_de_años']}:  {r['cv_auc']:.2f}" for _, r in drift.iterrows())
txt += "\n(≈0.5 = años indistinguibles · ≈1 = cohortes distintas)"
ax.text2D(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5,
          bbox=dict(boxstyle="round", fc="#fffbe6", ec="#caa", alpha=0.95), va="bottom")
fig.tight_layout()
for out in out_paths("fig1_eda_pca3d.png"):
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"GUARDADO: {out}")
for out in out_paths("fig1_eda_pca3d_matplotlib.png"):
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"GUARDADO: {out}")

points = pd.DataFrame({
    "GID": plot_df["GID"].astype(str),
    "Year": years,
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
    "PC3": Z[:, 2],
})
for out in out_paths("fig1_centroids.csv"):
    cen.to_csv(out, index=False)
    print(f"GUARDADO: {out}")
for out in out_paths("fig1_drift_auc.csv"):
    drift.to_csv(out, index=False)
    print(f"GUARDADO: {out}")
for out in out_paths("fig1_eda_pca3d_points.csv"):
    points.to_csv(out, index=False)
    print(f"GUARDADO: {out}")
for out in out_paths("fig1_eda_pca3d_interactive.html"):
    write_interactive_html(points, cen, drift, evr, out)
    print(f"GUARDADO: {out}")
