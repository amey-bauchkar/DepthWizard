/**
 * DepthWizard offline 3D scene — the entry point of the standalone export (one HTML file, no server, no internet).
 *
 * The backend embeds the job's heightfields (16-bit quantized), texture, LoD-1 buildings and provenance as JSON in
 * <script id="dw-scene">. This page reuses the app's HeightfieldViewer, so the 3D experience is identical; every
 * number shown here is sampled from the embedded heightfields (a downsampled copy of the job rasters) and says so.
 */
import { HeightfieldViewer } from "./viewer";
import type { HeightfieldMeta } from "./api";
import type { LoD1Data } from "./lod1";
import { resolveState, STATE_DISPLAY } from "./resultState";
import "./standalone.css";

interface Quantized { b64: string; offset: number; scale: number; nodata: number; max_rounding_error: number }
/** q = display heightfield (edge-sharpened mesh); m = plain block mean of the full-resolution raster (readings) */
interface SceneLayer { label: string; metric: boolean; units: string; vertical_reference: string | null; meta: HeightfieldMeta; q: Quantized; m?: Quantized }
interface Building {
  id: number; height_m: number; height_median_m?: number; height_p10_m?: number; height_p90_m?: number; base_elev_m: number;
  ground_min_m?: number; roof_elev_m?: number; area_m2: number; volume_m3?: number; floors_range?: [number, number];
  quality_flags?: string[]; coords: [number, number][];
}
interface Scene {
  format: string; version: number; title: string; exported_at: string; app_version: string;
  job: { id?: string; input_filename?: string; input_sha256?: string; processed_at?: string; model?: Record<string, any>; config_hash?: string; method_version?: string };
  source: { label?: string; attribution?: string } | null;
  mode: string; tier: string; quality: string; quality_triggers: string[]; flags: string[]; notes: string[];
  vertical_reference: string | null; units: string; gsd_m: number | null; crs: string | null; transform: number[] | null; size: [number, number];
  lonlat_grid: { n: number; width: number; height: number; lon: number[]; lat: number[] } | null;
  dem: { product?: string; posting_m?: number; vertical_crs?: string } | null;
  uncertainty: Record<string, any>; accuracy: string[]; validation: Record<string, any> | null;
  layers: Record<string, SceneLayer>; default_layer: string;
  texture: { mime: string; b64: string } | null;
  buildings: (LoD1Data & { buildings: Building[]; height_error?: { typical_m?: number | null; source?: string } }) | null;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));

function decode(q: Quantized): Float32Array {
  const bin = atob(q.b64);
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); // little-endian uint16
    out[i] = v === q.nodata ? NaN : q.offset + v * q.scale;
  }
  return out;
}

const LAYER_NAMES: Record<string, string> = { city: "3D city (terrain + LoD-1 buildings)", dsm: "DSM surface (ground + objects)", terrain: "Terrain (bare ground)", ndsm: "nDSM (height above ground)", relative: "Relative structure (no units)" };

function shell(sc: Scene, stateLabel: string, stateFull: string): string {
  const layerOpts = [...(sc.buildings && sc.layers.terrain ? ["city"] : []), ...["dsm", "terrain", "ndsm", "relative"].filter((k) => sc.layers[k])]
    .map((k) => `<option value="${k}"${k === sc.default_layer ? " selected" : ""}>${esc(LAYER_NAMES[k])}</option>`).join("");
  const metric = sc.mode === "B" && sc.tier !== "R";
  return `
  <header class="sa-top">
    <div class="sa-brand"><span class="sa-logo">DW</span><div><div class="sa-name">DepthWizard</div><div class="sa-sub">Offline 3D scene</div></div></div>
    <div class="sa-title" title="${esc(sc.title)}">${esc(sc.title)}</div>
    <div class="sa-badges">
      <span class="sa-badge sa-state" title="${esc(stateFull)}">${esc(stateLabel)}</span>
      <span class="sa-badge">Tier ${esc(sc.tier)}${sc.vertical_reference ? ` · ${esc(sc.vertical_reference)}` : ""}</span>
      <span class="sa-badge sa-q-${esc(sc.quality)}">Quality ${esc(sc.quality)}</span>
      <button id="sa-about-btn" class="sa-btn sa-btn-ghost" type="button">About this scene</button>
    </div>
  </header>
  <main class="sa-main">
    <section class="sa-stage">
      <div id="sa-viewer" class="sa-viewer"><div id="sa-msg" class="sa-msg">Loading 3D scene…</div></div>
      <div class="sa-help">Drag: rotate · Right-drag: pan · Wheel: zoom · Double-click: focus · Click: read heights${metric ? " · Walk: W A S D, Shift = run, Esc = exit" : ""}</div>
    </section>
    <aside class="sa-panel">
      <div class="sa-card">
        <div class="sa-h">View</div>
        <label class="sa-field">Layer <select id="sa-layer">${layerOpts}</select></label>
        ${sc.buildings ? `<label class="sa-check"><input type="checkbox" id="sa-bld" checked> LoD-1 buildings</label>` : ""}
        <div class="sa-row" role="group" aria-label="Camera">
          <button class="sa-btn active" id="sa-orbit" type="button">Orbit</button>
          <button class="sa-btn" id="sa-walk" type="button"${metric ? "" : " disabled"}>Walk</button>
          <button class="sa-btn" id="sa-fly" type="button">Fly-through</button>
        </div>
        <div class="sa-row" role="group" aria-label="Viewpoint">
          <button class="sa-btn" data-preset="nadir" type="button">Top</button>
          <button class="sa-btn" data-preset="oblique" type="button">Oblique</button>
          <button class="sa-btn" data-preset="horizon" type="button">Horizon</button>
        </div>
        <div class="sa-row" role="group" aria-label="Shading">
          <button class="sa-btn active" data-shade="aerial" type="button">Photo</button>
          <button class="sa-btn" data-shade="heatmap" type="button">Elevation colours</button>
        </div>
        <label class="sa-field" id="sa-exag-wrap">Vertical exaggeration <span id="sa-exag-val" class="sa-mono">×1.0 (true scale)</span>
          <input type="range" id="sa-exag" min="1" max="5" step="0.5" value="1"></label>
      </div>
      <div class="sa-card">
        <div class="sa-h">Point readout <button id="sa-measure" class="sa-btn sa-btn-small" type="button">Measure A→B</button></div>
        <div id="sa-readout" class="sa-readout sa-empty">Click the surface to read its heights.</div>
      </div>
      <div class="sa-card hidden" id="sa-bcard">
        <div class="sa-h">Building <span id="sa-bid" class="sa-mono"></span></div>
        <div id="sa-bbody"></div>
      </div>
      <div class="sa-card sa-note">
        <div class="sa-h">How far to trust it</div>
        <ul>${sc.accuracy.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
      </div>
    </aside>
  </main>
  <dialog id="sa-about" class="sa-dialog"><div id="sa-about-body"></div><form method="dialog"><button class="sa-btn" type="submit">Close</button></form></dialog>`;
}

function aboutHtml(sc: Scene, stateFull: string): string {
  const L = Object.values(sc.layers)[0];
  const q = L?.q;
  const rows: [string, string][] = [
    ["What this is", stateFull],
    ["Input image", `${esc(sc.job.input_filename)}${sc.job.input_sha256 ? ` <span class="sa-mono">sha256 ${esc(sc.job.input_sha256.slice(0, 16))}…</span>` : ""}`],
    ...(sc.source?.attribution ? [["Imagery source", esc(sc.source.attribution)] as [string, string]] : []),
    ["Ground sampling", sc.gsd_m ? `${fmt(sc.gsd_m, 2)} m per pixel, ${sc.size[0]} × ${sc.size[1]} px${sc.crs ? `, ${esc(sc.crs)}` : ""}` : "not georeferenced"],
    ["Vertical datum", esc(sc.vertical_reference ?? "none (relative heights)")],
    ["DEM", esc(sc.dem?.product ?? "none")],
    ["Height model", esc(sc.job.model?.tiles ? `${sc.job.model.tiles.name}@${sc.job.model.tiles.version}` : sc.job.model ? `${sc.job.model.name}@${sc.job.model.version}` : "—")],
    ["Quality", `${esc(sc.quality)}${sc.quality_triggers.length ? ` — ${esc(sc.quality_triggers.join("; "))}` : ""}`],
    ["Processed", `${esc(sc.job.processed_at ?? "—")} · method ${esc(sc.job.method_version ?? "—")}`],
    ["Exported", `${esc(sc.exported_at)} · DepthWizard ${esc(sc.app_version)} · job ${esc(sc.job.id)}`],
    ["Embedded heights", L ? `${L.meta.width} × ${L.meta.height} samples (${fmt((sc.gsd_m ?? 1) * (L.meta.downsample_factor || 1), 2)} m spacing), 16-bit, rounding ≤ ${fmt((q?.max_rounding_error ?? 0) * 100, 2)} cm. The 3D mesh is a display copy (edges sharpened for legibility); readings use the plain cell mean of the full-resolution rasters. Full-resolution GeoTIFFs are in the DepthWizard GIS package.` : "—"],
  ];
  const v = sc.validation;
  const val: string[] = [];
  if (v?.raster) val.push(`Against a reference ${esc(v.raster.layer)} raster (${esc(v.raster.reference ?? "reference")}): RMSE ${fmt(v.raster.rmse_m, 2)} m, bias ${fmt(v.raster.me_m, 2)} m over ${esc(v.raster.n)} pixels${v.raster.baseline_rmse_m != null ? `; input DEM alone ${fmt(v.raster.baseline_rmse_m, 2)} m` : ""}.`);
  if (v?.points) val.push(`Against ${esc(v.points.n_in_grid)} independent checkpoints (${esc(v.points.source)}): ${Object.entries(v.points.metrics as Record<string, any>).map(([k, m]) => `${esc(k.replace(/_/g, " "))} RMSE ${fmt(m.rmse_m, 2)} m (n=${esc(m.n)})`).join("; ")}.`);
  return `<h2>${esc(sc.title)}</h2>
    <table class="sa-table">${rows.map(([k, x]) => `<tr><th>${k}</th><td>${x}</td></tr>`).join("")}</table>
    <h3>Measured accuracy</h3><ul>${sc.accuracy.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
    ${val.length ? `<h3>Validation of this scene</h3><ul>${val.map((x) => `<li>${x}</li>`).join("")}</ul>` : ""}
    ${sc.notes.length ? `<h3>Notes</h3><ul>${sc.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    <p class="sa-fine">Heights from one image are a fast first look for planning and response, not a replacement for stereo or LiDAR survey. Floors are a range (3.0–3.5 m per storey), never a count.</p>`;
}

function main() {
  const el = document.getElementById("dw-scene");
  const root = $("dw-app");
  if (!el || !root) return;
  const sc = JSON.parse(el.textContent || "{}") as Scene;
  const rs = resolveState({ mode: sc.mode, calibration_tier: sc.tier, flags: sc.flags, object_scale_source: null });
  const sd = STATE_DISPLAY[rs];
  document.title = `${sc.title} · DepthWizard offline 3D scene`;
  root.innerHTML = shell(sc, sd.label, sd.full);
  $("sa-about-body").innerHTML = aboutHtml(sc, sd.full);
  $("sa-about-btn").addEventListener("click", () => ($("sa-about") as HTMLDialogElement).showModal());

  if (!HeightfieldViewer.webglAvailable()) { $("sa-msg").textContent = "WebGL is not available in this browser: open the file in a current Chrome, Edge, Firefox or Safari."; return; }
  const viewer = new HeightfieldViewer($("sa-viewer"));
  const textureUrl = sc.texture ? `data:${sc.texture.mime};base64,${sc.texture.b64}` : "";
  const decoded: Record<string, Float32Array> = {};
  const hf = (k: string) => (decoded[k] ??= decode(sc.layers[k].q));
  const readings: Record<string, Float32Array> = {};
  const rd = (k: string) => (readings[k] ??= sc.layers[k].m ? decode(sc.layers[k].m!) : hf(k));
  let current = sc.default_layer;
  let measuring = false; let ptA: { col: number; row: number } | null = null;

  const T = sc.transform;
  const toCrs = (col: number, row: number) => (T ? { x: T[0] + col * T[1] + row * T[2], y: T[3] + col * T[4] + row * T[5] } : null);
  const toLonLat = (col: number, row: number): { lon: number; lat: number } | null => {
    const g = sc.lonlat_grid;
    if (!g) return null;
    const n = g.n, fc = Math.min(n - 1 - 1e-9, Math.max(0, (col / g.width) * (n - 1))), fr = Math.min(n - 1 - 1e-9, Math.max(0, (row / g.height) * (n - 1)));
    const c0 = Math.floor(fc), r0 = Math.floor(fr), tc = fc - c0, tr = fr - r0;
    const at = (a: number[], r: number, c: number) => a[r * n + c];
    const bil = (a: number[]) => (1 - tr) * ((1 - tc) * at(a, r0, c0) + tc * at(a, r0, c0 + 1)) + tr * ((1 - tc) * at(a, r0 + 1, c0) + tc * at(a, r0 + 1, c0 + 1));
    return { lon: bil(g.lon), lat: bil(g.lat) };
  };
  const sampleLayer = (k: string, col: number, row: number): number | null => {
    const L = sc.layers[k];
    if (!L) return null;
    const f = L.meta.downsample_factor || 1;
    const c = Math.min(L.meta.width - 1, Math.max(0, Math.floor(col / f))), r = Math.min(L.meta.height - 1, Math.max(0, Math.floor(row / f)));
    const v = rd(k)[r * L.meta.width + c];
    return Number.isFinite(v) ? v : null;
  };
  const unc = (k: string, v: number): number | null => {
    const u = sc.uncertainty?.[k];
    if (!u) return null;
    if (k === "ndsm") return v >= (u.object_threshold_m ?? 2.5) ? u.object_m : (u.ground_m ?? null);
    return u.value_m ?? null;
  };

  function readoutHtml(col: number, row: number): string {
    const p = toCrs(col, row), ll = toLonLat(col, row);
    const pos = [p && sc.crs ? `${esc(sc.crs)} E ${fmt(p.x, 1)} N ${fmt(p.y, 1)}` : `pixel ${Math.floor(col)}, ${Math.floor(row)}`, ll ? `${fmt(ll.lat, 6)}° N, ${fmt(ll.lon, 6)}° E` : ""].filter(Boolean);
    const rows = (["dsm", "terrain", "ndsm", "relative"] as const).filter((k) => sc.layers[k]).map((k) => {
      const v = sampleLayer(k, col, row);
      const L = sc.layers[k];
      const e = v !== null && L.metric ? unc(k, v) : null;
      const val = v === null ? "nodata" : L.metric ? `${fmt(v, 2)} m` : fmt(v, 3);
      return `<tr><td><div class="sa-k">${esc(LAYER_NAMES[k].split(" (")[0])}</div><div class="sa-q">${esc(L.vertical_reference ?? (L.metric ? "" : "no units"))}</div></td>
        <td class="sa-v">${esc(val)}${e !== null ? `<div class="sa-u">± ${fmt(e, 1)} m typical</div>` : L.metric && v !== null ? `<div class="sa-u sa-none">± not measured</div>` : ""}</td></tr>`;
    }).join("");
    const ref = sc.layers.dsm ?? sc.layers[current === "city" ? "terrain" : current];
    const cell = fmt((sc.gsd_m ?? 1) * (ref?.meta.downsample_factor || 1), 1);
    const edge = ref?.meta.downsample_factor > 1 && ref?.meta.residual_vs_source?.max_abs ? ` Single ${fmt(sc.gsd_m, 1)} m pixels at building edges can differ by up to ${fmt(ref.meta.residual_vs_source.max_abs, 1)} m; exact values are in the full-resolution GeoTIFFs.` : "";
    return `<div class="sa-pos">${pos.map(esc).join("<br>")}</div><table class="sa-rt">${rows}</table>
      <div class="sa-fine">Mean of the ${cell} m × ${cell} m cell of the full-resolution rasters, embedded in this file.${esc(edge)}</div>`;
  }

  function measureHtml(a: { col: number; row: number }, b: { col: number; row: number }): string {
    const la = toLonLat(a.col, a.row), lb = toLonLat(b.col, b.row);
    let dist: number; let unit = "m";
    if (la && lb) {
      const R = 6371008.8, rad = Math.PI / 180;
      const dphi = (lb.lat - la.lat) * rad, dl = (lb.lon - la.lon) * rad;
      const h = Math.sin(dphi / 2) ** 2 + Math.cos(la.lat * rad) * Math.cos(lb.lat * rad) * Math.sin(dl / 2) ** 2;
      dist = 2 * R * Math.asin(Math.sqrt(h));
    } else { dist = Math.hypot(b.col - a.col, b.row - a.row); unit = "px"; }
    const parts = [`Horizontal distance <b>${fmt(dist, 1)} ${unit}</b>`];
    for (const k of ["dsm", "terrain", "ndsm"]) {
      const va = sampleLayer(k, a.col, a.row), vb = sampleLayer(k, b.col, b.row);
      if (va === null || vb === null || !sc.layers[k]?.metric) continue;
      const dz = vb - va;
      const grade = k !== "ndsm" && unit === "m" && dist > 0 ? ` (${fmt((Math.atan2(dz, dist) * 180) / Math.PI, 1)}° average slope)` : "";
      parts.push(`Δ ${esc(LAYER_NAMES[k].split(" (")[0])} <b>${dz >= 0 ? "+" : ""}${fmt(dz, 2)} m</b>${grade}`);
    }
    return `<div class="sa-measure">${parts.join("<br>")}</div>`;
  }

  viewer.onPick((col, row) => {
    const ro = $("sa-readout");
    ro.classList.remove("sa-empty");
    if (measuring) {
      if (!ptA) { ptA = { col, row }; ro.innerHTML = `<div class="sa-fine">Point A set. Click point B.</div>${readoutHtml(col, row)}`; return; }
      ro.innerHTML = measureHtml(ptA, { col, row }) + readoutHtml(col, row);
      measuring = false; ptA = null; $("sa-measure").classList.remove("active");
      return;
    }
    ro.innerHTML = readoutHtml(col, row);
  });

  const bById = new Map<number, Building>((sc.buildings?.buildings ?? []).map((b) => [b.id, b]));
  viewer.onBuildingPick((id) => {
    const b = bById.get(id);
    if (!b) return;
    const typ = sc.buildings?.height_error?.typical_m;
    const h = b.height_median_m ?? b.height_m;
    $("sa-bcard").classList.remove("hidden");
    $("sa-bid").textContent = `#${id}`;
    const rows: [string, string][] = [
      ["Height (typical roof)", `${fmt(h, 1)} m${typ ? ` ± ${fmt(typ, 1)} m` : ""}`],
      ...(b.height_p10_m != null && b.height_p90_m != null ? [["Roof spread p10–p90", `${fmt(b.height_p10_m, 1)} – ${fmt(b.height_p90_m, 1)} m`] as [string, string]] : []),
      ["Ground elevation", `${fmt(b.base_elev_m, 1)} m`],
      ...(b.roof_elev_m != null ? [["Roof elevation", `${fmt(b.roof_elev_m, 1)} m`] as [string, string]] : []),
      ["Footprint", `${fmt(b.area_m2, 0)} m²`],
      ...(b.volume_m3 != null ? [["Volume", `${fmt(b.volume_m3, 0)} m³`] as [string, string]] : []),
      ...(b.floors_range ? [["Floors (range)", `${b.floors_range[0]}–${b.floors_range[1]}`] as [string, string]] : []),
      ...(b.quality_flags?.length ? [["Flags", b.quality_flags.join(", ")] as [string, string]] : []),
    ];
    $("sa-bbody").innerHTML = `<table class="sa-rt">${rows.map(([k, v]) => `<tr><td class="sa-k">${esc(k)}</td><td class="sa-v">${esc(v)}</td></tr>`).join("")}</table>
      <div class="sa-fine">${typ ? esc(sc.buildings?.height_error?.source ?? "") : "Building height error not calibrated for this scene."}</div>`;
  });

  async function show(k: string) {
    current = k;
    const isCity = k === "city";
    const key = isCity ? "terrain" : k;
    const L = sc.layers[key];
    const metric = L.metric;
    const exagEl = $<HTMLInputElement>("sa-exag");
    const exag = isCity ? 1 : Number(exagEl.value) || 1;
    if (isCity) { exagEl.value = "1"; $("sa-exag-val").textContent = "×1.0 (true scale)"; }
    $("sa-exag-wrap").classList.toggle("hidden", !metric);
    $("sa-msg").textContent = "Building mesh…";
    $("sa-msg").classList.remove("hidden");
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    const spacing = (sc.gsd_m ?? 1) * (L.meta.downsample_factor || 1);
    await viewer.load(hf(key), L.meta, textureUrl, {
      metric, spacing: metric ? spacing : 1, exaggeration: exag,
      hud: { state: metric ? sd.hudState : "RELATIVE SURFACE STRUCTURE", tier: metric ? `Tier ${sc.tier} · ${(L.meta.vertical_reference ?? "—").replace(/^height_above_(terrain|ground)$/, "height above ground")}` : "Tier R · non-metric", quality: sc.quality ?? "UNVALIDATED", layer: isCity ? "3D City (LoD-1)" : LAYER_NAMES[k], showNorth: metric },
    });
    const bChk = document.getElementById("sa-bld") as HTMLInputElement | null;
    if (sc.buildings && (isCity || key === "dsm" || key === "terrain")) {
      viewer.loadBuildings(sc.buildings as LoD1Data, isCity);
      if (bChk) { bChk.disabled = false; if (isCity) bChk.checked = true; viewer.setBuildingsVisible(bChk.checked); }
    } else {
      viewer.loadBuildings(null);
      if (bChk) bChk.disabled = true;
    }
    $("sa-msg").classList.add("hidden");
    setCam("orbit");
  }

  const setCam = (m: "orbit" | "walk" | "fly") => {
    for (const id of ["sa-orbit", "sa-walk", "sa-fly"]) $(id).classList.toggle("active", id === `sa-${m}`);
  };
  viewer.onCameraModeChange((m) => setCam(m));
  viewer.onFlythroughToggle((on) => setCam(on ? "fly" : "orbit"));
  $("sa-orbit").addEventListener("click", () => { viewer.setFlythrough(false); viewer.setCameraMode("orbit"); });
  $("sa-walk").addEventListener("click", () => viewer.setCameraMode("walk"));
  $("sa-fly").addEventListener("click", () => viewer.setFlythrough(!viewer.isFlythrough));
  document.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((b) => b.addEventListener("click", () => viewer.setPresetView(b.dataset.preset as "nadir" | "oblique" | "horizon")));
  document.querySelectorAll<HTMLButtonElement>("[data-shade]").forEach((b) => b.addEventListener("click", () => {
    viewer.setShaderMode(b.dataset.shade as "aerial" | "heatmap");
    document.querySelectorAll("[data-shade]").forEach((x) => x.classList.toggle("active", x === b));
  }));
  $("sa-exag").addEventListener("input", (e) => {
    const v = Number((e.target as HTMLInputElement).value);
    $("sa-exag-val").textContent = `×${v.toFixed(1)}${v === 1 ? " (true scale)" : " (visual only)"}`;
    viewer.setExaggeration(v);
  });
  $("sa-layer").addEventListener("change", (e) => void show((e.target as HTMLSelectElement).value));
  document.getElementById("sa-bld")?.addEventListener("change", (e) => viewer.setBuildingsVisible((e.target as HTMLInputElement).checked));
  $("sa-measure").addEventListener("click", () => {
    measuring = !measuring; ptA = null;
    $("sa-measure").classList.toggle("active", measuring);
    const ro = $("sa-readout");
    ro.classList.remove("sa-empty");
    ro.innerHTML = measuring ? `<div class="sa-fine">Measure: click point A on the surface.</div>` : "Click the surface to read its heights.";
  });

  void show(current).catch((e) => { $("sa-msg").textContent = `Could not build the 3D scene: ${e instanceof Error ? e.message : String(e)}`; });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
else main();
