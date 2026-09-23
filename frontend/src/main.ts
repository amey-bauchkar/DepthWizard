/**
 * DepthWizard — main application logic.
 *
 * Result state model (maps to existing backend tier/flags — no backend changes):
 *   RELATIVE       Mode A, tier R — relative surface structure, no units
 *   TERRAIN_ONLY   Mode B, tier T/A, NO_OBJECT_SCALE flag — terrain elev. absolute, object heights relative
 *   TERRAIN_SCALED Mode B, tier T,   OBJECT_SCALE_DEM_FIT flag — terrain + unvalidated object scale
 *   ANCHOR_REFINED Mode B, tier A,   OBJECT_SCALE_ANCHORS flag — terrain + anchor-calibrated objects
 *
 * Measurement architecture: measurements are ALWAYS sampled server-side from the DSM raster.
 * The Three.js mesh is a visual representation only; picks are converted to raster pixel coords.
 */
import { api, ApiFailure, pollUntilDone, type DemoItem, type Job, type Result, type Sample } from "./api";
import { HeightfieldViewer } from "./viewer";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ──────────────────────────────────────── Result state model
type ResultState = "RELATIVE" | "TERRAIN_ONLY" | "TERRAIN_SCALED" | "ANCHOR_REFINED";

interface StateDisplay {
  label: string;      // short badge label
  full: string;       // longer description
  cssClass: string;   // state-RELATIVE etc.
  hudState: string;   // HUD top-left
  hudTier: string;    // HUD tier line
}

const STATE_DISPLAY: Record<ResultState, StateDisplay> = {
  RELATIVE: {
    label: "RELATIVE SURFACE",
    full:  "Relative Surface Structure — no units, no scale, no elevation. Upload a GeoTIFF for absolute output.",
    cssClass: "state-RELATIVE",
    hudState: "RELATIVE SURFACE STRUCTURE",
    hudTier:  "Tier R · non-metric",
  },
  TERRAIN_ONLY: {
    label: "TERRAIN ELEVATION ONLY",
    full:  "Terrain Elevation (DEM-derived, absolute) + Relative Object Structure (non-metric). Object heights are NOT calibrated metres.",
    cssClass: "state-TERRAIN_ONLY",
    hudState: "TERRAIN ELEVATION ONLY",
    hudTier:  "",  // filled dynamically
  },
  TERRAIN_SCALED: {
    label: "TERRAIN + SCALED OBJECTS (UNVALIDATED)",
    full:  "Terrain Elevation (DEM-derived, absolute) + Object Layer scaled from DEM residual fit. Object scale is unvalidated — LIMITED quality.",
    cssClass: "state-TERRAIN_SCALED",
    hudState: "TERRAIN + SCALED OBJECTS",
    hudTier:  "",
  },
  ANCHOR_REFINED: {
    label: "TERRAIN + ANCHOR-CALIBRATED OBJECTS",
    full:  "Terrain Elevation (DEM-derived, absolute) + Object Layer calibrated by ground control anchors.",
    cssClass: "state-ANCHOR_REFINED",
    hudState: "ANCHOR-CALIBRATED DSM",
    hudTier:  "",
  },
};

function resolveState(res: Result): ResultState {
  if (res.mode !== "B") return "RELATIVE";
  const flags = res.flags ?? [];
  if (flags.includes("NO_OBJECT_SCALE")) return "TERRAIN_ONLY";
  if (flags.includes("OBJECT_SCALE_ANCHORS")) return "ANCHOR_REFINED";
  if (flags.includes("OBJECT_SCALE_DEM_FIT")) return "TERRAIN_SCALED";
  // Fallback: if scale source exists use it, else terrain-only
  if (res.object_scale_source === "anchors") return "ANCHOR_REFINED";
  if (res.object_scale_source) return "TERRAIN_SCALED";
  return "TERRAIN_ONLY";
}

// ──────────────────────────────────────── App state
interface State {
  file: File | null; dem: File | null; anchors: File | null; anchorsLabel: string;
  jobId: string | null; job: Job | null; result: Result | null; viewer: HeightfieldViewer | null;
  resultState: ResultState;
  layer: string; measuring: boolean; measurePts: { x: number; y: number }[];
  demo: DemoItem[]; references: string[];
  viewLayer: string;
  currentHud: { state: string; tier: string; quality: string; layer: string; showNorth: boolean } | null;
}
const state: State = {
  file: null, dem: null, anchors: null, anchorsLabel: "",
  jobId: null, job: null, result: null, viewer: null,
  resultState: "RELATIVE",
  layer: "rgb", measuring: false, measurePts: [],
  demo: [], references: [],
  viewLayer: "dsm",
  currentHud: null,
};

// ──────────────────────────────────────── Helpers
const isTiff = (f: File) => /\.tiff?$/i.test(f.name);
function setStatus(msg: string, kind: "" | "ok" | "err" = "", el = "status") {
  const e = $(el); e.textContent = msg; e.className = `status ${kind}`;
}
function userMessage(e: unknown): string {
  if (e instanceof ApiFailure) return `${e.err.message} (${e.err.code})${e.err.detail ? " — " + e.err.detail : ""}`;
  if (e instanceof Error && e.message === "WEBGL_UNAVAILABLE") return "3D rendering is unavailable on this device. Raster outputs remain available.";
  if (e instanceof Error) return e.message;
  return "Unable to process request.";
}
const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d);
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ──────────────────────────────────────── Layer definitions
interface LayerDef {
  label: string;
  desc: string;
  art: string;
  units: string;
  tier: string;
  modeB: boolean;
}
const LAYER_DEFS: Record<string, LayerDef> = {
  rgb:      { label: "RGB Input",            art: "input_preview",   desc: "Source image (metric-grid resampled for Mode B).",                                     units: "—",        tier: "—",   modeB: false },
  depth:    { label: "Model Output",         art: "depth_preview",   desc: "Depth Anything V2 relative inverse depth (normalised). Non-metric. Zero-shot.",          units: "0–1 (rel)",tier: "R",   modeB: false },
  rdsm:     { label: "Relative Structure",   art: "rdsm_preview",    desc: "rDSM: relative surface structure derived from depth. No units. No scale. No elevation.", units: "0–1 (rel)",tier: "R",   modeB: false },
  dsm:      { label: "DSM",                  art: "dsm_preview",     desc: "Digital Surface Model. Terrain layer + object layer. Absolute elevation where calibrated.",units: "metres",   tier: "T/A", modeB: true  },
  terrain:  { label: "Terrain Layer",        art: "terrain_preview", desc: "DEM-derived terrain reconstruction. Ground-masked. NOT a certified DTM. Absolute.",       units: "metres",   tier: "T",   modeB: true  },
  ndsm:     { label: "Object Height (nDSM)", art: "ndsm_preview",    desc: "Height above local ground. Metric only when object scale source is available.",           units: "metres",   tier: "T/A", modeB: true  },
  hillshade:{ label: "Hillshade",            art: "hillshade_preview",desc: "Hillshade of the DSM for visual display only. Not a measurement layer.",                 units: "(display)",tier: "—",   modeB: true  },
  slope:    { label: "Slope",                art: "slope_preview",   desc: "Slope in degrees (Horn 3×3 kernel, applied to DSM). Steeper = brighter.",                units: "degrees",  tier: "T",   modeB: true  },
  flags:    { label: "Quality Flags",        art: "flags_preview",   desc: "Per-pixel quality flags: border · raw-DEM fallback · DEM void · nodata · no object scale · low ground support.", units: "(flags)", tier: "—", modeB: true },
};

// ──────────────────────────────────────── System / demo init
async function initSystem() {
  const badge = $("system-badge");
  try {
    const h = await api.health();
    const m = h.model;
    if (m?.available) {
      badge.textContent = `${m.name}@${m.version} · ${m.device}`;
      badge.className = "badge ok";
      $("m-model").textContent = `Depth Anything V2 Small (${m.name}@${m.version}, ${m.device}) — zero-shot relative depth · no metric head · Tier H unavailable in this build`;
    } else {
      badge.textContent = "model weights not installed";
      badge.className = "badge bad";
      $("m-model").textContent = "Model weights not installed. Run scripts/fetch_model.py.";
      setStatus("Model weights are not installed. Run scripts/fetch_model.py.", "err");
    }
  } catch {
    badge.textContent = "backend unreachable";
    badge.className = "badge bad";
    setStatus("Backend unreachable. Is the server running?", "err");
  }
  try {
    const d = await api.demo();
    state.demo = d.items; state.references = d.references;
    const wrap = $("demo-buttons"); wrap.innerHTML = "";
    d.items.forEach((it, i) => {
      const b = document.createElement("button");
      b.className = "linkbtn";
      b.textContent = it.label;
      b.title = it.source ?? "";
      b.addEventListener("click", () => loadDemo(it));
      wrap.appendChild(b);
      if (i < d.items.length - 1) wrap.append(" · ");
    });
    const aw = $("anchor-demo-buttons"); aw.innerHTML = "";
    ["urban", "rural"].forEach((t, i) => {
      const b = document.createElement("button"); b.className = "linkbtn"; b.textContent = t;
      b.addEventListener("click", () => loadDemoAnchors(t)); aw.appendChild(b);
      if (i === 0) aw.append(" · ");
    });
    const sel = $("ref-bundled") as HTMLSelectElement;
    d.references.forEach((r) => { const o = document.createElement("option"); o.value = r; o.textContent = r; sel.appendChild(o); });
  } catch { /* demo assets optional */ }
}

async function loadDemo(it: DemoItem) {
  try {
    const r = await fetch(`/demo/${it.file}`);
    if (!r.ok) throw new Error("demo missing");
    const blob = await r.blob();
    showInput(new File([blob], it.file, { type: blob.type || (it.mode === "B" ? "image/tiff" : "image/jpeg") }));
    if (it.mode === "B" && it.reference_dsm) {
      ($("ref-bundled") as HTMLSelectElement).value = it.reference_dsm;
      ($("ref-vcrs") as HTMLSelectElement).value = it.reference_vertical_crs ?? "same";
    }
    if (it.anchors) {
      const type = it.anchors.includes("urban") ? "urban" : "rural";
      await loadDemoAnchors(type);
    }
  } catch { setStatus("Demo asset not available on this server.", "err"); }
}

async function loadDemoAnchors(t: string) {
  try {
    const r = await fetch(`/demo/anchors_${t}_simulated.csv`);
    if (!r.ok) throw new Error("missing");
    state.anchors = new File([await r.blob()], `anchors_${t}_simulated.csv`, { type: "text/csv" });
    state.anchorsLabel = `simulated ${t} anchors (sampled from LiDAR — NOT surveyed ground control)`;
    updateOptSummary();
  } catch { setStatus("Demo anchors not available.", "err"); }
}

function updateOptSummary() {
  const parts: string[] = [];
  if (state.dem) parts.push(`user DEM: ${state.dem.name} (${($("dem-vcrs") as HTMLSelectElement).value})`);
  if (state.anchors) parts.push(`anchors: ${state.anchorsLabel || state.anchors.name}`);
  $("opt-summary").textContent = parts.length ? parts.join(" · ") : "none (bundled Copernicus GLO-30 DEM auto-used when AOI is covered)";
}

// ──────────────────────────────────────── Input
function showInput(file: File) {
  state.file = file; state.jobId = null; state.result = null; state.job = null;
  const tif = isTiff(file);
  const wrap = $("input-preview-wrap"); const img = $("input-preview") as HTMLImageElement;
  if (!tif) {
    const url = URL.createObjectURL(file);
    img.onload = () => { $("m-image").textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight} px · ${(file.size / 1024).toFixed(0)} KB`; URL.revokeObjectURL(url); };
    img.src = url; wrap.classList.remove("hidden");
  } else {
    wrap.classList.add("hidden"); img.removeAttribute("src");
    $("m-image").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB · GeoTIFF (preview available after processing)`;
  }
  $("m-mode").textContent = tif
    ? "B — Georeferenced GeoTIFF → DEM-anchored calibration → terrain + object layers"
    : "A — Non-georeferenced → relative surface structure only (no units, no elevation)";
  $("m-geo").textContent   = tif ? "Expected — read from GeoTIFF tags at ingest" : "No";
  $("m-metric").textContent = tif ? "Horizontal: yes (GSD from CRS) · Vertical: decided after calibration" : "No — output is unitless relative structure";
  $("m-tier").textContent  = tif ? "T (DEM terrain) / A (anchors) — determined after calibration" : "R (relative only)";
  ($("run-btn") as HTMLButtonElement).disabled = false;
  $("run-btn").textContent = tif ? "Generate Calibrated Surface (Mode B)" : "Generate Relative Surface (Mode A)";
  ($("open3d-btn") as HTMLButtonElement).disabled = true;
  ($("validate-btn") as HTMLButtonElement).disabled = true;
  $("result-body").classList.add("hidden"); $("result-empty").classList.remove("hidden");
  $("val-result").classList.add("hidden");
  setStatus(tif ? "Ready — click Generate Calibrated Surface." : "Ready — click Generate Relative Surface.");
}

// ──────────────────────────────────────── Run pipeline
async function run() {
  if (!state.file) return;
  const btn = $("run-btn") as HTMLButtonElement; btn.disabled = true;
  try {
    setStatus("Uploading…");
    const job = await api.createJob(state.file, {
      dem: state.dem,
      demVerticalCrs: ($("dem-vcrs") as HTMLSelectElement).value,
      anchors: state.anchors,
    });
    state.jobId = job.job_id;
    await api.run(job.job_id);
    const stageLabels: Record<string, string> = {
      PREPROCESSING: "ingest / georeference / validate",
      INFERENCE:     "Depth Anything V2 Small — zero-shot relative depth",
      CALIBRATION:   "terrain layer · datum · object scale · derivatives",
      RASTERIZING:   "rDSM + heightfield",
    };
    const done = await pollUntilDone(job.job_id, (j: Job) =>
      setStatus(`Processing: ${j.status}${stageLabels[j.status] ? " — " + stageLabels[j.status] : ""}`)
    );
    state.job = done;
    if (done.status === "FAILED") {
      setStatus(`${done.error?.message ?? "Processing failed."} (${done.error?.code ?? "FAILED"})${done.error?.detail ? " — " + done.error.detail : ""}`, "err");
      return;
    }
    const res = await api.result(job.job_id);
    state.result = res;
    state.resultState = resolveState(res);
    await renderResult(done, res);
    const ms = done.stages_ms;
    const summary = res.mode === "B"
      ? `Tier ${res.calibration_tier} · ${STATE_DISPLAY[state.resultState].label} · quality ${res.quality} · ${res.vertical_reference}`
      : "Relative surface structure — no metres.";
    setStatus(
      `READY in ${ms.total_ms} ms (inference ${ms.inference_ms} ms on ${done.model?.device}${ms.calibration_ms ? `, calibration ${ms.calibration_ms} ms` : ""}). ${summary}`,
      res.quality === "INVALID" ? "err" : "ok"
    );
  } catch (e) { setStatus(userMessage(e), "err"); }
  finally { btn.disabled = false; }
}

// ──────────────────────────────────────── Result rendering
async function renderResult(job: Job, res: Result) {
  const id = job.job_id;
  const rs = state.resultState;
  const sd = STATE_DISPLAY[rs];

  $("result-empty").classList.add("hidden");
  $("result-body").classList.remove("hidden");

  // ── Primary: state badge + quality badge
  const qClass = res.quality ?? "UNVALIDATED";
  $("result-primary").innerHTML = `
    <div class="state-badge ${sd.cssClass}">${sd.label}</div>
    <div class="q-badge q-${qClass}">${qClass}</div>
    ${res.mode === "B" ? `<div class="badge info">TIER ${res.calibration_tier}</div>` : ""}
  `;

  // ── State description (one honest sentence)
  $("result-state-desc").textContent = sd.full;

  // ── Secondary: key metrics grid
  const g = res.grid;
  const dem = res.dem ? `${res.dem.product} (${res.dem.vertical_crs}, ${res.dem.posting_m} m)` : "none";
  const scaleStr = res.object_scale_source
    ? `${res.object_scale_source}${res.object_scale_m_per_unit ? ` (${fmt(res.object_scale_m_per_unit, 3)} m/unit)` : ""}`
    : "none — object heights remain relative";

  const fields: [string, string, boolean?][] = res.mode === "B" ? [
    ["Grid",        `${g.width}×${g.height} px`],
    ["GSD",         `${fmt(res.gsd_m, 2)} m`],
    ["CRS",         g.crs],
    ["DEM",         dem],
    ["Vertical ref",res.vertical_reference ?? "—"],
    ["Object scale",scaleStr, !res.object_scale_source],
    ["Tier",        res.calibration_tier],
  ] : [
    ["Grid",   `${g.width}×${g.height} px`],
    ["Output", "Unitless relative structure [0–1]"],
    ["Tier",   "R (relative only)"],
    ["Model",  "Depth Anything V2 Small — zero-shot"],
  ];

  $("result-fields").innerHTML = fields.map(([label, val, warn]) =>
    `<div class="result-field">
       <div class="result-field-label">${label}</div>
       <div class="result-field-value${warn ? " warn" : ""}">${esc(String(val))}</div>
     </div>`
  ).join("");

  // ── Quality card: why is quality X?
  const triggers = res.quality_triggers ?? [];
  const notes    = res.notes ?? [];
  const qualityCard = $("quality-card");
  qualityCard.className = `quality-card q-${qClass}`;
  const triggerItems = triggers.length
    ? triggers.map(t => `<li>${esc(t)}</li>`).join("")
    : "<li>No specific triggers recorded.</li>";
  const noteItems = notes.map(n => `<li>${esc(n)}</li>`).join("");
  qualityCard.innerHTML = `
    <div class="quality-card-header">
      <span class="q-badge q-${qClass}">${qClass}</span>
      <span class="quality-card-title">${triggers.length ? "Why this quality?" : "Quality status"}</span>
    </div>
    <ul class="quality-triggers">${triggerItems}</ul>
    ${notes.length ? `<div class="quality-notes"><ul class="quality-triggers">${noteItems}</ul></div>` : ""}
  `;

  // ── Trust / provenance panel
  const flags = res.flags ?? [];
  $("trust-panel").innerHTML = `
    <div class="trust-header">Provenance &amp; Trust</div>
    <div class="trust-grid">
      <div class="trust-item"><div class="trust-label">Model</div><div class="trust-value">Depth Anything V2 Small</div></div>
      <div class="trust-item"><div class="trust-label">Inference</div><div class="trust-value">Zero-shot — no metric head (Tier H unavailable)</div></div>
      <div class="trust-item"><div class="trust-label">Input type</div><div class="trust-value">${res.mode === "B" ? "GeoTIFF (georeferenced)" : "PNG / JPEG (non-georeferenced)"}</div></div>
      <div class="trust-item"><div class="trust-label">Calibration tier</div><div class="trust-value ${res.calibration_tier === "A" ? "ok" : ""}">${res.calibration_tier}</div></div>
      ${res.mode === "B" ? `
      <div class="trust-item"><div class="trust-label">DEM source</div><div class="trust-value">${esc(dem)}</div></div>
      <div class="trust-item"><div class="trust-label">Vertical ref</div><div class="trust-value">${esc(res.vertical_reference ?? "—")}</div></div>
      <div class="trust-item"><div class="trust-label">Object scale</div><div class="trust-value ${!res.object_scale_source ? "warn" : "ok"}">${esc(scaleStr)}</div></div>
      ` : ""}
      <div class="trust-item"><div class="trust-label">Quality</div><div class="trust-value ${qClass === "GOOD" ? "ok" : qClass === "INVALID" ? "bad" : "warn"}">${qClass}</div></div>
      <div class="trust-item"><div class="trust-label">Flags</div><div class="trust-value ${flags.length ? "warn" : ""}">${flags.join(", ") || "none"}</div></div>
      <div class="trust-item"><div class="trust-label">Job ID</div><div class="trust-value">${esc(job.job_id)}</div></div>
      <div class="trust-item"><div class="trust-label">Processed</div><div class="trust-value">${new Date(job.updated_at).toLocaleString()}</div></div>
    </div>
  `;

  // ── Layer tabs
  const tabs = $("layer-tabs"); tabs.innerHTML = "";
  const available = Object.keys(LAYER_DEFS).filter((k) => res.artifacts[LAYER_DEFS[k].art]);
  available.forEach((k) => {
    const b = document.createElement("button");
    b.textContent = LAYER_DEFS[k].label;
    b.dataset.layer = k;
    b.addEventListener("click", () => showLayer(k));
    tabs.appendChild(b);
  });

  // ── GeoTIFF preview for Mode B
  if (res.mode === "B") {
    ($("input-preview") as HTMLImageElement).src = api.artifactUrl(id, res.artifacts.input_preview);
    $("input-preview-wrap").classList.remove("hidden");
    ($("validate-btn") as HTMLButtonElement).disabled = false;
  }

  // ── Default layer
  showLayer(res.mode === "B" ? "dsm" : "rdsm");

  // ── Downloads
  const dl = $("downloads"); dl.innerHTML = "<b>Download:</b> ";
  const files: [string, string][] = res.mode === "B"
    ? [["dsm.tif","DSM GeoTIFF"],["terrain.tif","Terrain layer"],["ndsm.tif","nDSM"],["slope.tif","Slope"],["aspect.tif","Aspect"],["flags.tif","Flags"],["relative.tif","Relative structure"],["calib_report.json","Calibration report"],["log.jsonl","Job log"]]
    : [["rdsm.tif","rDSM (no CRS)"],["relative_depth.npy","Raw model output"],["log.jsonl","Job log"]];
  const arts = new Set(Object.values(res.artifacts));
  files.filter(([f]) => arts.has(f) || f === "log.jsonl").forEach(([f, l]) => {
    const a = document.createElement("a"); a.href = api.artifactUrl(id, f); a.download = f;
    a.textContent = l; a.className = "linkbtn"; dl.appendChild(a);
  });

  // ── Technical details (collapsed)
  const md = await api.metadata(id);
  $("result-json").textContent = JSON.stringify({ job: { stages_ms: job.stages_ms, model: job.model, inputs: job.inputs }, result: { ...res, layers: undefined, artifacts: undefined }, calib_report: md.calib_report, prep: md.prep, input: md.meta }, null, 2);

  // ── 3D layer options
  const vl = $("view-layer") as HTMLSelectElement; vl.innerHTML = "";
  const opts: [string, string][] = res.mode === "B"
    ? [
        ["dsm", "DSM (metres, terrain + buildings)"],
        ["ndsm", "nDSM (metres, height above ground — flat ground, buildings only)"],
        ["terrain", "Terrain layer (metres, bare ground)"],
        ["relative", "Relative structure (tier R, non-metric)"]
      ]
    : [["relative", "Relative structure (tier R, non-metric)"]];
  opts.filter(([k]) => res.layers?.[k]?.heightfield || (res.mode === "A" && k === "relative")).forEach(([k, l]) => {
    const o = document.createElement("option"); o.value = k; o.textContent = l; vl.appendChild(o);
  });
  state.viewLayer = vl.value;
  ($("open3d-btn") as HTMLButtonElement).disabled = false;

  // ── Reset readout
  const ro = $("readout-body"); ro.innerHTML = "Click a point on the image or 3D mesh."; ro.className = "readout-body empty";
  $("measure-out").textContent = "";
  state.measurePts = []; state.measuring = false;
  $("measure-btn").classList.remove("active");
}

// ──────────────────────────────────────── Layer display
function showLayer(k: string) {
  if (!state.result || !state.jobId) return;
  state.layer = k;
  document.querySelectorAll<HTMLButtonElement>("#layer-tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.layer === k)
  );
  const def = LAYER_DEFS[k];
  const img = $("layer-img") as HTMLImageElement;
  img.src = api.artifactUrl(state.jobId, state.result.artifacts[def.art]);

  // Layer header
  const li = state.result.layers?.[k];
  const leg = li?.legend;
  const minMax = li && Number.isFinite((li as any).min) ? ` · ${fmt((li as any).min, 1)} – ${fmt((li as any).max, 1)} ${def.units}` : "";
  $("layer-name").textContent = def.label;
  $("layer-desc").textContent = def.desc;
  $("layer-meta").innerHTML = [
    `<span class="layer-meta-item">units: ${def.units}</span>`,
    def.tier !== "—" ? `<span class="layer-meta-item">tier: ${def.tier}</span>` : "",
    minMax ? `<span class="layer-meta-item">${minMax}</span>` : "",
    li?.label ? `<span class="layer-meta-item">${esc(li.label)}</span>` : "",
  ].filter(Boolean).join("");
  $("layer-legend").textContent = leg
    ? `Legend: ${Object.entries(leg).map(([a, b]) => `${a}=${typeof b === "number" ? fmt(b) : JSON.stringify(b)}`).join(" · ")}`
    : "";
}

// ──────────────────────────────────────── Point readout / measurement
function renderSample(s: Sample): string {
  if (s.in_bounds === false) return `<div class="readout-pos">outside raster bounds</div>`;
  const posLine = s.position
    ? `<div class="readout-pos">${esc(s.position.crs)}  x=${fmt(s.position.x, 1)}  y=${fmt(s.position.y, 1)}${s.position.lon !== undefined ? `  lon ${fmt(s.position.lon, 5)}  lat ${fmt(s.position.lat, 5)}` : ""}</div>`
    : "";
  const pixLine = s.pixel ? `<div class="readout-pos">pixel col ${s.pixel.col}  row ${s.pixel.row}</div>` : "";
  const rows = Object.entries(s.values).map(([k, v]) => {
    const meta = [v.tier ? `tier ${v.tier}` : "", v.vertical_reference ?? "", v.scale_source ?? ""].filter(Boolean).join(" · ");
    const valStr = v.valid && v.value !== null ? `${fmt(v.value, v.units === "m" ? 2 : 3)} ${v.units}` : "nodata";
    return `<tr>
      <td><div class="rk">${esc(k)}</div>${meta ? `<div class="rq">${esc(meta)}</div>` : ""}</td>
      <td class="rv">${esc(valStr)}</td>
    </tr>`;
  }).join("");
  const flagsLine = s.flags?.length ? `<div class="readout-flags">⚑ ${s.flags.join(", ")}</div>` : "";
  return `${pixLine}${posLine}<table class="readout-table">${rows}</table>${flagsLine}`;
}

async function pickAt(col: number, row: number) {
  if (!state.jobId || !state.result) return;
  const ro = $("readout-body");
  ro.className = "readout-body";
  try {
    if (state.measuring) {
      state.measurePts.push({ x: col, y: row });
      if (state.measurePts.length < 2) {
        $("measure-out").textContent = "Point A set — click Point B";
        return;
      }
      const m = await api.measure(state.jobId, state.measurePts);
      const s = m.segments[0];
      const parts = [`Δ distance ${fmt(s.horizontal_distance, 1)} ${s.distance_units}`];
      if (s.dz_dsm !== undefined)      parts.push(`Δ DSM ${fmt(s.dz_dsm, 2)} m${s.grade_dsm_deg !== undefined ? ` (${fmt(s.grade_dsm_deg, 1)}° slope)` : ""}`);
      if (s.dz_ndsm !== undefined)     parts.push(`Δ nDSM ${fmt(s.dz_ndsm, 2)} m`);
      if (s.dz_rdsm !== undefined)     parts.push(`Δ relative ${fmt(s.dz_rdsm, 3)} (no units)`);
      if (s.dz_relative !== undefined) parts.push(`Δ relative ${fmt(s.dz_relative, 3)} (no units)`);
      $("measure-out").textContent = `${parts.join("  ·  ")}  —  tier ${m.calibration_tier}  ·  ${m.authority}`;
      ro.innerHTML = `<div class="rq">A</div>${renderSample(m.points[0])}<div class="rq" style="margin-top:6px">B</div>${renderSample(m.points[1])}`;
      state.measurePts = []; state.measuring = false; $("measure-btn").classList.remove("active");
    } else {
      const s = await api.sample(state.jobId, col, row);
      ro.innerHTML = renderSample(s);
    }
  } catch (e) { ro.textContent = userMessage(e); }
}

function wireImagePick() {
  const img = $("layer-img") as HTMLImageElement;
  const wrap = img.parentElement as HTMLElement;
  img.addEventListener("click", (e) => {
    if (!state.result) return;
    const r = img.getBoundingClientRect();
    const g = state.result.grid; const W = Number(g.width), H = Number(g.height);
    const col = ((e.clientX - r.left) / r.width) * W;
    const row = ((e.clientY - r.top) / r.height) * H;
    wrap.querySelectorAll(".pick-dot").forEach((d, i) => { if (!state.measuring || i > 0) d.remove(); });
    const dot = document.createElement("div");
    dot.className = "pick-dot";
    dot.style.left = `${e.clientX - r.left}px`;
    dot.style.top  = `${e.clientY - r.top}px`;
    wrap.appendChild(dot);
    pickAt(col, row);
  });
}

// ──────────────────────────────────────── Validation
async function runValidation() {
  if (!state.jobId || !state.result) return;
  const btn = $("validate-btn") as HTMLButtonElement; btn.disabled = true;
  const refFile = ($("ref-input") as HTMLInputElement).files?.[0] ?? null;
  const bundled = ($("ref-bundled") as HTMLSelectElement).value || null;
  try {
    if (!refFile && !bundled) { setStatus("Choose a reference raster or a bundled reference.", "err", "val-status"); return; }
    setStatus("Validating — reproject → datum-align → co-register → mask → metrics…", "", "val-status");
    const v = await api.validate(state.jobId, {
      reference: refFile, bundled: refFile ? null : bundled,
      refType: ($("ref-type") as HTMLSelectElement).value,
      verticalCrs: ($("ref-vcrs") as HTMLSelectElement).value,
    });
    renderValidation(v);
    const hist = await api.validation(state.jobId);
    $("val-history").textContent = `${hist.runs.length} validation run(s) stored for this job.`;
    setStatus(`Validation complete in ${v.elapsed_ms} ms.`, "ok", "val-status");
  } catch (e) { setStatus(userMessage(e), "err", "val-status"); }
  finally { btn.disabled = false; }
}

function metricRow(m: Record<string, any>): string {
  return `<td>${m.n}</td><td>${fmt(m.ME)}</td><td>${fmt(m.RMSE)}</td><td>${fmt(m.MAE)}</td><td>${fmt(m.NMAD)}</td><td>${fmt(m.LE90)}</td><td>${fmt(m.LE95)}</td><td>${fmt(m.pearson_r, 3)}</td><td>${fmt(m.spearman_rho, 3)}</td>`;
}

function renderValidation(v: Record<string, any>) {
  $("val-result").classList.remove("hidden");
  const head = `<thead><tr><th>Set</th><th>n</th><th>ME (m)</th><th>RMSE (m)</th><th>MAE (m)</th><th>NMAD (m)</th><th>LE90</th><th>LE95</th><th>r</th><th>ρ</th></tr></thead>`;
  $("val-verdict-title").textContent = `${v.verdict?.band ?? ""} — ${v.verdict?.text ?? ""}`;
  $("val-context").innerHTML = `
    Compared layer: <b>${esc(v.compared_layer)}</b> (${esc(v.reference?.ref_type ?? "")}).
    Reference posting: ${v.reference?.native_posting_m?.[0]} m → reprojected to job grid.
    Datum handling: ${esc(v.reference?.datum_handling ?? "—")}.
    Job tier: ${esc(v.job?.calibration_tier ?? "—")}, quality: ${esc(v.job?.quality ?? "—")}.
  `;
  $("val-overall").innerHTML = `${head}<tbody>
    <tr><td>Overall (metres)</td>${metricRow(v.metrics_overall)}</tr>
    <tr><td><i>Oracle affine (diagnostic only)</i> — scale ${fmt(v.oracle_affine?.scale, 3)}, shift ${fmt(v.oracle_affine?.shift, 2)}</td>${metricRow(v.oracle_affine?.metrics_after_alignment ?? {})}</tr>
  </tbody>`;
  const strata = (title: string, obj: Record<string, any>) =>
    `<table class="metrics">${head}<tbody>${Object.entries(obj ?? {}).map(([k, m]) => `<tr><td>${esc(title)}: ${esc(k)}</td>${metricRow(m as any)}</tr>`).join("")}</tbody></table>`;
  $("val-strata").innerHTML = strata("slope", v.metrics_by_slope) + strata("object class", v.metrics_by_object) + strata("height bin", v.height_bins);
  $("val-json").textContent = JSON.stringify({ alignment: v.alignment, mask: v.mask, reference: v.reference, residual_stats: v.residual_stats, caveats: v.caveats }, null, 2);
  const img = $("val-residual") as HTMLImageElement;
  if (state.jobId && v.artifacts?.residual_preview)
    img.src = api.artifactUrl(state.jobId, v.artifacts.residual_preview) + `?t=${Date.now()}`;
}

// ──────────────────────────────────────── 3D viewer
async function open3d() {
  if (!state.jobId || !state.result) return;
  const msg = $("viewer-msg"); msg.classList.remove("hidden");
  try {
    if (!state.viewer) {
      msg.textContent = "Initialising WebGL…";
      state.viewer = new HeightfieldViewer($("viewer"));
      state.viewer.onPick(pickAt);
      state.viewer.onCameraModeChange((mode) => {
        $("cam-orbit-btn").classList.toggle("active", mode === "orbit");
        $("cam-walk-btn").classList.toggle("active", mode === "walk");
        $("walk-hud").classList.toggle("hidden", mode !== "walk");
      });
      state.viewer.onWalkStats((eyeZ, speedKmH) => {
        $("walk-hud-stats").textContent = `${eyeZ.toFixed(2)} m Eye Height · ${speedKmH.toFixed(1)} km/h`;
      });
    }
    const res = state.result;
    const layer = ($("view-layer") as HTMLSelectElement).value || "relative";
    const metric = res.mode === "B" && layer !== "relative";
    const hfName  = res.mode === "B" ? (res.layers?.[layer]?.heightfield ?? "heightfield_relative.f32") : "heightfield.f32";
    const metaName = res.mode === "B" ? (res.layers?.[layer]?.heightfield_meta ?? "heightfield_relative.json") : null;

    msg.textContent = "Loading heightfield…";
    const [hf, meta] = await Promise.all([
      api.heightfield(state.jobId, hfName),
      metaName ? api.heightfieldMeta(state.jobId, metaName) : Promise.resolve(res.heightfield),
    ]);
    const spacing = (res.gsd_m ?? 1) * (meta.downsample_factor || 1);

    // Build HUD info from honest state
    const rs = state.resultState;
    const sd = STATE_DISPLAY[rs];
    const tierLine = metric
      ? `Tier ${meta.calibration_tier} · ${meta.vertical_reference ?? "—"}`
      : "Tier R · non-metric";
    const hudInfo = {
      state:    metric ? sd.hudState : "RELATIVE SURFACE STRUCTURE",
      tier:     tierLine,
      quality:  res.quality ?? "UNVALIDATED",
      layer:    LAYER_DEFS[layer]?.label ?? layer.toUpperCase(),
      showNorth: metric,   // only meaningful for georeferenced output
    };
    state.currentHud = hudInfo;

    const exag = Number(($("exag") as HTMLInputElement).value);
    const info = await state.viewer.load(hf, meta, api.artifactUrl(state.jobId, res.artifacts.texture), {
      metric, spacing, exaggeration: exag, hud: hudInfo,
    });

    msg.textContent = ""; msg.classList.add("hidden");

    // Show/hide appropriate controls
    $("zscale-wrap").classList.toggle("hidden", metric);
    $("exag-wrap").classList.toggle("hidden", !metric);

    // Reset camera mode & mesh mode UI toggles to defaults
    $("cam-orbit-btn").classList.add("active");
    $("cam-walk-btn").classList.remove("active");
    $("walk-hud").classList.add("hidden");
    ($("mesh-mode-sel") as HTMLSelectElement).value = "regular";
    $("rtin-tol-wrap").classList.add("hidden");
    $("mesh-reduction-pill").classList.add("hidden");

    // Mesh info line
    $("mesh-info").textContent = [
      `${meta.width}×${meta.height} vertices (${info.triangles} triangles)`,
      meta.method,
      meta.downsample_factor > 1 ? `downsample ×${meta.downsample_factor}, residual RMSE ${meta.residual_vs_source.rmse.toFixed(3)} ${meta.units}` : "",
      metric ? `spacing ${spacing.toFixed(2)} m` : "",
      `nodata dropped: ${info.nodataDropped}`,
      "click mesh or image to sample raster",
    ].filter(Boolean).join(" · ");

    // Anti-smear state sync
    const antiSmearChk = $("antismear") as HTMLInputElement | null;
    if (antiSmearChk) {
      state.viewer.setAntiSmear(antiSmearChk.checked);
    }

    // LoD-1 3D Vector Building Blocks
    const lod1Wrap = $("lod1-wrap");
    const lod1Chk = $("lod1-chk") as HTMLInputElement;
    if (res.artifacts?.buildings_json) {
      try {
        const bData = await fetch(api.artifactUrl(state.jobId, res.artifacts.buildings_json)).then((r) => r.json());
        state.viewer.loadBuildings(bData);
        lod1Wrap.classList.remove("hidden");
        lod1Chk.checked = true;
      } catch (err) {
        console.warn("Failed to load 3D buildings", err);
        lod1Wrap.classList.add("hidden");
        state.viewer.loadBuildings(null);
      }
    } else {
      lod1Wrap.classList.add("hidden");
      state.viewer.loadBuildings(null);
    }

    $("panel-3d").scrollIntoView({ behavior: "smooth" });
  } catch (e) { msg.classList.remove("hidden"); msg.textContent = userMessage(e); }
}

// ──────────────────────────────────────── Wire events
function wire() {
  const dz = $("dropzone"); const input = $("file-input") as HTMLInputElement;
  input.addEventListener("change", () => { if (input.files?.[0]) showInput(input.files[0]); });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); const f = e.dataTransfer?.files?.[0]; if (f) showInput(f); });
  ($("dem-input") as HTMLInputElement).addEventListener("change", (e) => { state.dem = (e.target as HTMLInputElement).files?.[0] ?? null; updateOptSummary(); });
  ($("dem-vcrs") as HTMLSelectElement).addEventListener("change", updateOptSummary);
  ($("anchors-input") as HTMLInputElement).addEventListener("change", (e) => { state.anchors = (e.target as HTMLInputElement).files?.[0] ?? null; state.anchorsLabel = ""; updateOptSummary(); });
  $("anchors-clear").addEventListener("click", () => { state.anchors = null; state.anchorsLabel = ""; ($("anchors-input") as HTMLInputElement).value = ""; updateOptSummary(); });
  $("run-btn").addEventListener("click", run);
  $("open3d-btn").addEventListener("click", open3d);
  ($("view-layer") as HTMLSelectElement).addEventListener("change", () => { if (state.viewer && state.result) open3d(); });
  $("validate-btn").addEventListener("click", runValidation);
  $("measure-btn").addEventListener("click", () => {
    state.measuring = !state.measuring; state.measurePts = [];
    $("measure-btn").classList.toggle("active", state.measuring);
    $("measure-out").textContent = state.measuring ? "Click two points (image or 3D mesh) — measurements are sampled from the server raster" : "";
  });
  const zs = $("zscale") as HTMLInputElement;
  zs.addEventListener("input", () => { $("zscale-val").textContent = Number(zs.value).toFixed(2); state.viewer?.setZScale(Number(zs.value)); });
  const ex = $("exag") as HTMLInputElement;
  ex.addEventListener("input", () => {
    const v = Number(ex.value);
    $("exag-val").textContent = `×${v.toFixed(1)}${Math.abs(v - 1) < 1e-6 ? " (true scale)" : ""}`;
    state.viewer?.setExaggeration(v);
  });
  ($("wire") as HTMLInputElement).addEventListener("change", (e) => state.viewer?.setWireframe((e.target as HTMLInputElement).checked));
  const antiSmear = $("antismear") as HTMLInputElement | null;
  if (antiSmear) {
    antiSmear.addEventListener("change", (e) => state.viewer?.setAntiSmear((e.target as HTMLInputElement).checked));
  }
  const lod1Chk = $("lod1-chk") as HTMLInputElement | null;
  if (lod1Chk) {
    lod1Chk.addEventListener("change", (e) => state.viewer?.setBuildingsVisible((e.target as HTMLInputElement).checked));
  }
  $("reset-cam").addEventListener("click", () => {
    state.viewer?.setCameraMode("orbit");
    state.viewer?.resetCamera();
  });

  // Camera Mode buttons
  $("cam-orbit-btn").addEventListener("click", () => state.viewer?.setCameraMode("orbit"));
  $("cam-walk-btn").addEventListener("click", () => state.viewer?.setCameraMode("walk"));

  // Mesh Mode & Adaptive RTIN controls
  const meshSel = $("mesh-mode-sel") as HTMLSelectElement;
  const tolWrap = $("rtin-tol-wrap");
  const tolInput = $("rtin-tol") as HTMLInputElement;
  const tolVal = $("rtin-tol-val");
  const redPill = $("mesh-reduction-pill");

  const applyMeshMode = () => {
    if (!state.viewer) return;
    const mode = meshSel.value as "regular" | "rtin";
    const tol = Number(tolInput.value);
    tolVal.textContent = `${tol.toFixed(1)} m`;
    tolWrap.classList.toggle("hidden", mode !== "rtin");

    const stats = state.viewer.setMeshMode(mode, tol);
    if (mode === "rtin") {
      redPill.classList.remove("hidden");
      redPill.textContent = `⚡ RTIN: ${stats.reductionPct}% triangles reduced (${stats.triangles.toLocaleString()} tris, tol ${tol.toFixed(1)}m)`;
    } else {
      redPill.classList.add("hidden");
    }
  };

  meshSel.addEventListener("change", applyMeshMode);
  tolInput.addEventListener("input", applyMeshMode);

  wireImagePick();
  updateOptSummary();
}

wire();
initSystem();
