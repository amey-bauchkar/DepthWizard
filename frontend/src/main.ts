/**
 * DepthWizard — main application logic.
 *
 * Result state model (maps to backend tier/flags):
 *   RELATIVE       Mode A, tier R — relative surface structure, no units
 *   TERRAIN_ONLY   Mode B, tier T/A, NO_OBJECT_SCALE flag — DSM = DEM relief only (no model detail calibrated)
 *   TERRAIN_SCALED Mode B, tier T/A, OBJECT_SCALE_DEM_FIT flag — DEM + model detail scaled against the DEM band
 *   ANCHOR_REFINED Mode B, tier A,   OBJECT_SCALE_ANCHORS flag — DEM + model detail with anchor-fitted gain
 *
 * Measurement architecture: measurements are ALWAYS sampled server-side from the DSM raster.
 * The Three.js mesh is a visual representation only; picks are converted to raster pixel coords.
 */
import { api, ApiFailure, pollUntilDone, type DemoItem, type Job, type Result, type Sample } from "./api";
import { HeightfieldViewer } from "./viewer";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ──────────────────────────────────────── Result state model
type ResultState = "RELATIVE" | "TERRAIN_ONLY" | "TERRAIN_SCALED" | "ANCHOR_REFINED" | "METRIC_MODEL" | "METRIC_HEIGHTS";

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
    label: "DEM RELIEF ONLY",
    full:  "Absolute elevation from the DEM (datum-checked). No model detail could be calibrated for this scene, so sub-30 m structure (buildings, tree crowns) is not added.",
    cssClass: "state-TERRAIN_ONLY",
    hudState: "DEM RELIEF ONLY",
    hudTier:  "",  // filled dynamically
  },
  TERRAIN_SCALED: {
    label: "DEM + CALIBRATED MODEL DETAIL",
    full:  "Absolute DSM: the DEM keeps everything it resolves (≥ 30 m); Depth Anything V2 (tiled) adds the finer detail, scaled per tile against the DEM. Without anchors or a reference this scale is unvalidated — quality ≤ LIMITED.",
    cssClass: "state-TERRAIN_SCALED",
    hudState: "DEM + MODEL DETAIL",
    hudTier:  "",
  },
  ANCHOR_REFINED: {
    label: "DEM + ANCHOR-CALIBRATED DETAIL",
    full:  "Absolute DSM: DEM + tiled model detail whose gain was fitted to ground-control anchors (accepted only when leave-one-out error improves). Ground anchors also correct the terrain datum.",
    cssClass: "state-ANCHOR_REFINED",
    hudState: "ANCHOR-CALIBRATED DSM",
    hudTier:  "",
  },
  METRIC_MODEL: {
    label: "DEM + FINE-TUNED METRIC HEIGHTS",
    full:  "Absolute DSM: the DEM keeps everything ≥ 30 m; the fine-tuned DepthWizard nDSM model (heights in metres, validated on held-out regions) adds building and tree structure. Terrain = DSM − model heights.",
    cssClass: "state-TERRAIN_SCALED",
    hudState: "DEM + METRIC nDSM MODEL",
    hudTier:  "",
  },
  METRIC_HEIGHTS: {
    label: "METRIC HEIGHTS · NO ELEVATION (TIER H)",
    full:  "Heights above ground in metres from the fine-tuned nDSM model. No DEM (or no safe datum) for this area, so there is no absolute elevation — upload a DEM or anchors for tier T/A.",
    cssClass: "state-TERRAIN_ONLY",
    hudState: "METRIC HEIGHTS ABOVE GROUND",
    hudTier:  "",
  },
};

function resolveState(res: Result): ResultState {
  if (res.mode === "B" && res.calibration_tier === "H") return "METRIC_HEIGHTS";
  if (res.mode !== "B" || res.calibration_tier === "R") return "RELATIVE"; // Mode B without DEM / safe datum stays relative
  const flags = res.flags ?? [];
  if (flags.includes("METRIC_NDSM_MODEL") && !flags.includes("OBJECT_SCALE_ANCHORS")) return "METRIC_MODEL";
  if (flags.includes("NO_OBJECT_SCALE")) return "TERRAIN_ONLY";
  if (flags.includes("OBJECT_SCALE_ANCHORS")) return "ANCHOR_REFINED";
  if (flags.includes("OBJECT_SCALE_DEM_FIT")) return "TERRAIN_SCALED";
  // Fallback: if scale source exists use it, else terrain-only
  if (res.object_scale_source?.includes("anchor")) return "ANCHOR_REFINED";
  if (res.object_scale_source) return "TERRAIN_SCALED";
  return "TERRAIN_ONLY";
}

// ──────────────────────────────────────── App state
interface DisasterState {
  baseLayer: "rgb" | "terrain" | "hillshade" | "dsm";
  overlayVisible: boolean;
  opacity: number;
  shimmer: boolean;
  showBuildings: boolean;
  lastResult: any | null;
  autoFloodVal: number;
}

interface State {
  file: File | null; dem: File | null; anchors: File | null; anchorsLabel: string;
  jobId: string | null; job: Job | null; result: Result | null; viewer: HeightfieldViewer | null;
  resultState: ResultState;
  layer: string; measuring: boolean; measurePts: { x: number; y: number }[];
  demo: DemoItem[]; references: string[];
  viewLayer: string;
  currentHud: { state: string; tier: string; quality: string; layer: string; showNorth: boolean } | null;
  metricModel: boolean;
  disaster: DisasterState;
}
const state: State = {
  file: null, dem: null, anchors: null, anchorsLabel: "",
  jobId: null, job: null, result: null, viewer: null,
  resultState: "RELATIVE",
  layer: "rgb", measuring: false, measurePts: [],
  demo: [], references: [],
  viewLayer: "dsm",
  currentHud: null,
  metricModel: false,
  disaster: {
    baseLayer: "rgb",
    overlayVisible: true,
    opacity: 0.75,
    shimmer: true,
    showBuildings: true,
    lastResult: null,
    autoFloodVal: 0,
  },
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
  depth:    { label: "Model Output",         art: "depth_preview",   desc: "Depth Anything V2 relative inverse depth, whole-image pass (normalised). Non-metric. Zero-shot.", units: "0–1 (rel)",tier: "R",   modeB: false },
  rdsm:     { label: "Relative Structure",   art: "rdsm_preview",    desc: "rDSM: relative surface structure derived from depth. No units. No scale. No elevation.", units: "0–1 (rel)",tier: "R",   modeB: false },
  dsm:      { label: "DSM",                  art: "dsm_preview",     desc: "Digital Surface Model: DEM (≥ 30 m structure, datum-transformed) + calibrated model detail (< 30 m). Absolute elevation.", units: "metres",   tier: "T/A", modeB: true  },
  terrain:  { label: "Terrain Layer",        art: "terrain_preview", desc: "Ground estimate: ground-weighted DEM reconstruction combined with the morphological ground of the DSM. NOT a certified DTM.", units: "metres",   tier: "T/A", modeB: true  },
  ndsm:     { label: "Object Height (nDSM)", art: "ndsm_preview",    desc: "DSM − terrain layer: height of buildings / trees above the ground estimate.",           units: "metres",   tier: "T/A", modeB: true  },
  hillshade:{ label: "Hillshade",            art: "hillshade_preview",desc: "Hillshade of the DSM for visual display only. Not a measurement layer.",                 units: "(display)",tier: "—",   modeB: true  },
  slope:    { label: "Slope",                art: "slope_preview",   desc: "Slope in degrees (Horn 3×3 kernel, applied to DSM). Steeper = brighter.",                units: "degrees",  tier: "T",   modeB: true  },
  flags:    { label: "Quality Flags",        art: "flags_preview",   desc: "Per-pixel quality flags: border · raw-DEM fallback · DEM void · nodata · no object scale · low ground support.", units: "(flags)", tier: "—", modeB: true },
};

// ──────────────────────────────────────── System / demo init
async function initSystem() {
  initDemo(); // independent of model loading (first /health call loads + hashes the weights)
  const badge = $("system-badge");
  try {
    const h = await api.health();
    const m = h.model;
    if (m?.available) {
      badge.textContent = `${m.name}@${m.version} · ${m.device}`;
      badge.className = "badge ok";
      const mm = m.metric_model;
      $("m-model").textContent = mm?.available
        ? `Depth Anything V2 Small (${m.name}@${m.version}, ${m.device}) for relative depth + fine-tuned metric nDSM model ${mm.name}@${mm.version} for GeoTIFF tiles (tier H heights)`
        : `Depth Anything V2 Small (${m.name}@${m.version}, ${m.device}) — zero-shot relative depth, whole image + overlapping tiles · no fine-tuned metric model installed (tier H unavailable)`;
      state.metricModel = !!mm?.available;
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
}

async function initDemo() {
  try {
    const d = await api.demo();
    state.demo = d.items; state.references = d.references;
    const wrap = $("demo-buttons"); wrap.innerHTML = "";
    // Indian scenes first (SIH / ISRO context), then the Swiss LiDAR benchmark tiles
    [...d.items].sort((x, y) => Number(y.country === "IN") - Number(x.country === "IN")).forEach((it) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = `demo-pill demo-mode-${it.mode.toLowerCase()}`;
      const isHD = it.id.includes("hd");
      const isRural = it.id.includes("rural");
      const india = it.country === "IN";
      const icon = india ? "🇮🇳" : isRural ? (it.mode === "B" ? "🌲" : "🏞️") : (it.mode === "B" ? "🏙️" : "📷");
      const title = india
        ? it.id.replace(/^india_/, "").split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ").replace(/^North Sikkim /, "N. Sikkim ")
        : isRural
          ? (it.mode === "B" ? "Emmental Ridge" : "Emmental Photo")
          : (it.mode === "B" ? (isHD ? "Zürich Core HD" : "Zürich City 2m") : (isHD ? "Zürich Photo HD" : "Zürich Photo"));
      const tag = india ? "0.5m · ICESat-2" : it.mode === "B" ? (isHD ? "0.5m DSM" : "2m DSM") : "Mode A";

      pill.innerHTML = `<span class="demo-icon">${icon}</span><span class="demo-name">${title}</span><span class="demo-tag">${tag}</span>`;
      pill.title = `${it.label} — ${it.source ?? ""}`;
      pill.addEventListener("click", () => {
        wrap.querySelectorAll(".demo-pill").forEach((c) => c.classList.remove("active"));
        pill.classList.add("active");
        loadDemo(it);
      });
      wrap.appendChild(pill);
    });
    const aw = $("anchor-demo-buttons"); aw.innerHTML = "";
    ["urban", "rural"].forEach((t, i) => {
      const b = document.createElement("button"); b.className = "linkbtn"; b.textContent = t;
      b.addEventListener("click", () => loadDemoAnchors(t)); aw.appendChild(b);
      if (i === 0) aw.append(" · ");
    });
    const sel = $("ref-bundled") as HTMLSelectElement;
    d.references.forEach((r) => { const o = document.createElement("option"); o.value = r; o.textContent = r; sel.appendChild(o); });
    const psel = $("pts-bundled") as HTMLSelectElement;
    d.items.filter((it) => it.reference_points).forEach((it) => { const o = document.createElement("option"); o.value = it.reference_points!; o.textContent = `ICESat-2 · ${it.id.replace(/^india_/, "")}`; psel.appendChild(o); });
  } catch { /* demo assets optional */ }
}

async function loadDemo(it: DemoItem) {
  try {
    const r = await fetch(`/demo/${it.file}`);
    if (!r.ok) throw new Error("demo missing");
    const blob = await r.blob();
    showInput(new File([blob], it.file.split("/").pop() ?? it.file, { type: blob.type || (it.mode === "B" ? "image/tiff" : "image/jpeg") }));
    ($("ref-bundled") as HTMLSelectElement).value = it.mode === "B" && it.reference_dsm ? it.reference_dsm : "";
    if (it.mode === "B" && it.reference_dsm) ($("ref-vcrs") as HTMLSelectElement).value = it.reference_vertical_crs ?? "same";
    ($("pts-bundled") as HTMLSelectElement).value = it.reference_points ?? "";
    if (it.anchors) {
      const type = it.anchors.includes("urban") ? "urban" : "rural";
      await loadDemoAnchors(type);
    } else if (state.anchorsLabel.startsWith("simulated")) {
      // demo anchors belong to their own tile; never carry them over to another scene
      state.anchors = null; state.anchorsLabel = ""; updateOptSummary();
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

// ──────────────────────────────────────── Input check (before running)
let checkSeq = 0;
async function runInputCheck() {
  const box = $("input-check");
  if (!state.file) { box.classList.add("hidden"); return; }
  const seq = ++checkSeq;
  box.className = "input-check"; box.innerHTML = `<div class="ic-head">Input check <span class="ic-badge">checking…</span></div>`;
  try {
    const r = await api.inspect(state.file, !!state.dem, !!state.anchors);
    if (seq !== checkSeq) return;
    const badge = { ok: "READY", warn: "USABLE WITH LIMITS", bad: "NOT SUITABLE" }[r.verdict];
    const icon = { ok: "✓", warn: "!", bad: "✕" };
    const facts = [
      r.gsd_m ? `${fmt(r.gsd_m, 2)} m/px` : "",
      r.extent_km ? `${fmt(r.extent_km[0], 2)} × ${fmt(r.extent_km[1], 2)} km` : "",
      r.width ? `${r.width}×${r.height} px` : "",
      `mode ${r.mode} → expected tier ${r.expected_tier}`,
    ].filter(Boolean).join(" · ");
    const a = r.expected_accuracy;
    const acc = a ? `<div class="ic-acc"><div class="ic-acc-h">Expected typical error (RMSE, measured)</div>
      ${a.height_above_ground ? `<div title="${esc(a.height_above_ground.source)}">Height above ground: ± ${fmt(a.height_above_ground.objects_m, 1)} m buildings/trees · ± ${fmt(a.height_above_ground.ground_m, 1)} m open ground</div>` : ""}
      ${a.terrain_m !== undefined ? `<div title="${esc(a.elevation_source ?? "")}">Terrain elevation: ± ${fmt(a.terrain_m, 1)} m · surface (DSM): ± ${fmt(a.dsm_m ?? NaN, 1)} m <span class="ic-src">(Sikkim vs ICESat-2)</span></div>` : ""}
    </div>` : "";
    box.className = `input-check ${r.verdict}`;
    box.innerHTML = `<div class="ic-head">Input check <span class="ic-badge">${badge}</span></div>
      <div class="ic-facts">${esc(facts)}</div>
      <ul class="ic-list">${r.checks.map((c) => `<li class="${c.level}"><span class="ic-i">${icon[c.level]}</span><div><b>${esc(c.title)}</b><div class="ic-d">${esc(c.detail)}</div></div></li>`).join("")}</ul>${acc}`;
  } catch (e) {
    if (seq !== checkSeq) return;
    box.className = "input-check bad";
    box.innerHTML = `<div class="ic-head">Input check <span class="ic-badge">NOT READABLE</span></div><div class="ic-d">${esc(userMessage(e))}</div>`;
  }
}

function updateOptSummary() {
  if (state.file) void runInputCheck();
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
    ? "B if the TIFF carries a CRS (GeoTIFF → DEM + calibrated model detail → metric DSM); otherwise A (relative)"
    : "A — Non-georeferenced → relative surface structure only (no units, no elevation)";
  $("m-geo").textContent   = tif ? "Checked at ingest (CRS + geotransform)" : "No";
  $("m-metric").textContent = tif ? "Horizontal: yes if georeferenced (GSD from CRS) · Vertical: decided after calibration" : "No — output is unitless relative structure";
  $("m-tier").textContent  = tif ? "T (DEM) / A (anchors) if georeferenced — determined after calibration" : "R (relative only)";
  ($("run-btn") as HTMLButtonElement).disabled = false;
  $("run-btn").textContent = tif ? "Generate Surface (GeoTIFF → Mode B)" : "Generate Relative Surface (Mode A)";
  ($("open3d-btn") as HTMLButtonElement).disabled = true;
  ($("validate-btn") as HTMLButtonElement).disabled = true;
  ($("validate-pts-btn") as HTMLButtonElement).disabled = true;
  ($("view-layer") as HTMLSelectElement).innerHTML = "";
  $("result-body").classList.add("hidden"); $("result-empty").classList.remove("hidden");
  $("val-result").classList.add("hidden");
  $("pts-result").classList.add("hidden");
  $("panel-buildings").classList.add("hidden");
  $("job-progress-wrap").classList.add("hidden");
  $("viewer-msg").textContent = "Generate surface to build 3D heightfield.";
  $("viewer-msg").classList.remove("hidden");
  if (state.viewer) state.viewer.clear();
  setStatus(tif ? "Ready — click Generate Calibrated Surface." : "Ready — click Generate Relative Surface.");
  void runInputCheck();
}

// ──────────────────────────────────────── Run pipeline
async function run() {
  if (!state.file) return;
  const btn = $("run-btn") as HTMLButtonElement; btn.disabled = true;
  ($("open3d-btn") as HTMLButtonElement).disabled = true;
  ($("validate-btn") as HTMLButtonElement).disabled = true;
  state.jobId = null;
  state.result = null;
  state.job = null;
  ($("view-layer") as HTMLSelectElement).innerHTML = "";
  $("viewer-msg").textContent = "Processing surface… 3D heightfield will load upon completion.";
  $("viewer-msg").classList.remove("hidden");
  if (state.viewer) state.viewer.clear();

  const progWrap = $("job-progress-wrap");
  const progFill = $("job-progress-fill") as HTMLElement;
  const progPct = $("job-progress-pct");
  const progEta = $("job-progress-eta");
  const progLabel = $("job-progress-label");
  const progStage = $("job-progress-stage");

  const formatEta = (seconds: number | null | undefined): string => {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "⏱️ estimating…";
    const s = Math.max(1, Math.round(seconds));
    if (s < 60) return `⏱️ ~${s}s left`;
    const m = Math.floor(s / 60);
    const remS = s % 60;
    return `⏱️ ~${m}m ${remS}s left`;
  };

  try {
    progWrap.classList.remove("hidden");
    progFill.style.width = "5%";
    progFill.style.background = "";
    progPct.textContent = "5%";
    progEta.textContent = "⏱️ estimating…";
    progStage.textContent = "UPLOADING";
    progLabel.textContent = "Uploading image…";

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
      INFERENCE:     "Depth Anything V2 Small — whole image + overlapping tiles (CPU: up to ~1 min)",
      CALIBRATION:   "DEM · datum · detail fusion · anchors · terrain · derivatives · LoD-1",
      RASTERIZING:   "rDSM + heightfield",
    };

    const done = await pollUntilDone(job.job_id, (j: Job) => {
      setStatus(`Processing: ${j.status}${stageLabels[j.status] ? " — " + stageLabels[j.status] : ""}`);
      if (j.progress) {
        const pct = Math.min(100, Math.max(0, j.progress.percent));
        progFill.style.width = `${pct}%`;
        progPct.textContent = `${pct}%`;
        progEta.textContent = formatEta(j.progress.eta_s);
        progStage.textContent = j.progress.stage || j.status;
        progLabel.textContent = j.progress.message || stageLabels[j.status] || "Processing…";
      } else {
        progStage.textContent = j.status;
        progLabel.textContent = stageLabels[j.status] || "Processing…";
      }
    });

    state.job = done;
    if (done.status === "FAILED") {
      progFill.style.width = "100%";
      progFill.style.background = "var(--bad)";
      progStage.textContent = "FAILED";
      progEta.textContent = "Failed";
      setStatus(`${done.error?.message ?? "Processing failed."} (${done.error?.code ?? "FAILED"})${done.error?.detail ? " — " + done.error.detail : ""}`, "err");
      return;
    }

    progFill.style.width = "100%";
    progPct.textContent = "100%";
    progEta.textContent = "✓ Done";
    progStage.textContent = "READY";
    progLabel.textContent = `Completed in ${done.stages_ms.total_ms ?? 0} ms`;
    setTimeout(() => {
      progWrap.classList.add("hidden");
    }, 5000);

    const res = await api.result(job.job_id);
    state.result = res;
    state.resultState = resolveState(res);
    await renderResult(done, res);

    // Automatically load 3D view once processing is done
    try {
      await open3d();
    } catch (err) {
      console.warn("Auto open3d notice:", err);
    }

    const ms = done.stages_ms;
    const summary = res.mode === "B"
      ? (res.calibration_tier === "R" ? `Tier R · quality ${res.quality} — no absolute elevation (see quality card).` : `Tier ${res.calibration_tier} · ${STATE_DISPLAY[state.resultState].label} · quality ${res.quality} · ${res.vertical_reference}`)
      : "Relative surface structure — no metres.";
    setStatus(
      `READY in ${ms.total_ms} ms (inference ${ms.inference_ms} ms on ${done.model?.device}${ms.calibration_ms ? `, calibration ${ms.calibration_ms} ms` : ""}). ${summary}`,
      res.quality === "INVALID" ? "err" : "ok"
    );
  } catch (e) { 
    progWrap.classList.add("hidden");
    setStatus(userMessage(e), "err"); 
  }
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
    ? `${res.object_scale_source}${res.object_scale_m_per_unit ? ` (median gain ${fmt(res.object_scale_m_per_unit, 2)} m/unit)` : ""}`
    : "none — DEM relief only";

  const fields: [string, string, boolean?][] = res.mode === "B" ? [
    ["Grid",        `${g.width}×${g.height} px`],
    ["GSD",         `${fmt(res.gsd_m, 2)} m`],
    ["CRS",         g.crs],
    ["DEM",         dem],
    ["Vertical ref",res.vertical_reference ?? "—"],
    ["Model detail",scaleStr, !res.object_scale_source],
    ["Tier",        res.calibration_tier],
  ] : [
    ["Grid",   `${g.width}×${g.height} px`],
    ["Output", "Unitless relative structure [0–1]"],
    ["Tier",   "R (relative only)"],
    ["Model",  "Depth Anything V2 Small — zero-shot"],
    ["Detail", res.tile_refinement?.applied ? `whole image + ${res.tile_refinement.n_tiles} native-resolution tiles` : "whole image"],
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
      <div class="trust-item"><div class="trust-label">Inference</div><div class="trust-value">${(res.flags ?? []).includes("METRIC_NDSM_MODEL") ? "Fine-tuned metric nDSM model, tiled at ~0.5 m" : "Zero-shot, tiled — no metric model"}</div></div>
      <div class="trust-item"><div class="trust-label">Input type</div><div class="trust-value">${res.mode === "B" ? "GeoTIFF (georeferenced)" : "PNG / JPEG (non-georeferenced)"}</div></div>
      <div class="trust-item"><div class="trust-label">Calibration tier</div><div class="trust-value ${res.calibration_tier === "A" ? "ok" : ""}">${res.calibration_tier}</div></div>
      ${res.mode === "B" ? `
      <div class="trust-item"><div class="trust-label">DEM source</div><div class="trust-value">${esc(dem)}</div></div>
      <div class="trust-item"><div class="trust-label">Vertical ref</div><div class="trust-value">${esc(res.vertical_reference ?? "—")}</div></div>
      <div class="trust-item"><div class="trust-label">Model detail</div><div class="trust-value ${!res.object_scale_source ? "warn" : "ok"}">${esc(scaleStr)}</div></div>
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
    ($("validate-pts-btn") as HTMLButtonElement).disabled = false;
  }

  // ── Default layer
  showLayer(res.mode === "B" ? "dsm" : "rdsm");

  // ── Downloads
  const dl = $("downloads"); dl.innerHTML = "<b>Download:</b> ";
  const files: [string, string][] = res.mode === "B"
    ? [["dsm.tif","DSM GeoTIFF"],["terrain.tif","Terrain layer"],["dem.tif","Input DEM (on job grid)"],["ndsm.tif","nDSM"],["slope.tif","Slope"],["aspect.tif","Aspect"],["flags.tif","Flags"],["relative.tif","Relative structure"],["calib_report.json","Calibration report"],["log.jsonl","Job log"]]
    : [["rdsm.tif","rDSM (no CRS)"],["relative_depth.npy","Raw model output"],["heightfield.f32","Heightfield (float32)"],["log.jsonl","Job log"]];
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
        ...(res.artifacts?.buildings_json ? [["city", "🏙️ LoD-1 Digital Twin (Extruded 3D Building Blocks)"] as [string, string]] : []),
        ["dsm", "🗺️ DSM Surface Grid (Continuous 2.5D Raster Mesh)"],
        ["terrain", "Terrain layer (metres, bare ground)"],
        ["ndsm", "nDSM (metres, height above ground — flat ground, buildings only)"],
        ["relative", "Relative structure (tier R, non-metric)"]
      ]
    : [["relative", "Relative structure (tier R, non-metric)"]];
  opts.filter(([k]) => k === "city" || res.layers?.[k]?.heightfield || (res.mode === "A" && k === "relative")).forEach(([k, l]) => {
    const o = document.createElement("option"); o.value = k; o.textContent = l; vl.appendChild(o);
  });
  state.viewLayer = vl.value;
  ($("open3d-btn") as HTMLButtonElement).disabled = false;

  // ── Building intelligence panel
  loadBuildingsPanel();

  // ── Disaster slider initialization based on terrain elevation
  const tLeg = res.layers?.terrain?.legend || res.layers?.dsm?.legend;
  const fSlider = $("flood-level-slider") as HTMLInputElement | null;
  const fVal = $("flood-level-val");
  if (fSlider && tLeg && typeof tLeg.lo === "number" && typeof tLeg.hi === "number") {
    const minElev = Math.floor(tLeg.lo);
    const maxElev = Math.ceil(tLeg.hi);
    fSlider.min = String(minElev);
    fSlider.max = String(maxElev + 10);
    fSlider.step = "0.5";
    const defaultVal = Math.min(maxElev, minElev + 2.0);
    fSlider.value = defaultVal.toFixed(1);
    if (fVal) fVal.textContent = defaultVal.toFixed(1);
    state.disaster.autoFloodVal = defaultVal;
    const minLbl = $("flood-min-lbl"); if (minLbl) minLbl.textContent = `${minElev} m`;
    const maxLbl = $("flood-max-lbl"); if (maxLbl) maxLbl.textContent = `${maxElev + 10} m`;
  }
  initDisasterMap();

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
    const unc = v.valid && v.uncertainty_m !== undefined
      ? `<div class="ru" title="${esc(v.uncertainty_note ?? "")}">± ${fmt(v.uncertainty_m, 1)} m typical</div>`
      : v.valid && v.metric && v.units === "m" ? `<div class="ru none" title="No held-out measurement backs this layer for this job">± not measured</div>` : "";
    return `<tr>
      <td><div class="rk">${esc(k)}</div>${meta ? `<div class="rq">${esc(meta)}</div>` : ""}</td>
      <td class="rv">${esc(valStr)}${unc}</td>
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

async function runPointValidation() {
  if (!state.jobId || !state.result) return;
  const btn = $("validate-pts-btn") as HTMLButtonElement; btn.disabled = true;
  const file = ($("pts-input") as HTMLInputElement).files?.[0] ?? null;
  const bundled = ($("pts-bundled") as HTMLSelectElement).value || null;
  try {
    if (!file && !bundled) { setStatus("Choose a checkpoint CSV or bundled ICESat-2 checkpoints.", "err", "val-status"); return; }
    setStatus("Validating against checkpoints (datum-converted, 10 m footprint)…", "", "val-status");
    const v = await api.validatePoints(state.jobId, { points: file, bundled: file ? null : bundled });
    renderPointValidation(v);
    setStatus(`Checkpoint validation complete: ${v.n_in_grid} checkpoints in the scene.`, "ok", "val-status");
  } catch (e) { setStatus(userMessage(e), "err", "val-status"); }
  finally { btn.disabled = false; }
}

function renderPointValidation(v: Record<string, any>) {
  $("pts-result").classList.remove("hidden");
  const m = v.metrics ?? {};
  const t = m.terrain_vs_ground, b = m.input_dem_vs_ground;
  $("pts-title").textContent = t && b
    ? `Terrain vs independent checkpoints: RMSE ${fmt(t.RMSE)} m (input DEM alone: ${fmt(b.RMSE)} m) over ${t.n} points`
    : "Checkpoint comparison";
  $("pts-context").textContent = `${v.source ?? ""} · ${v.datum_handling} · job tier ${v.job?.calibration_tier ?? "—"}, quality ${v.job?.quality ?? "—"}`;
  const rows: [string, string][] = [
    ["input_dem_vs_ground", "Input DEM vs checkpoint ground (baseline)"],
    ["terrain_vs_ground", "DepthWizard terrain vs checkpoint ground"],
    ["input_dem_vs_top_of_surface", "Input DEM vs checkpoint top of surface (baseline)"],
    ["dsm_vs_top_of_surface", "DepthWizard DSM vs checkpoint top of surface"],
    ["ndsm_vs_canopy_height", "DepthWizard nDSM vs checkpoint canopy / structure height"],
  ];
  const head = `<thead><tr><th>Comparison</th><th>n</th><th>ME (m)</th><th>RMSE (m)</th><th>MAE (m)</th><th>NMAD (m)</th><th>LE90</th><th>LE95</th><th>r</th><th>ρ</th></tr></thead>`;
  $("pts-table").innerHTML = head + "<tbody>" + rows.filter(([k]) => m[k]?.RMSE !== undefined).map(([k, label]) => `<tr><td>${esc(label)}</td>${metricRow(m[k])}</tr>`).join("") + "</tbody>";
  const strata = Object.entries(v.strata ?? {}).map(([name, sm]: [string, any]) => `${esc(name)}: ` + Object.entries(sm).map(([k, x]: [string, any]) => `${esc(k)} RMSE ${fmt(x.RMSE)} m (n=${x.n})`).join(" · "));
  $("pts-strata").innerHTML = [...strata, ...(v.caveats ?? []).map((c: string) => `⚠ ${esc(c)}`)].join("<br>");
}

function renderValidation(v: Record<string, any>) {
  $("val-result").classList.remove("hidden");
  const head = `<thead><tr><th>Set</th><th>n</th><th>ME (m)</th><th>RMSE (m)</th><th>MAE (m)</th><th>NMAD (m)</th><th>LE90</th><th>LE95</th><th>r</th><th>ρ</th></tr></thead>`;
  $("val-verdict-title").textContent = `${v.verdict?.band ?? ""} — ${v.verdict?.text ?? ""}${v.verdict?.baseline_text ? "  " + v.verdict.baseline_text : ""}`;
  $("val-context").innerHTML = `
    Compared layer: <b>${esc(v.compared_layer)}</b> (${esc(v.reference?.ref_type ?? "")}).
    Reference posting: ${v.reference?.native_posting_m?.[0]} m → reprojected to job grid.
    Datum handling: ${esc(v.reference?.datum_handling ?? "—")}.
    Job tier: ${esc(v.job?.calibration_tier ?? "—")}, quality: ${esc(v.job?.quality ?? "—")}.
  `;
  $("val-overall").innerHTML = `${head}<tbody>
    <tr><td><b>DepthWizard ${esc(v.compared_layer ?? "")}</b> (metres)</td>${metricRow(v.metrics_overall)}</tr>
    ${v.metrics_baseline ? `<tr><td>Input DEM alone — baseline, same pixels</td>${metricRow(v.metrics_baseline)}</tr>` : ""}
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

// ──────────────────────────────────────── Building intelligence
let bldRows: Record<string, any>[] = [];
let bldTimer = 0;

async function loadBuildingsPanel() {
  const panel = $("panel-buildings");
  if (!state.jobId || !state.result?.artifacts?.buildings_json) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const minH = Number(($("bld-minh") as HTMLInputElement).value);
  const minA = Number(($("bld-mina") as HTMLInputElement).value) || 0;
  $("bld-minh-val").textContent = `${minH} m`;
  const q = `min_height=${minH}&min_area=${minA}`;
  ($("bld-geojson") as HTMLAnchorElement).href = `/api/jobs/${state.jobId}/buildings.geojson?${q}`;
  ($("bld-csv") as HTMLAnchorElement).href = `/api/jobs/${state.jobId}/buildings.csv?${q}`;
  try {
    const r = await api.buildings(state.jobId, minH, minA, 300);
    const s = r.summary; bldRows = r.buildings;
    const err = s.height_error ?? {};
    const errTxt = err.typical_m ? `±${fmt(err.typical_m, 1)} m typical (1 RMSE, held-out LiDAR)` : "not calibrated (zero-shot)";
    const cls = s.height_classes ?? {};
    const fields: [string, string][] = [
      ["Buildings", `${s.count_filtered} of ${s.count_total}`],
      ["Tallest / median", `${fmt(s.max_height_m, 1)} m / ${fmt(s.median_height_m, 1)} m`],
      ["Low · mid · high-rise", `${cls.low_lt10m ?? 0} · ${cls.mid_10_25m ?? 0} · ${cls.high_ge25m ?? 0}`],
      ["Footprint · volume", `${(s.total_footprint_m2 / 1e4).toFixed(2)} ha · ${(s.total_volume_m3 / 1e6).toFixed(2)} Mm³`],
      ["Height error", errTxt],
    ];
    $("bld-summary").innerHTML = fields.map(([l, v]) => `<div class="result-field"><div class="result-field-label">${l}</div><div class="result-field-value">${esc(v)}</div></div>`).join("");
    $("bld-table").innerHTML = `<thead><tr><th>#</th><th>Height (m)</th><th>Roof spread p10–p90</th><th>Floors (approx.)</th><th>Footprint (m²)</th><th>Volume (m³)</th><th>Ground elev. (m)</th></tr></thead><tbody>` +
      bldRows.map((b) => `<tr data-bid="${b.id}" style="cursor:pointer"><td>${b.id}</td><td><b>${fmt(b.height_m, 1)}</b>${b.height_interval_m ? ` <span class="hint">(${fmt(b.height_interval_m[0], 0)}–${fmt(b.height_interval_m[1], 0)})</span>` : ""}</td><td>${fmt(b.height_p10_m, 1)}–${fmt(b.height_p90_m, 1)}</td><td>${b.floors_range[0]}–${b.floors_range[1]}</td><td>${fmt(b.area_m2, 0)}</td><td>${b.volume_m3.toLocaleString()}</td><td>${fmt(b.ground_elev_m, 1)}</td></tr>`).join("") + "</tbody>";
    $("bld-notes").innerHTML = [err.source ? `Height error source: ${esc(err.source)}` : "", ...(s.notes ?? []).map((n: string) => esc(n))].filter(Boolean).join("<br>");
    $("bld-table").querySelectorAll<HTMLTableRowElement>("tr[data-bid]").forEach((tr) => tr.addEventListener("click", () => selectBuilding(Number(tr.dataset.bid), true)));
  } catch (e) { $("bld-summary").textContent = userMessage(e); }
}

function selectBuilding(id: number, fly: boolean) {
  const b = bldRows.find((x) => x.id === id);
  $("bld-table").querySelectorAll<HTMLTableRowElement>("tr[data-bid]").forEach((tr) => tr.classList.toggle("active", Number(tr.dataset.bid) === id));
  $("bld-table").querySelector<HTMLTableRowElement>(`tr[data-bid="${id}"]`)?.scrollIntoView({ block: "nearest" });
  const d = $("bld-detail");
  if (!b) { d.classList.remove("hidden"); d.innerHTML = `Building #${id} is outside the current filter.`; return; }
  d.classList.remove("hidden");
  d.innerHTML = `<div class="quality-card-header"><span class="q-badge q-GOOD">BUILDING #${b.id}</span><span class="quality-card-title">${fmt(b.height_m, 1)} m tall${b.height_interval_m ? ` (typical range ${fmt(b.height_interval_m[0], 0)}–${fmt(b.height_interval_m[1], 0)} m)` : ""} · approx. ${b.floors_range[0]}–${b.floors_range[1]} floors</span></div>
    <ul class="quality-triggers"><li>Ground ${fmt(b.ground_elev_m, 1)} m · roof ${fmt(b.roof_elev_m, 1)} m (${esc(state.result?.vertical_reference ?? "")})</li>
    <li>Footprint ${fmt(b.area_m2, 0)} m² · volume ≈ ${b.volume_m3.toLocaleString()} m³ · roof height spread ${fmt(b.height_p10_m, 1)}–${fmt(b.height_p90_m, 1)} m</li>
    <li>Location ${fmt(b.lat, 5)}° N, ${fmt(b.lon, 5)}° E</li></ul>`;
  if (fly && state.viewer) { state.viewer.setBuildingsVisible(true); ($("lod1-chk") as HTMLInputElement).checked = true; state.viewer.focusBuilding(id); }
}

// ──────────────────────────────────────── 3D viewer
async function open3d() {
  if (!state.jobId || !state.result) return;
  const msg = $("viewer-msg");
  if (state.job && state.job.status !== "READY") {
    msg.textContent = "Processing in progress… 3D view will be ready once job completes.";
    msg.classList.remove("hidden");
    return;
  }
  msg.classList.remove("hidden");
  try {
    if (!state.viewer) {
      msg.textContent = "Initialising WebGL…";
      state.viewer = new HeightfieldViewer($("viewer"));
      state.viewer.onPick(pickAt);
      state.viewer.onBuildingPick((id) => selectBuilding(id, false));
      state.viewer.onCameraModeChange((mode) => {
        $("cam-orbit-btn").classList.toggle("active", mode === "orbit");
        $("cam-walk-btn").classList.toggle("active", mode === "walk");
        $("cam-fly-btn").classList.remove("active");
        $("walk-hud").classList.toggle("hidden", mode !== "walk");
      });
      state.viewer.onFlythroughToggle((active) => {
        $("cam-fly-btn").classList.toggle("active", active);
        if (active) {
          $("cam-orbit-btn").classList.remove("active");
          $("cam-walk-btn").classList.remove("active");
        } else {
          $("cam-orbit-btn").classList.add("active");
        }
      });
      state.viewer.onWalkStats((eyeZ, speedKmH) => {
        $("walk-hud-stats").textContent = `${eyeZ.toFixed(2)} m Eye Height · ${speedKmH.toFixed(1)} km/h`;
      });
    }
    const res = state.result;
    const layer = ($("view-layer") as HTMLSelectElement).value || "relative";
    const isCity = layer === "city";
    // City mode: use terrain layer (smooth bare ground, no building spikes from nDSM).
    // LoD-1 buildings sit on top with correct absolute elevation alignment.
    const actualLayer = isCity ? "terrain" : layer;
    const metric = res.mode === "B" && layer !== "relative";
    const hfName = res.mode === "B"
      ? (res.layers?.[actualLayer]?.heightfield ?? "heightfield_relative.f32")
      : (res.artifacts?.heightfield ?? "heightfield.f32");
    const metaName = res.mode === "B"
      ? (res.layers?.[actualLayer]?.heightfield_meta ?? "heightfield_relative.json")
      : null;

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
      layer:    isCity ? "3D City (LoD-1)" : (LAYER_DEFS[layer]?.label ?? layer.toUpperCase()),
      showNorth: metric,   // only meaningful for georeferenced output
    };
    state.currentHud = hudInfo;

    // Default to true scale 1.0× so building heights are true-to-life architectural proportions
    const exagEl = $("exag") as HTMLInputElement;
    const currentExag = Number(exagEl.value) || 1.0;
    const exag = isCity ? 1.0 : (currentExag <= 1.0 ? 1.0 : currentExag);
    exagEl.value = String(exag);
    $("exag-val").textContent = `×${exag.toFixed(1)}${Math.abs(exag - 1) < 1e-6 ? " (true scale)" : ""}`;
    const info = await state.viewer.load(hf, meta, api.artifactUrl(state.jobId, res.artifacts.texture), {
      metric, spacing, exaggeration: exag, hud: hudInfo,
    });

    msg.textContent = ""; msg.classList.add("hidden");

    // Show/hide appropriate controls
    $("zscale-wrap").classList.toggle("hidden", metric);
    $("exag-wrap").classList.toggle("hidden", !metric);

    // Reset camera mode, presets & shader UI toggles to defaults
    $("cam-orbit-btn").classList.add("active");
    $("cam-walk-btn").classList.remove("active");
    $("cam-fly-btn").classList.remove("active");
    $("walk-hud").classList.add("hidden");
    $("preset-nadir-btn").classList.remove("active");
    $("preset-oblique-btn").classList.add("active");
    $("preset-horizon-btn").classList.remove("active");
    $("shader-aerial-btn").classList.add("active");
    $("shader-heatmap-btn").classList.remove("active");
    $("shader-cyber-btn").classList.remove("active");
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
    // LoD-1 blocks carry absolute base elevations: only meaningful over absolute surfaces (terrain / DSM)
    if (res.artifacts?.buildings_json && (isCity || layer === "dsm" || layer === "terrain")) {
      try {
        const bData = await fetch(api.artifactUrl(state.jobId, res.artifacts.buildings_json)).then((r) => r.json());
        state.viewer.loadBuildings(bData, isCity);
        lod1Wrap.classList.remove("hidden");
        const showBuildings = isCity || lod1Chk.checked;
        lod1Chk.checked = showBuildings;
        state.viewer.setBuildingsVisible(showBuildings);
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

// ──────────────────────────────────────── Disaster 2D Map & Hydrological Screening
let disasterDebounceTimer: any = null;

function initDisasterMap() {
  if (!state.result || !state.jobId) return;
  const placeholder = $("disaster-map-placeholder");
  if (placeholder) placeholder.classList.add("hidden");

  updateDisasterBaseMap();

  // Reset pick dot and inspector
  const pickDot = $("disaster-pick-dot");
  if (pickDot) pickDot.classList.add("hidden");
  resetHudInfo();

  // Auto-run baseline screening if Mode B and terrain exists
  if (state.result.mode === "B" && (state.result.artifacts.terrain_tif || state.result.artifacts.dsm_tif)) {
    runDisasterAnalysis(true);
  }
}

function updateDisasterBaseMap() {
  if (!state.result || !state.jobId) return;
  const baseImg = $("disaster-base-img") as HTMLImageElement;
  if (!baseImg) return;

  const key = state.disaster.baseLayer;
  let art = "input_preview";
  if (key === "terrain" && state.result.artifacts.terrain_preview) art = "terrain_preview";
  else if (key === "hillshade" && state.result.artifacts.hillshade_preview) art = "hillshade_preview";
  else if (key === "dsm" && state.result.artifacts.dsm_preview) art = "dsm_preview";
  else if (state.result.artifacts.input_preview) art = "input_preview";
  else art = Object.keys(state.result.artifacts)[0] || "input_preview";

  const filename = state.result.artifacts[art] || "input_preview.png";
  baseImg.src = api.artifactUrl(state.jobId, filename);

  document.querySelectorAll("#disaster-base-group button").forEach((btn) => {
    btn.classList.toggle("active", (btn as HTMLButtonElement).dataset.base === key);
  });
}

async function runDisasterAnalysis(silent = false) {
  if (!state.jobId || !state.result) {
    if (!silent) alert("Please generate a surface first.");
    return;
  }
  const runBtn = $("run-disaster-btn") as HTMLButtonElement;
  if (!silent && runBtn) runBtn.textContent = "Screening…";

  try {
    const disScen = $("disaster-scenario") as HTMLSelectElement;
    const mode = disScen ? disScen.value : "flood";
    const fLvl = $("flood-level-slider") as HTMLInputElement;
    const aSlp = $("access-slope-slider") as HTMLInputElement;
    const body = mode === "flood" ? { waterLevel_m: Number(fLvl.value || 0) }
      : mode === "landing_zones" ? { size: Number(($("hlz-size") as HTMLSelectElement).value), excludeFlooded: ($("hlz-exclude-flooded") as HTMLInputElement).checked }
      : { maxSlopeDeg: Number(aSlp.value || 15) };

    const res = await fetch(`/api/jobs/${state.jobId}/disaster/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const resp = await res.json();
    state.disaster.lastResult = resp;

    $("disaster-result").classList.remove("hidden");
    const placeholder = $("disaster-map-placeholder");
    if (placeholder) placeholder.classList.add("hidden");

    $("hlz-sites-wrap").classList.toggle("hidden", mode !== "landing_zones");
    $("disaster-download-extra").classList.toggle("hidden", mode !== "landing_zones");
    if (mode !== "landing_zones") $("hlz-vector-svg").innerHTML = "";

    if (mode === "flood") {
      const iso = resp.isolatedAreaM2 ? ` · ${fmtArea(resp.isolatedAreaM2)} isolated` : "";
      setStat("stat-area-card", "Inundated Area", fmtArea(resp.affectedAreaM2), `${resp.affectedAreaPct}% of valid terrain${iso}`);
      setStat("stat-buildings-card", "Affected Buildings", String(resp.affectedBuildingsCount ?? 0), `exposed · ${resp.contactBuildingsCount ?? 0} in contact`);
      setStat("stat-maxdepth-card", "Max Depth", `${resp.maxDepth_m} m`, "deepest flooded cell");
      setStat("stat-meandepth-card", "Mean Depth", `${resp.meanDepth_m} m`, "over flooded area");

      const dlLink = $("disaster-download-raster") as HTMLAnchorElement;
      if (dlLink) {
        dlLink.href = `/api/jobs/${state.jobId}/artifact/${resp.rasterResult}`;
        dlLink.textContent = "Download depth raster (GeoTIFF)";
      }

      const bWrap = $("disaster-buildings-wrap");
      const bList = $("disaster-buildings-list");
      const bCount = $("disaster-bldg-count");
      if (bCount) bCount.textContent = String(resp.affectedBuildingsCount ?? 0);

      if (resp.buildings && resp.buildings.length > 0) {
        bWrap.classList.remove("hidden");
        bList.innerHTML = resp.buildings.map((b: any) => `
          <div class="disaster-bldg-item" data-bldg-id="${b.id}">
            <div>
              <b>Building #${b.id}</b>
              <span class="hint" style="margin-left: 6px;">Ground ${b.base_elev_m} m · depth <span class="depth-val">${b.flood_depth_m} m</span>${b.wet_fraction != null ? ` · ${Math.round(b.wet_fraction * 100)}% wet` : ""}</span>
            </div>
            <span class="badge ${exposureBadge(b.exposure)}" style="font-size: 10px;">${b.exposure}</span>
          </div>
        `).join("");

        bList.querySelectorAll(".disaster-bldg-item").forEach((item) => {
          const el = item as HTMLElement;
          const id = Number(el.dataset.bldgId);
          el.addEventListener("mouseenter", () => highlightBuildingOnMap(id));
          el.addEventListener("mouseleave", () => unhighlightBuildingOnMap(id));
          el.addEventListener("click", () => focusBuildingOnMap(id));
        });
      } else {
        bWrap.classList.add("hidden");
      }

      // legend drawn from the backend's ramp + class definitions (single source of truth)
      const ramp = resp.previewRamp ?? { colours: ["#add8e6", "#00bfff", "#0000cd", "#000080"], stops_m: [0, 1.67, 3.33, 5] };
      $("disaster-legend-title").textContent = "Water depth above terrain (m)";
      $("disaster-ramp-bar").style.background = `linear-gradient(to right, ${ramp.colours.join(", ")})`;
      $("disaster-ramp-labels").innerHTML = ramp.stops_m.map((v: number, i: number) => `<span>${v.toFixed(1)}${i === ramp.stops_m.length - 1 ? "+" : ""}</span>`).join("");
      $("disaster-legend-classes").innerHTML = (resp.exposureRules ?? []).map((r: any) =>
        `<span class="l-item">${r.label} ${r.le_m == null ? `> ${r.gt_m}` : `≤ ${r.le_m}`}</span>`).join("");
      $("disaster-toggle-lbl").textContent = "Water overlay";
    } else if (mode === "landing_zones") {
      const r = resp.rules;
      const nClear = resp.sites.reduce((n: number, s: any) => n + s.clearBearings.length, 0);
      setStat("stat-area-card", "Candidate sites", String(resp.nSites), `${resp.nCandidate} with a clear approach · ${resp.blockedSites} rejected, all approaches blocked`);
      setStat("stat-buildings-card", "Pad", `${resp.padDiameterM} m`, `Size ${resp.size} · slope ≤ ${fmtNum(r.slopeMaxDegApplied - r.slopeMarginDeg)}° measured`);
      setStat("stat-maxdepth-card", "Feasible pad centres", fmtArea(resp.feasibleCentreAreaM2), `pad clear of detected obstacles + ${r.objectBufferM} m`);
      setStat("stat-meandepth-card", "Clear approach bearings", String(nClear), `of ${r.bearings} per site · 10:1 checked to ${r.approachLengthM} m`);

      const dlLink = $("disaster-download-raster") as HTMLAnchorElement;
      dlLink.href = resp.vectorResult ? `/api/jobs/${state.jobId}/artifact/${resp.vectorResult}` : "#";
      dlLink.textContent = "Download sites (GeoJSON, WGS84)";
      const dlExtra = $("disaster-download-extra") as HTMLAnchorElement;
      dlExtra.href = `/api/jobs/${state.jobId}/artifact/${resp.rasterResult}`;
      dlExtra.textContent = "Download reason raster (GeoTIFF)";

      $("disaster-buildings-wrap").classList.add("hidden");
      $("hlz-site-count").textContent = String(resp.nSites);
      const list = $("hlz-sites-list");
      list.innerHTML = resp.sites.length ? resp.sites.map((s: any) => `
        <div class="disaster-bldg-item" data-hlz-id="${s.id}">
          <div>
            <b>Site ${s.id}</b>
            <span class="hint hlz-item-meta">slope ${fmtNum(s.slopeDeg)}° · rough ${fmtNum(s.roughnessM)} m · ${s.clearBearings.length ? `clear ${s.clearBearings.map((b: number) => `${b}°`).join(", ")}` : "no clear bearing"}</span>
          </div>
          <span class="badge ${s.class === "CANDIDATE" ? "ok" : "warn"} badge-xs">${s.class}</span>
        </div>`).join("") : `<div class="hint hlz-empty">No site meets the pad rules in this scene.</div>`;
      list.querySelectorAll<HTMLElement>(".disaster-bldg-item").forEach((el) => {
        const id = Number(el.dataset.hlzId);
        el.addEventListener("mouseenter", () => highlightSite(id, true));
        el.addEventListener("mouseleave", () => highlightSite(id, false));
        el.addEventListener("click", () => showHudSiteInfo(id));
      });

      $("disaster-legend-title").textContent = "Landing-zone screening";
      $("disaster-ramp-bar").style.background = "rgba(29, 122, 79, 0.47)";
      $("disaster-ramp-labels").innerHTML = `<span>Feasible pad centres</span><span>${r.obstacleRatio}:1 · ${r.approachLengthM} m</span>`;
      $("disaster-legend-classes").innerHTML = `
        <span class="l-item"><i class="hlz-sw-candidate"></i>Candidate</span>
        <span class="l-item"><i class="hlz-sw-marginal"></i>Marginal</span>
        <span class="l-item"><i class="hlz-sw-clear"></i>Clear bearing</span>
        <span class="l-item"><i class="hlz-sw-unverified"></i>Unverified</span>
        <span class="l-item"><i style="background:#5a5a5a"></i>Detected obstacle</span>
      `;
      $("disaster-toggle-lbl").textContent = "Feasible-centre overlay";
    } else {
      $("stat-area-card").querySelector(".stat-label")!.textContent = "Accessible Area";
      $("stat-area-val").textContent = fmtArea(resp.accessibleAreaM2);
      $("stat-area-sub").textContent = `${resp.accessiblePctOfGround ?? "—"}% of open ground · buildings excluded`;

      $("stat-buildings-card").classList.add("hidden");
      $("stat-maxdepth-card").classList.add("hidden");
      $("stat-meandepth-card").classList.add("hidden");

      const dlLink = $("disaster-download-raster") as HTMLAnchorElement;
      if (dlLink) {
        dlLink.href = `/api/jobs/${state.jobId}/artifact/${resp.rasterResult}`;
        dlLink.textContent = "Download accessibility mask (GeoTIFF)";
      }

      $("disaster-buildings-wrap").classList.add("hidden");

      $("disaster-legend-title").textContent = "Terrain Accessibility";
      $("disaster-ramp-bar").style.background = "linear-gradient(to right, #22c55e, #ef4444)";
      $("disaster-ramp-labels").innerHTML = `<span>≤ ${resp.maxSlopeDeg}° (Safe)</span><span>> ${resp.maxSlopeDeg}° (Hazard)</span>`;
      $("disaster-legend-classes").innerHTML = `
        <span class="l-item"><i style="background:#22c55e"></i>Accessible</span>
        <span class="l-item"><i style="background:#ef4444"></i>Steeper</span>
        <span class="l-item"><i style="background:#5a5a5a"></i>Building</span>
      `;
      $("disaster-toggle-lbl").textContent = "Access overlay";
    }

    // Warnings
    const wDiv = $("disaster-warnings");
    if (resp.warnings && resp.warnings.length > 0) {
      wDiv.innerHTML = resp.warnings.map((w: string) => esc(w)).join("<br>");
      wDiv.classList.remove("hidden");
    } else {
      wDiv.classList.add("hidden");
    }

    // Update 2D Water Overlay Image on top of the 2D Map!
    if (resp.previewResult) {
      const overlayImg = $("disaster-overlay-img") as HTMLImageElement;
      if (overlayImg) {
        overlayImg.src = `/api/jobs/${state.jobId}/artifact/${resp.previewResult}?t=${Date.now()}`;
        overlayImg.style.display = state.disaster.overlayVisible ? "block" : "none";
        overlayImg.style.opacity = String(state.disaster.opacity);
        overlayImg.classList.toggle("water-shimmer", state.disaster.shimmer && mode === "flood");
      }
    }

    // Render building vector layer on 2D map
    renderDisasterSvg(resp.buildings);
    if (mode === "landing_zones") renderLandingSvg(resp.sites);

  } catch (e: any) {
    if (!silent) alert("Disaster Analysis Failed: " + (e.message || String(e)));
  } finally {
    if (!silent && runBtn) runBtn.textContent = "Run screening";
  }
}

function renderDisasterSvg(buildings?: any[]) {
  const svg = $("disaster-vector-svg") as unknown as SVGSVGElement;
  if (!svg || !state.result) return;
  const grid = state.result.grid;
  const W = Number(grid.width) || 100;
  const H = Number(grid.height) || 100;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";

  if (!state.disaster.showBuildings || !buildings || buildings.length === 0) return;

  const gsd = state.result.gsd_m || 0.5;
  const extX2 = ((W - 1) * gsd) / 2.0;
  const extY2 = ((H - 1) * gsd) / 2.0;

  buildings.forEach((b: any) => {
    let pts = "";
    if (b.pixel_coords && Array.isArray(b.pixel_coords) && b.pixel_coords.length >= 3) {
      pts = b.pixel_coords.map((c: [number, number]) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" ");
    } else if (b.coords && Array.isArray(b.coords) && b.coords.length >= 3) {
      // Scene coordinates are in centered meters: convert back to image pixel coordinates (col, row)
      pts = b.coords.map((c: [number, number]) => {
        const col = (c[0] + extX2) / gsd;
        const row = (extY2 - c[1]) / gsd;
        return `${col.toFixed(1)},${row.toFixed(1)}`;
      }).join(" ");
    } else if (b.pixel_bbox && b.pixel_bbox.length === 4) {
      const [x0, y0, x1, y1] = b.pixel_bbox;
      pts = `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1}`;
    }
    if (!pts) return;

    const isFlood = (b.flood_depth_m ?? 0) > 0;
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", pts);
    poly.setAttribute("class", isFlood ? "disaster-bldg-polygon" : "disaster-bldg-safe");
    poly.setAttribute("data-bldg-id", String(b.id));

    poly.addEventListener("mouseenter", () => {
      highlightBuildingOnMap(b.id);
      showHudBuildingInfo(b);
    });
    poly.addEventListener("mouseleave", () => {
      unhighlightBuildingOnMap(b.id);
      resetHudInfo();
    });
    poly.addEventListener("click", (e) => {
      e.stopPropagation();
      focusBuildingOnMap(b.id);
    });

    svg.appendChild(poly);
  });
}

function highlightBuildingOnMap(id: number) {
  const poly = document.querySelector(`.disaster-bldg-polygon[data-bldg-id="${id}"], .disaster-bldg-safe[data-bldg-id="${id}"]`);
  if (poly) poly.classList.add("highlight");
  const listItem = document.querySelector(`.disaster-bldg-item[data-bldg-id="${id}"]`);
  if (listItem) listItem.classList.add("active");
}

function unhighlightBuildingOnMap(id: number) {
  const poly = document.querySelector(`.disaster-bldg-polygon[data-bldg-id="${id}"], .disaster-bldg-safe[data-bldg-id="${id}"]`);
  if (poly) poly.classList.remove("highlight");
  const listItem = document.querySelector(`.disaster-bldg-item[data-bldg-id="${id}"]`);
  if (listItem) listItem.classList.remove("active");
}

function focusBuildingOnMap(id: number) {
  highlightBuildingOnMap(id);
  const listItem = document.querySelector(`.disaster-bldg-item[data-bldg-id="${id}"]`);
  if (listItem) listItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
  if (state.disaster.lastResult?.buildings) {
    const b = state.disaster.lastResult.buildings.find((x: any) => x.id === id);
    if (b) showHudBuildingInfo(b);
  }
}

function showHudBuildingInfo(b: any) {
  const hud = $("disaster-hud-content");
  if (!hud) return;
  const isFlood = (b.flood_depth_m ?? 0) > 0;
  hud.innerHTML = `
    <b>Building #${b.id}</b><br/>
    Ground ${b.base_elev_m} m${b.height_m ? ` · roof height ${b.height_m} m` : ""}<br/>
    Status: ${isFlood
      ? `<span class="hud-risk">Exposed — ${b.flood_depth_m} m on the low side${b.wet_fraction != null ? `, ${Math.round(b.wet_fraction * 100)}% of footprint wet` : ""}</span>`
      : `<span class="hud-dry">Above the water level</span>`}<br/>
    Exposure: <span class="badge ${exposureBadge(b.exposure)}">${b.exposure}</span>
  `;
}

function setStat(cardId: string, label: string, value: string, sub: string) {
  const c = $(cardId);
  c.classList.remove("hidden");
  c.querySelector(".stat-label")!.textContent = label;
  c.querySelector(".stat-value")!.textContent = value;
  c.querySelector(".stat-sub")!.textContent = sub;
}

function fmtNum(v: number): string {
  return String(Math.round(v * 100) / 100);
}

/** Pads and non-blocked approach corridors; the geometry comes from the API in pixel coordinates. */
function renderLandingSvg(sites: any[]) {
  const svg = $("hlz-vector-svg") as unknown as SVGSVGElement;
  if (!svg || !state.result) return;
  const g = state.result.grid;
  svg.setAttribute("viewBox", `0 0 ${Number(g.width) || 100} ${Number(g.height) || 100}`);
  const ns = "http://www.w3.org/2000/svg";
  svg.innerHTML = "";
  for (const s of sites) {
    const grp = document.createElementNS(ns, "g");
    grp.setAttribute("data-hlz-id", String(s.id));
    grp.setAttribute("class", `hlz-site ${s.class === "CANDIDATE" ? "hlz-candidate" : "hlz-marginal"}`);
    for (const a of s.approachPixelLines) {
      const ln = document.createElementNS(ns, "line");
      const [[x1, y1], [x2, y2]] = a.line;
      ln.setAttribute("x1", String(x1)); ln.setAttribute("y1", String(y1));
      ln.setAttribute("x2", String(x2)); ln.setAttribute("y2", String(y2));
      ln.setAttribute("class", a.status === "CLEAR" ? "hlz-appr-clear" : "hlz-appr-unverified");
      grp.appendChild(ln);
    }
    const pad = document.createElementNS(ns, "polygon");
    pad.setAttribute("points", s.padPixelRing.map((p: number[]) => `${p[0]},${p[1]}`).join(" "));
    pad.setAttribute("class", "hlz-pad");
    pad.addEventListener("mouseenter", () => highlightSite(s.id, true));
    pad.addEventListener("mouseleave", () => highlightSite(s.id, false));
    pad.addEventListener("click", (e) => { e.stopPropagation(); showHudSiteInfo(s.id); });
    grp.appendChild(pad);
    svg.appendChild(grp);
  }
}

function highlightSite(id: number, on: boolean) {
  document.querySelector(`#hlz-vector-svg g[data-hlz-id="${id}"]`)?.classList.toggle("highlight", on);
  document.querySelector(`#hlz-sites-list .disaster-bldg-item[data-hlz-id="${id}"]`)?.classList.toggle("active", on);
}

function showHudSiteInfo(id: number) {
  const resp = state.disaster.lastResult;
  const s = resp?.sites?.find((x: any) => x.id === id);
  const hud = $("disaster-hud-content");
  if (!s || !hud) return;
  document.querySelector(`#hlz-sites-list .disaster-bldg-item[data-hlz-id="${id}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const byStatus = (st: string) => s.approaches.filter((a: any) => a.status === st);
  const blocked = byStatus("BLOCKED");
  const nearest = blocked.reduce((m: any, a: any) => (!m || a.firstObstacle.distanceM < m.firstObstacle.distanceM ? a : m), null);
  hud.innerHTML = `
    <b>Site ${s.id}</b> <span class="badge ${s.class === "CANDIDATE" ? "ok" : "warn"}">${s.class}</span><br/>
    ${s.lat != null ? `${s.lat.toFixed(6)}, ${s.lon.toFixed(6)}<br/>` : ""}
    <span class="hud-k">Elevation</span> ${s.elevationM} m ${esc(resp.verticalCrs ?? "")}<br/>
    <span class="hud-k">Slope</span> ${fmtNum(s.slopeDeg)}° (${s.slopePct} %)${s.upslopeBearingDeg != null && s.flags.includes("UPSLOPE_ADVISORY") ? ` · land upslope, toward ${s.upslopeBearingDeg}°` : ""}<br/>
    <span class="hud-k">Roughness</span> ${fmtNum(s.roughnessM)} m RMS<br/>
    <span class="hud-k">Clear</span> ${byStatus("CLEAR").map((a: any) => `${a.bearingDeg}°`).join(", ") || "none"}<br/>
    <span class="hud-k">Unverified</span> ${byStatus("UNVERIFIED").map((a: any) => `${a.bearingDeg}°`).join(", ") || "none"}<br/>
    <span class="hud-k">Blocked</span> ${blocked.length} of ${s.approaches.length}${nearest ? ` · nearest ${nearest.firstObstacle.heightAbovePadM} m high at ${nearest.firstObstacle.distanceM} m` : ""}<br/>
    ${s.flags.length ? `<span class="hint">${s.flags.map(esc).join(" · ")}</span>` : ""}
  `;
}

function resetHudInfo() {
  const hud = $("disaster-hud-content");
  if (hud) hud.innerHTML = "Hover or click anywhere on the 2D map to inspect flood depth";
}

/** Applies the backend's exposure rules (resp.exposureRules) — the thresholds are never duplicated here. */
function classifyExposure(depth: number): string {
  const rules: { label: string; gt_m: number; le_m: number | null }[] = state.disaster.lastResult?.exposureRules ?? [];
  if (!(depth > 0) || !rules.length) return "NONE";
  return (rules.find((r) => depth > r.gt_m && (r.le_m == null || depth <= r.le_m)) ?? rules[rules.length - 1]).label;
}

function exposureBadge(label: string): string {
  return label === "NONE" ? "muted" : label === "LOW" ? "ok" : label === "MODERATE" ? "warn" : "bad";
}

/** Same quantity (m²) at a readable scale: m² below 1 ha, else km². */
function fmtArea(m2: number): string {
  return m2 < 1e4 ? `${Math.round(m2).toLocaleString()} m²` : `${(m2 / 1e6).toFixed(3)} km²`;
}

function wireDisasterMapInspector() {
  const stage = $("disaster-stage-wrap");
  const baseImg = $("disaster-base-img") as HTMLImageElement;
  const pickDot = $("disaster-pick-dot");
  const hud = $("disaster-hud-content");

  if (!stage || !baseImg) return;

  stage.addEventListener("mousemove", (e) => {
    if (!state.result || !state.jobId) return;
    const r = baseImg.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;

    const g = state.result.grid;
    const W = Number(g.width), H = Number(g.height);
    const col = Math.floor(((e.clientX - r.left) / r.width) * W);
    const row = Math.floor(((e.clientY - r.top) / r.height) * H);

    const fLvl = Number(($("flood-level-slider") as HTMLInputElement).value || 0);
    const scen = ($("disaster-scenario") as HTMLSelectElement).value;

    if (scen === "flood") {
      hud.innerHTML = `
        <span class="hud-k">Pixel</span> [${col}, ${row}]<br/>
        <span class="hud-k">Water plane</span> ${fLvl.toFixed(1)} m<br/>
        <span class="hint" style="font-size:10px;">Click to query elevation & depth</span>
      `;
    } else if (scen === "landing_zones") {
      hud.innerHTML = `
        <span class="hud-k">Pixel</span> [${col}, ${row}]<br/>
        <span class="hint" style="font-size:10px;">Hover or click a pad for its details</span>
      `;
    } else {
      const aSlp = Number(($("access-slope-slider") as HTMLInputElement).value || 15);
      hud.innerHTML = `
        <span class="hud-k">Pixel</span> [${col}, ${row}]<br/>
        <span class="hud-k">Max slope</span> ${aSlp}°<br/>
        <span class="hint" style="font-size:10px;">Click to query slope</span>
      `;
    }
  });

  stage.addEventListener("mouseleave", () => {
    resetHudInfo();
  });

  stage.addEventListener("click", async (e) => {
    if (!state.result || !state.jobId) return;
    const r = baseImg.getBoundingClientRect();
    const g = state.result.grid;
    const W = Number(g.width), H = Number(g.height);
    const col = Math.floor(((e.clientX - r.left) / r.width) * W);
    const row = Math.floor(((e.clientY - r.top) / r.height) * H);

    if (pickDot) {
      pickDot.style.left = `${e.clientX - r.left}px`;
      pickDot.style.top = `${e.clientY - r.top}px`;
      pickDot.classList.remove("hidden");
    }

    try {
      hud.innerHTML = `Sampling [${col}, ${row}]...`;
      const s = await api.sample(state.jobId, col, row);
      const fLvl = Number(($("flood-level-slider") as HTMLInputElement).value || 0);
      const tVal = s.values?.terrain?.value ?? s.values?.dsm?.value;

      if (tVal !== undefined && tVal !== null) {
        const depth = fLvl - tVal;
        const isFlood = depth > 0;
        hud.innerHTML = `
          <b>Location:</b> [${col}, ${row}]<br/>
          <b>Ground Elev:</b> ${tVal.toFixed(2)} m<br/>
          <b>Status:</b> ${isFlood 
            ? `<span class="hud-wet">Inundated (${depth.toFixed(2)} m depth)` 
            : `<span class="hud-dry">Dry ground (+${(-depth).toFixed(1)} m clear)`}</span><br/>
          ${isFlood ? `<b>Exposure:</b> <span class="badge ${depth > 3 ? 'bad' : depth > 1.5 ? 'warn' : 'ok'}">${classifyExposure(depth)}</span>` : ""}
        `;
      } else {
        hud.innerHTML = `Col ${col}, Row ${row}: Valid sample obtained.`;
      }
    } catch {
      hud.innerHTML = `Col ${col}, Row ${row}`;
    }
  });
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
  $("validate-pts-btn").addEventListener("click", runPointValidation);
  const bldReload = () => { clearTimeout(bldTimer); bldTimer = window.setTimeout(loadBuildingsPanel, 250); };
  $("bld-minh").addEventListener("input", bldReload);
  $("bld-mina").addEventListener("input", bldReload);
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
  $("cam-fly-btn").addEventListener("click", () => {
    if (!state.viewer) return;
    state.viewer.setFlythrough(!state.viewer.isFlythrough);
  });

  // Preset view buttons
  const presetMap: Record<string, "nadir" | "oblique" | "horizon"> = {
    "preset-nadir-btn": "nadir",
    "preset-oblique-btn": "oblique",
    "preset-horizon-btn": "horizon",
  };
  Object.entries(presetMap).forEach(([id, preset]) => {
    $(id).addEventListener("click", () => {
      if (!state.viewer) return;
      state.viewer.setPresetView(preset);
      Object.keys(presetMap).forEach((k) => $(k).classList.remove("active"));
      $(id).classList.add("active");
    });
  });

  // Floating Navigation Dock buttons (North, Zoom In/Out, Fit)
  const navNorth = $("nav-north-btn");
  if (navNorth) {
    navNorth.addEventListener("click", () => {
      if (state.viewer) state.viewer.alignNorth();
    });
  }
  const navZoomIn = $("nav-zoom-in");
  if (navZoomIn) {
    navZoomIn.addEventListener("click", () => {
      if (state.viewer) state.viewer.zoomBy(0.75);
    });
  }
  const navZoomOut = $("nav-zoom-out");
  if (navZoomOut) {
    navZoomOut.addEventListener("click", () => {
      if (state.viewer) state.viewer.zoomBy(1.33);
    });
  }
  const navFit = $("nav-fit-btn");
  if (navFit) {
    navFit.addEventListener("click", () => {
      if (state.viewer) {
        state.viewer.resetCamera();
        Object.keys(presetMap).forEach((k) => $(k).classList.remove("active"));
        $("preset-oblique-btn").classList.add("active");
      }
    });
  }

  // Shader mode buttons
  const shaderMap: Record<string, "aerial" | "heatmap" | "cyber"> = {
    "shader-aerial-btn": "aerial",
    "shader-heatmap-btn": "heatmap",
    "shader-cyber-btn": "cyber",
  };
  Object.entries(shaderMap).forEach(([id, mode]) => {
    $(id).addEventListener("click", () => {
      if (!state.viewer) return;
      state.viewer.setShaderMode(mode);
      Object.keys(shaderMap).forEach((k) => $(k).classList.remove("active"));
      $(id).classList.add("active");
    });
  });

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
    const unit = state.viewer.isMetric ? "m" : "scene units";
    tolVal.textContent = `${tol.toFixed(1)} ${unit}`;
    tolWrap.classList.toggle("hidden", mode !== "rtin");

    const stats = state.viewer.setMeshMode(mode, tol);
    if (mode === "rtin") {
      redPill.classList.remove("hidden");
      redPill.textContent = `⚡ RTIN: ${stats.reductionPct}% triangles reduced (${stats.triangles.toLocaleString()} tris, tol ${tol.toFixed(1)} ${unit})`;
    } else {
      redPill.classList.add("hidden");
    }
  };

  meshSel.addEventListener("change", applyMeshMode);
  tolInput.addEventListener("input", applyMeshMode);

  // ──────────────────────────────────────── Disaster Controls Binding
  const disScen = $("disaster-scenario") as HTMLSelectElement;
  const fLvl = $("flood-level-slider") as HTMLInputElement;
  const aSlp = $("access-slope-slider") as HTMLInputElement;

  if (disScen) {
    disScen.addEventListener("change", () => {
      $("flood-controls-wrap").classList.toggle("hidden", disScen.value !== "flood");
      $("access-slope-wrap").classList.toggle("hidden", disScen.value !== "accessibility");
      $("hlz-controls-wrap").classList.toggle("hidden", disScen.value !== "landing_zones");
      runDisasterAnalysis();
    });
  }

  if (fLvl) {
    fLvl.addEventListener("input", () => {
      $("flood-level-val").textContent = Number(fLvl.value).toFixed(1);
      const live = ($("disaster-live-scrub") as HTMLInputElement)?.checked;
      if (live) {
        clearTimeout(disasterDebounceTimer);
        disasterDebounceTimer = setTimeout(() => runDisasterAnalysis(true), 160);
      }
    });
    fLvl.addEventListener("change", () => runDisasterAnalysis());
  }

  if (aSlp) {
    aSlp.addEventListener("input", () => {
      $("access-slope-val").textContent = aSlp.value;
      const live = ($("disaster-live-scrub") as HTMLInputElement)?.checked;
      if (live) {
        clearTimeout(disasterDebounceTimer);
        disasterDebounceTimer = setTimeout(() => runDisasterAnalysis(true), 160);
      }
    });
    aSlp.addEventListener("change", () => runDisasterAnalysis());
  }

  // Step buttons (-1m, +1m, Auto)
  $("flood-minus-1")?.addEventListener("click", () => {
    fLvl.value = String(Math.max(Number(fLvl.min), Number(fLvl.value) - 1.0));
    $("flood-level-val").textContent = Number(fLvl.value).toFixed(1);
    runDisasterAnalysis();
  });
  $("flood-plus-1")?.addEventListener("click", () => {
    fLvl.value = String(Math.min(Number(fLvl.max), Number(fLvl.value) + 1.0));
    $("flood-level-val").textContent = Number(fLvl.value).toFixed(1);
    runDisasterAnalysis();
  });
  $("flood-auto-btn")?.addEventListener("click", () => {
    if (state.disaster.autoFloodVal) {
      fLvl.value = state.disaster.autoFloodVal.toFixed(1);
      $("flood-level-val").textContent = fLvl.value;
      runDisasterAnalysis();
    }
  });

  $("run-disaster-btn")?.addEventListener("click", () => runDisasterAnalysis());

  // Base map buttons
  document.querySelectorAll("#disaster-base-group button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const base = (btn as HTMLButtonElement).dataset.base as "rgb" | "terrain" | "hillshade" | "dsm";
      state.disaster.baseLayer = base;
      updateDisasterBaseMap();
    });
  });

  // Overlay toggle
  const overToggle = $("disaster-overlay-toggle") as HTMLInputElement;
  if (overToggle) {
    overToggle.addEventListener("change", () => {
      state.disaster.overlayVisible = overToggle.checked;
      const img = $("disaster-overlay-img") as HTMLImageElement;
      if (img) img.style.display = overToggle.checked ? "block" : "none";
    });
  }

  // Overlay opacity
  const opSlider = $("disaster-overlay-opacity") as HTMLInputElement;
  if (opSlider) {
    opSlider.addEventListener("input", () => {
      state.disaster.opacity = Number(opSlider.value) / 100;
      $("disaster-opacity-val").textContent = `${opSlider.value}%`;
      const img = $("disaster-overlay-img") as HTMLImageElement;
      if (img) img.style.opacity = String(state.disaster.opacity);
    });
  }

  // Shimmer button
  const shimBtn = $("disaster-shimmer-btn") as HTMLButtonElement;
  if (shimBtn) {
    shimBtn.addEventListener("click", () => {
      state.disaster.shimmer = !state.disaster.shimmer;
      shimBtn.classList.toggle("active", state.disaster.shimmer);
      const img = $("disaster-overlay-img") as HTMLImageElement;
      if (img) img.classList.toggle("water-shimmer", state.disaster.shimmer);
    });
  }

  // Buildings toggle button
  const bldgBtn = $("disaster-bldg-toggle-btn") as HTMLButtonElement;
  if (bldgBtn) {
    bldgBtn.addEventListener("click", () => {
      state.disaster.showBuildings = !state.disaster.showBuildings;
      bldgBtn.classList.toggle("active", state.disaster.showBuildings);
      const svg = $("disaster-vector-svg");
      if (svg) svg.style.display = state.disaster.showBuildings ? "block" : "none";
      if (state.disaster.showBuildings && state.disaster.lastResult?.buildings) {
        renderDisasterSvg(state.disaster.lastResult.buildings);
      }
    });
  }

  wireDisasterMapInspector();

  wireImagePick();
  updateOptSummary();
}

wire();
initSystem();

// Opt-in inspection hook for automated checks (only with ?debug=1 in the URL).
if (new URLSearchParams(location.search).has("debug")) (window as unknown as Record<string, unknown>).__depthwizard = state;
