export interface ApiError { code: string; message: string; detail?: string; recoverable?: boolean }
export interface Job {
  job_id: string; status: string; created_at: string; updated_at: string; input_filename?: string | null;
  input_sha256?: string | null; mode?: string | null; inputs?: Record<string, string>; stages_ms: Record<string, number>;
  error?: ApiError | null; model?: Record<string, unknown> | null;
}
export interface HeightfieldMeta {
  width: number; height: number; source_width: number; source_height: number; downsample_factor: number; method: string;
  nodata_value: string; valid_fraction: number; min: number; max: number; units: string; metric: boolean;
  calibration_tier: string; vertical_reference: string | null; texture_width: number; texture_height: number;
  residual_vs_source: { rmse: number; max_abs: number };
}
export interface LayerInfo {
  units?: string; vertical_crs?: string; tier?: string; preview?: string; legend?: Record<string, any>; label?: string;
  heightfield?: string; heightfield_meta?: string;
}
export interface Result {
  mode: string; mode_label: string; metric: boolean; metric_horizontal?: boolean; absolute_elevation?: boolean;
  calibration_tier: string; quality?: string; quality_triggers?: string[]; flags?: string[]; notes?: string[];
  units: string; vertical_reference: string | null; object_scale_source?: string | null; object_scale_m_per_unit?: number | null;
  grid: Record<string, any>; gsd_m?: number; dem?: Record<string, any> | null; rdsm_stats?: Record<string, unknown>;
  layers?: Record<string, LayerInfo>; heightfield: HeightfieldMeta; artifacts: Record<string, string>; timings_ms: Record<string, number>;
}
export interface SampleValue { value: number | null; valid: boolean; quantity: string; units: string; metric?: boolean; absolute?: boolean; tier?: string; vertical_reference?: string; scale_source?: string }
export interface Sample {
  mode: string; calibration_tier: string; quality?: string; vertical_reference?: string | null; in_bounds?: boolean;
  pixel?: { col: number; row: number }; position?: { x: number; y: number; crs: string; lon?: number; lat?: number };
  values: Record<string, SampleValue>; flags?: string[];
}
export interface Measure { mode: string; calibration_tier: string; points: Sample[]; segments: Record<string, any>[]; authority: string }
export interface DemoItem { id: string; label: string; file: string; mode: string; anchors?: string; reference_dsm?: string; reference_dtm?: string; reference_vertical_crs?: string; source?: string }

export class ApiFailure extends Error {
  constructor(public status: number, public err: ApiError) { super(err.message); }
}

async function handle<T>(r: Response): Promise<T> {
  if (r.ok) return (await r.json()) as T;
  let err: ApiError = { code: `HTTP_${r.status}`, message: `Request failed (${r.status}).` };
  try { const j = await r.json(); if (j?.error) err = j.error; } catch { /* ignore */ }
  throw new ApiFailure(r.status, err);
}

export interface CreateOptions { dem?: File | null; demVerticalCrs?: string; anchors?: File | null }

export const api = {
  health: () => fetch("/health").then((r) => handle<Record<string, any>>(r)),
  system: () => fetch("/api/system").then((r) => handle<Record<string, any>>(r)),
  demo: () => fetch("/api/demo").then((r) => handle<{ items: DemoItem[]; references: string[] }>(r)),
  createJob: async (file: File, opts: CreateOptions = {}) => {
    const fd = new FormData(); fd.append("file", file, file.name);
    if (opts.dem) { fd.append("dem", opts.dem, opts.dem.name); fd.append("dem_vertical_crs", opts.demVerticalCrs ?? "EGM2008"); }
    if (opts.anchors) fd.append("anchors", opts.anchors, opts.anchors.name);
    return handle<Job>(await fetch("/api/jobs", { method: "POST", body: fd }));
  },
  run: async (id: string) => handle<Job>(await fetch(`/api/jobs/${id}/run`, { method: "POST" })),
  job: async (id: string) => handle<Job>(await fetch(`/api/jobs/${id}`)),
  result: async (id: string) => handle<Result>(await fetch(`/api/jobs/${id}/result`)),
  metadata: async (id: string) => handle<Record<string, any>>(await fetch(`/api/jobs/${id}/metadata`)),
  artifactUrl: (id: string, name: string) => `/api/jobs/${id}/artifact/${name}`,
  sample: async (id: string, x: number, y: number, crs = "pixel") => handle<Sample>(await fetch(`/api/jobs/${id}/sample?x=${x}&y=${y}&crs=${crs}`)),
  measure: async (id: string, points: { x: number; y: number }[], crs = "pixel") =>
    handle<Measure>(await fetch(`/api/jobs/${id}/measure`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ points, crs }) })),
  validate: async (id: string, opts: { reference?: File | null; bundled?: string | null; refType: string; verticalCrs: string; sourceNote?: string }) => {
    const fd = new FormData();
    if (opts.reference) fd.append("reference", opts.reference, opts.reference.name);
    if (opts.bundled) fd.append("bundled", opts.bundled);
    fd.append("ref_type", opts.refType); fd.append("vertical_crs", opts.verticalCrs); fd.append("source_note", opts.sourceNote ?? "");
    return handle<Record<string, any>>(await fetch(`/api/jobs/${id}/validate`, { method: "POST", body: fd }));
  },
  validation: async (id: string) => handle<{ runs: Record<string, any>[]; latest: Record<string, any> | null }>(await fetch(`/api/jobs/${id}/validation`)),
  heightfield: async (id: string, name = "heightfield.f32"): Promise<Float32Array> => {
    const r = await fetch(`/api/jobs/${id}/artifact/${name}`);
    if (!r.ok) throw new ApiFailure(r.status, { code: "HEIGHTFIELD_MISSING", message: "Heightfield not available." });
    return new Float32Array(await r.arrayBuffer()); // little-endian float32, row-major, NaN = nodata
  },
  heightfieldMeta: async (id: string, name: string) => handle<HeightfieldMeta>(await fetch(`/api/jobs/${id}/artifact/${name}`)),
};

export async function pollUntilDone(id: string, onUpdate: (j: Job) => void, intervalMs = 400, timeoutMs = 600000): Promise<Job> {
  const t0 = Date.now();
  for (;;) {
    const j = await api.job(id);
    onUpdate(j);
    if (j.status === "READY" || j.status === "FAILED") return j;
    if (Date.now() - t0 > timeoutMs) throw new ApiFailure(504, { code: "TIMEOUT", message: "Processing timed out." });
    await new Promise((res) => setTimeout(res, intervalMs));
  }
}
