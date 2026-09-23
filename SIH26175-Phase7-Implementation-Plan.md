# Phase 7 — Implementation Planning, Codebase Design, Development Workflow & Build Execution Plan

SIH26175 DepthWizard · 2026-09-20

Scope note: Phase 7 converts the Phase 6 architecture (`SIH26175-Phase6-System-Architecture.md`, the source of truth: layers L1–L13, decision records D-01…D-15, module boundaries §16, API §13.1, output contracts §33, workstreams §27) into an executable development plan. No application or training code is written; no accuracy numbers are asserted. Tags `[E#]/[F#]/[G#]/[H#]` refer to the Phase 3–6 evidence ledgers; `[I#]` = verified in this phase.

**Inputs read:** Phase 1–6 documents in the project folder (six files; Phase 6 = 842 lines, re-read for decisions, module boundaries, API contracts, output contracts, workstreams, red team).

**Verified this phase**

| ID | Finding | Source | Confidence |
| --- | --- | --- | --- |
| I1 | Current releases (PyPI, 2026-09-20): torch 2.14.0 (py ≥ 3.10), torchvision 0.29.0, rasterio 1.5.1 (**py ≥ 3.12**), pyproj 3.8.0 (**py ≥ 3.12**), numpy 2.5.3 (**≥ 3.12**), scipy 1.18.1 (≥ 3.12), scikit-learn 1.9.1 (≥ 3.11), fastapi 0.141.1, uvicorn 0.53.0, pydantic 2.13.5, safetensors 0.8.0, pdal 3.5.5, earthaccess 0.19.0, pyinstaller 6.22.3 (py < 3.16). | PyPI JSON API | HIGH |
| I2 | npm: three 0.186.0, @mapbox/martini 0.2.0, vite 8.3.0, typescript 7.0.2; Node LTS v24.21 "Krypton". | npm registry, nodejs.org | HIGH |
| I3 | Phase 6 stack said "Python 3.11". Given I1, the geospatial/numeric stack now requires ≥ 3.12 → **minimal correction: pin Python 3.12** (torch, PyInstaller, FastAPI all support it). No other Phase 6 decision is changed. | I1 | HIGH |
| I4 | Exact CUDA wheel matrix for torch 2.14 not verified here; install via the official PyTorch selector for the machine's driver; CPU wheel is the default in the packaged build (Phase 6 D-07/§18). | — | MEDIUM (verify on dev machine) |

**Corrections to Phase 6 (minimal, explicit):** (1) Python 3.11 → 3.12 [I3]. No architectural change.

---

## 1. Executive Implementation Summary

The team builds a **modular Python monolith** (`depthwizard` package: `core/*` libraries + `backend/` FastAPI app with an in-process persisted job worker) and a **Vite/TypeScript + Three.js** frontend, packaged with PyInstaller into one offline executable. The critical path is: geo core → API/job skeleton with a stub model → tiled inference (zero-shot baseline first) → Level-T calibration (DEM → datum → ground-masked normalized convolution) → DSM + COG outputs → terrain tiles → viewer → fine-tuned model swap-in → validation harness → packaging. Everything that is not on that path (anchors, uncertainty, semantic head, profile/polygon tools, contours, glTF, cross-country scripts) is Advanced/Optional and is scheduled so it can be cut without breaking the demo. Fine-tuning (WS2) runs in parallel from day one on data prepared by the geo core, and the pipeline is designed to run end-to-end with the zero-shot baseline until the fine-tuned head is ready (Phase 6 §5 "baseline path"). Tests are fixture-driven (synthetic rasters with known planes, slopes, shifts, canopy bias and anchor transformations) so correctness is verified without large datasets.

---

## 2. Phase 6 → Phase 7 Traceability

| Phase 6 Component | Implementation Module | Technology | Input | Output | Dependencies | Test Required |
| --- | --- | --- | --- | --- | --- | --- |
| L1 `ingest` | `core/ingest/` | rasterio, Pillow | file path, options | `Meta`, mode | `core/geo` | Unit (formats/edge cases) |
| L2 `geo.meta/grid/reproject/resample/vertical/writer/check` | `core/geo/` | rasterio, pyproj, numpy | rasters, `Grid` | rasters, `Grid` | GDAL/PROJ grids | Unit (round-trips, datum) |
| L2 `preprocess` | `core/preprocess/` | numpy, rasterio | array + `Meta` | tiles, `Prep` | `core/geo` | Unit (determinism, inverse error) |
| L3 `inference` | `core/inference/` | torch, safetensors | tiles | `ndsm_canon` / `rel_canon`, `unc_model` | `models/`, `core/preprocess` | Integration (stub model) |
| L4 `ml.registry` | `ml/registry.py` | safetensors, json | name/version | model handle + card | `models/` | Unit (hash) |
| L4 `ml.train`, `ml.data`, `ml.eval` | `ml/` | torch, rasterio, pdal | datasets | weights, card, tables | `core/geo` | Offline validation |
| L5 `calib.discover` | `core/calib/discover.py` | json index, rasterio | bounds, config | `CalibInputs` | `assets/index.json` | Unit |
| L5 `calib.terrain` | `core/calib/terrain.py` | numpy, scipy | nDSM, DEM | `terrain`, support, report | `core/geo` | Unit (synthetic canopy bias) |
| L5 `calib.anchors` | `core/calib/anchors.py` | scikit-learn, scipy | anchors, terrain, nDSM | params, residuals | `core/geo` | Unit (outliers) |
| L5 `calib.tier` | `core/calib/tier.py` | pure python | reports | tier, flags | — | Unit (rule table) |
| L6 `dsm` | `core/dsm/` | numpy | terrain, nDSM | dsm, slope, aspect, uncertainty, flags | `core/geo` | Unit (analytic surfaces) |
| L7 geospatial consistency | `core/geo/check.py`, `writer.py` | rasterio | rasters | COGs | — | Unit + integration |
| L8 `validate` | `core/validate/` | numpy, scipy, rasterio | outputs + reference | `validation.json`, `residual.tif` | `core/geo` | Unit (synthetic shift) |
| L8 `quality` | `core/quality/` | pure python | reports | `quality.json` | — | Unit |
| L9 `terrain` (server) | `core/terrain/` | rasterio, numpy, Pillow | dsm/rdsm + RGB | tileset | GDAL overviews | Integration (continuity) |
| L9 `viewer/mesh` (client) | `frontend/src/viewer/mesh/` | @mapbox/martini, three | height tiles | meshes, residual | tileset | UI/unit (JS) |
| L10 `viewer` | `frontend/src/viewer/` | three, TS | API | scene, tools | API | UI (Playwright) |
| L11 `api`, `jobs`, `store`, `config`, `assets` | `backend/` | FastAPI, uvicorn, pydantic, sqlite3 | HTTP | JSON/files | `core/*` | API/integration |
| L13 packaging, demo mode | `scripts/package_*.py`, `assets/` | PyInstaller | build | executable | all | Deployment test |
| Observability | `backend/logging.py`, `log.jsonl` | stdlib logging (JSON) | events | logs | — | Smoke |

---

## 3. Final Technology Stack

| Component | Technology (version) | Why needed / module | Compatibility | Licence | GPU/CPU | Alternative |
| --- | --- | --- | --- | --- | --- | --- |
| Language | **Python 3.12.x** [I3] | Torch + GDAL + PROJ ecosystem; all `core/`, `backend/`, `ml/` | Required by rasterio 1.5 / pyproj 3.8 / numpy 2.5 | PSF | — | 3.11 with older rasterio 1.4 (not recommended) |
| DL framework | torch 2.14, torchvision 0.29 | fine-tune + inference (`ml/`, `core/inference`) | CUDA wheel per machine [I4]; CPU wheel in package | BSD-3 | GPU optional | ONNX Runtime for CPU inference (Advanced) |
| Weights format | safetensors 0.8 | safe, hashable weights (`ml/registry`) | — | Apache-2.0 | — | .pth |
| Backbone | Depth Anything V2-Small (ViT-S/14 + DPT) | D-01 | code vendored/pinned commit | Apache-2.0 [E26] | ~25 M params | — |
| Raster I/O | rasterio 1.5.1 (wheels bundle GDAL) | all raster read/write/COG (`core/geo`) | py ≥ 3.12 | BSD-3 (GDAL MIT/X) | CPU | GDAL Python bindings |
| CRS/datum | pyproj 3.8 (bundled PROJ) + grids `us_nga_egm96_15.tif`, `us_nga_egm08_25.tif` | vertical transforms (`core/geo/vertical`) | set `PROJ_DATA`/`pyproj.datadir` to bundled dir | MIT / grids PD | CPU | — |
| Numerics | numpy 2.5, scipy 1.18 | normalized convolution, Horn slope, robust stats | — | BSD | CPU | — |
| Robust fitting | scikit-learn 1.9 (RANSAC, TheilSen, Huber) | `core/calib/anchors` | — | BSD | CPU | scipy-only |
| Image decode | Pillow (latest) | PNG/JPG (`core/ingest`) | — | HPND | — | imageio |
| Point clouds (dev only) | pdal 3.5 (conda) | 3DEP EPT → DSM/DTM (`ml/data`) | conda-forge install | BSD | CPU | LAStools (proprietary) |
| API | fastapi 0.141, uvicorn 0.53, pydantic 2.13 | `backend/api` | — | MIT/BSD | — | Flask |
| Job state | sqlite3 (stdlib) + threading worker | `backend/jobs`, `backend/store` | — | PD | — | Celery/Redis (rejected D-05) |
| Config | pydantic-settings + YAML (PyYAML) | `backend/config` | — | MIT | — | — |
| Frontend build | Node 24 LTS, vite 8, typescript 7 | `frontend/` | — | MIT | — | — |
| 3D | three 0.186, @mapbox/martini 0.2 | D-04 viewer/mesh | WebGL2 | MIT / ISC | Browser GPU | Delatin |
| UI | plain TS + small component lib (or React if team prefers) | panels | — | MIT | — | — |
| Charts | uPlot or Chart.js | profiles/histograms | — | MIT | — | — |
| Tests | pytest, pytest-cov, hypothesis (optional), Playwright | all | — | MIT/Apache | — | — |
| Lint/format | ruff, mypy (loose), eslint, prettier | CI gates | — | MIT | — | — |
| Packaging | pyinstaller 6.22 (+ optional Tauri shell later) | D-07 | py < 3.16 | GPL w/ exception | — | Docker (eval only) |
| Data access (dev) | earthaccess 0.19 (ICESat-2), AWS anonymous S3 (Copernicus) | `ml/data`, `scripts/fetch_assets.py` | — | MIT | — | — |

---

## 4. Repository Structure

```
depthwizard/                      # git root; monorepo
  pyproject.toml                  # package metadata, deps (pinned), ruff/mypy/pytest config
  requirements/                   # lock files: base.txt, gpu.txt, dev.txt, package.txt
  README.md
  backend/                        # RUNTIME · Owner: Backend (M3)
    app.py                        # entrypoint: create FastAPI app, mount static frontend, start worker
    api/                          # routers: upload.py process.py jobs.py results.py rasters.py terrain.py validate.py metadata.py export.py system.py
    schemas/                      # pydantic models: Meta, JobState, ResultManifest, CalibReport, ValidationResult, QualityReport, ErrorResponse
    jobs/                         # state_machine.py worker.py stages.py retry.py
    store/                        # paths.py registry.py (sqlite) artefacts.py export.py retention.py
    config/                       # settings.py (pydantic-settings) defaults loader, hashing
    assets/                       # index.py (bundled asset lookup: weights, grids, DEM tiles, samples)
    logging.py                    # JSON logging, per-job log.jsonl
    errors.py                     # application error taxonomy
  core/                           # RUNTIME · pure library, no HTTP · Owners: M1 (geo/calib/dsm), M2 (preprocess/inference), M4 (validate/quality), M5 (terrain)
    geo/  meta.py grid.py reproject.py resample.py vertical.py writer.py check.py coords.py
    preprocess/  normalize.py canonical_gsd.py tiling.py qc.py
    inference/  model.py batching.py tta.py blend.py postprocess.py baseline.py
    calib/  discover.py dem.py terrain.py anchors.py tier.py report.py
    dsm/  compose.py derive.py uncertainty.py flags.py
    validate/  reference.py align.py coregister.py masks.py metrics.py stratify.py residual.py report.py leakage.py
    quality/  states.py
    terrain/  pyramid.py tileset.py tiles.py export.py
  ml/                             # DEV-ONLY (training/eval) · Owner: M2 (+M4 eval)
    registry.py                   # model card + weights loader (also used at runtime)
    data/  sources/{gamus.py dfc19.py swisstopo.py linz.py usgs3dep.py} derive_ndsm.py tiling.py splits.py overlap.py manifest.py
    train/  dataset.py model.py losses.py train.py card.py
    eval/  dfc19_protocol.py ablations.py crosscountry.py ause.py tables.py
  models/                         # weights + model cards (weights via release assets / fetch script; .gitignored)
  assets/                         # RUNTIME (bundled): geoid grids, demo DEM tiles, samples, demo references; index.json; LICENSES.md
  frontend/                       # RUNTIME (built bundle served by backend) · Owners: M5 (viewer), M6 (UI)
    src/  api/ store/ pages/{Upload,Status,Result} components/{Map2D,RasterView,Panels,Tools,Badges} viewer/{loader,mesh,camera,layers,tools,scene}
    tests/                        # vitest unit (mesh math), playwright e2e
  scripts/                        # fetch_assets.py fetch_model.py make_fixtures.py package_win.py package_linux.py clean_machine_test.md bench_log.py
  tests/                          # unit/ integration/ system/ fixtures/ (small synthetic rasters, generated by make_fixtures.py)
  configs/                        # default.yaml mvp.yaml full.yaml demo.yaml eval_config.json canonical_gsds.json
  docs/                           # architecture.md datum_and_tiers.md output_contracts.md api.md setup.md model.md data.md calibration.md geospatial.md validation.md demo.md troubleshooting.md experiments.md LICENSES.md
  data/                           # local only (.gitignored except README + manifests)
```

| Directory | Responsibility | Important files | Depends on | Owner | Runtime/Dev |
| --- | --- | --- | --- | --- | --- |
| `backend/` | HTTP, jobs, storage, config | `app.py`, `jobs/state_machine.py` | `core`, `ml/registry` | M3 | Runtime |
| `core/geo` | All CRS/transform/datum/raster I/O | `grid.py`, `vertical.py`, `writer.py` | rasterio, pyproj | M1 | Runtime |
| `core/preprocess` | Normalize, canonical GSD, tiling, QC | `canonical_gsd.py`, `tiling.py` | geo | M2 | Runtime |
| `core/inference` | Model exec, TTA, blend, inverse resample | `model.py`, `blend.py` | torch, registry | M2 | Runtime |
| `core/calib` | Tiers, terrain layer, anchors | `terrain.py`, `anchors.py`, `tier.py` | geo, scipy, sklearn | M1 | Runtime |
| `core/dsm` | Composition and derivatives | `compose.py`, `derive.py` | geo | M1 | Runtime |
| `core/validate`, `core/quality` | Blueprint harness, states | `coregister.py`, `metrics.py` | geo | M4 | Runtime |
| `core/terrain` | Tileset generation, export | `tileset.py` | geo, Pillow | M5 | Runtime |
| `ml/` | Data prep, training, eval | `train/train.py`, `data/derive_ndsm.py` | torch, pdal | M2, M4 | Dev (registry runtime) |
| `frontend/` | SPA + viewer | `viewer/mesh/rtin.ts`, `viewer/camera/*` | three, martini | M5, M6 | Runtime |
| `assets/`, `models/` | Offline artefacts | `index.json`, model cards | fetch scripts | M3 | Runtime |
| `scripts/`, `tests/`, `configs/`, `docs/` | Tooling, tests, config, docs | — | — | all | Dev (configs runtime) |

---

## 5. File-Level Design (key modules)

**`core/geo/`**
- `meta.py` — `read_meta(path, options) -> Meta` (CRS, transform, shape, bands, dtype, nodata, vertical CRS tag, RPC flag, GSD in metres via `coords.local_gsd`); `classify_mode(meta) -> "A"|"B"`. Inputs: path; outputs `Meta` (§7). Deps: rasterio, pyproj.
- `grid.py` — `Grid(crs, transform, width, height, nodata)`; `Grid.from_meta`, `.pixel_to_crs`, `.crs_to_pixel`, `.bounds`, `.assert_same(other)`, `.window(bounds)`.
- `reproject.py` — `reproject_to_grid(src_path|array+Grid, dst_grid, resampling) -> array`; used for DEM, anchors, references.
- `resample.py` — `to_canonical(array, grid, gsd_target) -> (array, grid, factor)`; `from_canonical(array, canon_grid, target_grid) -> array` (cubic); invertibility test hook.
- `vertical.py` — `transform_heights(z, x, y, src_vcrs, dst_vcrs) -> z'` using pyproj with bundled grids; `datum_registry` (EGM96=EPSG:5773, EGM2008=EPSG:3855, ellipsoidal, NAVD88, NAP, LN02, NZVD2016 mapping to PROJ pipelines); `ensure_grids()` checks presence.
- `writer.py` — `write_cog(path, array, grid, tags, nodata, dtype)`; sets `VERTICAL_CRS`, provenance tags; builds overviews.
- `check.py` — `assert_grid(path|array, grid)`, `validate_cog(path)`, `assert_finite_transform`.
- `coords.py` — `local_gsd(meta)`, pixel-centre helpers, bounds to EPSG:4326.

**`core/preprocess/`**
- `normalize.py` — `stretch(array, p_low, p_high) -> uint8`, `to_model_tensor(rgb8, mean, std)`.
- `canonical_gsd.py` — `choose_band(gsd, bands) -> gsd_c, out_of_range_flag`.
- `tiling.py` — `plan_tiles(shape, tile=518, overlap=64) -> [Window]`, `feather_weights(tile, overlap)`, `Mosaic` accumulator (weighted sum + weight, windowed to disk for large grids).
- `qc.py` — `quality_score(rgb) -> {nodata_frac, saturation, blockiness, contrast, cloud_flag}`; `facade_heuristic(rgb)` (Advanced).

**`core/inference/`**
- `model.py` — `load_model(card) -> nn.Module` (DA-V2-S encoder + nDSM head; or baseline relative head); device policy.
- `batching.py` — adaptive batch, OOM cascade (`halve → single → cpu`).
- `tta.py` — flips/rot90 set, inverse mapping, mean/var.
- `blend.py` — feathered mosaic assembly; `tile_consistency_stat`.
- `postprocess.py` — clamp ≥ 0, inverse canonical resample, `rel_normalize(array, p=(1,99))` for Mode A.
- `baseline.py` — zero-shot relative path (Concept A) for ablations/fallback.

**`core/calib/`**
- `discover.py` — `discover(bounds4326, config, uploads) -> CalibInputs{dem: DemSource|None, anchors: AnchorSet|None, grids_ok}`; reads `assets/index.json`.
- `dem.py` — `load_dem_for_grid(dem_source, grid, out_vcrs) -> dem_coarse(array at ~30 m, coarse_grid), void_mask`; datum transform via `vertical`.
- `terrain.py` — `ground_mask(ndsm, h_ground)`, `ground_support(ground, grid, coarse_grid) -> w`, `normalized_convolution(dem, w, sigma_cells, w_min) -> terrain_coarse, raw_fallback_mask`, `upsample_terrain(terrain_coarse, coarse_grid, grid) -> terrain`, `consistency_check(terrain, ndsm, dem) -> {ME, NMAD}`.
- `anchors.py` — `load_anchors(path) -> AnchorSet` (CSV/GeoJSON; ATL08 loader in Advanced), `transform_anchors(set, grid, out_vcrs)`, `split_holdout(set, frac, seed, stratify=True)`, `fit(terrain, ndsm, anchors) -> AnchorFit{offset, tilt?, scale, residuals, n_used}` (TheilSen/Huber/RANSAC), `accept(fit, sigma_anchor, k)`.
- `tier.py` — `decide(meta, calib_inputs, checks, fit) -> Tier, flags`.
- `report.py` — `CalibReport` builder (method version, sources, params, checks, anchors used/held out, timestamp).

**`core/dsm/`**
- `compose.py` — `compose(terrain, ndsm, nodata) -> dsm`.
- `derive.py` — `slope_aspect(dsm, gsd_x, gsd_y, prefilter=None) -> (slope_deg, aspect_deg)` (Horn).
- `uncertainty.py` — `combine(unc_model, priors_table, terrain_unc, anchor_resid) -> unc`.
- `flags.py` — bit constants; `build_flags(...)`.

**`core/validate/`**
- `reference.py` — `load_reference(spec) -> RefRaster|RefPoints` (requires declared vertical CRS).
- `align.py` — `common_grid(pred_grid, ref_grid) -> comparison grid`, aggregation (block mean/max).
- `coregister.py` — `nuth_kaab_shift(dh, slope, aspect, mask) -> (dx, dy, dz)`; `apply_shift`.
- `masks.py` — nodata/water/border/change/leakage exclusion.
- `metrics.py` — ME, RMSE, MAE, NMAD, LE90/95, pearson, spearman, `affine_fit(pred, ref)`; Mode A wrapper.
- `stratify.py` — terrain classes from reference DTM slope + land cover; class tables; height bins; RMSE-B; F1-HE.
- `residual.py`, `report.py`, `leakage.py`.

**`core/terrain/`**
- `pyramid.py` — overviews for DSM/RGB; `tileset.py` — quadtree scheme, `tileset.json` writer; `tiles.py` — 257² float32 (or uint16+scale) height tiles + JPEG texture tiles; `export.py` — glTF/OBJ (Advanced).

**`backend/jobs/`**
- `state_machine.py` — states/transitions (Phase 6 §13.2), persistence; `stages.py` — one function per stage calling `core`, idempotent, writes artefacts; `worker.py` — thread consuming queue, resume on start; `retry.py` — cascade policies.

**`frontend/src/viewer/`**
- `loader.ts` (tileset fetch/LOD), `mesh/rtin.ts` (martini wrapper, skirts, holes, residual), `camera/{orbit,fly,firstPerson}.ts` (clamp to height tiles), `layers/*.ts` (RGB, ramps, slope, flags, residual, terrain-only, contours), `tools/{point,twoPoint,profile,polygon}.ts` (call `/sample`), `scene.ts`, `badges.ts`.

---

## 6. Module Contracts

| Module | Responsibility | Inputs | Outputs | Types/Formats | Dependencies | Error conditions | Test strategy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ingest` | Decode, classify, meta | path, options{band_map, user_crs, user_gsd, vertical_crs} | `Meta`, mode, warnings | file → pydantic `Meta` | geo | `InvalidFile`, `UnsupportedFormat`, `InvalidGeoref`, `TooLarge` | unit fixtures per format & edge case |
| `geo` | Grid/CRS/datum/raster ops | arrays + `Grid` | arrays + `Grid`, COG files | float32 arrays, `Grid` | rasterio, pyproj | `GridMismatch`, `MissingGrid`, `CRSUnresolvable` | round-trips; PROJ reference values |
| `preprocess` | Normalize, canonical GSD, tiles, QC | rgb array, `Meta` | tiles iterator, `Prep`, weights | uint8 → float tensors | geo | `GSDOutOfRange` (flag) | determinism; inverse error < 1e-3 relative |
| `inference` | Predict per tile, TTA, blend, postprocess | tiles, model handle | `ndsm_canon`/`rel_canon`, `unc_model`, consistency stat | float32 arrays | torch | `ModelUnavailable`, `InferenceFailed`, OOM cascade | stub model integration |
| `calib.discover` | Locate sources | bounds, config, uploads | `CalibInputs` | dataclass | assets index | none (empty allowed) | unit |
| `calib.terrain` | Terrain layer | ndsm, dem_coarse, grids, config | terrain, support, raw_fallback, checks | float32 | scipy | `DatumSuspect` (flag) | synthetic canopy bias |
| `calib.anchors` | Robust fit | anchors, terrain, ndsm | `AnchorFit`, holdout residuals | dataclass | sklearn | `AnchorFitRejected` (flag), `TooFewAnchors` | synthetic outliers |
| `calib.tier` | Tier & flags | inputs, checks, fit | `Tier`, flags | enum, bitmask | — | — | rule table |
| `dsm` | Compose/derive | terrain, ndsm, unc | dsm, slope, aspect, unc, flags | float32/uint16 | geo | `GridMismatch` | analytic surfaces |
| `validate` | Blueprint metrics | outputs, reference spec | `validation.json`, residual | JSON, COG | geo | `ReferenceMissingVCRS`, `NoOverlap`, `ShiftUndetermined` (report) | synthetic shift; metric identities |
| `quality` | States | reports | `quality.json` | JSON | — | — | thresholds |
| `terrain` | Tileset | dsm/rdsm, rgb | tileset dir | JSON + binaries | Pillow | `TilesetFailed` | continuity |
| `api` | HTTP | requests | responses | JSON | jobs, store | HTTP 4xx/5xx | API tests |
| `jobs` | Orchestrate | job spec | states, artefacts | sqlite rows | core | `StageFailed{stage}` | integration with stub |
| `store` | Paths/registry/export | job id | paths, zip | fs | sqlite | `DiskFull` | unit |

---

## 7. Data Contracts

Terminology is fixed: **relative height** (unitless, Mode A), **nDSM / height above local ground** (metres), **terrain elevation** (metres, declared vertical CRS, DEM-derived low-frequency), **surface elevation / DSM** (metres, declared vertical CRS). The word "depth" appears only inside `core/inference` internals; "elevation" only for datum-referenced quantities.

| Artefact | Object / file | Dimensions | Units | CRS | dtype | nodata | Metadata / provenance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Input image | original file + `input/sha256` | native | — | as given | native | native tag/alpha | `meta.json` |
| `Meta` | pydantic → `meta.json` | — | — | EPSG/WKT, transform (6 coeffs), vertical_crs? | — | value | mode, gsd_m, bands, dtype, rotation, rpc_present, warnings |
| `Grid` | in-memory | W×H | — | crs + transform | — | value | — |
| Relative height (Mode A) | `pred/rel_canon.tif` → `out/rdsm.tif` | canonical → input W×H | unitless [0,1] | pixel grid (no CRS) | float32 | −9999 | normalization percentiles, tier R, scale_hint? |
| nDSM (Level H) | `pred/ndsm_canon.tif` → `out/ndsm.tif` | canonical → input | metres above local ground, ≥ 0 | input CRS | float32 | −9999 | model hash, gsd_c, TTA set |
| Model uncertainty | `pred/unc_model.tif` | same | metres (TTA std) | input CRS | float32 | −9999 | TTA set |
| DEM coarse | `calib/dem_coarse.tif` | ~30 m grid over bounds | metres, output vertical CRS | input CRS | float32 | −9999 | source, tiles, src vcrs, grid file hashes |
| Ground support | `calib/ground_support.tif` | coarse grid | fraction [0,1] | input CRS | float32 | −1 | h_ground |
| Terrain elevation (Level T) | `calib/terrain.tif` | input W×H | metres, vertical CRS (EPSG:3855 default) | input CRS | float32 | −9999 | NC params, raw-fallback %, checks |
| Anchors | `calib/anchors_used.geojson`, `anchors_holdout.geojson` | points | metres (output vcrs) | input CRS | — | — | source, σ, ids, residuals |
| DSM (Level T/A) | `out/dsm.tif` | input W×H | metres, vertical CRS | input CRS + `VERTICAL_CRS` tag | float32 | −9999 | tier, terrain/ndsm hashes, anchors used, model hash, job id |
| Slope / aspect | `out/slope.tif`, `out/aspect.tif` | input | degrees | input CRS | float32 | −9999 | method Horn, prefilter |
| Uncertainty | `out/uncertainty.tif` | input | metres | input CRS | float32 | −9999 | priors table version |
| Flags | `out/flags.tif` | input | bitmask | input CRS or pixel | uint16 | 0 | bit dictionary |
| `CalibReport` | `calib/calib_report.json` | — | — | — | — | — | method_version, sources, params, checks, tier, timestamp |
| `ValidationResult` | `validation/validation.json`, `residual.tif` | comparison grid | metres | job CRS | float32 | −9999 | ref hash, vcrs, shift, masks, eval_config hash |
| `QualityReport` | `quality.json` | — | states + triggers | — | — | — | thresholds version |
| Tileset | `terrain/tileset.json`, `{z}/{x}/{y}.hgt` (float32 LE or uint16+scale/offset in header), `{z}/{x}/{y}.jpg` | 257² heights, 256² texture | metres (B) / relative (A) | job CRS or pixel; bounds in JSON | float32/uint16 | NaN/sentinel → dropped | tolerance default, exaggeration default, tier, datum |
| Mesh (client) | in GPU memory; optional glTF export | — | metres/relative | local metres | — | — | tolerance, residual |
| `JobState` | sqlite row + `job.json` | — | — | — | — | — | state, stage, progress, config_hash, model_hash, timestamps, error |

---

## 8. Configuration Strategy

- **Files:** `configs/default.yaml` (all scientific defaults), profile overlays `mvp.yaml`, `full.yaml`, `demo.yaml`; `configs/canonical_gsds.json` (must match model card); `configs/eval_config.json` (metrics, thresholds, terrain-class rules).
- **Environment variables** (prefix `DW_`): `DW_CONFIG`, `DW_PROFILE`, `DW_DATA_DIR`, `DW_MODELS_DIR`, `DW_ASSETS_DIR`, `DW_DEVICE` (auto|cuda|cpu), `DW_HOST/PORT`, `DW_LOG_LEVEL`, `PROJ_DATA`.
- **Secrets:** none required at runtime (fully offline). Dev-only: Earthdata credentials via `~/.netrc` for `earthaccess` (never in repo).
- **Runtime overrides:** per-request `options` in `POST /process` (calibration dem_source, output_vertical_crs, anchors.holdout_fraction, tta, cleanup, tolerance_m, aoi) validated against schema; the effective config is hashed → `config_hash` in `report.json`.
- **Must not be hard-coded:** model name/version/path; canonical GSD bands; tile size/overlap; stretch percentiles; mean/std; TTA set; device policy; `max_pixels`; `h_ground`; NC `sigma_cells`, `w_min`; `datum_sanity_m`; DEM priority list; output vertical CRS; anchor `holdout_fraction`, `min_n`, `k_accept`; cleanup on/off; slope prefilter; confidence thresholds; validation aggregation rule and class thresholds; tileset tile size/max levels/default tolerance/exaggeration; retention days; log level; demo flags.

---

## 9. Environment Setup

Assumptions: Windows 11 or Ubuntu 22.04+; Python **3.12**; Node **24 LTS**; optional NVIDIA GPU with recent driver. Package manager: `uv` or `pip` in a venv (conda only for PDAL in the dev/data environment).

Backend/core (all platforms):
```bash
git clone <repo> depthwizard && cd depthwizard
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate ; Linux: source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements/base.txt          # rasterio, pyproj, numpy, scipy, sklearn, fastapi, uvicorn, pydantic, pillow, pyyaml, safetensors
pip install torch torchvision --index-url <official PyTorch index for your CUDA or cpu>   # [I4] verify on machine
pip install -r requirements/dev.txt           # pytest, ruff, mypy, playwright
python scripts/fetch_assets.py --grids --samples          # geoid grids (EGM96 2.6 MB, EGM2008 76.9 MB) + sample inputs
python scripts/fetch_assets.py --dem-demo                 # Copernicus tiles for demo AOIs (~40 MB each)
python scripts/fetch_model.py --name dw-ndsm-small --version latest   # or baseline DA-V2-S weights
python scripts/make_fixtures.py                            # synthetic test rasters
pytest -q tests/unit
```
Frontend:
```bash
cd frontend && npm ci && npm run build   # outputs to frontend/dist served by backend
```
Run (dev): `uvicorn backend.app:app --reload` then open `http://127.0.0.1:8000`. Training/data environment (dev machines only): `conda create -n dw-data python=3.12 pdal gdal -c conda-forge` for 3DEP processing; everything else via pip in the same env.

---

## 10. GPU / ML Environment

- **GPU (optional):** NVIDIA with ≥ 6–8 GB VRAM for fine-tuning the Small backbone at batch 8–16 of 518² tiles in FP16 — *estimate; benchmark on the dev machine before fixing batch sizes*. Inference in FP16 needs < 2 GB VRAM at batch 1 (estimate).
- **CUDA:** install torch via the official selector matching the driver [I4]; record `torch.version.cuda` in `report.json`.
- **CPU fallback:** identical code path, FP32, batch 1; expected minutes per multi-megapixel scene (to be measured in Phase 8; logged per stage).
- **Precision:** FP16 autocast on GPU, FP32 on CPU; outputs always float32.
- **Weights:** `models/<name>/<version>/weights.safetensors` + `model_card.json`; ~100 MB class for Small.
- **Memory optimization:** windowed mosaic; adaptive batch; TTA optional; `torch.inference_mode()`; pinned memory off for CPU.
- **Determinism:** fixed seeds; TTA set fixed; note that GPU/CPU results may differ at float precision — reproducibility is asserted per device class.

---

## 11. Data Directory Strategy

```
data/                          # .gitignored except README.md and manifests/
  README.md
  manifests/                   # tracked: dataset manifests (source URLs, dates, licences, tile lists, splits, overlap-check results, hashes)
  raw/<source>/<version>/      # downloads as-is (GAMUS h5, DFC19, swisstopo tiles, LINZ tiles, 3DEP EPT/LAZ, NAIP)
  processed/<source>/<gsd>/    # co-registered RGB–nDSM tiles (COG), per-tile terrain class, footprints
  splits/<split_id>/           # train/val/test tile lists with buffers and rationale
  calibration/                 # DEM tiles (Copernicus/SRTM/…), geoid grids copy, anchor sets (ATL08 granules) — runtime copies live in assets/
  reference/<source>/          # LiDAR DSM/DTM/nDSM references for validation (never used in training)
  validation/<run_id>/         # harness outputs (validation.json, residuals, plots)
  outputs/<job_id>/            # runtime job directories (or configurable DW_DATA_DIR)
  demo/                        # curated demo inputs + precomputed demo jobs (flagged) — copied into assets/ at packaging
```
Naming: `<source>_<region>_<gsd>m_<yyyymmdd>_<tileid>.tif`; every folder has a `MANIFEST.json` (source, licence, download date, hash). Git tracks only `manifests/`, small fixtures in `tests/fixtures/`, and configs. Large artefacts move via `scripts/fetch_*.py` and release assets.

---

## 12. Model Management

- **Registry:** `models/<name>/<semver>/` with `weights.safetensors`, `model_card.json` (backbone, licence, training manifest hash, canonical GSDs, preprocessing params, loss, epochs, seed, validation tables per terrain/class, date), `preprocess.json`.
- **Names:** `da-v2-small-baseline` (zero-shot relative head, Apache-2.0), `dw-ndsm-small` (fine-tuned nDSM head), versions `0.1.0`, `0.2.0`….
- **Checksums:** sha256 of weights and card in `models/INDEX.json`; loader verifies before use.
- **Download/caching:** `scripts/fetch_model.py` pulls from the project release assets (or HF mirror) into `models/`; offline package bundles the selected version.
- **Licensing:** card records licence; CI check refuses packaging any non-Apache/CC-BY-compatible weight (e.g., DA-V2 Base/Large CC-BY-NC).
- **Reproducibility:** every job writes `model_hash`, `preprocess_hash`, `config_hash`; every GeoTIFF gets `MODEL_HASH` tag; `GET /system` lists loaded model versions.

---

## 13. Dependency-Aware Implementation Order

| Task ID | Task | Dependencies | Deliverable | Definition of done | Complexity | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | Environment + repo skeleton + CI gates | — | repo, venv, `pyproject`, ruff/pytest running | `pytest` green on empty suite; frontend builds | Low | version conflicts (py 3.12) |
| T02 | Fixtures generator (synthetic rasters) | T01 | `scripts/make_fixtures.py`, `tests/fixtures/` | plane/slope/step/canopy/anchors fixtures with known truth | Low | — |
| T03 | `core/geo` (Grid, meta, resample, vertical, writer, check) | T01, T02 | package + unit tests | round-trips exact; datum values match PROJ within 1 cm; COG valid | Medium | PROJ grid paths in packaged env |
| T04 | `core/ingest` | T03 | package + tests | all format/edge cases behave per Phase 6 §3 | Medium | exotic TIFFs |
| T05 | Backend skeleton: app, config, store, sqlite, job state machine, `/upload`, `/job`, `/system` | T01, T04 | running server | upload → job CREATED; state persisted | Medium | — |
| T06 | `core/preprocess` (normalize, canonical GSD, tiling, mosaic) | T03 | package + tests | tiles cover grid; inverse resample error tiny; deterministic | Medium | — |
| T07 | `core/inference` with **stub** and **baseline zero-shot** DA-V2-S | T06, model fetch | mosaic from real baseline weights | end-to-end relative mosaic on sample; OOM cascade unit-tested | Medium | torch install |
| T08 | Stage wiring: PREPROCESSING → INFERENCE in worker; `/process`, `/result`, `/raster` (PNG preview) | T05, T07 | Mode A end-to-end (relative) | upload PNG → `rdsm.tif` + preview | Medium | — |
| T09 | `core/terrain` tileset + frontend viewer core (loader, RTIN mesh, orbit, texture) | T03, T08 (or fixture DSM) | 3D view of rdsm/dsm | mesh residual computed; RGB draped | High | LOD/skirts |
| T10 | `core/calib` Level T: discover, DEM load + datum, ground mask, NC terrain, checks, tier, report; bundled DEM index | T03, T07 | `terrain.tif`, `calib_report.json` | synthetic canopy-bias test passes; tier rules tested | High | datum/grid handling |
| T11 | `core/dsm` compose + slope/aspect + flags; COG writing with vertical tag; `geo.check` in pipeline | T10 | `dsm.tif`, `slope.tif`, `flags.tif` | analytic slope test; grid asserts; QGIS opens with correct CRS | Medium | — |
| T12 | Mode B end-to-end in worker (GeoTIFF → DSM → tileset) | T08, T10, T11, T09 | live Mode B job | tier badge T shown; metadata panel | Medium | — |
| T13 | Frontend: upload/options, status, 2D views, badges, point & 2-point tools via `/sample`, metadata panel, export | T08, T12 | usable UI | measurements labelled by tier; Mode A watermark | Medium | — |
| T14 | First-person controls with terrain clamp; layers (DSM ramp, slope) | T09 | navigation | clipping tests pass | Medium | — |
| T15 | `ml/data` prep for GAMUS + one LiDAR source (CH or NZ or US): derive nDSM, tiles, blocked splits, overlap check, manifests | T03 | processed tiles + manifests | manifest complete; split leakage check passes | High | download volume; datum of DTM/DSM |
| T16 | `ml/train` fine-tune DA-V2-S nDSM head v0.1 + model card + validation tables | T15, GPU | `dw-ndsm-small/0.1.0` | beats baseline on held-out blocks (no threshold asserted; ablation reported) | High | GPU time; loss choice |
| T17 | Swap fine-tuned model into inference (`Level H`), keep baseline for ablation | T16, T07 | metric nDSM in pipeline | tier H reachable | Low | — |
| T18 | `core/validate` harness: reference load, align, co-register, masks, metrics, stratify, residual, leakage check; `/validate` endpoints; UI panel | T11 | validation.json + residual overlay | synthetic shift recovered; metrics match independent recomputation | High | reference datums |
| T19 | `core/quality` states; uncertainty (TTA) raster + overlay | T17, T18 | `quality.json`, `uncertainty.tif` | states triggered correctly on fixtures | Medium | — |
| T20 | Anchors module (GCP CSV/GeoJSON; hold-out; robust fit; acceptance) + UI upload | T10 | tier A reachable | synthetic anchor transform recovered within tolerance; hold-out residuals reported | Medium | — |
| T21 | Packaging (PyInstaller) with bundled assets; clean-machine offline test; demo mode | T12, T13, T14 | executable | runs offline on clean VM; demo jobs labelled | High | torch/GDAL bundling |
| T22 | DFC19 protocol runner; ablations C0–C8 scripts; experiment tracking | T17, T18 | `ml/eval` outputs | scripts run end-to-end on subsets | Medium | data access |
| T23 | Advanced: ICESat-2 ATL08 loader; semantic head; profile/polygon tools; contours; glTF export; facade heuristic | T20, T16, T14 | features | each with tests | Medium–High | time |
| T24 | Docs (all guides), licence manifest, reproducibility check | all | `docs/` | reviewer can reproduce a demo job | Medium | — |

---

## 14. MVP Development Plan

| ID | Objective | Dependency | Deliverable | Acceptance criterion |
| --- | --- | --- | --- | --- |
| MVP-01 | Geo core + fixtures + ingestion | T01–T04 | `core/geo`, `core/ingest`, tests | All fixture formats classified correctly; CRS/transform round-trip exact; datum test values match PROJ |
| MVP-02 | Backend skeleton with persisted jobs | T05 | server + sqlite | `POST /upload` → job; restart preserves state |
| MVP-03 | Mode A end-to-end with baseline model | T06–T08 | `rdsm.tif`, preview | PNG → relative raster, RELATIVE label in manifest |
| MVP-04 | 3D viewer on rdsm | T09 | tileset + Three.js scene | mesh renders; RGB aligned; mesh residual displayed |
| MVP-05 | Level T calibration + DSM + COG | T10–T11 | `terrain.tif`, `dsm.tif`, tags | synthetic canopy test passes; QGIS shows correct CRS/vertical tag |
| MVP-06 | Mode B end-to-end | T12 | live job | tier T badge; datum label; slope layer |
| MVP-07 | Measurement + navigation UI | T13–T14 | tools, first-person | point/2-point values from `/sample`, labelled; first-person clamps to terrain |
| MVP-08 | Fine-tuned model v0.1 in pipeline | T15–T17 | `dw-ndsm-small/0.1.0` | tier H metric nDSM; ablation vs baseline reported (no threshold asserted) |
| MVP-09 | Validation harness + panel | T18 | validation.json, residual overlay | synthetic shift recovered; per-terrain tables when class raster present |
| MVP-10 | Packaged offline demo | T21 | executable + demo assets | runs on clean machine offline; demo jobs flagged |

---

## 15. Full Implementation Plan

- **CORE (MVP-01…10 above)** — must all be complete before any Advanced work is merged to `main`.
- **ADVANCED (in this order):** A1 anchors module + UI (T20); A2 uncertainty raster + AUSE + overlay (T19); A3 quality states + flags overlay (T19); A4 DFC19 protocol + ablations + experiment tracking (T22); A5 semantic head + class-wise metrics (T23); A6 ICESat-2 loader (T23); A7 profile/polygon tools, contours, residual overlay (T23); A8 cross-country & India assessment scripts (T22/T23); A9 observability panel, resume-on-restart hardening, retention (backend).
- **OPTIONAL:** glTF/OBJ export; facade heuristic; histogram matching; scalar GSD embedding (D-10 Advanced); LoD1 extrusion; Tauri shell; Linux build; Docker for evaluation.

---

## 16. Backend Implementation Plan

- **Entrypoint:** `backend/app.py` — loads settings, ensures dirs/grids/models, opens sqlite, starts worker thread, mounts routers under `/api/v1`, serves `frontend/dist` at `/`.
- **Routing:** one router per resource (upload, process, jobs, results, rasters, terrain, validate, metadata, export, system).
- **Services:** thin service layer calling `core/*`; `jobs/stages.py` maps state → stage function; each stage receives `JobContext(job_dir, meta, config, logger)` and returns artefact paths.
- **Job state:** sqlite `jobs(id, state, stage, progress, mode, tier, created, updated, config_hash, model_hash, error_json)`; `job.json` mirror; events appended to `log.jsonl`.
- **File handling:** uploads streamed to `jobs/<id>/input/`; magic-byte + driver whitelist; sha256; size cap.
- **Model inference interface:** `core.inference.Predictor(card, device_policy).predict(tile_iter) -> Mosaic`.
- **Calibration interface:** `core.calib.run(job_ctx, ndsm_path) -> CalibResult(terrain_path, tier, report)`.
- **Validation interface:** `core.validate.run(job_ctx, reference_spec) -> ValidationResult`.
- **Error handling:** `backend/errors.py` taxonomy (§28) → uniform `{error: {code, message, details, recoverable, fallback}}`.
- **Logging:** JSON lines; per-job file + console.

| Method | Path | Request | Response | Status | Error | Sync |
| --- | --- | --- | --- | --- | --- | --- |
| POST | `/api/v1/upload` | multipart image (+dem, anchors, reference, options JSON) | `{job_id, mode, meta, warnings}` | 201 | 400 InvalidFile/Unsupported; 413 TooLarge; 422 InvalidGeoref | Sync |
| POST | `/api/v1/process/{job_id}` | `{profile, calib{dem_source, output_vertical_crs, anchors{holdout_fraction}}, tta, cleanup, tolerance_m, aoi}` | `{job_id, state}` | 202 | 404; 409 running | Async |
| GET | `/api/v1/job/{job_id}` | — | `{state, stage, progress, warnings, error}` | 200 | 404 | Sync |
| GET | `/api/v1/result/{job_id}` | — | manifest | 200 | 404; 409 incomplete | Sync |
| GET | `/api/v1/raster/{job_id}/{layer}` | `?format=tif|png&overview=` | file | 200 | 404 | Sync |
| GET | `/api/v1/raster/{job_id}/{layer}/sample` | `?x&y&crs` or `?polyline`/`?polygon` (GeoJSON) | `{values, units, label, tier, datum}` | 200 | 400 | Sync |
| GET | `/api/v1/terrain/{job_id}/tileset.json`; `/terrain/{job_id}/{z}/{x}/{y}.{hgt,jpg}` | — | JSON / binary | 200 | 404 | Sync |
| POST | `/api/v1/validate/{job_id}` | multipart reference + `{type, vertical_crs, class_mask?, footprints?}` | `{validation_id, state}` | 202 | 422 MissingVCRS; 404 | Async |
| GET | `/api/v1/validation/{job_id}` | — | validation.json (+ residual URL) | 200 | 404; 204 not_validated | Sync |
| GET | `/api/v1/metadata/{job_id}` | — | meta + calib report + model card + quality | 200 | 404 | Sync |
| GET | `/api/v1/export/{job_id}` | `?include=` | zip stream | 200 | 404 | Sync (stream) |
| GET | `/api/v1/system` | — | device, versions, assets, demo flag | 200 | — | Sync |
| DELETE | `/api/v1/job/{job_id}` | — | — | 204 | 404 | Sync |

---

## 17. Frontend Implementation Plan

| # | Area | Components | State | API | Data | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Upload | `DropZone`, `OptionsForm` (band map, CRS/GSD override, output datum, DEM source, anchors file, tolerance) | draft options | `/system`, `/upload`, `/process` | meta preview, warnings | Invalid file → clear message; Mode shown before processing |
| 2 | Processing state | `JobStatus`, stage stepper, cancel | polling job | `/job` | stage/progress | Stages advance; failure shows reason + fallback |
| 3 | Result dashboard | `ResultManifest`, tier/datum badges, downloads | manifest | `/result` | artefacts | Tier badge correct per job |
| 4 | 2D views | `RasterView` (RGB, rDSM/nDSM/DSM ramps, slope, flags), legend, hover readout | layer, ramp | `/raster/*.png`, `/sample` | PNG overviews | Hover shows labelled value |
| 5 | 3D viewer | `Scene`, `TileLoader`, `RTINMesh`, `CameraModes`, `Layers`, `Badges`, `ToleranceControl` | camera, layer, tolerance, exaggeration | `/terrain/*` | tiles | Residual displayed; exaggeration badge; texture aligned |
| 6 | Height analysis | `PointTool`, `TwoPointTool` (MVP), `ProfileTool`, `PolygonTool` (Adv.) | picks | `/sample` | values | Labels ABSOLUTE / METRIC RELATIVE / RELATIVE by tier |
| 7 | Slope analysis | slope layer + legend + pick readout | — | `/raster/slope` | degrees | Degrees hidden in Mode A |
| 8 | Validation panel | `ReferenceUpload`, `MetricsTable` (overall/terrain/class), `Histogram`, `ResidualOverlay` | validation state | `/validate`, `/validation` | validation.json | "not validated" state explicit |
| 9 | Metadata panel | `MetaView`, `CalibReportView`, `ModelCardView`, `FlagsSummary` | — | `/metadata` | JSON | All provenance visible |
| 10 | Export | `ExportDialog` | selection | `/export` | zip | Download completes; manifest hashes |

Sequence: 1–3 (stubbed API) → 4 → 5 (with fixture tileset) → 6/7 → 9 → 8 → 10.

---

## 18. 3D Engine Implementation Plan

**GEOMETRY:** `tileset.json` schema (bounds, crs, gsd, levels, tileSize=257, heightEncoding, nodata, exaggerationDefault, tier, datum) → `TileLoader` (quadtree, screen-space error selection, frustum culling, cache) → `RTINMesh` per tile (martini: `new Martini(257)`, `createTile(heights)`, `getMesh(tolerance_m / exaggeration)`), vertices in local metres (x,y from tile origin × GSD; z height), skirts, nodata triangles dropped, normals via `computeVertexNormals` → `residual.ts` samples original heights vs mesh (barycentric) → displayed RMSE/max.
**TEXTURE:** JPEG texture per tile; UV = normalized tile coords; `SRGBColorSpace`; layer textures (ramps) generated server-side as PNG tiles or client-side from heights via shader (slope/ramps computed in fragment shader from height texture for fidelity).
**INTERACTION:** `OrbitControls` (aerial), `FlyControls`-style WASD (aerial fly, altitude clamp), `PointerLockControls` first-person with eye height and per-frame terrain clamp (height sampled from height tiles, not mesh); mode switch with smooth transition; bookmarks; loading progress bar.
**ANALYSIS:** raycast pick → local coords → pixel/CRS via meta → `/sample`; two-point overlay line with Δz/distance; (Adv.) polyline profile chart; polygon median; slope layer legend; uncertainty/flags overlays; terrain-only layer toggle.
**LARGE SCENES:** max levels from config; tile cache eviction; tolerance auto-raise when frame time exceeds budget (announced in UI).

---

## 19. Geospatial Implementation Plan

| Task | Implementation | Test |
| --- | --- | --- |
| GeoTIFF reading | rasterio open, tags, nodata, dtype, band count; windowed reads | fixture GeoTIFFs (UTM, geographic, rotated, no-CRS, nodata, uint16) |
| CRS extraction | `rasterio.crs.CRS` → pyproj `CRS`; EPSG lookup; vertical component detection | resolve/unresolvable cases |
| Affine transform | rasterio `Affine`; rotation detection; pixel-centre convention | forward/inverse pixel↔CRS exact for corners/centres |
| Bounds | `Grid.bounds`, to EPSG:4326 for discovery | fixture with known bounds |
| Reprojection | `rasterio.warp.reproject` into `Grid` (bilinear/cubic/average) | synthetic plane reprojection preserves gradient |
| Raster alignment | `Grid.assert_same`; all outputs written on input grid | integration assert on every artefact |
| Resampling | canonical GSD (average down / cubic up) + exact inverse | inverse error on smooth field < 1e-3 relative; discontinuity test documented |
| Nodata | masks propagated; −9999 written; alpha → mask | nodata counts equal across outputs |
| Vertical reference | pyproj transformers EPSG:4979↔9707/9518, NAVD88/NAP/LN02/NZVD2016 pipelines; grid presence check | Delhi/Kanyakumari/Leh EGM96 & EGM2008 values match PROJ within 1 cm [F13] |
| DSM writing | COG (LZW, overviews), tags (`VERTICAL_CRS`, `TIER`, hashes) | `validate_cog`; tags round-trip; QGIS manual check |
| Metadata preservation | `meta.json` → tags; bounds equality | equality tests |

---

## 20. Calibration Implementation Plan

| Task | Implementation | Logged in `calib_report.json` |
| --- | --- | --- |
| Input discovery | `assets/index.json` (tile bounds → files); user uploads; anchors file parse | sources, versions, tile IDs |
| DEM loading | reproject to job CRS at native posting; void mask; vertical transform to output vcrs | src vcrs, dst vcrs, grid file hashes, void % |
| Ground mask & support | `ndsm < h_ground`; block-average to coarse grid → `w` | h_ground, support histogram |
| Normalized convolution | Gaussian kernel σ cells; `T = (K∗(w·D))/(K∗w)`; where `K∗w < w_min` → raw DEM + flag | σ, w_min, raw-fallback % |
| Upsampling | bicubic to job grid | method |
| Consistency check | `lowpass(terrain + ndsm) − dem` ME/NMAD; datum sanity threshold | ME, NMAD, DATUM_SUSPECT |
| GCP input | CSV/GeoJSON schema: id, x, y, z, vcrs, type{ground,object_top}, sigma | count, source |
| Control-point validation | inside bounds; finite; duplicate removal; datum transform | rejected ids |
| Hold-out split | if N ≥ min_n: stratified spatial split (k-means on xy), fraction from config, seed | ids used/held, seed |
| Fitting | offset via TheilSen on ground residuals; tilt (plane) if N ≥ 10 and spread ok; object scale via RANSAC on (nDSM, z−terrain) if ≥ 5 object anchors | params, n_used |
| Outlier rejection | RANSAC inliers; residual > 3σ flagged as blunders (ASPRS) [F7] | blunder ids |
| Confidence | post-fit NMAD vs k·σ_anchor; hold-out ME/NMAD/RMSE | values, accept/reject |
| Fallback | reject → tier T; no DEM → tier H; no GSD → tier R | tier, reasons |
| Versioning | `method_version`, timestamp, config hash | — |

---

## 21. Validation Implementation Plan

Independence: `core/validate` imports nothing from `core/calib` fitting; it reads `calib/anchors_used.geojson` only to **exclude** those points/radii (`leakage.py`). Training/tuning never reads `data/reference/`.

| Step | Implementation |
| --- | --- |
| Reference loading | GeoTIFF (dsm/ndsm/dtm) or points CSV; declared `vertical_crs` mandatory (422 otherwise); type must match claimed target (dsm↔dsm, ndsm↔ndsm; dtm only for terrain-layer check) |
| Vertical normalization | `geo.vertical.transform_heights` to job output vcrs |
| Spatial alignment | reproject reference to job CRS; comparison grid = coarser posting; block mean (default) or block max (option) |
| Resolution matching | aggregation factor recorded; never upsample coarser reference |
| Co-registration | Nuth–Kääb on ground pixels with slope > 5°; report (dx, dy, dz); `ShiftUndetermined` if flat |
| Valid-pixel mask | nodata both; water (if mask); border band; change mask (optional); anchor exclusion radius |
| Metrics | ME, RMSE, MAE, NMAD, LE90, LE95, Pearson r, Spearman ρ, affine (s,t) fit reported; Mode A: metrics after affine only |
| Terrain subsets | class raster from reference DTM slope thresholds + land cover (config) → urban/sparse/hilly/forested tables; class tables; height bins; RMSE-B and F1-HE when footprints/labels supplied |
| Residual maps | `residual.tif` (pred − ref), histogram PNG, error-vs-slope table |
| Report | `validation.json` with eval_config hash, reference hash, shift, masks, counts |
| Points mode | ≥ 30 points per class recommended [F7]; bilinear sampling; same metrics |

---

## 22. Testing Strategy

| Category | What to test | Minimum set | Expected result | Failure interpretation |
| --- | --- | --- | --- | --- |
| Unit | geo transforms, datum, resample inverse, tiling coverage, NC terrain, anchor fit, tier rules, slope, metrics, masks, flags | ~60 tests | exact/tolerance asserts | math or convention bug |
| Integration | ingest→preprocess→stub inference→mosaic; ndsm+dem→terrain→dsm→COG; dsm→tileset; upload→result via API (stub) | ~15 | artefacts on same grid, tags present | wiring/contract bug |
| End-to-end | Mode A and B on samples with baseline and fine-tuned models; CPU and GPU; restart/resume; export | 6 | COMPLETE state, manifest complete | stage failure |
| Geospatial | fixtures: UTM, geographic, rotated, no-CRS, nodata, uint16, BigTIFF | 7 | correct mode/meta/outputs | metadata handling bug |
| ML pipeline | stub model determinism; TTA inverse mapping; OOM cascade (monkeypatched); model card hash check | 6 | identical outputs; cascade path taken | inference bug |
| Calibration | synthetic canopy bias reduction; raw-fallback where no ground; datum-suspect trigger; anchor recovery with outliers; hold-out disjoint | 8 | recovered params within tolerance | calibration bug |
| 3D | RTIN residual ≤ tolerance on fixture; skirts; nodata holes; UV alignment (checkerboard texture) | 5 (vitest) | asserts | mesh bug |
| Frontend | upload flow, badges per tier, watermark Mode A, tool labels, first-person clamp, validation states | 8 (Playwright) | UI states | UX/contract bug |
| API | schemas, status codes, error taxonomy, sample endpoint units | 12 | per contract | API bug |
| Deployment | packaged run offline on clean VM (Windows), demo job reproducible hash | 2 manual | runs; hashes match | packaging bug |

---

## 23. Test Fixtures (generated by `scripts/make_fixtures.py`, committed, each ≤ 200 KB)

- `rgb_64.png`, `rgb_64.jpg` — tiny checkerboard/gradient RGB.
- `geo_utm_64.tif` — 64² RGB GeoTIFF, EPSG:32643, 1 m pixels, nodata alpha.
- `geo_wgs84_64.tif` — geographic CRS, tests local GSD derivation.
- `geo_rotated_64.tif` — rotated transform.
- `tiff_nocrs_64.tif`, `tiff_uint16_4band_64.tif`.
- `plane_dsm_128.tif` — z = a·x + b·y + c (known slope/aspect).
- `step_ndsm_128.tif` — ground 0 m with 10 m and 30 m blocks (buildings), 15 m "canopy" region.
- `dem_coarse_canopy.tif` — 30 m DEM = true terrain + 4 m over canopy cells (synthetic bias).
- `terrain_true_128.tif` — ground truth terrain for NC test.
- `anchors_ground.csv` — 12 ground points on true terrain + 2 outliers; `anchors_object.csv` — 6 roof points.
- `ref_shifted.tif` — plane_dsm shifted by (2, −1) pixels for co-registration test.
- `stub_model` — deterministic function returning `step_ndsm`-like output for pipeline tests.
- `tileset_fixture/` — tiny 2-level tileset from `plane_dsm_128`.

---

## 24. Synthetic Test Data (mathematical correctness)

| Verifies | Construction | Assertion |
| --- | --- | --- |
| Scale recovery (anchors) | `ndsm_true × 1.2` fed as prediction; object anchors from truth | RANSAC scale ≈ 1/1.2 within 1 % |
| Offset recovery | terrain − 3.0 m; ground anchors | TheilSen offset ≈ +3.0 within 0.05 m; outliers excluded |
| Affine alignment (Mode A metrics) | rdsm = (z − t)/s | fitted (s,t) recovered; residual ≈ 0 |
| Slope | plane fixture | slope = arctan(√(a²+b²)) within 0.01°; aspect within 0.1° |
| Height (nDSM) | step fixture | picks return 10/30 m; polygon median exact |
| DSM generation | terrain_true + step_ndsm | dsm equals sum; nodata propagates |
| Normalized convolution | dem_coarse_canopy vs terrain_true | terrain error over canopy cells reduced ≥ 50 % vs raw DEM (fixture-specific), zero over ground cells |
| Datum | PROJ reference points [F13] | EGM96/EGM2008 N within 1 cm |
| Co-registration | ref_shifted | recovered shift (2, −1) px within 0.2 px |
| Mesh geometry | RTIN of plane at tolerance 0.1 m | max |mesh − raster| ≤ 0.1 m; UV maps checkerboard corners exactly |
| Tiling | random shapes | union of windows = grid; weights sum to 1 in overlaps |

---

## 25. Acceptance Criteria (module-level, objective)

- **Ingestion:** all fixture valid files accepted with correct mode; corrupt/unsupported rejected with taxonomy codes.
- **Georeferencing:** every output raster passes `assert_grid` against input; tags round-trip; datum values within 1 cm of PROJ.
- **Preprocessing:** deterministic; tile union covers grid; inverse canonical resampling error < 1e-3 relative on smooth field.
- **Inference:** stub deterministic; OOM cascade unit-tested; baseline and fine-tuned models load with hash check.
- **Calibration:** synthetic offset/scale recovered within tolerances above; canopy-bias reduction demonstrated on fixture; tier/flag rules match table; report schema valid.
- **DSM:** dimensions/CRS/transform/nodata identical to input; vertical tag present; slope on plane within 0.01°.
- **Validation:** metrics equal an independent numpy recomputation; synthetic shift recovered; leakage check excludes anchors.
- **3D:** mesh residual ≤ tolerance on fixtures; texture corners align; first-person camera never below terrain in scripted path.
- **API:** contracts honoured; error responses uniform.
- **Deployment:** offline run on a clean machine; demo job hash reproducible on same device class.
- **ML accuracy:** no numeric threshold asserted here (Phase 4 constraint); acceptance = ablation report produced on spatially blocked held-out data comparing fine-tuned vs baseline, per terrain.

---

## 26. Experiment Tracking

Lightweight: each training/eval run writes `experiments/<run_id>/run.json` (dataset manifest hash, split id, model version/hash, config hash, calibration method version, reference hashes, hardware, torch/CUDA versions, seed, start/end, metrics tables, artefact paths) plus `metrics.csv`; `ml/eval/tables.py` aggregates runs into `docs/experiments.md`. Optional: MLflow local file store if the team already uses it — not required.

---

## 27. Logging / Debugging

- **INFO:** job created (mode, size, GSD); stage start/end with durations; device/batch chosen; DEM source/tiles; tier decision; output paths/hashes; validation headline metrics; export.
- **WARNING:** flags (GSD out of range, low quality, raw-DEM fallback %, datum suspect, anchor rejected, shift undetermined, vertical assumed); OOM cascade steps; retention deletions.
- **ERROR:** stage failures with exception, stage, job id, input meta summary; write failures; asset missing.
- **DEBUG:** tile windows, per-tile timings, NC parameters, fit iterations, residual samples, tileset levels.
Per-job `log.jsonl` + `report.json` make any failure reproducible from the job directory alone.

---

## 28. Error Handling (taxonomy in `backend/errors.py`)

| Code | User message | Internal log | Recoverable | Fallback |
| --- | --- | --- | --- | --- |
| `INVALID_FILE` | "File could not be decoded." | decoder exception | No | — |
| `UNSUPPORTED_FORMAT` | "Supported: PNG, JPEG, TIFF/GeoTIFF." | driver/mime | No | — |
| `MISSING_CRS` | "No coordinate system found — processed as relative (Mode A). You may supply CRS/GSD." | meta dump | Yes | Mode A / user georef |
| `INVALID_GEOREF` | "Georeferencing invalid (reason). Absolute mode unavailable." | CRS/transform details | Yes | Mode A |
| `IMAGE_TOO_LARGE` | "Image exceeds N megapixels — select an area or allow downsampling." | dims | Yes | AOI / canonical downsample |
| `MODEL_UNAVAILABLE` | "Height model not found; using baseline relative model." | registry error | Yes | baseline (tier ≤ R + labelled) |
| `GPU_OOM` | "GPU memory insufficient — continuing on CPU (slower)." | torch error, batch | Yes | CPU |
| `CALIBRATION_FAILED` | "Absolute calibration not defensible (reason); delivering height above ground only." | checks | Yes | tier H |
| `INSUFFICIENT_REFERENCE` | "Reference lacks vertical datum / no overlap — validation not run." | spec | Yes | not_validated |
| `OUTPUT_WRITE_FAILED` | "Could not write outputs (disk?)." | OS error | No | — |
| `MESH_FAILED` | "3D tiles could not be generated; rasters remain available." | exception | Yes | 2D views |
| `VALIDATION_FAILED` | "Validation error (reason)." | exception | Yes | retry with options |

---

## 29. CI / Quality Gates

GitHub Actions (or equivalent), single workflow: `ruff check` + `ruff format --check`; `mypy core backend --ignore-missing-imports` (advisory); `pytest tests/unit tests/integration -q` (CPU, stub model, fixtures); `npm ci && npm run lint && npm run test && npm run build` in `frontend/`; licence check script (`scripts/check_licenses.py`) refusing non-compatible weights in `models/INDEX.json`; artefact size guard (no files > 5 MB outside `assets/` fetched at runtime). E2E/Playwright and packaging run manually before milestones.

---

## 30. Dependency Management

- `pyproject.toml` with pinned ranges; `requirements/base.txt` (exact pins, CPU torch), `gpu.txt` (torch index note), `dev.txt`, `package.txt` (PyInstaller set). Regenerate locks with `pip-compile`/`uv pip compile`.
- **System deps:** none beyond Python/Node on Windows/Linux for runtime (rasterio/pyproj wheels bundle GDAL/PROJ); dev data env uses conda for PDAL.
- **Known conflict points:** Python 3.12 requirement [I1]; numpy 2.x ABI vs any old wheel (avoid); rasterio and pyproj bundle *different* PROJ copies — set `PROJ_DATA` for pyproj (used for all vertical transforms) and avoid GDAL-side vertical transforms; torch CPU vs CUDA wheel selection [I4]; PyInstaller hidden imports for rasterio drivers and torch.
- **Frontend:** `package-lock.json` committed; pinned three/martini/vite/typescript versions [I2].
- **Model deps:** vendored DA-V2 model definition (pinned commit) inside `ml/train/model.py`/`core/inference/model.py` to avoid upstream drift.

---

## 31. Docker / Packaging Decision

- **Primary deliverable: PyInstaller executable (Windows first)** per Phase 6 D-07 — evaluators run it offline. Build recipe: `scripts/package_win.py` → one-folder bundle (more robust than one-file for GDAL/torch), including `assets/`, `models/<selected>`, `frontend/dist`, PROJ grids, with `PROJ_DATA` set at startup; smoke test launches server and runs a demo job headless.
- **Docker: yes, but only for evaluation reproducibility** (`Dockerfile.eval`: CPU torch, harness, fixed model) — not for the demo. Rationale: judges' machines cannot be assumed to have Docker; Phase 4 clean-machine criterion favours a native bundle.
- No separate containers for frontend/inference (monolith).

---

## 32. Demo Deployment Plan

- **Machine:** laptop, 16 GB RAM, Windows 11, optional NVIDIA GPU; 20 GB free; backup laptop with identical bundle.
- **Startup:** run `DepthWizard.exe` → server on 127.0.0.1:8000 → browser opens; `GET /system` shows device, model version, assets OK.
- **Model loading:** bundled `dw-ndsm-small/<ver>` + baseline; hash check at start.
- **Test images:** bundled samples: (a) PNG city crop (Mode A); (b) GeoTIFF from an open source (e.g., swisstopo/LINZ/NAIP subset) with bundled DEM tiles (Mode B, tier T); (c) same with anchors CSV (tier A, if Advanced done); (d) reference DSM subset for the validation panel (licence noted).
- **Flow:** upload → process → 3D flythrough → measurements → validation panel → export; then remove DEM to show honest tier downgrade.
- **Fallback data:** precomputed demo jobs (flagged `precomputed_demo`) for CPU-only situations; 2D views if WebGL fails.
- **Network:** none required; verify by disabling Wi-Fi during rehearsal.
- **Recovery:** restart executable (jobs persist); rerun from precomputed job; switch to backup laptop (same bundle hash).

---

## 33. Artifact Versioning

Models: semver + sha256 in `models/INDEX.json`; datasets: manifests with hashes and download dates (tracked), data untracked; configs: tracked, hashed per job; outputs: per job dir with `report.json` (all hashes), retention policy; calibration data: DEM tile IDs/versions and grid hashes recorded per job; validation reports: `validation/<run_id>` with eval_config hash; mesh assets: derived, regenerable, not versioned beyond tileset params in `tileset.json`. Git LFS is not used; large binaries via release assets + fetch scripts.

---

## 34. Documentation Plan (in `docs/`)

README (what/why/quick start) · setup.md · model.md (versions, card, licence, canonical GSDs, baseline vs fine-tuned) · data.md (sources, manifests, splits, licences) · calibration.md (tiers R/H/T/A, NC terrain, anchors, report schema) · geospatial.md (Grid, CRS, datum handling, EGM values, COG tags) · validation.md (blueprint, metrics, stratification, leakage rules) · api.md · demo.md (script, fallbacks) · troubleshooting.md (error codes) · architecture.md (Phase 6 summary + diagrams) · experiments.md (Phase 8 results, generated) · output_contracts.md · LICENSES.md.

---

## 35. Team Parallelization (roles M1–M6 from Phase 6 §27.1; no names in project docs)

| Developer | Workstream | Tasks | Dependencies | Deliverables | Handoff |
| --- | --- | --- | --- | --- | --- |
| A (M1 geo/calib) | Geo core, calibration, DSM | T02–T04, T10, T11, T20 | — | `core/geo`, `core/calib`, `core/dsm`, fixtures | `Grid`/`Meta` contract day 1; calib report schema to F |
| B (M2 ML) | Data prep, training, inference | T06, T07, T15–T17, T19 (uncertainty) | A's geo for data prep | `core/preprocess`, `core/inference`, model v0.1 + card | canonical GSDs to A/C; weights to C |
| C (M3 backend) | API, jobs, store, packaging, demo | T05, T08, T12, T21 | A's ingest; B's predictor interface (stub first) | server, executable, demo assets | API stubs to E/F day 1 |
| D (M4 validation) | Harness, quality, experiments | T18, T19 (quality), T22 | A's geo; C's outputs | `core/validate`, eval scripts, experiments.md | validation.json schema to F |
| E (M5 3D) | Tileset + viewer mesh/cameras/layers | T09, T14 | fixture DSM; tileset schema with C | `core/terrain`, `frontend/src/viewer` | tileset.json schema day 2 |
| F (M6 frontend) | Pages, panels, tools, UX, SUS | T13, panels, tests | C's stubs; E's viewer | SPA | integration with C |

---

## 36. Git Workflow

`main` (protected, always demo-runnable) ← `develop` (integration) ← feature branches `feat/<area>-<short>`; fixes `fix/…`; experiment branches `exp/…` (may be discarded). Commits: Conventional Commits (`feat(calib): normalized convolution terrain`). PRs into `develop` require CI green + one review; merge order follows critical path (geo → backend skeleton → inference → calib → dsm → terrain → frontend). Tags: `m1-input`, `m2-baseline`, …, `demo-v1`; `release/demo` branch frozen 24 h before demo with the packaged bundle hash recorded in `docs/demo.md`.

---

## 37. Implementation Milestones

| Milestone | Deliverables | Blockers | Acceptance | Demo capability |
| --- | --- | --- | --- | --- |
| M0 Environment Ready | T01, T02 | py 3.12 toolchain | CI green; fixtures | — |
| M1 Input & Geo Core | T03, T04, T05 | — | unit tests; upload creates job | upload + metadata panel |
| M2 Baseline Relative Pipeline | T06–T08 | torch install | Mode A end-to-end with baseline | rDSM preview |
| M3 3D Viewer | T09, T14 | tileset schema | mesh + residual + navigation | flythrough on rDSM |
| M4 Level-T Calibration & DSM | T10–T12 | DEM assets, grids | tier T job; QGIS check | Mode B flythrough with datum badge |
| M5 UI & Tools | T13 | API | labelled measurements | height/slope picks |
| M6 Fine-tuned Model | T15–T17 | GPU, data | tier H; ablation report | metric nDSM |
| M7 Validation | T18 | reference subsets | panel with tables/residual | validation demo |
| M8 Robustness & Advanced | T19, T20, retries, flags | — | anchors tier A; uncertainty overlay | honest downgrade demo |
| M9 Demo Ready | T21, T24 | packaging | offline clean-machine run | full script |

---

## 38. Critical Path

T01 → T03 (geo) → T05 (skeleton) → T06/T07 (tiles + baseline inference) → T08 (Mode A E2E) → T10 (Level T) → T11 (DSM/COG) → T12 (Mode B E2E) → T09/T14 (viewer) → T13 (UI) → T21 (packaging) — with T15→T16→T17 (fine-tuned model) as a **parallel critical chain** that must land before M9.

| Task | Why critical | Risk | Fallback |
| --- | --- | --- | --- |
| T03 geo core | Everything asserts on `Grid`; datum transforms | PROJ grid paths | pyproj network off; bundled grids tested day 1 |
| T07 baseline inference | Unblocks E2E before fine-tuning | torch/CUDA install | CPU wheel |
| T10 Level T | The PS's absolute-DSM requirement | datum/NC bugs | raw-DEM fallback path still yields tier T with flag |
| T12 Mode B E2E | Core demo | integration | keep Mode A demo |
| T16 fine-tune | Metric nDSM (tier H) | GPU time/data | ship baseline + explicit "relative + DEM offset" labelling; ablation still reportable |
| T21 packaging | "Standalone deployment" criterion | torch/GDAL bundling | fallback: venv + start script on demo laptop (documented, less ideal) |

---

## 39. Parallelization Plan

| Workstream | Can start after | Parallel with | Blocks |
| --- | --- | --- | --- |
| Geo core (A) | T01 | B data prep, C skeleton, E viewer (fixtures) | calib, dsm, validate, terrain |
| Backend skeleton (C) | T01 | A, E, F | E2E wiring |
| Data prep + training (B) | T03 partial (grid/resample) | everything | fine-tuned model only (baseline decouples) |
| Viewer (E) | T01 + fixture tileset | A, B, C | UI 3D tab |
| Frontend pages (F) | API stubs (day 1) | all | demo polish |
| Validation (D) | T03 | A, B | validation panel, experiments |
| Packaging (C) | M4 | Advanced work | M9 |

---

## 40. Scope-Control Rules

KEEP if it is on the MVP list or directly produces PS-scored evidence (DSM accuracy tables, faithful 3D, standalone run). DEFER if it needs data not yet in hand (ICESat-2 granules, semantic labels beyond GAMUS), or if the MVP path has any red test. DROP if it cannot be validated with available references, needs external services, adds infrastructure (queues, databases, cloud), or has consumed > 2× its estimate without a passing test. Cut order when time runs out: glTF export → contours → polygon/profile tools → semantic head → ICESat-2 loader → uncertainty overlay (keep raster) → anchors UI (keep CSV path) → Linux build. Never cut: tier labelling, datum tagging, validation harness core, offline packaging, Mode A watermark.

---

## 41. Complete Implementation Backlog (all STATUS = NOT STARTED)

| ID | Title | Description | Module | Deps | Prio | Cx | Deliverable | Acceptance | Tests | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | Repo skeleton & CI | pyproject, requirements, ruff/pytest, GH workflow, frontend scaffold | repo | — | P0 | L | green CI | lint+tests+build pass | smoke | C |
| B02 | Fixture generator | synthetic rasters/anchors/tileset | scripts, tests | B01 | P0 | L | fixtures | all fixtures generated deterministically | unit | A |
| B03 | `Grid`/`Meta`/coords | grid math, meta reading, GSD | core/geo | B02 | P0 | M | module | round-trips exact | unit | A |
| B04 | Vertical datum module | PROJ pipelines, grids check, registry | core/geo | B03 | P0 | M | module | PROJ reference values | unit | A |
| B05 | Resample/reproject/writer/check | canonical GSD, reproject, COG, asserts | core/geo | B03 | P0 | M | module | inverse error; COG valid | unit | A |
| B06 | Ingest | decode/classify/edge cases | core/ingest | B03 | P0 | M | module | fixture matrix | unit | A |
| B07 | Config & settings | YAML + env + overrides + hashing | backend/config | B01 | P0 | L | module | hash stable | unit | C |
| B08 | Store & sqlite registry | job dirs, paths, index, export zip | backend/store | B07 | P0 | M | module | CRUD; zip manifest | unit | C |
| B09 | Job state machine & worker | states, persistence, resume, retry cascade | backend/jobs | B08 | P0 | M | module | transitions; resume test | integration | C |
| B10 | API: upload/job/system | routers + schemas + errors | backend/api | B06, B09 | P0 | M | endpoints | contracts | api | C |
| B11 | Preprocess | stretch, normalize, canonical GSD, tiling, mosaic, QC | core/preprocess | B05 | P0 | M | module | coverage/determinism | unit | B |
| B12 | Inference (stub + baseline) | loader, batching, TTA, blend, postprocess, baseline | core/inference | B11 | P0 | M | module | stub determinism; OOM cascade | integration | B |
| B13 | Stage wiring Mode A | PREPROCESS→INFERENCE→OUTPUT; `/process`, `/result`, `/raster` | backend | B10, B12 | P0 | M | Mode A E2E | rdsm + preview | e2e | C |
| B14 | Tileset generator | pyramid, quadtree, height/texture tiles, tileset.json | core/terrain | B05 | P0 | M | module | continuity test | integration | E |
| B15 | Viewer core | loader, RTIN mesh, skirts, holes, residual, orbit, texture | frontend/viewer | B14 | P0 | H | scene | residual ≤ tol; UV align | vitest | E |
| B16 | Calib discover + DEM load | assets index, reproject, datum | core/calib | B04, B05 | P0 | M | module | tiles found; datum applied | unit | A |
| B17 | NC terrain + checks + tier + report | ground mask, support, NC, upsample, consistency, tier rules, report | core/calib | B16 | P0 | H | module | canopy fixture; rules | unit | A |
| B18 | DSM compose/derive/flags + COG tags | dsm, slope, aspect, flags, writer tags | core/dsm | B17 | P0 | M | module | plane slope; tags | unit | A |
| B19 | Stage wiring Mode B | CALIBRATION→DSM→GEO_CHECK→TERRAIN | backend | B13, B18, B14 | P0 | M | Mode B E2E | tier T job | e2e | C |
| B20 | Frontend pages & panels | upload/options, status, result, 2D views, metadata, export | frontend | B10, B13 | P0 | M | SPA | flows | playwright | F |
| B21 | Measurement tools | point/2-point via `/sample`; labels by tier | frontend, backend/api | B19, B15 | P0 | M | tools | labels correct | playwright | F |
| B22 | First-person + layers | pointer-lock, clamp, DSM/slope layers, legends, badges | frontend/viewer | B15 | P0 | M | navigation | clamp test | playwright | E |
| B23 | Data prep GAMUS | h5 → tiles, manifest | ml/data | B05 | P0 | M | tiles | manifest | script test | B |
| B24 | Data prep LiDAR source (CH/NZ/US) | download AOIs, DSM−DTM, ortho co-registration, tiles, terrain classes | ml/data | B05 | P0 | H | tiles | manifest; datum recorded | script test | B |
| B25 | Splits & overlap check | blocked splits, buffers, cross-source footprint exclusion | ml/data | B23, B24 | P0 | M | split manifests | leakage check pass | unit | B/D |
| B26 | Fine-tune v0.1 + card | training loop, loss, tables | ml/train | B25, GPU | P0 | H | model | card + ablation vs baseline | eval | B |
| B27 | Model registry & swap-in | hash check, card load, tier H | ml/registry, core/inference | B26 | P0 | L | module | tier H | integration | B |
| B28 | Validation harness | reference, align, coregister, masks, metrics, stratify, residual, leakage | core/validate | B05, B18 | P0 | H | module | synthetic shift; metric identity | unit | D |
| B29 | Validation API + panel | `/validate`, `/validation`, UI tables/residual | backend, frontend | B28, B20 | P0 | M | panel | states | e2e | D/F |
| B30 | Packaging + demo mode + clean-machine test | PyInstaller bundle, assets, demo flags, checklist | scripts, backend | B19–B22 | P0 | H | executable | offline run | deployment | C |
| B31 | Quality states + uncertainty raster | TTA var × priors + calib resid; overlay | core/quality, core/dsm, frontend | B27, B28 | P1 | M | quality.json, unc.tif | fixture triggers | unit | B/D |
| B32 | Anchors module + UI | CSV/GeoJSON, split, robust fit, accept, tier A, upload UI | core/calib, frontend | B17 | P1 | M | tier A | synthetic recovery | unit | A/F |
| B33 | Experiment tracking + DFC19 protocol + ablations | run.json, tables, scripts C0–C8 | ml/eval | B27, B28 | P1 | M | experiments.md | scripts run | script | D |
| B34 | Observability & hardening | log.jsonl fields, resume, retention, error taxonomy complete | backend | B19 | P1 | M | logs | diagnosable failure | integration | C |
| B35 | Docs set | all guides + licence manifest | docs | all | P1 | M | docs | reproducible demo | review | all |
| B36 | Semantic head + class metrics | GAMUS labels aux head; class masks; class tables | ml, core | B26, B28 | P2 | H | model v0.2 | class tables | eval | B/D |
| B37 | ICESat-2 loader | ATL08 → anchors with datum | core/calib | B32 | P2 | M | loader | fixture granule | unit | A |
| B38 | Profile/polygon tools, contours, residual overlay | analysis | frontend | B21 | P2 | M | tools | labels | playwright | F |
| B39 | glTF/OBJ export | terrain export | core/terrain | B14 | P2 | L | files | opens in viewer | integration | E |
| B40 | Facade/cloud heuristics; histogram matching | QC/adaptation | core/preprocess | B11 | P2 | M | flags | fixture | unit | B |
| B41 | Cross-country & India scripts | eval | ml/eval | B33 | P2 | M | reports | run | script | D |
| B42 | Linux build; Docker eval image | packaging | scripts | B30 | P2 | M | artefacts | runs | deployment | C |

---

## 42. Sorted Build Order (by dependency & critical path)

B01 → B02 → B07 → B03 → B04 → B05 → B06 → B08 → B09 → B10 → B11 → B12 → B13 ‖ (B23, B24 start after B05) → B14 → B15 → B16 → B17 → B18 → B19 → B20 → B21 → B22 → B25 → B26 → B27 → B28 → B29 → B30 → B31 → B32 → B33 → B34 → B35 → B36 → B37 → B38 → B39 → B40 → B41 → B42.

---

## 43. First 24 Hours Plan (6 developers, realistic pace)

| Hours | Goal | Tasks | Deliverable | Deps | Risk |
| --- | --- | --- | --- | --- | --- |
| 0–2 | Environment | B01 (C), B02 (A), Node/Vite scaffold (F), torch install check (B), fixture tileset plan (E), reference data inventory (D) | CI green; venvs | — | py 3.12/torch wheel issues |
| 2–6 | Contracts | `Grid`/`Meta` (A B03); settings/store (C B07/B08); preprocess skeleton (B B11); tileset.json schema + loader stub (E); API stubs + upload page (F); validation metrics module (D, part of B28) | contracts merged to `develop` | 0–2 | contract churn |
| 6–10 | First pixels | vertical datum + writer (A B04/B05); job state machine (C B09); baseline inference on fixture (B B12); RTIN mesh on fixture DSM (E B15); status page (F); coregister function (D) | baseline mosaic; mesh renders | 2–6 | torch device |
| 10–14 | Skeleton E2E | ingest (A B06); `/upload` `/job` `/process` wiring with stub model (C B10/B13); inference blend + postprocess (B); orbit controls + texture (E); result page (F); masks/stratify (D) | upload PNG → rdsm → tileset → 3D view (stub/baseline) | 6–10 | integration bugs |
| 14–18 | Calibration start | DEM discover/load/datum (A B16); `/raster` + `/sample` (C); GAMUS data prep start (B B23); layers/badges (E); 2D views + hover (F); residual raster + report (D) | Mode A demo-able; DEM loads | 10–14 | datum grid paths |
| 18–22 | NC terrain | NC + checks + tier (A B17); DSM compose/derive (A B18 start); LiDAR source download begins (B B24); first-person clamp (E B22); tools UI (F B21 stub); harness E2E on fixtures (D) | terrain.tif on fixture passes canopy test | 14–18 | NC numerics |
| 22–24 | Consolidate | merge to `develop`; tag `skeleton-e2e`; write handoffs; plan day 2 (B19 Mode B E2E, B20, B26 training kickoff) | tagged skeleton; burn-down | all | fatigue — stop, don't rush merges |

---

## 44. Demo-Day Readiness Checklist

- [ ] Environment: executable launches on demo + backup laptop; `GET /system` OK; Wi-Fi off test passed.
- [ ] Models: `dw-ndsm-small/<ver>` + baseline bundled; hashes match `INDEX.json`; model card visible in UI.
- [ ] Datasets/manifests: training manifests and split IDs in docs; no reference data used in training (leakage check log).
- [ ] Sample inputs: PNG (Mode A), GeoTIFF (Mode B), GeoTIFF + anchors (if A tier), all open-licensed; copies on USB.
- [ ] Reference data: validation subset with declared vertical CRS; licence note.
- [ ] Calibration: DEM tiles for demo AOIs bundled; geoid grids present; datum sanity passes; tier badge correct; removing DEM shows downgrade.
- [ ] DSM output: opens in QGIS with CRS, transform, nodata, `VERTICAL_CRS` tag; slope layer sane.
- [ ] GeoTIFF export zip verified.
- [ ] Validation: panel produces tables + residual on sample; metrics equal offline harness.
- [ ] 3D mesh: residual ≤ tolerance shown; texture aligned; no holes except nodata.
- [ ] Viewer: orbit, fly, first-person all work; badges/watermark; exaggeration = 1.0 default.
- [ ] Measurements: point/2-point values labelled; match QGIS pixel values.
- [ ] Error handling: corrupt file, no-CRS TIFF, oversized image → clear messages (rehearsed).
- [ ] Performance: stage durations logged for demo scenes on the demo machine (numbers from logs, not promises).
- [ ] Offline fallback: precomputed demo jobs flagged and loadable; 2D fallback tested.
- [ ] Documentation: README, demo.md, troubleshooting.md printed/offline.
- [ ] Backup machine: identical bundle hash; rehearsal completed on both.

---

## 45. Implementation Red-Team

| Risk | Impact | Detection | Mitigation | Fallback |
| --- | --- | --- | --- | --- |
| Torch/CUDA install or PyInstaller bundling fails | No demo | Day-1 install check; early packaging spike (M4) | CPU wheel default; one-folder bundle; pinned versions | venv + start script on demo laptop |
| PROJ grids not found in packaged app → silent ellipsoid/geoid error (24–99 m) | Wrong absolute DSM | `ensure_grids()` at startup; datum unit test in package smoke test; datum-sanity check | bundle grids; set `PROJ_DATA`; refuse tier T if grids missing | tier H with explicit message |
| DSM−DTM data prep for LiDAR source takes too long | No multi-terrain fine-tune | Day-2 checkpoint on AOI download | AOI subsets (few hundred km²); one source first | GAMUS-only fine-tune + baseline; report limitation |
| Fine-tuned head no better than baseline | Weak metric claim | ablation on held-out blocks | loss/long-tail tuning; more epochs | ship baseline relative + DEM offset, labelled honestly |
| NC terrain mis-weights (canopy mislabelled ground) | Terrain bias in forest | raw-fallback %, forest validation | conservative `h_ground`; flag; semantic head later | raw DEM + flag |
| Rasterio vs pyproj PROJ version mismatch | Wrong vertical transform | unit tests with reference values | all vertical transforms via pyproj only | — |
| Pixel-centre/corner mistake in tileset or `/sample` | Half-pixel misalignment everywhere | checkerboard UV test; `/sample` vs QGIS | single `Grid` helper for all conversions | — |
| Random tile split slips into training manifests | Inflated ablation | overlap/leakage check in CI script | blocked splits with buffers; footprint exclusion | re-run splits |
| Anchors reused as checkpoints | Fake accuracy | `leakage.py` radius check | ID + radius exclusion enforced in `validate` | — |
| Viewer performance on large tilesets | Laggy demo | frame-time log | LOD cap; tolerance auto-raise; demo AOI sizes | 2D views |
| Silent wrong output from band order (BGR/4-band) | Wrong predictions | band-map UI + histogram preview | explicit band mapping; warning | — |
| Scope creep (Advanced before MVP) | MVP incomplete | burn-down review daily | rules §40; `main` demo-runnable always | cut list |
| GPU OOM during demo | Stall | OOM cascade logs | small batch default; CPU cascade | precomputed job |
| Hardest to debug: calibration tier decisions | Confusing results | full `calib_report.json`, flags | rule table tests; UI shows triggers | — |
| Most error-prone geospatial op: reprojection of DEM into rotated/geographic grids | Misaligned terrain | plane reprojection test; visual DEM overlay layer | reproject to UTM early (recorded) | — |

---

## 46. Implementation Readiness Review

**Can a competent developer start from this document without major architectural decisions?** Yes for MVP-01…MVP-07 and MVP-09/10: modules, contracts, formats, config, environment, tests, fixtures and order are specified.

**Still unspecified (deliberately, to be fixed by experiment or team preference, not architecture):**
1. Exact loss and long-tail handling for the nDSM head (B26) — choose by ablation on held-out blocks (Phase 6 §6 defers this to Phase 7/8; Phase 7 defers the *choice* to the first training runs, the *candidates* are fixed: L1/SiLog ± height-bin reweighting or HTC-DC-style classification-regression [E4]).
2. Numeric defaults that need empirical tuning: `h_ground` (start 1.0 m), NC `sigma_cells` (start 1.5), `w_min` (0.1), `datum_sanity_m` (15), anchor `k_accept` (3), default mesh tolerance (0.5 m or 0.5·GSD) — all config, all logged.
3. Which LiDAR source is prepared first (CH vs NZ vs US) — decide by download speed/AOI licence on day 1–2; all three are documented as equivalent in Phase 4.
4. UI component library (plain TS vs React) — team preference; contracts unaffected.
5. Exact CUDA wheel for the dev GPUs [I4] — verify on hardware.
6. Availability of organiser-supplied imagery/reference — UNKNOWN; demo assets are chosen from open sources regardless.
No missing item requires an architectural decision.

---

## PHASE 7 COMPLETE

**1. Final Technology Stack.** Python 3.12; torch 2.14/torchvision 0.29 (CPU wheel default, CUDA per machine); safetensors; Depth Anything V2-Small (Apache-2.0) + fine-tuned nDSM head; rasterio 1.5 (GDAL), pyproj 3.8 (PROJ + EGM96/EGM2008 grids); numpy 2.5, scipy 1.18, scikit-learn 1.9; Pillow; FastAPI 0.141 + uvicorn 0.53 + pydantic 2.13; sqlite3 + in-process worker; Node 24 LTS, Vite 8, TypeScript 7, Three.js 0.186, @mapbox/martini 0.2; pytest/ruff/mypy/Playwright; PyInstaller 6.22 (Windows first); PDAL (dev/data only); earthaccess (dev).

**2. Repository Structure.** §4 (`backend/`, `core/{geo,preprocess,inference,calib,dsm,validate,quality,terrain}`, `ml/{registry,data,train,eval}`, `models/`, `assets/`, `frontend/`, `scripts/`, `tests/`, `configs/`, `docs/`, `data/`).

**3. Top 30 P0/P1 Tasks.** B01–B30 (P0) and B31–B35 (P1) as listed in §41.

**4. Critical Path.** B01→B03→B05→B09/B10→B11→B12→B13→B16→B17→B18→B19→B15/B22→B20/B21→B30, with B23→B24→B25→B26→B27 as the parallel model chain.

**5. MVP Build Sequence.** MVP-01 geo/ingest → MVP-02 backend skeleton → MVP-03 Mode A baseline E2E → MVP-04 3D viewer → MVP-05 Level-T calibration + DSM/COG → MVP-06 Mode B E2E → MVP-07 tools + navigation → MVP-08 fine-tuned model → MVP-09 validation harness/panel → MVP-10 packaged offline demo.

**6. Team Work Allocation.** A geo/calib/dsm; B data/training/inference; C backend/packaging/demo; D validation/quality/experiments; E tileset/viewer; F frontend pages/tools/UX (§35).

**7. First 24 Hours.** §43: environment → contracts → first pixels (baseline mosaic, RTIN mesh) → skeleton E2E (PNG → rDSM → 3D with stub/baseline) → DEM loading → NC terrain on fixture → consolidated tag `skeleton-e2e`.

**8. Required Development Environment.** Windows 11/Ubuntu 22.04+, Python 3.12 venv, Node 24, optional NVIDIA GPU (≥ 6–8 GB VRAM for fine-tuning — estimate to verify), 16 GB RAM, bundled PROJ grids and demo DEM tiles, conda env with PDAL for 3DEP data prep only.

**9. Top 15 Implementation Risks.** Torch/GDAL packaging; PROJ grids missing in package; LiDAR data prep time; fine-tune not beating baseline; NC canopy mis-weighting; PROJ version mismatch rasterio/pyproj; pixel-centre errors; split leakage; anchor/checkpoint contamination; viewer performance; band-order errors; scope creep; GPU OOM at demo; calibration debuggability; DEM reprojection into non-UTM grids.

**10. Top 15 Acceptance Criteria.** Fixture format matrix passes; `assert_grid` on every artefact; datum values within 1 cm of PROJ; inverse canonical resampling error < 1e-3; stub inference deterministic; OOM cascade tested; synthetic offset/scale recovered; canopy bias reduced on fixture; slope on plane within 0.01°; vertical tag present and QGIS-valid; validation metrics equal independent recomputation; synthetic shift recovered; mesh residual ≤ tolerance and UV corners aligned; first-person never below terrain; offline clean-machine run with reproducible demo hash.

**11. Demo-Day Checklist.** §44.

**12. What Phase 8 Must Validate.** Execute the Phase 4 blueprint on spatially blocked held-out LiDAR blocks (CH/NZ/US) per terrain with the full metric set and object metrics; report fine-tuned vs baseline ablation; run calibration ablations C0–C8 (oracle, DEM, anchors N-curves, combined, datum-sensitivity, anchor-noise); measure canopy-bias reduction of NC terrain vs raw DEM add-back on forest/built-up strata; run the DFC19 Track-1 protocol; cross-country degradation (train/test country swaps); India ICESat-2-bounded assessment on open imagery; uncertainty AUSE; mesh-vs-DSM residuals, projection residuals, frame-time percentiles, crash-free runs, clean-machine install, SUS sessions; and record measured stage durations for the demo hardware.

READY FOR PHASE 8 — VALIDATION & EXPERIMENT EXECUTION
