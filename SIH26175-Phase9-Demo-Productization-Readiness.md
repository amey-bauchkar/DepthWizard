# Phase 9 — Demo Engineering, Productization, Standalone Deployment & Evaluator Readiness

SIH26175 DepthWizard · 2026-09-21

**Source of truth for this phase: the Phase 8 Validation Report (2026-09-21).** Its controlling finding, re-confirmed by directory inspection at the start of Phase 9: the project contains the seven planning documents, the Phase 8 report, and the Level-0 validation suite (`validation/phase8/`, 5 runnable + 3 interactive scripts, results, 4 synthetic GeoTIFFs). **There is no application code, no model weights, no assets, no outputs, no packaging.** No component of the product is IMPLEMENTED. Every Phase 8 "system-level" result is NOT EXECUTED; the only validated items are design-level mathematical checks.

Consequently Phase 9 does two things and refuses to do a third: (1) it records the true product state (nothing demo-able today); (2) it specifies, in evaluator-ready detail, the demo product, trust layer, 3D experience, packaging, fallbacks, scripts and readiness gates that the team must build — constrained by what Phase 8 established (including the two design corrections: silent PROJ "ballpark" datum fallback and gradient-kernel orientation/iteration); (3) it does **not** invent any measured accuracy, latency, dataset result, screenshot, or precomputed output. Every placeholder that must later hold a measured number is written as `[MEASURED]` and must be filled only from the validation panel or `experiment_log.json` of the built system.

Status vocabulary used throughout: IMPLEMENTED + VALIDATED · IMPLEMENTED + PARTIALLY VALIDATED · IMPLEMENTED + UNVALIDATED · PLANNED · EXPERIMENTAL · BROKEN · REMOVED · DEFERRED.

---

## 1. Current Product State Audit

| Component | Current state | Phase 8 evidence | Demo-safe? | Reliability | Action |
| --- | --- | --- | --- | --- | --- |
| Image ingestion (PNG/JPG/TIFF) | PLANNED (design only) | §2: NOT FOUND; L0-02 verified metadata handling logic on synthetic GeoTIFFs | No (nothing to run) | n/a | Build B06 (Phase 7); port L0-02 checks as unit tests |
| GeoTIFF support (CRS/transform/nodata/tags) | PLANNED; design math VALIDATED at L0 | L0-02 PASS (round-trip exact; rotation detected; geographic anisotropy 0.98×1.11 m at 28.6°N) | No | Design sound | Build `core/geo`; enforce UTM reprojection for geographic CRS |
| PNG/JPG mode (rDSM) | PLANNED | NOT EXECUTED | No | n/a | Build B13 |
| GSD extraction | PLANNED; design VALIDATED at L0 | L0-02 (transform GSD; geodesic local GSD) | No | Design sound | Build in `geo.coords` |
| Depth inference (DA-V2-S) | PLANNED; **no weights present** | §3 environment: no model files; CPU-only machine | No | Unknown | Fetch baseline weights; build B12 |
| Fine-tuned nDSM model | PLANNED (cannot be trained on current machine — no GPU) | §3: `torch.cuda.is_available()=False` | No | Unknown | Train elsewhere (B26) or ship baseline honestly (tier ≤ R + labelled) |
| nDSM output | PLANNED | — | No | Unknown | Depends on model |
| DEM discovery / bundled tiles | PLANNED; **no DEM tiles, no geoid grids locally** | §3, §11 CAL-5: offline PROJ returns 0 m silently | No | **Unsafe until guard exists** | Bundle Copernicus tiles + EGM grids; implement ballpark rejection (C-1) |
| Terrain reconstruction (NC layer) | PLANNED; mechanism VALIDATED at L0 (synthetic) | L0-04: canopy bias 7.79→2.01 m, buildings 12.16→−0.48 m, DSM RMSE 5.69→2.00 m; +2 m residual under closed canopy; ground NMAD 1.40→1.90 | No | Design sound on synthetic; real terrain UNTESTED | Build `calib.terrain`; port L0-04 as regression test |
| Calibration (tiers R/H/T/A, anchors) | PLANNED; estimators VALIDATED at L0 | L0-05: median offset error ≤0.13 m (N≥5); RANSAC scale ≤1 % (N≥5); **Huber failed** (10.99 m); hold-out RMSE 6.81 vs NMAD 0.83 with one blunder | No | Design sound with median/RANSAC | Build `calib.anchors/tier`; forbid Huber; blunder rule |
| DSM composition | PLANNED; VALIDATED at L0 (nodata propagation) | L0-10 | No | Design sound | Build `dsm.compose` |
| Validation harness | PLANNED; metrics/co-registration VALIDATED at L0 **after correction** | L0-08 identities; L0-07 grid search (2,−1)→(1.979,−1.020); gradient estimator diverged with `convolve`, converged with `correlate` (2.000,−1.012) | No | Design sound if C-2 applied | Build `core/validate` with synthetic-shift unit test |
| 3D mesh (RTIN) | PLANNED; RTIN NOT EXECUTED | L0-11: uniform decimation loses 20 m at building edges | No | Unknown | Build B14/B15 with residual display |
| Texture projection | PLANNED | NOT EXECUTED | No | Unknown | Build; checkerboard alignment test |
| Navigation (first-person/aerial) | PLANNED | NOT EXECUTED | No | Unknown | Build B15/B22 |
| Measurements (raster-backed) | PLANNED | NOT EXECUTED | No | Unknown | Build B21 |
| Slope analysis | PLANNED; Horn VALIDATED at L0 (interior exact; edges 3.18° error) | L0-06 | No | Design sound; edges must be flagged | Build `dsm.derive` |
| Uncertainty raster | PLANNED (Advanced) | NOT EXECUTED | No | Unknown | Defer until MVP works |
| Failure flags / tier badges | PLANNED | — | No | — | Build with MVP (non-negotiable for honesty) |
| Export (COG zip, glTF) | PLANNED | — | No | — | COG zip in MVP; glTF DEFERRED |
| Offline deployment (PyInstaller bundle) | PLANNED; **PyInstaller not installed; Python 3.14/Node 25 on dev machine** | §3 | No | Unknown | Pin 3.12/Node 24; packaging spike early (Phase 7 M4) |
| Level-0 design-verification suite | IMPLEMENTED + VALIDATED (design checks only; not a product feature) | `validation/phase8/`, deterministic (L10) | Yes — as *evidence artefact*, not as demo | High | Ship in `docs/`/CI; show in provenance panel |

Implementation status vs scientific validation, summarized: **implementation = 0 product components; scientific validation = design mathematics only (synthetic).** No accuracy, latency or terrain result exists.

---

## 2. DepthWizard Demo Contract

**Contract status: NOT YET SATISFIABLE.** The contract below defines what the demo *may* claim **once the corresponding component is IMPLEMENTED and the Phase 8 Level ≥ 4 tests have been run**. Until then, every line is a target, not a promise.

**Supported inputs (target):** PNG, JPG/JPEG, TIFF without CRS (Mode A); GeoTIFF with resolvable CRS and finite transform (Mode B); optional coarse DEM (bundled Copernicus GLO-30 for demo AOIs, or user GeoTIFF with declared vertical CRS); optional anchors CSV/GeoJSON with declared vertical CRS; optional reference DSM/nDSM GeoTIFF for the validation panel. Not supported: multi-page TIFF, exotic compressions, strongly off-nadir imagery (flag only), GSD outside canonical bands (flag).

**Supported outputs (target), each labelled by tier:** `rdsm.tif` (relative, unitless, Mode A); `ndsm.tif` (metres above local ground, tier H); `terrain.tif` (DEM-derived low-frequency terrain, declared vertical CRS, tier T/A — *not a certified DTM*); `dsm.tif` (absolute, declared vertical CRS, tier T/A); `slope.tif` (degrees, Mode B only); `flags.tif`; `calib_report.json`; `quality.json`; `validation.json` + `residual.tif` (only when a reference was supplied); terrain tileset + interactive 3D scene; `export.zip`. `uncertainty.tif` — only if the Advanced work is finished and tested.

**Supported accuracy claims:** **NONE at present.** Phase 8 §14: URBAN/SPARSE/HILLY/FORESTED all NOT VALIDATED — DATA UNAVAILABLE. When the harness has run on spatially blocked LiDAR blocks, the only permitted form is: "On [dataset/region], [N] pixels, terrain class [X], reference [LiDAR product, vertical CRS], co-registered shift [dx,dy], the system achieved ME [m], RMSE [m], MAE [m], NMAD [m], r [value], with [tier] calibration." Synthetic Level-0 numbers (e.g., composed-DSM RMSE 5.69→2.00 m) may be shown **only** as "design verification on synthetic data", never as system accuracy.

**Explicit non-claims (mandatory in UI text and speech):** no metres for non-georeferenced imagery; no absolute elevation without a DEM or anchors; no metre-level accuracy claim for Indian scenes (no Indian LiDAR reference exists; Phase 4); no guarantee on off-nadir imagery; no guarantee outside the trained GSD bands; the coarse DEM is a calibration input, not ground truth; the terrain layer is not a DTM; visualization fidelity says nothing about elevation accuracy; hidden surfaces are not reconstructed; no claim of novelty for any standard component (Phase 3 §20).

---

## 3. Final Product Architecture As Actually Implemented

**As implemented: none.** The only runnable artefact is the Level-0 suite (`run_all.py` → `results/*.json`). The architecture the demo will present is the Phase 6 architecture **with the Phase 8 corrections baked in**:
- C-1 `geo.vertical.ensure_grids()` must verify grid files exist **and** reject any PROJ transformer whose description contains "ballpark"; tier T is impossible without grids; a datum-sanity check against the DEM mean is mandatory.
- C-2 Co-registration must use kernel *correlation* (sign-checked on a plane), iterate to convergence with reference smoothing, or use grid search + parabolic refinement; unit-tested on a synthetic (2, −1) px shift.
- C-3 Offset by median/Theil–Sen, scale by RANSAC, N ≥ 5 for scale; ASPRS 3×RMSE blunder investigation and NMAD/LE95 reporting on hold-out anchors; Huber is not used.
- C-4 Resampling acceptance threshold recalibrated from the measured 1.06e-3 (or area-average downsampling); edge blur on canonical downsampling documented in the UI ("edges may be softened at this GSD").
Everything else (modular monolith, tiers R/H/T/A, NC terrain, COG outputs with vertical tag, Three.js RTIN viewer, PyInstaller offline bundle) is unchanged and PLANNED.

---

## 4. Final UI / UX Structure (information architecture)

| Screen | Purpose | Primary action | Most important information | Visually dominant | Secondary/hidden | 5-second understanding |
| --- | --- | --- | --- | --- | --- | --- |
| Home | Orient; choose demo scene or upload | "Load demo scene" / "Upload" | Mode badges legend (RELATIVE / METRIC nDSM / ABSOLUTE); demo-mode banner | Two large buttons | Settings, About, Provenance | "One image in → height surface + 3D out; labels tell me what the numbers mean" |
| Input | Confirm what was detected | "Process" | Detected mode, CRS, GSD, size, DEM found?, anchors found?, warnings | Metadata card with green/amber/red states | Band mapping, overrides | "It read my georeferencing and knows whether it can produce metres" |
| Processing | Show honest progress | Cancel | Stage stepper (preprocess → inference → calibration → DSM → tiles), device (GPU/CPU), elapsed | Stepper | Log tail (dev toggle) | "It is really computing, on this machine, at this stage" |
| Result dashboard | Decide what to inspect | Open 3D / Open validation | **Tier badge + datum + quality state** first; thumbnails RGB / nDSM / terrain / DSM / slope | Tier badge | Hashes, config | "Absolute DSM, EGM2008, calibration LIMITED because ground support was low" |
| DSM inspection (2D) | Read values | Hover/click | Colour ramp with units + legend; hover readout with label | Raster | Layer opacity | "Heights are in metres above ground / elevation above EGM2008" |
| Validation panel | Show numbers vs reference | Upload reference / view | Reference name + vertical CRS + shift applied; ME/RMSE/MAE/NMAD/r table overall + per terrain/class; residual map; "NOT VALIDATED" when absent | Metrics table + residual | Histogram, bins | "These are real comparisons on N pixels; bias and spread are both shown" |
| 3D viewer | Explore geometry | Walk / fly / measure | Tier badge, datum, exaggeration (=1.0), mesh-vs-DSM residual, layer toggles | Terrain | Debug wireframe | "The 3D surface *is* the DSM; the residual number proves it" |
| Measurement tools | Ask "how tall / how steep" | Click | Value + unit + label (ABSOLUTE / METRIC RELATIVE / RELATIVE) + uncertainty if available | Value chip | Raw pixel coords | "This building is 23 m above local ground (metric relative); elevation 245 m EGM2008" |
| Metadata / calibration | Inspect how metres were produced | Expand | Tier, DEM source/version/datum, ground-support %, raw-DEM fallback %, anchors used/held-out residuals, checks | Calibration card | JSON | "Terrain came from Copernicus GLO-30 (EGM2008); model gave heights above ground; anchors adjusted offset by x m" |
| Error / quality | Understand limits | Toggle overlays | Quality state with triggers; flags overlay (border, raw-DEM fallback, GSD out of range) | Hatching overlay | Thresholds | "Here is where the system is unsure and why" |
| Export | Take results away | Download | What's in the zip + hashes | Button | — | "GeoTIFFs open in QGIS with CRS and vertical reference" |
| Settings / About | Configure, credit, licences | — | Model version/hash, software version, licences, PROJ grids present ✔/✘ | — | Advanced params | "Reproducible, licensed, offline" |

---

## 5. Trust Layer (calibration / quality / measurement context)

| Element | Values | Evaluator-facing wording | Source |
| --- | --- | --- | --- |
| Calibration tier | R / H / T / A | R "Relative only — no scale"; H "Metric heights above local ground — no absolute elevation"; T "Absolute elevation — terrain from coarse DEM"; A "Absolute elevation — refined with N control points (hold-out NMAD x m)" | `calib_report.json` |
| Vertical reference | EGM2008 (default) / EGM96 / ellipsoidal / national datum / none | "Elevations are metres above the EGM2008 geoid (≈ mean sea level)"; Mode A/H: "No vertical reference" | `geo.vertical`; guard C-1 |
| Data sources | DEM product + version; anchor source; validation reference | "Terrain: Copernicus GLO-30 (2024_1), EGM2008 → converted"; "Reference: swisstopo swissSURFACE3D 0.5 m (LN02 → EGM2008)" | discover + validation |
| Quality state | GOOD / LIMITED / WARNING / INVALID / UNVALIDATED | GOOD = tier A with hold-out NMAD ≤ threshold or tier T with ground support ≥ 0.5 and consistency |ME| ≤ 3 m; LIMITED = partial ground support or consistency 3–10 m; WARNING = raw-DEM fallback > 50 %, GSD out of range, facade suspect; INVALID = datum-sanity failure or grids missing ("ballpark" detected); UNVALIDATED = no reference supplied | `quality.json` (thresholds visible in Settings) |
| Measurement context | RELATIVE / METRIC RELATIVE (nDSM) / ABSOLUTE | Every number carries one of these three words | tools |
| Validation context | reference, N pixels, terrain class, ME, RMSE, MAE, NMAD, r, shift | Shown as a table; "correlation alone does not certify accuracy" tooltip (Phase 8 L0-08: r = 1.0 with 60 m bias) | `validation.json` |
| Design-verification evidence | L0 suite results | "Mathematical checks (synthetic): datum values, NC terrain, anchor recovery, shift recovery — reproducible" | `validation/phase8/results` |

Rule: no metres are displayed unless tier ≥ H; no elevation unless tier ≥ T; Mode A shows a persistent RELATIVE watermark.

---

## 6. 3D Viewer Specification (faithfulness first)

- **Initial camera:** aerial oblique view over the scene centre at ~45°, whole scene in frame, north up; exaggeration 1.0 (badge visible whenever ≠ 1.0).
- **Modes:** Aerial (orbit/pan/zoom, fly WASD with altitude clamp); First-person (pointer-lock, eye height 1.7 m × GSD-neutral, camera clamped to height tiles — never below terrain); Reset view; bookmarks.
- **Layers:** RGB texture (default); DSM hypsometric ramp with legend and datum; nDSM ramp; terrain-only (DEM-derived layer) — the "what did the DEM contribute" toggle; slope (°) with legend; flags hatching; residual (if validated); wireframe/debug showing RTIN triangles and tile borders.
- **Mesh quality indicator:** tolerance slider (metres) with live **mesh-vs-DSM residual** (RMSE/max) computed from the height tiles; default tolerance documented; uniform decimation is not offered (Phase 8 L0-11: destroys building edges).
- **Tiles/LOD:** quadtree by screen-space error; skirts; nodata rendered as holes, never filled; scene bounds drawn; progressive loading bar with tile counts.
- **Measurement:** point (elevation + height above ground + slope), two-point (Δz, distance), later profile/polygon (DEFERRED unless built and tested). Values come from `/sample` on rasters — a tooltip states "measured on the elevation raster, not the mesh".
- **No** sky/atmosphere effects, water animation, procedural buildings, generated facades, smoothing shaders.
- **Proof that the 3D comes from the DSM:** (1) residual indicator; (2) toggle DSM ramp ↔ texture on identical geometry; (3) click a point in 3D and in the 2D DSM view → identical value and pixel coordinates; (4) wireframe shows triangles following building edges.

---

## 7. Demo Dataset Recommendation

**Phase 8 measured accuracy for every candidate: NONE (NOT EXECUTED).** No scene is validated; none is "demo-worthy" yet. The candidates below are chosen because Phase 4 verified that open imagery *and* LiDAR-grade absolute references exist for them, so scientific claims can later be defended.

| Candidate | Source | Region | Terrain | Imagery GSD | Reference | Phase 8 accuracy | Calibration tier possible | Known limitations | Why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Urban | swisstopo SWISSIMAGE + swissSURFACE3D/ALTI3D (OGD) or LINZ (CC BY 4.0) | Zürich/Geneva or Auckland/Christchurch | Urban | 0.1–0.5 m (resample to canonical) | LiDAR DSM 0.5–1 m, LN02/NZVD2016 → EGM2008 | NOT EXECUTED | T (bundled Copernicus); A with synthetic anchors from LiDAR (labelled) | LN02 datum conversion; long-tail high-rises | Open, commercial-OK, dense reference |
| Relief | swisstopo Alps or LINZ Southern Alps/West Coast | CH / NZ | Hilly | 0.5–1 m | LiDAR DSM + DTM | NOT EXECUTED | T | DEM slope error; co-registration matters | Only open hilly LiDAR |
| Forest | Open-Canopy (France, 1.5 m) or swisstopo/LINZ forest tiles | FR / CH / NZ | Forested | 0.5–1.5 m | ALS canopy height / DSM | NOT EXECUTED | T | closed canopy → raw-DEM fallback (L0-04 showed +2 m residual synthetic) | Forest evidence required by PS |
| Difficult / failure-aware | Same regions: dense CBD or closed forest; plus a PNG copy of an urban tile | — | Mixed | — | same | NOT EXECUTED | shows downgrade T→H→R when DEM removed / CRS stripped | demonstrates honesty | Evaluator trust |
| India (bounded) | LISS-IV (5.8 m, open) or organiser Cartosat if supplied | India | any | 5.8 m | ICESat-2 ATL08 sparse; CartoDEM 8 m LE90 | NOT EXECUTED | T/A (sparse) | coarse GSD; no LiDAR | Only Indian evidence possible |

Selection rule: a scene enters the demo only after the harness has produced its per-terrain table on that scene's held-out blocks.

---

## 8. Primary Wow Moment

**Primary (target, 45 s):** on the same 3D geometry, toggle **terrain-only → nDSM → DSM → RGB texture**, then click a building: the tool returns "23.4 m above local ground · METRIC RELATIVE" and "312.7 m above EGM2008 · ABSOLUTE", while the calibration card beside it shows "Terrain: Copernicus GLO-30 · tier T · quality LIMITED (ground support 41 %)". Then remove the DEM and re-run: the badge drops to **H**, elevation values disappear, heights above ground remain. Genuine capability being shown: the two-layer decomposition and honest tier downgrade.
**Secondary 1:** validation panel showing ME/RMSE/MAE/NMAD/r per terrain beside the residual map on the same scene. **Secondary 2:** first-person walk down a street with the mesh-vs-DSM residual indicator visible. None of these can be shown today.

---

## 9. Failure-Resilient Demo Plan

| Path | Trigger | Behaviour |
| --- | --- | --- |
| Primary | GPU or CPU available; bundled DEM & grids; demo GeoTIFF | Live Mode B job → tier T → 3D → validation panel with bundled reference subset |
| Fallback 1 — inference fails | OOM/exception after cascade | Job FAILED{INFERENCE} shown honestly; switch to precomputed job for the same scene (banner "PRECOMPUTED DEMONSTRATION OUTPUT · job hash …"); explain the cascade |
| Fallback 2 — no GPU | `torch.cuda.is_available()=False` (the current dev machine) | Same code path on CPU; start the live job, show the stepper, and narrate on a precomputed twin while it completes; never claim GPU timings |
| Fallback 3 — no internet | Always assumed | Everything bundled (weights, EGM grids, DEM tiles, samples, reference subset); `GET /system` shows "offline OK / grids present"; if grids missing the app must display INVALID rather than metres (C-1) |
| Fallback 4 — scene too slow | stage duration > budget | Switch to a smaller bundled AOI; raise mesh tolerance (badge updates); keep the large scene as precomputed |
| Emergency | app cannot start | Open exported GeoTIFFs in QGIS + the HTML `validation.json` table + precomputed tileset served by a static file server; state plainly that this is precomputed and why |
| Absolute rule | — | Precomputed ≠ live: banner, job hash, timestamp; the live pipeline must be runnable on the same input so hashes can be compared |

**Today:** none of these paths exists; the emergency path is impossible because there are no precomputed outputs. The only thing that can be shown live today is the Level-0 suite (`python run_all.py`, ~15 s) — as *design evidence*, not as the product.

---

## 10. Offline Packaging Plan & Clean-Machine Test

**Actual status:** nothing packaged; PyInstaller not installed; dev machine Python 3.14 / Node 25 / no GPU (Phase 8 §3). Plan (Phase 6 D-07, Phase 7 §31) with Phase 8 constraints:

| Item | Requirement | Note |
| --- | --- | --- |
| Python runtime | bundled by PyInstaller (target 3.12) | 3.14 worked in-session for all libs, but pin 3.12 for wheel guarantees |
| Torch | CPU wheel in bundle; GPU wheel optional installer | CUDA absent on dev machine; never assume GPU |
| GDAL/rasterio, PROJ/pyproj | wheels bundled; hidden imports for drivers | two PROJ copies: route vertical transforms via pyproj only |
| `PROJ_DATA` + geoid grids | set at launch; **bundle `us_nga_egm96_15.tif` (2.6 MB) and `us_nga_egm08_25.tif` (76.9 MB)**; startup self-test transforms a known point and asserts non-ballpark, non-zero undulation | Phase 8 P0 finding |
| DEM assets | Copernicus GLO-30 tiles for demo AOIs (~40 MB each), index.json with datum | offline |
| Model weights | baseline + fine-tuned (if trained) with hashes and licences (Apache-2.0 Small only) | — |
| Frontend | built static bundle served by backend | no Node at runtime |
| Backend | FastAPI/uvicorn bound to 127.0.0.1, free port probe | — |
| Local DB / storage | SQLite + job dirs under a writable user data dir | cleanup/retention |
| Config / env | `configs/*.yaml` bundled; `DW_*` env optional | — |
| Launch | `DepthWizard.exe` → server → default browser; `--headless-selftest` runs datum + L0 checks and a bundled demo job | — |
| Clean-machine test | Fresh Windows VM, no Python/Node/internet: install, run self-test, run demo job, open 3D, export, open COG in a portable GIS | required before GO |

If a one-folder bundle proves impossible in time, the documented minimum is: Python 3.12 + `pip install -r requirements/package.txt` from a bundled wheelhouse + start script — still offline, but not "no Python".

---

## 11. Performance Benchmark Plan

**Measured: nothing** (no system). Plan: log per stage for 1024², 2048², 4096² (and one strip-size AOI) on the demo laptop, CPU and (if available) GPU: cold start, model load, preprocessing, inference (per tile and total), calibration, DSM composition, tile generation, browser load, 3D first frame, peak RSS/VRAM, CPU %, output sizes, frame-time p50/p95 during navigation. Numbers go into `experiment_log.json` and the About panel; slides quote them only from there. Expected bottleneck (inference on CPU) is an *inference*, not a measurement, until logged.

---

## 12. Observability & Logging (user-facing vs developer)

User-facing: stage names, elapsed time, device ("running on CPU"), tier badge, quality state with plain-language triggers, warnings ("No DEM found — heights above ground only"), error codes with one-sentence explanations, hashes in the provenance panel. Developer (toggle/log file): job state transitions, tile counts, batch sizes, OOM cascade events, DEM tiles used, ground-support statistics, consistency ME/NMAD, anchor residuals, PROJ pipeline descriptions (must never contain "ballpark"), config/model hashes, per-stage durations, memory peaks, WebGL context events.

---

## 13. Product Hardening Checklist (minimum before demo)

Invalid/corrupt/unsupported inputs → taxonomy errors; missing CRS → Mode A with offer; invalid CRS → refusal; missing PROJ grids → INVALID state, no metres; DEM coverage gap → tier H with message; vertical-datum mismatch of user reference → 422 asking for datum; huge files → AOI/downsample with record; cancellation → clean job dir; repeated runs → idempotent stages, no stale artefacts; browser refresh → job persists; interrupted processing → resume from last stage; partial outputs → marked FAILED with stage; temp-file buildup → retention job; WebGL loss → 2D fallback; race conditions → single worker by default. **Status of all items: PLANNED (nothing exists to harden).**

---

## 14. Provenance / Traceability Panel

Shows, per job: input filename + sha256; mode; CRS/transform/GSD; preprocessing hash (stretch, canonical GSD, tile plan); model name/version/weights sha256/licence; calibration tier, DEM source/version/tile IDs/datum + grid file hashes, ground support %, raw-DEM fallback %, consistency ME/NMAD, anchors used (IDs) and held-out residuals; composition method; software version + torch/GDAL/PROJ versions; timestamp; validation reference hash/datum/shift if present; output hashes; link to the Level-0 design-verification results. Answer to "where did this number come from?" = click the number → highlights the chain INPUT → PREPROCESSING → MODEL → CALIBRATION → TERRAIN → DSM → VALIDATION → VISUALIZATION with the relevant provenance fields.

---

## 15. MUST / SHOULD / CUT

**MUST FINISH (credible demo):** `core/geo` with ballpark guard and UTM reprojection; ingestion; preprocessing/tiling; inference with baseline weights (fine-tuned if trainable elsewhere); Level-T calibration (NC terrain) with datum handling; DSM/slope/flags + COG tags; tier badges and Mode A watermark; validation harness with corrected co-registration and metric set; terrain tiles + Three.js viewer (orbit, first-person, layers, residual, point/2-point tools); API/jobs/store; bundled assets (grids, DEM tiles, samples, one reference subset); offline package + clean-machine test; provenance panel; at least one per-terrain validation table on real blocked data.
**SHOULD FINISH:** anchors (tier A) with hold-out reporting; uncertainty raster + overlay; quality states with triggers; DFC19 protocol run; residual overlay; export zip.
**FREEZE / CUT:** semantic head; ICESat-2 loader; profile/polygon tools; contours; glTF export; scalar-GSD embedding; Linux build; Docker; any visual effect; any feature that cannot show a residual or a test.

---

## 16. Final MVP Definition

**One sentence:** DepthWizard takes one nadir RGB image and, when it is georeferenced and a coarse DEM is available, produces a metric height-above-ground map and an EGM2008-referenced DSM as GeoTIFFs with a stated calibration tier, validates them against a supplied reference, and renders exactly that surface as a navigable textured 3D scene with raster-backed height and slope measurements — otherwise it produces an explicitly relative surface.

**End-to-end journey:** 1. Launch the app (offline). 2. Load demo scene or upload. 3. Read the detected mode/CRS/GSD/DEM card. 4. Process; watch stages. 5. Read the tier badge, datum and quality state. 6. Inspect nDSM/terrain/DSM/slope in 2D. 7. Open validation panel; upload/choose reference; read ME/RMSE/MAE/NMAD/r per terrain and the residual map. 8. Open 3D; orbit; walk; toggle terrain-only/DSM/texture; read the mesh residual. 9. Click to measure height (labelled) and slope. 10. Open provenance; export the zip; open a COG in QGIS.

---

## 17. Demo Scripts (targets; every `[MEASURED]` must come from the built system's panels)

### 3-minute
| Time | Click/show | Evaluator sees | Say | Technical point | Evidence | Do not say |
| --- | --- | --- | --- | --- | --- | --- |
| 0:00–0:20 | Home → Load demo GeoTIFF | metadata card: CRS, GSD, "DEM found: Copernicus GLO-30" | "One georeferenced satellite/aerial image; the app detects it can produce absolute elevation because a coarse DEM covers it." | Mode detection, calibration discovery | `meta.json`, `calib_inputs.json` | "high-precision", "LiDAR-quality" |
| 0:20–0:50 | Process (live, CPU/GPU stated) | stepper | "The image model predicts height above local ground; the coarse DEM gives only the low-frequency terrain, sampled where the model sees bare ground; we add them." | Two-layer composition; tiers | `calib_report.json` | "the DEM is ground truth" |
| 0:50–1:30 | Result: tier badge T, EGM2008, quality state; toggle nDSM/terrain/DSM | badges + rasters | "Tier T means absolute elevation from DEM terrain; heights above ground are metric from the model." | Trust layer | quality triggers | any metre claim for Mode A |
| 1:30–2:20 | 3D: orbit → first-person → click building; toggle texture/DSM; residual indicator | measured values with labels | "This geometry *is* the DSM — residual [MEASURED] m at the chosen tolerance; the value comes from the raster." | Faithful visualization | `/sample`, residual | "the 3D proves accuracy" |
| 2:20–3:00 | Validation panel | per-terrain table + residual map | "Against [reference], N=[MEASURED] pixels, RMSE [MEASURED], ME [MEASURED], NMAD [MEASURED]; correlation alone would hide bias, so we show all." | Blueprint metrics | `validation.json` | numbers not on screen |

### 5-minute
Adds: 0:20 upload the *same tile as PNG* → RELATIVE watermark, no metres (honest mode); 3:00 remove DEM → tier drops to H, elevation disappears; 3:40 provenance panel: model hash, DEM version, grid hashes, "PROJ pipeline: non-ballpark ✔"; 4:20 slope layer + pick; 4:40 export → open COG in QGIS showing vertical CRS tag.

### 10-minute
Adds: the Level-0 design evidence (30 s: `run_all.py` output — datum values, NC bias reduction on synthetic, anchor recovery, shift recovery); anchors (tier A) if built, with hold-out NMAD; forest and relief scenes with their tables and flags (raw-DEM fallback hatching under closed canopy); a deliberately unsupported input (invalid CRS) showing refusal; the ablation table fine-tuned vs zero-shot if run; limitations slide spoken aloud.

---

## 18. Twenty Hard Evaluator Questions

| # | Question | Best concise answer | Phase 8 evidence | Do not claim |
| --- | --- | --- | --- | --- |
| 1 | Why monocular at all? | The PS asks for an agile single-view alternative; we treat it as a calibrated estimate with stated tiers, not a stereo replacement. | Phase 3/4 (CartoDSM 8 m LE90 as ISRO's own stereo bar) | parity with stereo/LiDAR |
| 2 | Why Depth Anything V2? | PS mandates a pretrained monocular backbone; the Small model is Apache-2.0 with training code; fine-tuning such backbones for nadir height is demonstrated in literature. | Phase 5 G2/G5; Phase 3 E14/E16 | zero-shot metric capability (δ₁≈0 % at nadir, Phase 3) |
| 3 | Why predict nDSM, not DSM? | All available truth and prior results are height above ground; absolute elevation needs an external datum source anyway. | Phase 4 §6 | that the model knows sea level |
| 4 | Why a DEM? Why not SRTM-only? | The DEM supplies low-frequency terrain and datum; it cannot resolve buildings (30 m posting, +1.6/+3–5 m bias). SRTM alone has no object detail. | L0-04 (raw add-back bias 7.8–12.2 m vs NC ≤ 2 m, synthetic) | DEM as truth |
| 5 | Why not stereo? | Out of scope of the PS; stereo (Cartosat-1) is the accuracy benchmark, not the method. | Phase 3 E23 | superiority over stereo |
| 6 | How is scale recovered? | Learned metric head + true GSD (tier H); anchors refine offset/scale (tier A). | L0-05 estimator recovery (synthetic) | real-data scale accuracy (untested) |
| 7 | Which vertical datum? | EGM2008 by default; all inputs transformed via PROJ grids; the app refuses tier T if grids are missing. | L0-01/01b (ballpark 0 m without grids; −24…−99 m over India) | "WGS84 heights" without specifying |
| 8 | How do you separate terrain from objects? | Ground mask from predicted nDSM weights the DEM cells; normalized convolution; raw-DEM fallback flagged. | L0-04 (+2 m residual under closed canopy) | a certified DTM |
| 9 | Forest? | Worst case by design: canopy blocks ground support; we flag and report separately (VVA-style). | L0-04; Phase 4 F7 | forest accuracy (untested) |
| 10 | Hilly terrain? | DEM slope error dominates the terrain layer; co-registration is mandatory; per-slope error tables planned. | L0-07 (shift recovery), Phase 3 E21 | hilly accuracy (untested) |
| 11 | GSD dependence? | Canonical GSD bands; out-of-range flagged; downsampling softens edges. | L0-03 (13 m edge error on step field) | GSD invariance |
| 12 | Aerial → satellite transfer? | Unknown until tested; training includes satellite-like DFC19; sensor-shift test planned. | Phase 4 §21 | transfer to Cartosat |
| 13 | Indian generalization? | No Indian LiDAR reference exists; we can only bound with ICESat-2/CartoDEM; sub-5 m ISRO imagery is priced. | Phase 4 F6/F10 | Indian metre-level accuracy |
| 14 | How validated? | Phase 4 blueprint: blocked splits, co-registration, masks, ME/RMSE/MAE/NMAD/LE95/r, per terrain/class. Currently NOT EXECUTED. | Phase 8 §14 | any current accuracy |
| 15 | Leakage? | Blocked splits with buffers, footprint-overlap exclusion, anchor–checkpoint exclusion — designed, not yet exercised. | Phase 8 §16 | "no leakage" (unproven) |
| 16 | Why report RMSE and NMAD? | RMSE is blunder-dominated (one +20 m point → 6.8 m RMSE vs 0.83 m NMAD); both are needed. | L0-05 hold-out | RMSE alone |
| 17 | Isn't correlation enough? | No: r = 1.000 with 60 m bias. | L0-08 | correlation as accuracy |
| 18 | Absolute vs relative output? | Tiers R/H/T/A label every number; PNG gives relative only. | design; Phase 8 scorecard | metres for PNG |
| 19 | Uncertainty? | Planned per-pixel raster (TTA + priors) — SHOULD, not yet built. | Phase 8 §32 | calibrated uncertainty |
| 20 | Standalone? | Planned one-folder offline bundle with grids/DEM/weights; clean-machine test required before claiming. | Phase 8 §3 (PyInstaller absent) | "runs anywhere" |

---

## 19. 60-Second Technical Explanation

"A single satellite photo can't tell you how high the ground is above sea level, but a depth model taught on laser-scanned cities and forests can tell you how tall the things on the ground are. Free elevation maps like Copernicus or SRTM know the ground's rough shape at 30 metres but not buildings — and they're a few metres too high under trees. So we split the job: the image gives heights above ground; the coarse map gives the terrain, read only where the image shows bare ground; we convert everything to one vertical reference — a step that matters by 24 to 99 metres in India — and add them to get a DSM in real metres, labelled with how it was calibrated. If the photo has no coordinates we say so and give relative heights only. We validate against airborne laser data with bias, RMSE and robust spread per terrain, and the 3D flythrough is built from exactly that surface, with the mesh error shown on screen."

## 20. 30-Second Elevator Pitch

"DepthWizard turns one georeferenced RGB image into a height-above-ground map and a sea-level-referenced surface model, labels every number with how it was calibrated, checks it against reference elevation data, and lets you fly through the exact surface it computed — offline. Without coordinates it gives an honest relative surface instead of fake metres."

---

## 21. PPT ↔ Product Consistency Audit

| Issue | Action |
| --- | --- |
| Any accuracy number on slides | Forbidden until `validation.json` exists; quote only from the panel with dataset/N/terrain |
| "Handles urban, sparse, hilly, forested" | Reword to "designed and *to be tested* across…"; show which have tables |
| Architecture diagrams with semantic head, ICESat-2, uncertainty, glTF | Remove or mark "planned" per §15 CUT/SHOULD |
| Synthetic Level-0 results | Show as "design verification (synthetic)" with the ballpark-datum and kernel-orientation findings — real, memorable evidence of rigour |
| Features that exist but are missing from the story | The L0 suite and the honest tier downgrade are the only current assets — feature them |
| Metrics that must appear on screen | ME, RMSE, MAE, NMAD, r, N, reference, shift, tier |
| Failure cases to show | Mode A watermark; DEM removed → tier H; closed-canopy hatching; invalid CRS refusal |
| Timings | Only from `experiment_log.json` of the demo machine |

---

## 22. Evaluator Perception Test (for the target product)

10 s: "Image in, height surface and 3D out; it tells me whether the numbers are metres." 30 s: "It detected georeferencing and a DEM; it says where absolute elevation comes from." 2 min: "The 3D is the computed surface (residual shown); heights are labelled relative/metric/absolute; there is a real validation table." 5 min: "Provenance and limitations are explicit; the team distinguishes prediction from measurement." Likely misunderstandings: that tier T accuracy equals the DEM's; that the terrain layer is a DTM; that visual smoothness implies accuracy. Skepticism trigger: any number without N/reference; a smooth mesh over a city. Resolving experiment: click the same building in 2D and 3D → identical raster value; toggle terrain-only vs DSM; show the residual map beside the metrics. **Today's evaluator would see nothing; the perception test cannot be run.**

---

## 23. Product Red Team

| Attack | Failure mode | Probability | Impact | Mitigation | Demo fallback |
| --- | --- | --- | --- | --- | --- |
| "Show me the software" | Nothing exists | **Certain today** | Total | Build MVP (Phase 7 B01–B30) | none |
| Missing geoid grids in bundle | Silent 0 m datum → wrong "absolute" DSM | High if untested | Severe (24–99 m) | Bundle grids; startup self-test; refuse tier T on "ballpark" | Show INVALID state honestly |
| Co-registration sign bug | Inflated/diverging validation | Medium if ported blindly | High | `correlate` + plane sign test + synthetic-shift test | Grid search |
| Huber estimator chosen by habit | Wrong offset with outliers | Medium | High | Median/RANSAC per L0-05 | — |
| No GPU on demo laptop | Slow live job | High | Medium | CPU path + precomputed twin (labelled) | Precomputed |
| Fine-tuned model not trained | Only relative baseline | High (no GPU locally) | High (no tier H) | Train on external GPU; else present baseline + DEM offset honestly | Mode A/relative demo |
| Reference datum untagged | Wrong metrics | Medium | High | 422 requiring vertical CRS | — |
| Evaluator asks Indian accuracy | No evidence | Certain | Medium | Bounded ICESat-2 statement; say "untested" | — |
| Mesh looks smooth over city | Perceived accuracy > real | Medium | Medium | Residual indicator; wireframe; no smoothing | — |
| PyInstaller bundle fails on clean VM | No standalone | Medium | High | Early packaging spike; wheelhouse fallback | venv + start script |
| Large image OOM | Crash | Medium | Medium | Cascade; AOI cap | Smaller AOI |
| WebGL missing | No 3D | Low | Medium | 2D fallback | QGIS |

---

## 24. Final Product Readiness Scorecard (actual implementation, not the plan)

| Category | Score / 10 | Explanation |
| --- | --- | --- |
| Scientific correctness in product | 0 | No product; design math verified at L0 only |
| Functional completeness | 0 | No component implemented |
| Input robustness | 0 | No ingestion code |
| DSM pipeline reliability | 0 | Not built |
| Calibration reliability | 0 (design 6) | Estimators/NC verified synthetically; nothing runs in a product |
| Geospatial correctness | 0 (design 7) | Round-trip logic verified; ballpark guard designed, not implemented |
| Validation visibility | 0 | No panel |
| 3D fidelity | 0 | No viewer |
| UI/UX | 0 | No UI |
| Performance | 0 | Nothing measured |
| Offline deployment | 0 | No bundle; PyInstaller absent |
| Failure resilience | 0 | No error handling exists |
| Reproducibility | 2 | L0 suite deterministic with provenance; system n/a |
| Demo readiness | 0 | Nothing to demo except L0 evidence |
| Evaluator clarity | 1 | Documents are clear; product absent |
| **Overall product readiness** | **0** | The research and design are complete; the product does not exist |

---

## 25. GO / NO-GO Decision

## NO-GO

The system cannot be demonstrated to an evaluator: no software exists, no output exists, no accuracy has been measured, and no packaging has been attempted. Demonstrating slides or synthetic checks as if they were the product would violate the Phase 8/9 rules.

**P0 — must fix before any demo:** implement the Phase 7 MVP (B01–B30) with the Phase 8 corrections C-1 (ballpark guard + bundled grids), C-2 (kernel orientation + iterative/grid-search co-registration with synthetic-shift test), C-3 (median/RANSAC + blunder rule); obtain baseline weights and, on an external GPU, the fine-tuned head (or ship baseline honestly at tier ≤ R with DEM offset labelled); bundle Copernicus tiles + geoid grids + one open reference subset; run the validation harness on at least one spatially blocked real scene per available terrain and put the table on screen; produce the tier badges/Mode A watermark; package offline and pass the clean-machine test; produce labelled precomputed twins for the demo scenes.
**P1 — should fix:** anchors with hold-out reporting; quality states with triggers; residual overlay; export zip; performance logging on the demo laptop; provenance panel complete; DFC19 protocol run.
**P2 — optional:** uncertainty raster; contours/profile; glTF; Linux build; ablation slides.

---

## 26. Final Phase 9 Deliverables Index

1 Product state audit §1 · 2 Demo contract §2 · 3 Architecture as implemented §3 · 4 UI/UX §4 · 5 Trust layer §5 · 6 3D spec §6 · 7 Demo datasets §7 · 8 Wow moment §8 · 9 Failure-resilient plan §9 · 10 Packaging §10 · 11 Performance plan §11 · 12 Hardening §13 · 13 Provenance §14 · 14 MUST/SHOULD/CUT §15 · 15 MVP §16 · 16–18 Demo scripts §17 · 19 Questions §18 · 20 60-s explanation §19 · 21 30-s pitch §20 · 22 PPT audit §21 · 23 Red team §23 · 24 Scorecard §24 · 25 Decision §25 · 26 P0/P1/P2 §25.

**Phase 8 determined what is true: the design is sound where it could be checked, two real defects were found and corrected in the design, and the product does not yet exist. Phase 9 therefore delivers the complete evaluator-facing product specification and an unambiguous NO-GO until the MVP is built, validated on real blocked data, and packaged offline.**
