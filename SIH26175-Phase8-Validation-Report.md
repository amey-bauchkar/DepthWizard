# Phase 8 — Validation, Experiment Execution, Benchmarking, Error Analysis & Red-Team Review

SIH26175 DepthWizard · executed 2026-09-20 → 2026-09-21

**Headline finding (read first).** The project directory contains **only the seven planning documents (Phases 1–7)**. There is no source code, no model weights, no configuration, no dataset, no fixture, no output artefact, no experiment log and no git repository. Consequently the Phase 6 architecture has **not been implemented**, and Levels 1–10 of the Phase 8 protocol (unit → deployment → reproducibility of the *system*) are **NOT EXECUTED**. The only validation that could truthfully be performed is **Level 0 — mathematical correctness of the specified algorithms** — using seeded synthetic data and self-contained reference scripts written for this phase. Those results are reported below with provenance; they verify the *design*, not a codebase. Nothing in this report is an estimate dressed up as a measurement.

Artifacts created: `validation/phase8/` (5 runnable scripts + 3 interactive scripts, `run_all.py`, `README.md`, `experiment_log.json`, `results/*.json`, 4 synthetic GeoTIFFs). All scripts are seeded; re-running produced identical output hashes (§22).

---

## 1. Validation Objective

Determine with evidence whether the implemented SIH26175 system executes, produces the intended intermediate representations (relative height → metric height above ground → terrain layer → absolute DSM), preserves geospatial metadata, achieves defensible metric accuracy across urban/sparse/hilly/forested terrain, renders faithfully in 3D, and can be demonstrated standalone — and where it fails.

## 2. Implementation Under Test

| Item | Status | Evidence |
| --- | --- | --- |
| Source code (`core/`, `backend/`, `frontend/`, `ml/`) | **NOT FOUND** | `find . -type f` → 7 `.md` files only; 0 files with code/data extensions |
| Model weights / model cards | NOT FOUND | no `models/` |
| Configuration / fixtures / tests | NOT FOUND | no `configs/`, `tests/` |
| Datasets / reference data / DEM tiles / geoid grids | NOT FOUND (DATA UNAVAILABLE locally) | no `data/`, `assets/` |
| Output artefacts / experiment logs | NOT FOUND | — |
| Git repository | NOT FOUND | `git rev-parse` → "not a git repository" |

Therefore the "system under test" for Levels ≥ 1 does not exist. What was tested: reference implementations (written 2026-09-20/21, `validation/phase8/*.py`) of the algorithms specified in Phase 6 §7–§12 and Phase 7 §24.

### Phase 6/7 traceability audit

| Planned component | Implemented? | Matches architecture? | Tested? | Evidence | Deviation |
| --- | --- | --- | --- | --- | --- |
| `ingest`, `geo` (meta/grid/reproject/resample/vertical/writer/check) | No | n/a | **L0 only** (round-trip, datum, resample) | `l0_02`, `l0_01`, `l0_06_12` | Not implemented — CRITICAL for demo |
| `preprocess` (normalize, canonical GSD, tiling, QC) | No | n/a | L0 only (tiling weights, resample error) | `l0_06_12` | Not implemented — CRITICAL |
| `inference` (DA-V2-S + nDSM head, TTA, blend) | No | n/a | NOT EXECUTED | — | Not implemented; no weights — CRITICAL |
| `ml.data/train` (fine-tuning) | No | n/a | NOT EXECUTED | — | Not implemented — CRITICAL for tier H |
| `calib` (discover, DEM datum, NC terrain, anchors, tier) | No | n/a | L0 only (NC, anchors, datum) | `l0_04`, `l0_05`, `l0_01` | Not implemented — CRITICAL |
| `dsm` (compose, slope, flags) | No | n/a | L0 only | `l0_06_12` | Not implemented |
| `validate`, `quality` | No | n/a | L0 only (metrics, co-registration) | `l0_06_12`, `l0_07*` | Not implemented |
| `terrain` tileset + viewer (Three.js/RTIN) | No | n/a | L0 partial (regular-grid decimation only; RTIN NOT EXECUTED) | `l0_06_12` | Not implemented |
| `api`, `jobs`, `store`, packaging, demo mode | No | n/a | NOT EXECUTED | — | Not implemented |
| Deviations found in the *design* during L0 | — | — | — | see §11, §13, §27 | Two design-level corrections required (PROJ ballpark detection; Horn kernel orientation/iteration); one acceptance threshold needs recalibration |

## 3. Environment (actual, measured 2026-09-21)

| Item | Value | Comparison with Phase 7 plan |
| --- | --- | --- |
| OS | Windows 11, build 10.0.26200 | matches |
| Python | **3.14.0** (`Python314`) | plan: 3.12 → DEVIATION (rasterio 1.5.1, pyproj 3.8.0, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.9.0, torch 2.11.0+cpu, torchvision 0.26.0, fastapi 0.141.1, pydantic 2.13.4, pillow 12.2.0, safetensors 0.8.0 all installed and importable on 3.14 — no conflict observed in this session; torchvision excludes only 3.14.1) |
| Node / npm | v25.6.1 / 11.9.0 | plan: 24 LTS → DEVIATION (current release, not LTS; frontend not built, so untested) |
| GPU / CUDA | **none** (`nvidia-smi` not found; `torch.cuda.is_available()` = False; torch build CPU-only) | plan profile C (CPU fallback) applies; fine-tuning not possible on this machine |
| RAM / disk | 15.7 GB / 81.7 GB free of 458.9 GB | meets profile B/C RAM; disk adequate for demo assets |
| GDAL / PROJ | GDAL 3.12.4 (rasterio wheel), PROJ 9.8.1 (pyproj wheel); PROJ data dir = pyproj's bundled `share/proj` | two PROJ copies (rasterio vs pyproj) as anticipated in Phase 7 §30 |
| Geoid grids | **absent** locally (`*egm*` not in PROJ data dir or user dir) | see §11 finding on silent ballpark transforms |
| PyInstaller | NOT INSTALLED | packaging untested |
| Model weights | NOT FOUND | — |
| Browser/WebGL | not exercised | — |

## 4. Dataset Inventory

| Dataset | Local availability | Status |
| --- | --- | --- |
| GAMUS, DFC2019, GeoNRW, swisstopo, LINZ, AHN4, 3DEP+NAIP, Open-Canopy, NEON | none downloaded | DATA UNAVAILABLE (locally) — public availability was verified in Phase 4 |
| Copernicus GLO-30 / SRTM / AW3D30 / CartoDEM tiles | none downloaded | DATA UNAVAILABLE (locally) |
| ICESat-2 ATL08 / GEDI | none | DATA UNAVAILABLE |
| Organiser-supplied imagery/reference | none (repository still README-only as of Phase 7) | UNKNOWN / NOT PROVIDED |
| Synthetic test scenes | created this phase | available in `validation/phase8/results/*.tif` and in-script arrays |

## 5. Reference Data

No LiDAR, DEM or GCP reference exists locally. All Level-0 "references" are **synthetic ground truth** (analytic terrain, block buildings, canopy region) with known parameters; they validate mathematics only.

## 6. Test Protocol

Levels per Phase 8 brief: L0 executed with seeded synthetic data (`numpy.random.default_rng(3/7/42)`), each script writing JSON to `results/`; runner `run_all.py` captures environment and script hashes. L1–L10 NOT EXECUTED (no implementation). No result was altered after computation; two *test scripts* were corrected for bugs in the test itself (an out-of-footprint pixel index in L0-10; a kernel-orientation bug in the co-registration reference) — both originals and corrections are retained in `results/` and described in §13 and §24.

## 7. Unit Validation (Level 1, system code)

NOT EXECUTED — no codebase. Level-0 equivalents below.

## 8. Module Validation (Level 0 results, per specified module)

| Spec (Phase 6 §) | Test | Result (actual) | Status |
| --- | --- | --- | --- |
| `geo.meta/grid/writer` (§9.3, §12) | `l0_02`: synthetic UTM GeoTIFF write/read | CRS 32643 and transform round-trip exact; pixel (col 10,row 20) centre → (500010.5, 2999979.5) = expected; inverse index exact; bounds exact; nodata −9999 count 1; tags `VERTICAL_CRS=EPSG:3855`, `TIER`, `MODEL_HASH` round-trip; overviews [2,4] | PASS |
| rotation / geographic / no-CRS handling | `l0_02` b–d | rotated transform detected (GSD from transform 0.906 m); geographic CRS at 28.6°N local GSD **0.978 m (E) × 1.108 m (N)** — anisotropic, confirming reproject-to-UTM rule; no-CRS TIFF → `crs is None` (Mode A) | PASS |
| `geo.vertical` (§12, D-08) | `l0_01` online | EGM96 / EGM2008 undulation (m): Kanyakumari −98.554/−98.274; Bengaluru −86.421/−86.164; Mumbai −68.521/−67.672; Delhi −52.602/−52.555; Kolkata −56.863/−56.946; Leh −23.646/−24.322; Guwahati −49.000/−47.994; Dehradun −43.777/−43.089; EGM96−EGM2008 within −1.006…+0.676 m | PASS (values) |
| `geo.vertical` offline behaviour | `l0_01` offline, `l0_01b` | **With grids absent PROJ returns 0.0 m correction via a "ballpark vertical transformation" — no error, even with `only_best=True`** | **FAIL of naive design assumption → design correction required (§27 C-1)** |
| `preprocess.canonical_gsd` inverse error (§4) | `l0_06_12` L0-03 | smooth field: RMSE 0.0054 m, max 0.073 m, relative RMSE **1.06e-3** (Phase 7 threshold 1e-3 → marginal); step field (20 m buildings): max error **13.24 m**, RMSE 0.545 m | PARTIAL — threshold needs recalibration; edge blur on down-then-up resampling is intrinsic |
| `preprocess.tiling` (§4) | L0-12 | tile starts [0,48,96,136] cover 200 px fully; raw feather weight sums 0–1.74 → normalized blend required (as designed) | PASS |
| `calib.terrain` NC vs raw add-back (§7) | `l0_04` (300×300 @1 m, 30 m DEM, canopy penetration 0.6, DEM noise 1 m) | terrain-layer error over **canopy**: raw ME +7.79 m / RMSE 7.97 → NC ME +2.01 / RMSE 2.27 (bias −74.3 %); over **buildings**: raw +12.16 / 14.28 → NC −0.48 / 1.39 (−96.1 %); over **ground**: raw ME 0.93 RMSE 3.17 NMAD 1.40 → NC ME 0.45 RMSE 1.77 **NMAD 1.90**; composed DSM error all pixels: raw ME 3.01 RMSE 5.69 → NC ME 0.88 RMSE 2.00; ground support mean 0.69, raw-fallback cells 0; consistency check ME 1.88 / NMAD 2.26 | PASS (mechanism works as designed on synthetic data); note residual +2 m under closed canopy and slight ground smoothing |
| `calib.anchors` offset (§7) | `l0_05` (noise 0.5 m, 2 gross +20 m outliers for N ≥ 5) | median offset error: N=3 0.445 m (no outliers, no redundancy); N=5 0.053 m; N=10 0.132 m; N=30 0.030 m; blunders flagged 2/2 for N ≥ 5; **Huber at N=5 with 40 % outliers → 10.999 m (fails)** | PASS for median/Theil–Sen; Huber NOT robust at high outlier fraction |
| `calib.anchors` object scale | `l0_05` RANSAC | expected 0.8333; N=3 0.7955 (4.5 % error); N=5 0.8330; N=8 0.8248; N=20 0.8321; inliers 3/4/7/19 | PASS for N ≥ 5 |
| anchor hold-out reporting (§7, D-12) | `l0_05` | fit offset 3.019 (21 pts); hold-out (9 pts, contains 1 injected +20 m outlier): ME 2.215, **NMAD 0.83**, RMSE 6.81 | PASS with lesson: hold-out statistics must include blunder handling/NMAD |
| `dsm.derive` Horn slope (§8) | `l0_06_12` L0-06 | plane a=0.10,b=0.05: expected 6.3794°; interior max error **0.0°**; edge max error **3.18°** (nearest padding) | PASS interior; edges must be flagged (as designed) |
| `dsm.compose` nodata (§8) | L0-10 (+fix) | 100 + 100 NaN → 200 NaN in DSM; value check correct after test-index fix | PASS |
| `validate.metrics` (§10) | L0-08 (Gaussian e ~ N(0.5, 2²)) | ME 0.496, RMSE 2.061 (= √(ME²+var) 2.061 ✓), MAE 1.646, NMAD 2.010 (σ=2 ✓), LE90 3.40, LE95 4.04, r 0.9313, ρ 0.9319 | PASS |
| correlation blindness (Phase 4 §12) | L0-08 | prediction = 1.5·ref + 10: **r = 1.000, ρ = 1.000 while ME 60.43 m, RMSE 60.49 m** | PASS (demonstrates constraint) |
| ground-pixel dominance (Phase 4 §13) | L0-09 (57 % zeros) | all-zero prediction and doubled-heights prediction have **identical MAE 3.385 / RMSE 6.632**; object-only MAE 7.871 both | PASS (reproduces SynRS3D critique numerically) |
| `validate.coregister` (§10, D-14) | `l0_06_12` L0-07 grid search | true (2, −1) px → integer best (2, −1); sub-pixel (1.979, −1.020); RMSE 0.520 → 0.424 m | PASS |
| gradient / Nuth–Kääb estimator | `l0_07b/c/d` | first reference implementation: (−1.04, +0.42) then **divergent** iteration → cause: `ndimage.convolve` flips the Horn kernel (sign error); corrected (`correlate`): converges to (2.000, −1.012) in 3 iterations with reference smoothing σ=2; (1.979, −0.999) in 10 iterations without smoothing (regression dilution) | PASS after correction; **design lesson → §27 C-2** |
| `terrain` mesh fidelity (§11) | L0-11 regular-grid decimation | plane f=2 max 0.15 m; hills f=8 max 1.72 m; **buildings f=2 max 17.9 m, f=4/8 max 20.0 m** (full building height at edges) | PASS as demonstration that uniform decimation destroys building edges → error-bounded RTIN + residual display are necessary; RTIN itself NOT EXECUTED |

## 9. Integration Validation — NOT EXECUTED (no implementation).
## 10. End-to-End Validation — NOT EXECUTED.

## 11. Calibration Validation

Executed only at Level 0 (§8 rows `calib.*`). Summary of *actual* experiments:

| Experiment | Input | Reference | Method | Parameters | Expected | Actual | Error | Failure mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAL-1 NC terrain vs raw | synthetic nDSM (noisy, σ 0.8) + 30 m DEM (canopy 0.6, noise 1 m) | analytic terrain | ground-weighted normalized convolution | h_ground 1.0, σ 1.5 cells, w_min 0.1 | bias reduction over objects | canopy ME 7.79→2.01; buildings 12.16→−0.48; ground NMAD 1.40→1.90 | residual +2.0 m under closed canopy | closed canopy has no interior ground support; NC interpolates from outside |
| CAL-2 offset recovery | terrain −3 m | ground anchors σ 0.5 + 2×(+20 m) | median | N ∈ {3,5,10,30} | +3.0 | 2.555 / 3.053 / 2.868 / 2.970 | 0.445 / 0.053 / 0.132 / 0.030 | N=3 no redundancy; Huber fails at 40 % outliers |
| CAL-3 scale recovery | nDSM ×1.2 | object anchors + 1 outlier | RANSAC (no intercept) | N ∈ {3,5,8,20} | 0.8333 | 0.7955 / 0.8330 / 0.8248 / 0.8321 | 4.5 % / 0.04 % / 1.0 % / 0.1 % | N=3 unreliable |
| CAL-4 hold-out | 30 ground anchors, 3 outliers | — | 70/30 split | seed 7 | ≈ 0 residual | hold ME 2.22, NMAD 0.83, RMSE 6.81 | outlier in hold set | RMSE must be accompanied by NMAD / blunder rule |
| CAL-5 datum offline | Delhi (77.21, 28.61) | PROJ EGM2008 | pyproj transform | grids absent | error or NaN | **0.0 m, "ballpark vertical transformation"** | −52.6 m silent | **silent datum failure** |
| CAL-6 DEM-assisted with real DEM; CAL-7 GCP distributions on real scenes; CAL-8 anchor-noise curves | — | — | — | — | — | NOT EXECUTED | — | DATA UNAVAILABLE / no implementation |

## 12. DSM Validation — NOT EXECUTED on real data. Synthetic composition and nodata propagation verified (§8).

## 13. Geospatial Validation

PASS at Level 0 for CRS/transform/pixel-centre/bounds/nodata/tag round-trip and rotation/geographic/no-CRS handling (`l0_02`). **FAIL (design assumption)** for offline vertical transforms: PROJ substitutes a 0-m ballpark transform when grids are missing; the Phase 6 `ensure_grids()` idea is necessary but must additionally reject any pipeline whose description contains "ballpark" (verified in `l0_01b`: `only_best=True` did *not* prevent it). Real-data reprojection, DEM alignment, and reference alignment: NOT EXECUTED.

## 14. Terrain Validation

| Terrain | Status |
| --- | --- |
| URBAN | NOT VALIDATED — DATA UNAVAILABLE (no implementation, no reference) |
| SPARSE | NOT VALIDATED — DATA UNAVAILABLE |
| HILLY | NOT VALIDATED — DATA UNAVAILABLE |
| FORESTED | NOT VALIDATED — DATA UNAVAILABLE |

Height-bin and object-class analyses (Parts 14–15): NOT EXECUTED. The only class-resolved evidence is synthetic (CAL-1: canopy vs buildings vs ground).

## 15. Generalization Validation — NOT EXECUTED (in-domain and out-of-domain).

## 16. Leakage Audit

| Risk | Evidence | Impact | Remediation |
| --- | --- | --- | --- |
| Train/test overlap, duplicates, adjacent tiles | No training has occurred; no manifests exist | none yet | Phase 7 B25 (blocked splits, overlap check) must be implemented before any accuracy claim |
| Calibration/test contamination | `l0_05` shows why: an anchor in the hold-out set dominates RMSE; the design's ID+radius exclusion is untested | potential | implement `leakage.py`, unit-test with synthetic anchors |
| Reference-data leakage into training | no data downloaded | none | keep `data/reference/` out of `ml/data` loaders |
| Preprocessing fitted on test data | none exists | none | stretch percentiles per image only (as designed) |
| Fine-tuning on test data | none exists | none | manifests + hashes |

## 17. 3D Geometry Validation

RTIN/martini mesh: NOT EXECUTED (no frontend). Regular-grid decimation experiment (L0-11) shows the geometric stakes: plane/hills residuals grow with decimation factor (0.15 → 1.05 m plane; 0.26 → 1.72 m hills) and **building edges lose the full 20 m at any factor**, so a fixed decimation is inadmissible and error-bounded simplification with a displayed residual (Phase 6 §11) is the correct requirement.

## 18. Texture Projection Validation — NOT EXECUTED. (Design guarantees alignment by shared grid; unverified in practice.)
## 19. UX Validation — NOT EXECUTED (no UI; no SUS session).
## 20. Performance Validation — NOT EXECUTED for the system. Only L0 script timings exist (0.6–8.7 s each; irrelevant to the pipeline). No inference time, VRAM, frame rate or E2E time has been measured; none is claimed.
## 21. Failure-Safety Validation — NOT EXECUTED for the system. One failure-safety fact established at L0: PROJ fails *unsafely* (silently) when grids are absent — the system must add its own detection.

## 22. Reproducibility Validation

L0 suite: three seeded scripts run twice → identical SHA-256 of outputs (`l10_reproducibility_l0_suite.json`: all `deterministic: true`). Python 3.14.0, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.9.0, rasterio 1.5.1/GDAL 3.12.4, pyproj 3.8.0/PROJ 9.8.1 recorded in `experiment_log.json`. System reproducibility: NOT EXECUTED.

## 23. Baseline Comparison

The Phase 5 baseline (zero-shot DA-V2 + affine fit) and the proposed system: NOT EXECUTED (no weights, no implementation). The only baseline-vs-proposal comparison performed is synthetic CAL-1: **raw DEM add-back (Concept B-style) vs ground-masked NC terrain (proposed)** — composed-DSM RMSE 5.69 m → 2.00 m on the synthetic scene. This is evidence that the mechanism does what it is designed to do, not evidence about real terrain.

## 24. Experiment Log

| ID | Date | Code version | Model | Dataset | Config | Hardware | Input | Reference | Method | Result | Error | Artifacts | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0-01 | 2026-09-21 | `l0_01_datum.py` (sha 16-hex in log) | — | — | PROJ CDN on/off | Win11, CPU | 8 Indian points | PROJ EGM grids | pyproj | online values; offline 0.0 ballpark | — | `results/l0_01_datum.json` | grids must be bundled and ballpark detected |
| L0-01b | 2026-09-20 | interactive | — | — | offline | same | Delhi | — | Transformer.description | "ballpark vertical transformation" | — | `l0_01b_*.json` | `only_best` insufficient |
| L0-02 | 2026-09-21 | `l0_02_geotiff_roundtrip.py` | — | synthetic | — | same | 4 synthetic GeoTIFFs | analytic | rasterio | all checks PASS | 0 | `l0_02_*.json`, `*.tif` | geo core spec sound |
| L0-04 | 2026-09-21 | `l0_04_nc_terrain.py` | — | synthetic (seed 42) | σ1.5, w_min 0.1, h_ground 1 | same | 300² scene | analytic terrain | NC vs raw | canopy bias −74 %, buildings −96 %, DSM RMSE 5.69→2.00 | residual +2 m canopy | `l0_04_*.json` | mechanism valid on synthetic |
| L0-05 | 2026-09-21 | `l0_05_anchors.py` | — | synthetic (seed 7) | noise 0.5, outliers | same | anchors | analytic | median/Huber/RANSAC | see §11 | — | `l0_05_*.json` | median+RANSAC robust; Huber not |
| L0-06…12 | 2026-09-21 | `l0_06_12_misc.py` | — | synthetic (seed 3) | — | same | fields | analytic | numpy/scipy | see §8 | — | `l0_06_12_*.json` | slope/metrics/tiling sound; resample threshold marginal |
| L0-07b | 2026-09-20 | interactive | — | synthetic | — | same | shifted hills | analytic | NK cosine form / linearised | wrong sign & magnitude | — | `l0_07b_*.json` | convention-sensitive |
| L0-07c | 2026-09-21 | interactive | — | synthetic | — | same | same | analytic | iterative (buggy kernel) | **diverged** | — | `l0_07c_*.json` | sign error → divergence |
| L0-07d | 2026-09-21 | interactive | — | synthetic | — | same | same | analytic | iterative, correlate, σ∈{0,1,2} | (2.000, −1.012) | ≤ 0.02 px | `l0_07d_*.json` | corrected estimator converges |
| L0-10fix | 2026-09-20 | interactive | — | synthetic | — | same | — | — | index fix | PASS | — | `l0_10_fix.json` | test bug, not design |
| L10 | 2026-09-21 | interactive | — | — | — | same | 3 scripts ×2 | — | sha256 | deterministic | — | `l10_*.json` | L0 suite reproducible |
| SYS-* (Levels 1–10) | — | — | — | — | — | — | — | — | — | **NOT EXECUTED** | — | — | no implementation |

## 25. Validation Scorecard

| Area | Status | Evidence |
| --- | --- | --- |
| Image ingestion | NOT TESTED (system); L0 metadata handling PASS | `l0_02` |
| Non-georeferenced pipeline | NOT TESTED | no implementation |
| Georeferenced pipeline | NOT TESTED | no implementation |
| Depth inference | NOT TESTED | no weights/code |
| Calibration | PARTIAL (L0 mechanism PASS; real data NOT TESTED; offline datum FAIL of assumption) | `l0_04`, `l0_05`, `l0_01b` |
| Metric elevation | NOT TESTED | — |
| DSM generation | PARTIAL (L0 composition PASS) | `l0_06_12` |
| Geospatial correctness | PARTIAL (L0 round-trip PASS; datum detection FAIL; real data NOT TESTED) | `l0_02`, `l0_01b` |
| Quantitative validation (RMSE/MAE/r vs LiDAR) | NOT TESTED | DATA UNAVAILABLE |
| Terrain robustness | NOT TESTED | DATA UNAVAILABLE |
| 3D generation | NOT TESTED (RTIN); decimation stakes shown | `l0_06_12` L0-11 |
| Texture projection | NOT TESTED | — |
| Navigation | NOT TESTED | — |
| Measurement | NOT TESTED | — |
| Slope analysis | PARTIAL (L0 Horn interior exact; edges 3.18° error must be flagged) | L0-06 |
| Performance | NOT TESTED | — |
| Stability | NOT TESTED | — |
| Deployment | NOT TESTED (PyInstaller not installed; no build) | environment |

## 26. PS Requirement Matrix

| SIH Requirement | Implementation | Test | Result | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Accept PNG/JPG | none | — | — | — | NOT TESTED |
| Accept GeoTIFF/TIFF | none (spec verified) | L0 round-trip | metadata preserved in spec | `l0_02` | NOT TESTED (system) |
| rDSM for non-georeferenced | none | — | — | — | NOT TESTED |
| Absolute DSM with metric heights (SRTM/GCP) | none (mechanism verified) | L0 CAL-1..4 | mechanism valid on synthetic | `l0_04`, `l0_05` | NOT TESTED (system) |
| Pretrained monocular depth backbone | none | — | — | — | NOT TESTED |
| Scale-calibration module | none | L0 | tiers/rules not implemented | — | NOT TESTED |
| Standard geospatial output | none | L0 COG tags | tags round-trip | `l0_02` | NOT TESTED (system) |
| RGB projected on 3D mesh | none | — | — | — | NOT TESTED |
| Rendering engine, first-person, aerial | none | — | — | — | NOT TESTED |
| Structural height / slope analysis | none | L0 slope | Horn correct interior | L0-06 | NOT TESTED (system) |
| Upload → visualize → validate | none | — | — | — | NOT TESTED |
| RMSE/MAE/correlation vs LiDAR | none | L0 metric identities | formulas correct | L0-08 | NOT TESTED (no reference) |
| Stability across four terrains | none | — | — | — | NOT TESTED |
| Software stability | none | — | — | — | NOT TESTED |
| Standalone deployment | none | — | — | — | NOT TESTED |
| Source + documentation | 7 planning docs; L0 scripts | — | — | repo | PARTIAL (documentation exists; source does not) |

## 27. Scientific Validity Audit (hostile reviewer)

| Question | Evidence | Verdict | Required correction |
| --- | --- | --- | --- |
| Is the reference appropriate? | No real reference used; synthetic truth only | Not yet assessable | Acquire LiDAR DSM/DTM subsets per Phase 4 before any claim |
| Is this truly a DSM? | Design composes terrain + nDSM; synthetic test confirms semantics | Design-valid; system untested | — |
| Is metric scale demonstrated? | Only synthetic scale recovery (RANSAC within 1 % for N ≥ 5) | NOT demonstrated on real data | Run C0–C8 ablations (Phase 4 §19) once implemented |
| Is absolute elevation established? | Terrain layer mechanism verified synthetically; **offline datum transform silently fails without grids** | Not established; **C-1**: `ensure_grids()` must verify grid files AND reject transformers whose description contains "ballpark"; add datum-sanity check vs DEM mean | Implement + unit-test before any tier-T output |
| Is the vertical datum consistent? | Values for 8 Indian points computed; EGM96↔EGM2008 ≤ 1 m; ellipsoid mix-up 24–99 m | Consistent in design; unverified in system | — |
| Is spatial alignment correct? | Grid search recovers shift to 0.02 px; gradient estimator needed iteration and had a sign bug in the reference | **C-2**: implementation must use `correlate` (or verify sign on a plane), iterate to convergence, and unit-test on synthetic shifts | Add plane-gradient sign test and shift-recovery test (Phase 7 §24 already lists the latter) |
| Is the benchmark contaminated? | No benchmark run | N/A | Blocked splits + overlap check before training |
| Is the test set independent? | None exists | N/A | — |
| Are metrics computed correctly? | Identities hold (RMSE² = ME² + var; NMAD ≈ σ) | Yes (reference implementation) | Port with unit tests |
| Are outliers handled? | Median/RANSAC robust; Huber failed at 40 % outliers; hold-out RMSE dominated by one blunder | **C-3**: use median/Theil–Sen + RANSAC, apply ASPRS 3×RMSE blunder investigation to hold-out points, report NMAD alongside RMSE | — |
| Does correlation conceal bias? | r = ρ = 1.000 with ME 60 m | Yes — never report r alone | Enforce metric set |
| Are terrain categories represented? | None tested | No | Data acquisition |
| Are improvements meaningful? | Synthetic only (5.69 → 2.00 m RMSE) | Not yet | Real-data ablations with per-terrain tables |
| Are visuals hiding geometry errors? | Decimation destroys 20 m building edges | Risk confirmed | Error-bounded RTIN + residual display (already designed) |
| Resampling acceptance threshold | smooth-field relative RMSE 1.06e-3 vs threshold 1e-3 | **C-4**: threshold marginally violated by the reference method (order-1 down / order-3 up) | Recalibrate threshold or use area-average downsampling; document edge blur on discontinuities |

## 28. Engineering Red Team

| Finding | Evidence | Severity |
| --- | --- | --- |
| Silent datum no-op when grids absent | `l0_01b` | **P0** |
| Kernel orientation (convolve vs correlate) flips gradient sign → divergent co-registration | `l0_07c/d` | **P0** (if ported unchecked) |
| Huber offset estimator fails at 40 % outliers | `l0_05` | P1 |
| Hold-out RMSE dominated by single blunder | `l0_05` | P1 (reporting rule) |
| Geographic-CRS anisotropic GSD (0.98 × 1.11 m at 28.6°N) | `l0_02` | P1 (must reproject) |
| Canonical downsampling blurs edges (13 m max on step) | L0-03 | P1 (document; prefer canonical band ≤ input GSD) |
| Horn edge error 3.18° with nearest padding | L0-06 | P2 (flag border) |
| NC low-pass raises ground NMAD 1.40 → 1.90 m | `l0_04` | P2 (σ tuning trade-off) |
| Uniform mesh decimation loses full building height | L0-11 | P1 (RTIN mandatory) |
| Python 3.14 / Node 25 on dev machine vs plan 3.12 / 24 | environment | P2 (works in session; pin per Phase 7) |
| Two PROJ copies (rasterio, pyproj) | environment | P2 (route vertical transforms through pyproj only) |
| No GPU on available machine | environment | P1 (fine-tuning must run elsewhere) |
| Hardcoded assumptions, race conditions, memory leaks, stale artefacts | no code | NOT ASSESSABLE |

## 29. Error Budget (qualitative; no measured decomposition exists)

Real-data error budget: NOT MEASURABLE yet. Synthetic decomposition (CAL-1): with a perfect-scale nDSM (noise 0.8 m) and a 1 m-noise DEM, composed DSM RMSE 2.00 m splits into terrain-layer error (RMSE 1.77 m ground / 2.27 m canopy / 1.39 m buildings) plus model noise (0.8 m) plus NC smoothing; the raw add-back budget was dominated by DEM object bias (7.8–12.2 m ME). On real data the expected dominant terms (Phase 3/4 evidence) are model scale/morphology error, DEM slope error in relief, reference datum/registration error — none measured here.

## 30. What Actually Works

**Reliably (Level 0, synthetic):** GeoTIFF metadata round-trip logic; pixel-centre conventions; PROJ datum values (online); Horn slope interior; metric formulas; tiling coverage with normalized blending; nodata propagation; median/Theil–Sen offset recovery and RANSAC scale recovery for N ≥ 5; ground-masked NC terrain reduces object-induced DEM bias substantially on the synthetic scene; grid-search co-registration; L0 suite determinism.

## 31. What Does Not Work / Fails

- **The system**: does not exist → every system-level requirement is unmet today.
- **Offline vertical transform** without bundled grids: PROJ silently returns 0 m (would label ellipsoidal heights as EGM2008).
- **Huber offset** at 40 % gross outliers.
- **Single-pass gradient co-registration** without iteration/smoothing (attenuated) and with wrong kernel orientation (divergent).
- **Uniform mesh decimation** for building scenes.
- Resampling acceptance threshold as written in Phase 7 (marginal miss).

**Under conditions:** NC terrain needs ground support around objects (closed canopy retains ~2 m bias here); anchor fits need N ≥ 5 with outlier rejection.

**Fragile:** anything depending on grid files, kernel conventions, or pixel-centre conventions — all silent-failure classes.

## 32. What Remains Untested

Everything at Levels 1–10: ingestion of real files, inference (success, time, memory, tiling seams, border artefacts), fine-tuned vs zero-shot accuracy, Mode A and Mode B end-to-end, DEM-assisted calibration on real DEMs, GCP/ICESat-2 anchors on real scenes, DSM accuracy (RMSE/MAE/r) against LiDAR, all four terrain classes, height bins, object classes, generalization, leakage in real manifests, RTIN mesh residual, texture alignment, navigation, measurement labelling, UX/SUS, performance, large inputs, failure-safety of the application, output reopening in GIS, system reproducibility, packaging/offline deployment.

## 33. Issues (P0–P3)

| Prio | Issue | Evidence | Impact | Fix | Validation after fix |
| --- | --- | --- | --- | --- | --- |
| P0 | No implementation exists | directory listing | nothing can be demonstrated or validated | Execute Phase 7 backlog B01→B30 (MVP) | Levels 1–4 tests |
| P0 | Silent PROJ ballpark vertical transform | `l0_01b` | 24–99 m datum error labelled as EGM2008 | bundle grids; `ensure_grids()` checks files **and** rejects "ballpark" descriptions; datum-sanity vs DEM | unit test offline → must raise |
| P0 | Gradient kernel orientation / non-iterated co-registration | `l0_07c/d` | wrong shift → inflated validation error or divergence | use `correlate`, plane sign test, iterate to convergence (or grid search + sub-pixel) | synthetic shift test (2,−1) within 0.05 px |
| P1 | Robust estimator choice | `l0_05` | Huber fails at high outlier fraction | median/Theil–Sen for offset; RANSAC for scale; N ≥ 5 for scale | synthetic outlier tests |
| P1 | Hold-out reporting | `l0_05` | RMSE dominated by blunders | ASPRS 3×RMSE investigation; report NMAD + LE95 with RMSE | fixture |
| P1 | Geographic-CRS inputs | `l0_02` | anisotropic GSD → wrong metric scale | reproject to UTM (as designed) and unit-test | fixture |
| P1 | No GPU on dev machine | environment | fine-tuning blocked locally | cloud/colleague GPU; CPU inference path | — |
| P1 | Uniform decimation in any viewer fallback | L0-11 | building edges vanish | RTIN with tolerance + residual | vitest residual test |
| P2 | Resampling threshold | L0-03 | false test failures | set threshold from measured 1.06e-3 (e.g. 2e-3) or use area-average down-sampling | rerun |
| P2 | Horn edge handling | L0-06 | 3° edge error | one-sided differences + BORDER flag | fixture |
| P2 | NC smoothing trade-off | `l0_04` | ground detail loss | tune σ on real DTM low-pass comparison | Phase 8 rerun on data |
| P2 | Python/Node version drift | environment | packaging surprises | pin 3.12 / Node 24 per Phase 7 | build test |
| P3 | Documentation of conventions | this report | onboarding errors | `docs/geospatial.md` section on convolve/correlate, pixel centre, datum | review |

## 34. Evidence-Based Recommendations

1. Implement the MVP (Phase 7 B01–B30) — no validation beyond Level 0 is possible until then.
2. Port the L0 reference logic into `core/` **with its tests** (`tests/unit`): datum values, ballpark rejection, plane-gradient sign, shift recovery, NC canopy-bias reduction, anchor recovery with outliers, metric identities, nodata propagation, tiling coverage.
3. Bundle `us_nga_egm96_15.tif` and `us_nga_egm08_25.tif`; make tier T impossible when grids are missing.
4. Use median/Theil–Sen (offset) and RANSAC (scale); require N ≥ 5 for scale; report NMAD/LE95 with RMSE; apply blunder rule to hold-out.
5. Implement co-registration as iterative gradient fit with reference smoothing *or* grid search + parabolic refinement; test both against synthetic shifts.
6. Reproject geographic-CRS inputs to UTM before GSD-dependent steps.
7. Prefer canonical GSD bands ≤ input GSD where possible; document edge blur; recalibrate the resampling acceptance threshold from measurement.
8. Keep error-bounded RTIN with residual display; never ship uniform decimation.
9. Acquire at least one multi-terrain LiDAR subset (Phase 4 §4) and one coarse DEM tile set before the next validation pass; without them terrain validation remains impossible.
10. Re-run this L0 suite in CI on every commit to `core/` (it takes ~15 s).

---

## PHASE 8 COMPLETE

**1. Overall Validation Status: NOT YET VALIDATED.** No implementation exists; only Level-0 design verification was executed.

**2. What Actually Works (Level 0, synthetic):** GeoTIFF/CRS/transform/pixel-centre/tag round-trip; PROJ datum values (online); Horn slope (interior); metric definitions and identities; normalized blending/tiling coverage; nodata propagation; median offset and RANSAC scale recovery (N ≥ 5, with outliers); ground-masked NC terrain (object bias −74 % canopy / −96 % buildings, composed RMSE 5.69 → 2.00 m on the synthetic scene); grid-search and corrected iterative co-registration; deterministic suite.

**3. Works Only Under Conditions:** NC terrain requires ground support around objects (≈ +2 m residual under a closed synthetic canopy block); anchor scale needs N ≥ 5; single-pass gradient co-registration needs reference smoothing and iteration; slope at raster edges needs one-sided differences/flags.

**4. Currently Fails:** the system (absent); offline vertical transforms without grids (silent 0 m); Huber offset at 40 % outliers; non-iterated/mis-oriented gradient co-registration (divergence); uniform mesh decimation on buildings; the 1e-3 resampling threshold as written.

**5. Not Tested:** all Level 1–10 items (§32) — ingestion, inference, fine-tuning, both modes end-to-end, real calibration, DSM accuracy, all four terrains, generalization, RTIN, texture, navigation, UX, performance, large inputs, failure-safety, outputs in GIS, packaging.

**6. Actual DSM Metrics:** none against real reference. Synthetic composed-DSM (CAL-1): raw add-back ME 3.01 / RMSE 5.69 m; NC terrain ME 0.88 / RMSE 2.00 m (all pixels, 90,000 px). Not a system accuracy figure.

**7. Terrain-wise Results:** URBAN / SPARSE / HILLY / FORESTED — NOT VALIDATED — DATA UNAVAILABLE.

**8. Calibration Results (synthetic):** offset recovery error 0.445 / 0.053 / 0.132 / 0.030 m for N = 3/5/10/30 (median; 2 outliers at N ≥ 5); Huber 10.999 m at N=5 (fail); scale 0.7955 / 0.8330 / 0.8248 / 0.8321 vs 0.8333 for N = 3/5/8/20; hold-out (9 pts, 1 blunder) ME 2.22, NMAD 0.83, RMSE 6.81 m; NC canopy ME 7.79 → 2.01 m, buildings 12.16 → −0.48 m.

**9. Geospatial Validation Results:** L0 round-trip PASS; geographic GSD anisotropy 0.978 × 1.108 m at 28.6°N; datum undulations −98.6 m (Kanyakumari) … −23.6 m (Leh), EGM96−EGM2008 within ±1.0 m; **offline ballpark transform returns 0 m silently (FAIL of assumption).**

**10. 3D / Visualization Results:** NOT TESTED; decimation experiment: buildings lose up to 20 m at edges under uniform decimation; plane/hills residuals 0.15–1.72 m for factors 2–8.

**11. Performance Results:** NOT MEASURED (no system). Environment: CPU-only, Python 3.14.0, 15.7 GB RAM.

**12. Top 10 P0/P1 Issues:** (1) no implementation; (2) silent PROJ ballpark datum; (3) gradient kernel orientation / non-iterated co-registration; (4) Huber not robust — use median/RANSAC; (5) hold-out RMSE blunder handling; (6) geographic-CRS anisotropy → reproject; (7) no GPU locally; (8) uniform decimation inadmissible; (9) edge blur from canonical downsampling; (10) two PROJ copies — route vertical through pyproj only.

**13. Top 10 Evidence-Based Fixes:** build MVP; bundle grids + reject "ballpark"; correlate + iterate (or grid search); median/Theil–Sen + RANSAC with N ≥ 5; blunder rule + NMAD on hold-out; UTM reprojection; RTIN with residual; recalibrated resampling threshold / area-average downsampling; one-sided edge gradients + BORDER flag; port L0 suite into CI.

**14. Strongest Evidence Supporting the System:** the core calibration mechanism (ground-masked normalized convolution) does exactly what the design claims on controlled synthetic data, and the geospatial/metric mathematics specified in Phases 6–7 are sound and reproducible.

**15. Weakest Evidence / Biggest Uncertainty:** there is no evidence at all about real-world accuracy, terrain robustness, the fine-tuned model, or the visualization — the entire 50 % + 50 % of the PS score is unmeasured.

**16. Final Scientific Validity Assessment:** The *design* has passed the mathematical checks that could be run and has produced two genuine, evidence-backed corrections (silent datum fallback; kernel orientation/iteration). The *system* has no scientific validity to assess yet, because it has not been built. No accuracy statement about DepthWizard is defensible today.

**17. What Must Be Fixed Before Demo:** implement B01–B30; incorporate corrections C-1…C-4 (§27) into `core/geo`, `core/validate`, `core/calib`; acquire one LiDAR-referenced multi-terrain subset and demo DEM tiles; package offline.

**18. What Must Be Revalidated After Fixes:** the full Phase 8 protocol Levels 1–10 — in particular Mode A/B end-to-end, real DEM calibration with the ballpark guard active, per-terrain RMSE/MAE/r/NMAD tables against LiDAR on blocked splits, the fine-tuned vs zero-shot ablation, RTIN residuals, texture alignment, UX protocol with SUS, performance on the demo laptop, and a clean-machine offline run.

READY FOR PHASE 9 — DEMO / PRODUCTIZATION
