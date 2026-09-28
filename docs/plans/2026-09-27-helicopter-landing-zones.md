# Helicopter Landing-Zone Screening: Implementation Plan

Date: 2026-09-27. Status: plan, nothing built yet.
Scope: a new Disaster Management scenario that finds **candidate** helicopter landing zones (HLZs) in a processed job.
It follows the math-audit discipline: every constant is justified and centralised in `core/screening_params.py`,
analytic golden tests come first, and the method is measured against LiDAR before any claim is made.

---

## 0. What the feature must and must not say

It **is**: a screening layer. It lists places where, *according to this surface model*, a pad of the chosen size is
flat enough, free of detected obstacles, and has at least one approach corridor that is clear of detected
obstacles.

It **is not**: a landing clearance. The UI, API `warnings` and GeoJSON properties always carry:

- "Candidate landing zones: screening only. Ground or air reconnaissance is required before use."
- "Not detectable: wires, cables, poles, antennas, loose debris, and obstacles lower than the object threshold
  (~{object_threshold_m} m for this job)." Wires are the leading HLZ hazard, and a raster DSM cannot see them.
- "Surface condition (soft soil, water, snow, dust, rotor-wash debris) is not assessed."
- "Imagery date: {acquisition date if known}. Conditions may have changed." The Sikkim demo imagery is *pre-event*
  (March 2022).

---

## 1. Corrections to the one-line pitch (read first)

1. **Slope units.** Field HLZ rules are stated in **percent grade**, not degrees. The provisional defaults (to be
   verified, §2) are:
   - ≤ 7 % (≈ 4.0°): land in any direction;
   - 7–15 % (≈ 4.0–8.5°): upslope landing only (MARGINAL);
   - > 15 %: no touchdown.

   The pitch's "about 7°" was wrong: 7° ≈ 12.3 %.
2. **Obstacles are not only nDSM objects.** In a Sikkim valley the main approach obstacle is **terrain**: the valley
   wall. The approach check therefore uses the **DSM relative to the pad elevation**, not nDSM alone. nDSM is used
   only to find objects *on* the pad.
3. **The pad size depends on the aircraft.** A single 25 m square is not correct for all helicopters. The size is
   chosen by aircraft class (§2).

---

## 2. Constants: verify against sources BEFORE coding (task T0)

Every value below goes into `core/screening_params.py` under a new `# helicopter landing zones` section. Each entry
has a category (PHYSICAL / STATISTICAL / POLICY / ALGORITHMIC, as in the existing file) and a **source citation**.
Values marked *verify* are my recollection of US Army Pathfinder doctrine (ATP 3-21.38 / FM 3-21.38) and **must be
checked against the current manual text and, where available, IAF / NDMA helicopter SOPs** before merging. If a
source disagrees, the source wins and this plan is updated.

| Constant | Default | Category | Source / justification |
|---|---|---|---|
| `HLZ_PAD_DIAMETER_M` by class | light 25, utility 35, medium 50, heavy 80 | POLICY | Pathfinder landing-point sizes 1–4 (*verify*). Suggested Indian examples (*verify*): Chetak/Cheetah = light, ALH Dhruv = utility, Mi-17 = medium, CH-47F = heavy |
| `HLZ_SLOPE_ANY_DIR_PCT` | 7.0 | POLICY | Pathfinder: land in any direction below this (*verify*) |
| `HLZ_SLOPE_MAX_PCT` | 15.0 | POLICY | Pathfinder: no touchdown above this (*verify*) |
| `HLZ_OBSTACLE_RATIO` | 10.0 (10 m horizontal per 1 m height) | POLICY | Pathfinder obstacle-clearance ratio (*verify*). ICAO Annex 14 Vol II heliport approach surfaces are stricter (4.5–12.5 %) and apply to certified heliports, not field HLZs; state this in the docs |
| `HLZ_APPROACH_LENGTH_M` | 300 | POLICY | Checked corridor length. Beyond it and beyond the scene edge the result is UNVERIFIED |
| `HLZ_CORRIDOR_DIVERGENCE` | 0.10 per side | POLICY | Corridor widens from the pad diameter (*verify* against the chosen doctrine; if the doctrine gives none, keep it and label it a DepthWizard choice) |
| `HLZ_BEARINGS` | 16 (22.5° step) | ALGORITHMIC | Angular resolution of the approach search; a 16-bearing sweep with a corridor width ≥ pad diameter leaves no angular gap within 300 m for pads ≥ 25 m (prove in docs; see T4) |
| `HLZ_OBJECT_MARGIN_M` | = `result.uncertainty.ndsm.object_m` (5.5 m on current jobs) | STATISTICAL | Obstacle heights come from the model with the measured object RMSE. They are added as a 1σ vertical buffer to object pixels in the approach test. The value is read from the job, not hard-coded |
| `HLZ_OBJECT_THRESHOLD_M` | = `result.uncertainty.ndsm.object_threshold_m` (2.5 m) | STATISTICAL | Below this, nDSM cannot distinguish an object from ground noise (ground RMSE 2.47 m). On-pad obstacle mask = nDSM > threshold. The limitation is disclosed, not hidden |
| `HLZ_ROUGHNESS_MAX_M` | set from data (T7) | STATISTICAL | Pad plane-fit residual RMS limit = P95 of the same statistic over LiDAR-flat areas in the Zürich scenes (dataset-derived calibration) |
| `HLZ_MIN_PAD_PIXELS_ACROSS` | 10 | ALGORITHMIC | Below 10 px across the pad, the disk, slope and obstacle tests are too coarse; the request is refused with 422 |
| `HLZ_MAX_SITES` | 50 | ALGORITHMIC | Caps the approach-check cost and the list length |

---

## 3. Inputs and refusals

Reads from the job directory: `dsm.tif`, `ndsm.tif`, `terrain.tif`, `building_labels.tif` (or the
`buildings.json` polygon fallback via the existing `flood._footprint_sets`), optionally `flood_depth.tif`, and
`result.json`.

Refuses with **HTTP 422** (`INVALID_PARAMETER` or new code `HLZ_NOT_SUPPORTED`) in these cases:

- Tier **R**: no metric heights. "Landing-zone screening needs metric heights; this job is relative only."
- Tier **H** / no `terrain.tif` / no `dsm.tif`: there is no terrain to measure slope on.
- GSD too coarse for the chosen pad (`pad_diameter / gsd < HLZ_MIN_PAD_PIXELS_ACROSS`).
- Unknown aircraft class, or a slope or length parameter outside its physical range.

Always records `result.dem.posting_m`. When the DEM posting exceeds the pad diameter (Copernicus 30 m vs a 25 m pad),
every site carries the flag `TERRAIN_COARSER_THAN_PAD`: the large-scale slope comes from a DEM that cannot resolve
features the size of the pad. This is exactly the case for the Sikkim demo, so the flag will show there.

---

## 4. Mathematics (all distances in CRS metres via the pixel→CRS affine J; exact under rotation and shear)

Notation: pixel offsets (Δc, Δr) map to world offsets J·(Δc, Δr). The pad kernel K is the set of offsets with
|J·(Δc, Δr)| ≤ R, where R = D/2. K is centrally symmetric, so Σ_K Δx = Σ_K Δy = 0.

### 4.1 Pad clear of detected obstacles
Obstacle mask:

    O = (nDSM > h_obj) ∪ building footprint ∪ ¬valid ∪ (wet, if excludeFlooded)

Pixels outside the raster count as O (pad with 1s), so a pad can never hang off the edge.
Clear pad test:

    clear(p) = [ (O ⊛ K)(p) = 0 ]

The convolution is done with FFT and then rounded to an integer. Its error is ≪ 0.5, so the rounding is exact.
Assert this in a test.

### 4.2 Pad slope and roughness: least-squares plane over the disk, computed for every centre with convolutions
For each centre p fit z = a·Δx + b·Δy + c over K, with z = DSM (see 4.4 for why DSM).

Because K is symmetric, the normal equations decouple:

    c = S_z / n
    [a b]ᵀ = M⁻¹ [S_xz S_yz]ᵀ,   M = [[S_xx, S_xy], [S_xy, S_yy]]

M is a constant of the kernel. S_z = z ⊛ K, S_xz = z ⊛ (K·Δx), S_yz = z ⊛ (K·Δy).

- Slope: `slope_pct = 100·√(a² + b²)`, `slope_deg = atan(√(a² + b²))`.
- Roughness: RMS residual `√(RSS / n)`, with `RSS = S_zz − c·S_z − a·S_xz − b·S_yz`.

**Precision trap (must handle).** Elevations are ~400–4000 m. The FFT has a relative error of about 1e-13 of the
largest value, and RSS is a difference of numbers of size ~n·z². Therefore:

- subtract the scene median elevation from z before any convolution;
- clamp RSS at 0.

A test compares against `np.linalg.lstsq` on 200 random windows (tolerance 1e-6 m) and runs again at +3000 m offset.

Only centres where `clear(p)` holds are used, so every window is fully valid and the fit is well-posed.

### 4.3 Approach and departure corridors
For each site and each bearing θ:

- the corridor starts at the pad edge (s = R) and runs to s = R + L;
- its half-width is w(s) = R + divergence·(s − R).

A cell q in the corridor, at along-track distance s, **violates** if

    DSM(q) + m(q) − Z_pad > (s − R) / ratio

where Z_pad = c from 4.2, m(q) = `HLZ_OBJECT_MARGIN_M` if nDSM(q) > h_obj, else 0.

Status of a bearing:
- **BLOCKED**: any violation. Report the first violation's s and its height above the pad.
- **UNVERIFIED**: no violation, but the corridor leaves the raster before s = R + L.
- **CLEAR**: otherwise.

Implementation: max-pool DSM + m to 1 m cells (conservative: a thin obstacle can never be missed). Sample each
corridor on a 1 m along × lateral lattice using the pixel footprint → pooled-cell mapping; each cell counts at its
maximum height. Vectorise over bearings.

Terrain cells get no margin, because relative DEM error over ≤ 300 m is not measured in this project. This is stated
as an assumption in the docs.

### 4.4 Why DSM (and not terrain.tif) for pad slope
Inside a clear pad every pixel has nDSM ≤ h_obj, so DSM ≈ terrain + ground noise. terrain.tif below the DEM posting
is an interpolated 30 m DEM and cannot show terraces or embankments; the DSM carries the model's sub-posting detail.
Whether that detail is *accurate* is measured in T7 (DSM-pad slope vs LiDAR-DTM-pad slope). If T7 shows the DSM slope
is worse than terrain.tif, switch to terrain.tif. That is a one-line change, and the decision is recorded in the docs.

### 4.5 Site selection
Feasible mask F = clear ∧ slope ≤ max ∧ roughness ≤ max.

Rank feasible centres by:
1. slope class (≤ 7 % before 7–15 %);
2. slope;
3. roughness.

Greedy non-maximum suppression with spacing D, so pads don't overlap, keeps ≤ `HLZ_MAX_SITES`. Run 4.3 on those only.

Site class:
- **SUITABLE**: slope ≤ 7 % and ≥ 1 CLEAR bearing.
- **MARGINAL**: slope 7–15 % (upslope only), or its only non-blocked bearings are UNVERIFIED.
- Dropped: every bearing BLOCKED. Counted in the summary as `blockedSites`.

---

## 5. Outputs and API contract (additive only; no existing contract changes)

`POST /api/jobs/{job_id}/disaster/landing_zones`
body: `{ "aircraftClass": "light|utility|medium|heavy", "maxSlopePct"?: number, "excludeFlooded"?: bool }`

Response (the rule values come from the server, and the UI never duplicates them, same as `exposureRules`):

```
scenario, method ("hlz-screening-1"), aircraftClass, padDiameterM, rules {slopeAnyDirPct, slopeMaxPct,
obstacleRatio, approachLengthM, divergence, objectThresholdM, objectMarginM, roughnessMaxM},
feasibleAreaM2, nSites, blockedSites,
sites: [{ id, class, x, y, crs, lon, lat, col, row, elevationM, verticalCrs, slopePct, slopeDeg, roughnessM,
          approaches: [{ bearingDeg, status, firstObstacle: {distanceM, heightAbovePadM} | null }],
          flags: [...] }],
rasterResult "landing_feasible.tif", previewResult "landing_preview.png", vectorResult "landing_zones.geojson",
quantityCategory, warnings
```

Per-site flags: `TERRAIN_COARSER_THAN_PAD`, `UPSLOPE_ONLY`, `APPROACH_UNVERIFIED_BEYOND_SCENE`,
`SINGLE_APPROACH_ONLY`, `FLOOD_NOT_CONSIDERED` (when excludeFlooded is false but a flood result exists).

Artifacts written to the job directory:
- `landing_feasible.tif`: uint8 **reason bitmask** per centre: 1 slope > max, 2 obstacle in pad, 4 rough,
  8 wet, 16 invalid/edge, 0 = feasible. This answers "why not here?".
- `landing_preview.png`: RGBA overlay.
- `landing_zones.geojson`: pad polygons, centre points, and CLEAR corridor lines, in the job CRS. A WGS84 copy
  is written if the CRS transform is available, via pyproj, which the project already uses.
- `disaster_landing_zones.json`: the response, like the other scenarios.

---

## 6. Files

| File | Change |
|---|---|
| `core/screening_params.py` | new HLZ section (T0) |
| `core/disaster/landing_zones.py` | **new**: `pad_kernel`, `clear_mask`, `plane_fit_fields`, `approach_status`, `select_sites`, `run_landing_zone_screening` |
| `backend/api/routes.py` | new endpoint, same try/except → 422 pattern as flood/accessibility |
| `frontend/index.html` | `<option value="landing">` in `#disaster-scenario`; aircraft-class select + slope input (hidden unless landing is selected) |
| `frontend/src/main.ts` | body branch in the run handler; result branch (stats, site list reusing the `.disaster-bldg-item` pattern, legend from `resp.rules`); `renderLandingSvg`: pad circles + CLEAR corridor ticks in **pixel coords from the API** (no centred-metre conversion) |
| `frontend/src/style.css` | site-class colours using existing tokens (`--ok` suitable, `--warn` marginal) |
| `tests/unit/test_landing_zones.py` | **new**: goldens + invariants (T2–T5) |
| `scripts/math_audit.py` | add `hlz_study()` (T7) |
| `docs/landing_zones.md` | **new**: method, assumptions, limits, measured results |

---

## 7. Tasks, in order, each with its exit check

**T0: Verify constants.** Read the current Pathfinder manual (ATP 3-21.38) and any public IAF / NDMA HLZ guidance.
Fill in the citations in `screening_params.py`.
*Exit:* every HLZ constant has a category and a source, or is explicitly labelled a DepthWizard POLICY choice.

**T1: Kernel and clear-area test (4.1).**
*Exit:* tests for disk membership under north-up, rotated 30°, sheared and non-square transforms (the disk is
round in *world* metres); FFT counts equal direct counts exactly.

**T2: Plane-fit fields (4.2).** Goldens:
- G2 tilted plane of grade g on all four grid types gives slope = g to 1e-9;
- G7 plane + checkerboard ±r gives roughness = r;
- lstsq agreement on 200 random windows;
- vertical translation +3000 m gives identical results (precision trap);
- horizontal origin at 2.6e6 m gives identical results.

*Exit:* all pass.

**T3: Obstacles on the pad.** Goldens:
- G5: a building pixel at world distance R − ε from the centre makes it infeasible; at R + ε it stays feasible;
- G6: one NaN inside the disk → infeasible;
- a pad touching the raster edge → infeasible;
- G9: wet cells are excluded only when excludeFlooded is true.

**T4: Approach corridors (4.3).** Goldens:
- G4: a post of height h at along-track distance d past the pad edge is BLOCKED iff h + margin > d/ratio. Test at
  ±1 % of the boundary. Only bearings whose corridor covers the post are blocked;
- G8: a terrain wall rising at 20 % to the north blocks N, NNE and NNW, and leaves S CLEAR;
- a corridor leaving the scene is UNVERIFIED, never CLEAR;
- the no-gap property: a post anywhere in the annulus R..R+L at a height that violates is caught by ≥ 1 bearing.

*Exit:* all pass; the no-gap argument is written in the docs.

**T5: Site selection and invariants.**
- G1 flat scene: sites are non-overlapping (pairwise distance ≥ D) and all interior;
- a 90° rotation of raster + transform gives the same set of sites, with bearings rotated by 90°;
- the class boundaries at exactly 7 % and 15 % are handled per the rule text (≤ vs <, pinned by a test).

**T6: API, UI and artifacts.**
- API test: tier-R job → 422; a normal job → schema keys present, artifacts exist, GeoJSON validates, reason
  bitmask is 0 exactly at feasible centres;
- the existing 143 tests still pass;
- `tsc --noEmit` is clean;
- browser check on the Chungthang demo: select the scenario, run it, check that pads, corridors, list, legend and
  warnings render; 375 px width has no horizontal scroll; zero console errors.

**T7: Measure on real data (the evidence the pitch will use).** Zürich scenes, where swissALTI3D (DTM) and
swissSURFACE3D (DSM) are pixel-aligned with the jobs:
1. **Roughness calibration:** P95 of the pad residual RMS on LiDAR-flat areas → `HLZ_ROUGHNESS_MAX_M`.
2. **Slope error:** DepthWizard pad slope vs LiDAR-DTM pad slope over all DepthWizard-feasible pads: bias, NMAD,
   P95. Decide between DSM and terrain.tif (4.4).
3. **False-clear rate (the safety metric):** the share of DepthWizard SUITABLE/MARGINAL pads where the LiDAR
   nDSM has an object > h_obj inside the disk, or a LiDAR-DSM violation in a bearing reported CLEAR. Report it
   plainly, whatever it is. If it is high, the UI wording and defaults get stricter. The threshold is never tuned
   to hide it.
4. Runtime on 2000² (Zürich) and 2400² (Sikkim) rasters; target < 5 s end-to-end.

The results go into `docs/landing_zones.md` and `math_audit_results.md`.

**T8: Chungthang demo run.** Run on the Chungthang and Teesta scenes. Capture: sites, the blocked-by-valley-wall
bearings, and the `TERRAIN_COARSER_THAN_PAD` flag. Check at least the top 5 sites visually against the imagery (open
ground, not a riverbed or a roof), and document any false positives as they are.

---

## 8. Risks and how the plan handles them

| Risk | Handling |
|---|---|
| Declaring a pad clear that has a wire or a small obstacle | Undetectable class stated in every output; T7 false-clear rate measured and shown |
| Model under-estimates tree height in forest | 1σ object margin in the approach test; the forest scenes (chungthang_west) are in T8 |
| 30 m DEM hides terraces | `TERRAIN_COARSER_THAN_PAD` flag; DSM slope used and measured (T7.2) |
| FFT precision at high elevations | median subtraction + translation-invariance test |
| Scene too small for a 300 m corridor | UNVERIFIED status, never CLEAR |
| Doctrine numbers misremembered | T0 blocks coding until the constants are cited |
| Pre-event imagery used as if current | Acquisition date shown in the warnings |

Estimated effort: T0 0.5 day · T1–T5 1.5–2 days · T6 1 day · T7–T8 1 day. About 4–5 working days in total.
