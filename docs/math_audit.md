# DepthWizard Mathematical Audit & Improvement Report

Scope: Building Intelligence and Hazard Screening (flood, accessibility, slope/aspect). Measured numbers come from
`scripts/math_audit.py` → `docs/math_audit_results.md` (seeded, reproducible). Golden fixtures and invariants are in
`tests/unit/test_building_math.py` and `tests/unit/test_hazard_math.py`. Every constant is in `core/screening_params.py`
with its justification category (PHYSICAL / STATISTICAL / POLICY / ALGORITHMIC).

Notation: T = terrain layer (`terrain.tif`), S = DSM, h = nDSM, W = water level, J = 2×2 linear part of the
pixel→CRS affine, A_px = |det J|, P = a building's pixel set, V ⊆ P its valid pixels.

---

## A. Existing mathematics (before the audit)

| Quantity | Implemented formula | Location |
|---|---|---|
| nDSM | h = max(S − T, 0) per pixel | `pipeline_b.py` |
| Building height (3D extrusion) | P85 of h over P | `lod1.py` |
| Building height (table) | median of h over P | `lod1.py` / `buildings.py` |
| Ground elevation | median of T over P | `lod1.py` |
| Roof elevation | median(T) + median(h) | `buildings.py` |
| Footprint area | shoelace area of the **outer ring** of the largest polygon × gsd² | `lod1.py` |
| Volume | area × median(h) | `buildings.py` |
| Flood indicator / depth | I = [T ≤ W], D = max(W − T, 0) on T | `flood.py` |
| Flooded area | N_wet × gsd_x² (x pixel size only) | `flood.py` |
| Building flood exposure | affected ⇔ W > median(T over P); depth = W − median(T) | `flood.py` |
| Exposure classes | bins 0.5 / 1.5 / 3.0 m, duplicated in `main.ts` | `flood.py`, `main.ts` |
| Slope | atan‖(∂z/∂col / ‖col-vector‖, ∂z/∂row / ‖row-vector‖)‖, Horn, `nearest` padding | `coregister.py` |
| Aspect | atan2(p, −q) | `coregister.py` |
| Accessibility | slope(**DSM**) ≤ θ, area N × gsd_x² | `accessibility.py` |
| Connectivity | none | — |

## B. Problems found

1. **Aspect was the uphill direction.** atan2(p, −q) with p = ∂z/∂x, q = ∂z/∂row gives the azimuth of the *gradient*;
   the GIS aspect (GDAL/ArcGIS) is the downslope/facing direction. Every value was off by 180° (a west-facing slope
   read 90°), and by 180° + θ on a grid rotated by θ. (§5 of the results.)
2. **Slope/aspect divided by column-vector norms.** Correct only for north-up grids; on a 20° sheared grid a
   11.31° plane read 10.64°–11.94°.
3. **Edge and NoData bias in slope.** `nearest` padding and median-filled NoData halve the Horn gradient on the outer
   ring (5.71° → 2.86°), i.e. the stencil was evaluated where it is undefined.
4. **Accessibility used DSM slope.** Flat roofs counted as accessible ground; walls as cliffs.
5. **Extrusion height P85** is the worst estimator in every test: +3.4 m bias on synthetic roofs (27 m RMSE on
   podium + tower), +4.0 m bias on Zürich LiDAR. Table (median) and 3D block (P85) disagreed by construction.
6. **Volume = A × median** is not the volume of a variable-height roof: −23.6 % bias on stepped buildings; 50.2 %
   RMSE vs Zürich LiDAR.
7. **Area counted courtyard holes** (outer ring only) and ignored disconnected parts, while heights used the pixel set:
   area and statistics referred to different sets.
8. **Roof elevation = median(T) + median(h)** — medians are not additive; wrong on sloped sites when h and T are not
   comonotone (golden fixture C: 120.75 vs true 120.25 m).
9. **Building flood exposure used one scalar ground.** A building was "affected" only if > 50 % of its footprint was
   below W. On Zürich this missed 72–182 partially inundated buildings per scenario (§3b).
10. **Pixel area = gsd_x²** — wrong for non-square, rotated or sheared grids. Correct: |det J|.
11. **No connectivity information.** 77 % of the wet area at the lowest Zürich scenario lies in depressions with no
    surface path to the scene boundary; it was reported identically to connected water.
12. **Legend did not match the raster.** The preview ramp is linear 0–5 m (stops 0/1.67/3.33/5) but the legend
    labelled the stops 0/0.5/1.5/3.0 m and conflated them with the exposure classes; the UI re-implemented the
    class thresholds.
13. **Segmentation edge defects** (found by the golden tests): morphology treated outside-raster as background
    (edge buildings lost a pixel column, −10 % area), and watershed silently dropped mask components without a
    seed marker (Zürich: 11 buildings ≥ 15 m², 540 m²).
14. **Water level had no stated vertical datum.**

## C. Improved mathematics

Building (`core/terrain/building_stats.py`, on the exact label raster `building_labels.tif`):

- A = |P| · A_px, A_px = |det J|
- V = A_px · Σ_{i∈V} h_i · |P|/|V|  (∫_B h dA with mean imputation of invalid pixels)
- H_block = V / A = mean_{V} h  → the LoD-1 extrusion; the block has exactly volume V
- H_roof = median_{V} h  → the table "height", plus P10, P90, NMAD (distribution, not a single number)
- Z_ground = median_{V} T; Z_roof = median_{V} (T + h) (pixelwise first)
- ground tilt: least squares T = a(x−x̄) + b(y−ȳ) + c on **centred** CRS coordinates, slope = atan√(a²+b²);
  null if cond(design) > 1e8
- Flags: SMALL_SAMPLE (|V| < 20), LOW_VALID_FRACTION (|V|/|P| < 0.8), TRUNCATED_BY_RASTER_EDGE, GROUND_TILT_UNDEFINED

Flood (`core/disaster/flood.py`):

- I = [T ≤ W], D = max(W − T, 0), A_wet = N_wet · |det J| (kept; see §E)
- connected = wet cells 8-connected within the wet set to an open boundary (raster edge or NoData neighbour);
  `isolatedAreaM2` always reported; `connectedOnly` restricts the map
- per building: f_wet, D_mean(wet), D_max, and D_exp = max(W − Q10(T_P), 0); class = bin(D_exp)

Slope/aspect (`core/dsm/derive.py`):

- ∇_{xy}z = J^{−T} (∂z/∂col, ∂z/∂row)ᵀ, J^{−T} = (1/det J)[[e, −d], [−b, a]] — exact for any affine
- slope = atan‖∇z‖; aspect = atan2(−∂z/∂x, −∂z/∂y) mod 360 (downslope, clockwise from grid north); NaN if ‖∇z‖ < 1e−6
- undefined (NaN) where the 3×3 stencil is incomplete (outer ring, next to NoData)

Accessibility: slope of **T**, buildings excluded as obstacles and reported separately; areas via |det J|.

## D. Why each change was made

| Change | Reason (mathematical) | Evidence |
|---|---|---|
| Aspect sign | definition: aspect is the azimuth of −∇z | analytic planes: old 90° for a W-facing plane, new 270.0° (exact) |
| J^{−T} gradient | chain rule through the affine; norms are not the Jacobian | sheared grid: old 10.64–11.94°, new 11.3099° (exact) |
| Stencil validity | a 3×3 derivative is undefined without 9 samples; padding is extrapolation | edge cells 2.86° → NaN (not 5.71° guessed) |
| Terrain slope for accessibility | "terrain accessibility" is a property of ground, not roofs | semantic; roofs no longer count as ground |
| H_block = V/A for extrusion | the only height for which the block volume equals ∫h dA; lowest real-data error | Zürich RMSE 4.37–4.63 m vs P85 5.99–7.25 m under all three truth definitions |
| Keep median for table height | 50 % breakdown; returns the dominant roof level | synthetic RMSE 0.38 m (P85 4.12, mean 2.49) |
| Volume = ∫h dA | exact for variable roofs | stepped buildings −23.6 % → −4.6 % bias; Zürich RMSE 50.2 % → 42.2 % |
| Area from pixel set | area and statistics must refer to the same set; holes are not building | definition |
| Roof = median(T+h) | medians are not additive | fixture C: 120.25 m exact vs 120.75 m |
| Exposure from footprint pixels, P10 ground | depth where water first reaches the building; P10 has 10 % breakdown | Zürich: +72 / +182 / +162 exposed buildings found that the old rule missed |
| Connectivity diagnostic | a cell below W is not necessarily reachable | pit fixture: 9 m² isolated exactly; Zürich: 38 486 of 50 045 m² isolated |
| Morphology edge padding / orphan labels | outside-raster is unknown, watershed must partition the mask | edge block 144 → 160 m² (exact); Zürich 11 buildings recovered |

Not changed, with proof:
- **Binary cell counting for wet area**: vs exact polygon clipping on an oblique plane, error ≤ 0.01 % at 0.5–2 m
  GSD and 0.07 % at 10 m; area-weighted partial cells would change nothing measurable at DepthWizard resolutions.
  (Error reaches −11 % only for a small wet area on 30 m cells.)
- **Pixelwise terrain-relative nDSM instead of a ring ground estimator (§2.1)**: of the ring methods only a RANSAC
  plane is unbiased on slopes (0.00 m at slopes 0–0.3); the scalar ring median reaches −1.70 m bias at slope 0.3 and
  LSQ planes −3.3 m under neighbour-roof contamination. The pipeline's h = S − T per pixel equals the RANSAC-plane
  result when T is correct and needs no ring at all, and DSM = T + nDSM holds exactly (max error 0.0 m on Zürich).
- **Bathtub thresholding on T (not S)**: depth is terrain-relative; flood never reads the DSM.
- **Depression filling / breaching**: not implemented. The terrain layer is not hydrologically conditioned, and
  filling would change terrain silently; the connectivity diagnostic reports the same information explicitly.
- **Polygon fallback for old jobs**: exact labels vs rasterised simplified polygons differ by ≤ 2 exposed buildings
  and 6–16 class changes (≤ 1.2 %) on Zürich — kept as a flagged fallback.

## E. Old vs new results (summary; full tables in `math_audit_results.md`)

Building height, Zürich vs independent LiDAR (truth = reference median over footprint, n = 1769):

| Method | MAE | RMSE | Bias | NMAD |
|---|---:|---:|---:|---:|
| P85 (old extrusion) | 5.05 | 6.09 | 3.96 | 3.74 |
| median (table, kept) | 3.89 | 4.98 | 1.64 | 4.02 |
| V/A = mean (new extrusion) | 3.17 | 4.37 | 0.21 | 3.62 |

Volume, Zürich: A × median bias 27.0 %, RMSE 50.2 % → ∫h dA bias 17.4 %, RMSE 42.2 %.

Flood exposure, Zürich (1948 buildings):

| W (m) | old affected | new affected | < 50 % wet (missed by old) | isolated / wet area (m²) |
|---:|---:|---:|---:|---:|
| 401.69 | 113 | 185 | 72 | 38 486 / 50 045 |
| 405.74 | 567 | 749 | 182 | 90 373 / 249 561 |
| 408.04 | 1120 | 1283 | 162 | 102 449 / 499 423 |

## F. Synthetic ground truth (exact expected vs actual; all pass)

| Fixture | Expected | Actual |
|---|---|---|
| A flat, h = 12, 30 cells | area 30, V 360, roof 412 | exact |
| B plane (a, b), h = 9 | h = 9, tilt atan√(a²+b²) | exact / ±0.01° |
| C stepped on slope | roof 120.25, V 280, H_block 17.5, P10/P90 10/20 | exact |
| D 12 % trees | median 12 | exact |
| E 30 % NoData | V 800, flag LOW_VALID_FRACTION | exact |
| F edge building | 160 m², TRUNCATED flag | exact |
| G/H 3×3 threshold/depth | 5 wet cells, D matrix, mean 0.45 | exact |
| I partial inundation | f 0.5, D_mean 0.25, D_max 0.45, P10 100.29, D_exp 0.36 → LOW | exact |
| J/K planes × 4 grids | slope atan‖(a,b)‖, aspect 270/180/90/0/225 | < 1e−6° |
| L/M rotation, non-square, shear | A_px = |det J| | exact |
| N pixel↔CRS round trip | identity | < 1e−6 px |
| O datum +47.3 m | heights, areas, volumes unchanged; ground/roof +47.3 | exact |
| Invariants | T+c, W+c → identical I, D; estimator scale/translation equivariance; 4-connected ⊆ 8-connected (randomised); NoData outside footprint irrelevant | pass |

## G. Real-data results

Zürich tile 2682-1247 (SWISSIMAGE 0.5 m), reference swissSURFACE3D − swissALTI3D (0.5 m, LN02; the datum cancels in a
height difference). The simulated anchors only calibrate the terrain datum, never object heights, so the reference is
independent. Building heights: **empirically validated (one urban scene)** — model error dominates (NMAD ≈ 3.6–4.0 m),
estimator choice changes RMSE by up to 1.7 m. Flood and accessibility have no independent ground truth: **mathematically
and software verified, not empirically validated.**

Caveat: footprints come from the prediction, so the "true" height depends on the truth definition; every estimator was
scored against three definitions and the ranking did not change.

## H. Error propagation

- Roof: Z_roof = T + h per pixel. T and h are **not** independent: h = S − T, so Cov(T, h) = Cov(T, S) − σ_T²; if S is
  exact, σ_roof² = σ_T² + σ_h² + 2Cov = 0 — the roof error is the DSM error, not σ_T² + σ_h². DepthWizard therefore
  reports no separate roof-elevation uncertainty (DSM RMSE is reported per layer where validated).
- Height interval: H ± σ_h with σ_h the model card's per-object RMSE (held-out LiDAR). Not per-building calibrated.
- Volume: V = A · H_block. Fully correlated error → σ_V = A·σ_h (reported as `volume_range_m3`, conservative);
  independent pixel errors → A_px·σ_h·√N (much smaller). Real behaviour lies between; the correlation length is not
  calibrated, so only the conservative bound is shown.
- Area: from counted cells, no sampling error; the footprint uncertainty (segmentation) is not modelled.
- Flood: D inherits the terrain-layer error directly (∂D/∂T = −1); terrain RMSE is reported by the pipeline.

## I. Numerical stability

| Case | Behaviour |
|---|---|
| NaN/Inf water level, slope outside [0, 90] | ValueError → HTTP 422 |
| all footprint pixels NoData | building dropped (no statistics invented) |
| < 3 or collinear pixels | ground tilt null + GROUND_TILT_UNDEFINED |
| UTM-magnitude coordinates | centred plane fit (uncentred cond ~1e12) |
| singular transform | ValueError in `world_gradients` |
| flat terrain | aspect NaN (not 0 or 90) |
| incomplete 3×3 stencil | slope/aspect NaN |
| < 20 samples | SMALL_SAMPLE (P10/P90 are extreme order statistics) |
| no wet cells | depths 0, area 0 |
| float32 raster storage | translation invariance holds to 2e−4 m at 1 km elevation offsets |

## J. Computational cost (2000 × 2000, 1948 buildings)

| Step | Old | New |
|---|---:|---:|
| per-building statistics | 0.45 s | 1.29 s (+0.85 s ≈ 1.5 % of a 55 s job) |
| flood connectivity labelling | — | 0.10 s (24 MB peak) |
| per-building exposure | < 1 ms (scalar) | 0.31 s |
| full flood screening incl. I/O | — | 1.0–1.7 s |

## K. Remaining limitations

- Footprints come from a rule-based RGB + nDSM segmentation (not a trained model); footprint errors dominate area and
  volume error and are not modelled in any uncertainty.
- Model height error (NMAD ≈ 3.6–4.0 m on Zürich) dominates building height; estimator changes cannot remove it.
- The terrain layer is a DEM-driven ground estimate, not a certified DTM; its slope under dense urban cover can be
  locally steep (e.g. 22° under one Zürich building) and flood depth inherits its error one-to-one.
- Exposure bins are policy, not a damage/hazard curve (no velocity, duration or vulnerability data).
- Accessibility uses slope only; no roughness, land cover or road network.
- Scene coordinates in `buildings.json` assume a north-up grid (display only; all statistics use the true affine).
- Only one scene has an independent building-height reference; no flood reference exists.
- Jobs processed before this audit use flagged fallbacks (`LEGACY_STATS_REPROCESS_JOB`, polygon footprints) until
  reprocessed.

## L. Final mathematical specification

```
A_px          = |det J|                                            (m², DERIVED)
h_i           = S_i − T_i ≥ 0                                      (m, PREDICTED)       DSM = T + nDSM exactly
A             = |P| · A_px                                         (m², DERIVED)
V             = A_px · Σ_{V} h_i · |P|/|V|                         (m³, DERIVED)
H_block       = V / A                                              (m, DERIVED, extrusion)
H_roof        = median_V h                                         (m, PREDICTED, table)
P10, P90      = percentiles_V h                                    (m, PREDICTED)
Z_ground      = median_V T                                         (m, REFERENCE-derived)
Z_roof        = median_V (T + h)                                   (m, DERIVED)
tilt          = atan √(a²+b²),  (a, b) = LSQ on centred CRS coords (deg, DERIVED)
floors        = [max(1, ⌊H_roof/3.5⌋), max(1, ⌈H_roof/3.0⌉)] if H_roof ≥ 2.5 else [0, 0]

∇z            = J^{−T} (∂z/∂col, ∂z/∂row)ᵀ   (Horn, valid 3×3 stencil only)
slope         = atan ‖∇z‖                                          (deg)
aspect        = atan2(−∂z/∂x, −∂z/∂y) mod 360, NaN if ‖∇z‖ < 1e−6 (deg, downslope, clockwise from grid north)

I             = [T ≤ W],  D = max(W − T, 0)                        (SCENARIO; W in T's vertical datum)
A_wet         = Σ I · A_px
connected     = 8-connected to raster edge or NoData within {I = 1}; isolated = I ∧ ¬connected
f_wet(B)      = mean_{B} I;  D_mean = mean_{B, I} D;  D_max = max_B D
D_exp(B)      = max(W − Q10(T_B), 0)
class(d)      = NONE d ≤ 0 | LOW d ≤ 0.5 | MODERATE d ≤ 1.5 | HIGH d ≤ 3.0 | VERY HIGH   (policy, returned by the API)

accessible    = slope(T) ≤ θ ∧ ¬building ∧ valid
```

## Acceptance answers

1. **Building math correct?** No, not before. 2. **What was wrong:** B5–B8, B13. 3. **Replaced by:** V/A extrusion,
∫h dA volume, pixel-set area, pixelwise roof, exact labels. 4. **Why better:** §D. 5. **Demonstrated:** §E/F.
6. **Flood correct?** The threshold/depth was correct; building exposure, area (|det J|) and connectivity were not.
Now fixed or diagnosed. 7. **Dimensionally correct:** yes, m / m² / m³ throughout; A_px = |det J|; km² only at display.
8. **Exposure meaningful:** continuous f_wet, D_mean, D_max, D_exp first; class from one policy function.
9. **Slope/aspect:** now exact on analytic planes for any affine. Before: aspect +180°, shear error.
10. **CRS/datum:** geographic input reprojected to UTM at ingest (existing test C-4); heights never mixed across
datums; W's datum now stated. 11. **NoData/edges:** §I. 12. **Frontend = backend:** the UI renders the backend's
numbers, classes, rules and ramp; no scientific quantity is recomputed. 13. **Exports = internal:** tested
(`test_exports_equal_api_records`). 14. **Remaining weak assumptions:** §K.
