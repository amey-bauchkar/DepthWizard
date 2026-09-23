# Phase 5 — Solution Ideation, Concept Development & Technical Solution Selection

SIH26175 DepthWizard · 2026-09-20

Scope note: Phase 5 selects and develops a solution *concept*. It does not fix the architecture, APIs, data schema, UI structure, framework or engine (Phase 6), and contains no implementation code. Evidence tags `[E#]` refer to the Phase 3 ledger, `[F#]` to the Phase 4 ledger, `[G#]` to items verified in this phase (below). Every major design decision is traced to a requirement, a gap, a prior-art finding, a data/validation constraint, or an explicitly stated trade-off.

**Inputs read:** Phase 1 (problem decomposition, 354 lines), Phase 2 (research, 316 lines), Phase 3 (prior art & gaps, 760 lines; 17 gaps GAP-01…GAP-17; 15 constraints), Phase 4 (data & validation, 895 lines; 15 constraints; 7 items Phase 5 must resolve).

**Verified this phase**

| ID | Finding | Source | Confidence |
| --- | --- | --- | --- |
| G1 | `IMG-PROCESS-SAC/SIH2026` unchanged (last push 2026-08-29, one README, no licence). | GitHub API | HIGH |
| G2 | Depth Anything V2 repository ships fine-tuning code (`metric_depth/train.py`, `dist_train.sh`, dataset loaders) under an Apache-2.0 repository licence; only the **Small** checkpoint is Apache-2.0 (Base/Large CC-BY-NC-4.0) [E26]. | GitHub contents API + LICENSE | HIGH |
| G3 | swissALTI3D is a 0.5 m LiDAR **DTM** (bare earth), OGD (free incl. commercial), LV95 / **LN02 = EPSG:5728**, same campaign as the swissSURFACE3D DSM → nDSM = DSM − DTM derivable nationwide incl. Alps and forest. | swisstopo / geocat / OpenTopography | HIGH |
| G4 | LINZ national 1 m **DEM (bare earth)** exists alongside the 1 m DSM composite, NZVD2016, CC BY 4.0 → nDSM derivable. | LINZ Data Service | HIGH |
| G5 | Fine-tuning a DA-V2-Small on nadir imagery is computationally light: Depth Any Canopy reports < $1.30 compute for its fine-tune [E14]. | Phase 3 E14 | HIGH |

**Inherited assumptions re-checked and carried forward unchanged:** SRTM = EGM96 geoid, Copernicus = EGM2008 [E2][E3]; India geoid separation −24…−99 m [F13]; GAMUS HF = 3 cities, nDSM, non-georeferenced [E6]; zero-shot foundation depth fails metrically at nadir [E15][E16]; fine-tuning closes much of the gap [E13][E14][E16]; DFC19 random-split vs official-test discrepancy [E4][E11]; coarse DEMs are surface-like and biased in forest/built-up [E19][E21]; no LiDAR reference over India; Cartosat imagery priced for NGEs [F10]; ASPRS Ed. 2 rules [F7].

---

## 1. Evidence-to-Design Synthesis

**A. What the PS explicitly requires** (Phase 1 §1, SAC README [E1]): single-view optical RGB in; PNG/JPG → rDSM; GeoTIFF → absolute metric DSM using SRTM-class DEM or limited GCPs; a pretrained monocular depth backbone for extraction; a scale-calibration module; DSM "in a standard geospatial format"; RGB projected onto a 3D terrain mesh; first-person navigation, arbitrary aerial views, structural-height and slope analysis; standalone deployment; RMSE/MAE/correlation against LiDAR/reference across urban, sparse, hilly, forested; 50/50 accuracy–visualization scoring.

**B. What existing research already solves reasonably well**
- Single nadir image → **height above ground (nDSM)** with LiDAR supervision: mature (IM2HEIGHT → HTC-DC Net → GlobalBuildingAtlas at 3 m, RMSE 5.5 m global) [E4][E12]; canopy height at 1 m, MAE 2.8 m [E13].
- Adapting a **foundation depth encoder** to nadir height by fine-tuning: demonstrated (Depth Any Canopy, Sat3R, Tolan et al. DINOv2) [E14][E16][E13].
- Global **scale-and-shift alignment** to sparse anchors and prior-conditioned dense fusion — in camera frame [E9][E27].
- **Heightmap → textured mesh → navigation**, LOD, standalone HTML export: commodity [E25].
- **DEM co-registration and robust accuracy statistics** [F8][E32][F7].

**C. What remains technically difficult** (Phase 3 GAP list): zero-shot metric scale at nadir (GAP-01); relative → absolute with a 30 m DEM (GAP-02); nDSM + terrain → certified absolute DSM (GAP-03); cross-morphology generalization (GAP-04); hilly/forested evidence (GAP-05); terrain/object separation (GAP-06); tall-object underestimation (GAP-07); GCP requirements (GAP-08); validation without LiDAR over India (GAP-09); datum discipline (GAP-11); GSD mismatch (GAP-12); uncertainty (GAP-15); packaging (GAP-16); faithful visualization (GAP-17).

**D. Strongest evidence-backed gaps that a solution can actually act on:** GAP-01 (adapt, don't trust zero-shot — evidence δ₁≈0 %, MAE 4.6 m with scale drift), GAP-02/03 (decide what the DEM is *for* and how nDSM becomes DSM), GAP-11 (declare and transform the vertical datum — a 24–99 m effect in India), GAP-15 (expose uncertainty), GAP-17 (mesh must be measured against the DSM). GAP-04/05/09 are data-limited and can only be *bounded*, not solved.

**E. What data is realistically available** (Phase 4 §§4–5, 29): open nDSM training pairs (GAMUS 0.33 m urban; DFC19 1.3 m; SynRS3D synthetic); open **absolute LiDAR DSM + DTM + orthophotos across all four terrain classes** from Switzerland (0.5 m, LN02), New Zealand (1 m, NZVD2016), USA (3DEP + NAIP, NAVD88), Netherlands (0.5 m, NAP, flat); coarse DEMs globally (Copernicus GLO-30 EGM2008 best-ranked; SRTM EGM96; CartoDEM India); sparse absolute anchors globally incl. India (ICESat-2 ATL08 ≈ 0.7–4 m); **no Indian RGB–LiDAR pairs; Cartosat imagery priced** [F2–F6][F10][G3][G4].

**F. What evaluation evidence can realistically be produced:** per-terrain, spatially blocked RMSE/MAE/ME/NMAD/LE95/r tables against LiDAR DSMs in CH/NZ/US; object-level metrics (RMSE-B, F1-HE); external fixed benchmark (DFC19 Track 1, official split); an India assessment bounded by ICESat-2/CartoDEM error; calibration-ablation curves (oracle vs DEM vs anchors vs none); mesh-vs-DSM residuals, projection residuals, performance percentiles, clean-machine install tests, SUS [F7][F11][F16] (Phase 4 §§19–26, 30–32).

**G. What the solution absolutely must accomplish:** (1) both modes with honest semantics (rDSM relative; DSM absolute *only* when calibration inputs exist); (2) a pretrained backbone that is *adapted* to nadir imagery, not used raw; (3) an explicit, inspectable calibration chain that separates relative depth → metric height above ground → absolute elevation → datum-tagged raster; (4) geospatially valid GeoTIFF output (CRS, transform, nodata, vertical CRS); (5) validation per the Phase 4 blueprint across four terrains; (6) a navigable textured 3D scene whose geometry is the DSM; (7) standalone operation.

**H. What the solution should NOT waste time on:** inventing a new depth architecture (backbones are solved; adaptation is the lever [E14][E16]); trying to make zero-shot metric depth work at nadir (negative evidence [E15]); recovering sub-DEM-resolution terrain from the DEM (Nyquist; GAP-02); photorealistic effects, procedural embellishment or exaggeration that is not measurement-faithful (Phase 1 §9, GAP-17); training on random patch splits for headline numbers (GAP-10); multi-view or stereo (out of scope); hidden-surface reconstruction (physically impossible from one nadir image, GAP-06).

---

## 2. Design Requirements

**HARD REQUIREMENTS (PS compliance)**
- H1 Accept PNG/JPG and GeoTIFF/georeferenced TIFF; detect mode from metadata. (Phase 1 §3–4)
- H2 Mode A → rDSM (relative), used directly for visualization. [E1]
- H3 Mode B → absolute DSM in metres, using a low-resolution DEM (SRTM-class) or limited GCPs to map scale-agnostic depth to absolute elevation. [E1]
- H4 Use a pretrained monocular depth model for elevation extraction. [E1]
- H5 Provide a scale-calibration module (scene statistics / DEM / semantic priors / GCPs). [E1]
- H6 Output DSM in a standard geospatial format (GeoTIFF with CRS + transform + nodata; vertical CRS declared). (Phase 1 §8)
- H7 Project the original RGB onto a 3D terrain mesh; render in an interactive engine; first-person navigation; arbitrary aerial viewpoints. [E1]
- H8 Structural-height and slope analysis in the viewer. [E1]
- H9 Upload → visualize → validate against reference in the interface. [E1]
- H10 Standalone deployment; software stability across input types/sizes. [E1]
- H11 Report RMSE, MAE, correlation vs LiDAR/reference, stratified urban/sparse/hilly/forested. [E1]
- H12 Source code + technical documentation. [E1]

**HIGH-VALUE TECHNICAL REQUIREMENTS (credibility)**
- V1 Domain adaptation of the backbone to nadir imagery with metric-height supervision (GAP-01; [E14][E16]).
- V2 Explicit two-quantity separation: height-above-ground layer vs terrain layer, composed into DSM with the DEM's limitations respected (GAP-02/03; [E19][E21]).
- V3 Vertical-datum pipeline: every input (DEM, anchors, reference) transformed to one declared vertical CRS; output tagged (GAP-11; [F13]).
- V4 Calibration hierarchy with confidence tiers and failure detection; never emit metres without a defensible source (Design principles 1–6, 12).
- V5 Per-pixel uncertainty and calibration-quality reporting (GAP-15).
- V6 GSD-aware processing (GSD from geotransform; canonical-GSD resampling or conditioning) (GAP-12; [E4]).
- V7 Validation harness implementing the Phase 4 blueprint (blocked splits, co-registration, masks, metric set, per-terrain, object metrics).
- V8 Mesh faithful to the DSM with reported simplification tolerance; measurements read from the raster (GAP-17).
- V9 Tiled inference with overlap blending for large rasters, with a single metric scale across tiles (Phase 1 §11D).

**OPTIONAL ENHANCEMENTS**
- O1 Semantic head (ground / building / tree / water / road) trained jointly (GAMUS labels) to sharpen ground masking and class-wise error reporting [E4][E5].
- O2 LoD1 building extrusion from footprints × nDSM for a truer urban 3D (GBA practice) [E12].
- O3 Test-time augmentation ensembles for uncertainty; multi-GSD inference.
- O4 Change-mask handling when reference and imagery dates differ [E36].
- O5 Off-nadir facade detection and warning (GAP-13).

**FEATURES TO AVOID**
- A1 Zero-shot "metric" depth models with camera intrinsics on orthophotos (no geometric basis; δ₁≈0 %) [E15].
- A2 Upsampling the 30 m DEM to image resolution and calling it terrain truth (GAP-02).
- A3 Post-hoc affine fit against the *test* reference presented as calibration (Phase 4 constraint 2).
- A4 Vertical exaggeration or mesh smoothing without disclosure; measurements taken from a simplified mesh (GAP-17).
- A5 Photorealistic sky/water/vegetation effects, procedural buildings, generative in-painting of unseen facades (misrepresents geometry).
- A6 Cloud/tile-server dependencies that break "standalone" (Phase 4 §25).
- A7 Feature sprawl (multi-temporal change detection, flood simulation, etc.) not asked for by the PS.

---

## 3. Solution Concept Generation

Five genuinely different strategies. Each is stated neutrally; selection follows in §5.

### Concept A — "Zero-Shot Backbone + Global Affine Calibration" (monocular-first, PS-literal)
- **Core idea:** run an off-the-shelf relative-depth foundation model on the image; invert depth to relative height; fit a single scale + shift to the coarse DEM (Mode B) or output normalized (Mode A).
- **Input / output:** RGB (+GeoTIFF metadata) → relative height (A) or DEM-fitted "DSM" (B).
- **Key mechanism:** DA-V2-class relative depth; least-squares (s, t) in height space against DEM samples; optional RANSAC.
- **Metric scale:** entirely from the DEM fit. **Absolute elevation:** the shift term; datum = DEM datum. **DSM:** the scaled relative map. **Domain gap:** unaddressed (zero-shot). **Georeferencing:** inherited from input. **Validation:** Phase 4 blueprint.
- **Advantages:** minimal effort; no training; literal reading of the PS text.
- **Limitations:** zero-shot nadir output has scale drift and blurred edges [E16]; fitting a per-image affine to a 30 m surface-like DEM transfers the DEM's canopy/roof bias and cannot fix spatially varying scale [E9][E19]; no terrain/object separation; nothing prevents 2–3× cross-domain error.
- **Dependencies:** backbone weights; DEM tiles. **Data:** none for training. **Compute:** inference only. **Failure modes:** DEM-driven bias; per-tile scale inconsistency; forest double-counting. **Novelty:** none (this is the naive baseline every reviewer expects). **Coverage:** formally covers H1–H7 but H3 is weakly defensible.

### Concept B — "Purpose-Trained nDSM Regressor + DEM Terrain Add-Back" (MHE-classic)
- **Core idea:** train/reuse an HTC-DC-Net-class network to regress metric nDSM from the image (the field's proven target), then add a terrain surface from the coarse DEM to obtain DSM.
- **Mechanism:** classification-regression MHE network trained on GAMUS/DFC19/GBH-type nDSM; DSM = DEM(resampled) + nDSM.
- **Metric scale:** learned (supervised in metres at the training GSD). **Absolute elevation:** DEM. **Domain gap:** handled by training on nadir data only (no natural-image pretraining beyond ImageNet). **Validation:** blueprint.
- **Advantages:** strongest published in-distribution accuracy [E4]; GBA shows it scales to global production [E12].
- **Limitations:** does not use a *monocular depth* foundation backbone as the PS phrases it (interpretive risk); HTC-DC Net has no licence [E4]; GSD-locked; naive DEM add-back double-counts canopy/roofs contained in the DEM [E19]; no uncertainty; Mode A needs a separate relative path.
- **Dependencies:** nDSM training data at target GSD. **Compute:** training a CNN from ImageNet init — moderate. **Failure modes:** cross-morphology drop (São Paulo 9.3 m) [E4]; long tail. **Novelty:** none in the regressor; DEM add-back is the undocumented step (GAP-03). **Coverage:** H3 defensible; H4 debatable.

### Concept C — "Two-Layer Calibrated Surface Model" (foundation backbone adapted to metric height-above-ground + terrain layer from de-biased coarse DEM + anchor refinement + uncertainty)
- **Core idea:** treat the DSM as the sum of two physically different layers that have different best sources: **(i) an object layer** — height above local ground — which a single nadir image *can* predict once the backbone is adapted with LiDAR nDSM supervision; **(ii) a terrain layer** — low-frequency absolute elevation — which only an external absolute source (DEM, anchors) can supply, and only at low spatial frequency. Calibrate each layer with the source that is competent for it; expose the tier reached.
- **Input / output:** Mode A: RGB → rDSM (normalized object+shape map) + relative measurements. Mode B: GeoTIFF (+DEM, ±anchors) → metric nDSM raster **and** absolute DSM raster (both GeoTIFF, vertical CRS declared) + uncertainty raster + calibration report.
- **Key mechanism:** DA-V2-Small (Apache-2.0) encoder fine-tuned with a metric nDSM head on multi-GSD, multi-terrain LiDAR-derived nDSM (GAMUS + CH/NZ/US DSM−DTM) with GSD normalization; terrain layer = DEM transformed to the declared vertical datum, sampled preferentially where the model predicts ground (nDSM ≈ 0) and low-pass reconstructed, which limits inheritance of the DEM's canopy/roof bias; optional robust affine refinement of object scale / terrain offset from sparse anchors (GCPs or ICESat-2), anchors partitioned from checkpoints; DSM = terrain + nDSM; uncertainty from TTA/ensemble variance + class/terrain priors + calibration residual.
- **Metric scale:** learned from nDSM supervision, conditioned on true GSD (Level H); optionally corrected by anchors (Level A). **Absolute elevation:** terrain layer (Level T); datum transformed and declared. **DSM:** explicit composition. **Domain gap:** fine-tuning [E14][E16] + multi-terrain data [G3][G4][F5]. **Georeferencing:** transform/CRS carried through; outputs on the input grid. **Validation:** blueprint, plus per-layer validation (nDSM vs LiDAR nDSM; terrain vs LiDAR DTM low-pass; DSM vs LiDAR DSM).
- **Advantages:** each claim is backed by the source competent to support it; degrades honestly (no DEM → metric nDSM only; no GSD → relative only); directly addresses GAP-01/02/03/11/15; uses the PS-named backbone; licence-clean; the object layer is the quantity for which abundant truth exists.
- **Limitations:** ground masking depends on model output (circular if the model mislabels canopy as ground); terrain layer is only as good as the DEM low frequencies (3–16 m σ in relief); requires a fine-tuning campaign and multi-terrain data preparation; two-layer composition is itself the undocumented step and must be validated.
- **Dependencies:** DA-V2-S weights; DEM tiles; geoid grids; optional anchors. **Data:** nDSM pairs (open). **Compute:** fine-tune 24.8 M-param model (light [G5]); inference CPU-feasible. **Failure modes:** forested slopes where ground pixels are rare (terrain layer falls back to raw DEM); high-rise underestimation (long tail); off-nadir inputs. **Novelty:** the *explicit, validated layer decomposition with tiered calibration and datum handling* — the GAP-03 step nobody documented [E4][E12][E16]. **Coverage:** all H1–H12; V1–V9.

### Concept D — "Geometry-Cue-First" (shadow / sun-angle / lean / RPC classical photogrammetry)
- **Core idea:** derive building heights from shadow length and solar geometry (metadata), building lean from off-nadir angle, and terrain from the DEM; use the depth model only for ordering/segmentation.
- **Mechanism:** shadow segmentation, sun elevation from acquisition time, per-building height = shadow length × tan(elevation); Intermap-patent-style edge geometry [E33].
- **Metric scale:** trigonometric (needs sun elevation + GSD). **Absolute elevation:** DEM. **Domain gap:** avoided (no learned depth). **Validation:** per-building.
- **Advantages:** physically interpretable; needs no training.
- **Limitations:** documented fragility (overlapping shadows, dense cores, vegetation, flat roofs) [Phase 2 §9]; metadata (acquisition time, sun angle) absent in PNG and often stripped from GeoTIFF; produces per-building heights, not a dense surface; nothing for forest/terrain; contradicts the PS's "pretrained monocular depth" instruction.
- **Novelty:** none (patent landscape). **Coverage:** partial — fails H4, weak on forest/hilly.

### Concept E — "Prior-Conditioned End-to-End Absolute DSM" (DEM-as-input learned fusion)
- **Core idea:** feed the coarse DEM (and optionally sparse anchors) as an extra input channel/prior into a depth foundation model fine-tuned to output absolute elevation directly (analogue of Prior Depth Anything / PromptDA with a DEM prior) [E27].
- **Mechanism:** RGB + upsampled DEM channel → network → absolute DSM in DEM datum; trained on LiDAR DSM pairs where a DEM is co-available.
- **Metric scale / absolute:** learned jointly from DEM channel + supervision. **Domain gap:** fine-tuning. **Validation:** blueprint.
- **Advantages:** single model; can learn to de-bias the DEM implicitly; potentially best accuracy where training covers the domain.
- **Limitations:** untested on RS (no prior art with DEM priors) [E27]; opaque calibration (an evaluator cannot inspect what the DEM contributed — violates principle 6/11); training requires DSM (not nDSM) truth co-registered with DEM in one datum — only CH/NZ/US/NL qualify, so GAMUS/DFC19 nDSM data are unusable; risk of the network learning to copy the upsampled DEM (scores well at 30 m scale, fails at object scale — Phase 4 red-team); no graceful fallback without DEM (needs a separate model); Mode A separate.
- **Novelty:** genuine but high-risk research. **Coverage:** H3 strong if it works; explainability weak.

---

## 4. Concept Red-Team

| Attack | A (zero-shot + fit) | B (MHE + add-back) | C (two-layer) | D (geometry cues) | E (prior-conditioned) |
| --- | --- | --- | --- | --- | --- |
| Biggest hidden assumption | Zero-shot structure is right and only scale is wrong — contradicted by [E15][E16] | Training GSD/morphology ≈ evaluation | Model can separate ground from objects well enough to sample the DEM on ground | Shadows visible, isolated, sun metadata present | Network won't shortcut by copying the DEM channel |
| Scale calibration fails when… | DEM bias ≠ model bias spatially; few DEM cells; relief | GSD differs from training | GSD out of training range; anchors clustered; ground rare | Overlapping shadows; flat roofs | Training domain ≠ test domain |
| Without DEM | Relative only (must not claim metres) | nDSM only | **nDSM only, clearly labelled** | Per-building heights only | No output (needs prior) or separate model |
| Without GCPs | Same as with (uses DEM) | Same | Level T reached; Level A skipped; wider uncertainty | Unaffected | Same |
| Forests | DEM canopy bias inherited; blurred | Double-counts canopy in DEM add-back | Ground-masked terrain reduces double-count; where no ground, falls back to raw DEM and flags it | Fails (no shadows/buildings) | Learns whatever training forests taught; unverifiable |
| Hills | Depth compression vs relief; affine fit absorbs slope | Add-back OK; nDSM unaffected by slope in principle | Terrain layer carries DEM slope error (5.6 ± 15.7 m high relief [E21]); flagged by slope | Shadow geometry on slopes wrong | Same DEM limits, hidden |
| Cities | Edge blur, scale drift | Strong in-distribution; long tail | Long tail persists (GAP-07); per-tile consistent scale | Dense cores fail | Strong if trained |
| Shadows | Confound | Mitigated by training | Same as B | It *is* the signal | Same as B |
| Water | Spurious relief | Trained ≈ 0 | Trained ≈ 0; water mask optional | N/A | Same |
| Large buildings | Underestimated | Underestimated | Underestimated; uncertainty high | Shadow truncation | Underestimated |
| Incomplete georeferencing (CRS but no vertical, rotated transform) | Fit still runs → silent bias | Add-back in wrong datum | **Detected: vertical datum declared/defaulted with flag; rotation handled by GDAL** | Metadata needed | Silent |
| Bad metadata (wrong CRS) | Wrong DEM sampled → garbage silently | Same | Consistency check: DEM/anchor residual NMAD → fail flag | Same | Silent |
| Very large image | Per-tile affine inconsistency, seams | Tiles OK (metric) | Tiles OK (metric nDSM) + blending; terrain layer global | Per-building | Tiles OK |
| Slow inference | Fast | Moderate | Small model → fast; CPU fallback | Fast | Moderate |
| Coarse DEM | Everything depends on it | Terrain only | Terrain only, low-pass, flagged | Terrain only | Prior quality limits |
| Visually convincing but wrong | Very likely (smooth relief) | Possible | Mitigated by uncertainty overlay + mesh tolerance disclosure | Possible | Likely (DEM copy looks right) |

Red-team verdicts: A fails principles 1–5 by construction; B is defensible but licence-blocked and not "backbone-based"; D is not general; E is scientifically interesting but opaque and data-starved; C survives every attack with an *honest* degraded output rather than a silent wrong one.

---

## 5. Concept Comparison

Scale: 1 (poor) – 5 (strong). Scores are judgments justified by the cited evidence; they are used only to structure the decision.

| Criterion (definition) | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| PS compliance (all hard reqs, incl. "pretrained monocular depth backbone") | 4 | 3 (backbone wording) | **5** | 2 | 4 |
| DSM accuracy potential (published analogues) | 2 (zero-shot MAE 4.6 m, drift [E16]) | 4 (HTC-DC/GBA [E4][E12]) | 4 (fine-tuned foundation ≈ MHE [E14][E16]) | 2 | 4–5 (untested) |
| Metric-scale credibility (is the metre defensible?) | 1 (DEM fit only) | 4 (supervised) | **5** (supervised + tiered + anchors) | 3 | 3 (opaque) |
| Geospatial correctness (datum, CRS, layering) | 2 | 3 | **5** (explicit V3) | 3 | 3 |
| Remote-sensing robustness (domain gap addressed) | 1 | 4 | 4 | 3 | 4 |
| Terrain robustness (hilly/forest handling) | 1 | 2 (double-count) | 4 (ground-masked terrain, flags) | 1 | 3 |
| Data availability (training data exists & is licence-clean) | 5 | 3 (nDSM yes; HTC-DC unlicensed) | 4 (nDSM open; CH/NZ/US derivable [G3][G4]) | 5 | 2 (needs DSM+DEM pairs only) |
| Implementation complexity (lower = better) | 5 | 3 | 3 | 3 | 2 |
| Computational feasibility | 5 | 3 | 4 (24.8 M params [G5]) | 5 | 3 |
| Validation feasibility (per Phase 4) | 4 | 4 | **5** (per-layer + ablations) | 3 | 3 |
| Explainability to ISRO evaluator | 3 | 4 | **5** | 4 | 2 |
| Demoability | 4 | 4 | 4 | 2 | 4 |
| Deployment feasibility (standalone, licences) | 4 | 2 (licence) | **5** (Apache-2.0 Small) | 4 | 3 |
| Differentiation from prior art | 1 | 2 | 4 (GAP-03 closure, tiers, uncertainty) | 1 | 4 |
| Risk (lower risk = higher score) | 3 (low effort, high claim risk) | 3 | 4 | 2 | 2 |
| **Sum (indicative)** | 45 | 48 | **65** | 43 | 46 |

Key evidence behind the decisive rows: metric credibility (A vs C) rests on [E15][E16] (zero-shot scale failure) and [E9] (single affine leaves spatial residual); terrain robustness (B vs C) on [E19][E21] (DEM canopy/roof bias → naive add-back double-counts); deployment (B vs C) on [E4][E26][G2] (HTC-DC no licence; DA-V2-Small Apache-2.0 with training code); explainability (E) on principle 6/11 and the Phase 4 red-team "DEM copy" failure.

---

## 6. Primary and Backup Concept Selection

**PRIMARY: Concept C — Two-Layer Calibrated Surface Model.**

Why: it is the only concept that (i) uses the PS-mandated pretrained monocular backbone while neutralising its documented nadir failure through fine-tuning [E14][E16]; (ii) assigns each physical quantity to a source with evidence of competence — object heights to a LiDAR-supervised image model [E4][E12][E13], low-frequency terrain to the DEM [E2][E32], offsets to sparse anchors [E9][F6]; (iii) makes the DSM = terrain + nDSM composition explicit, which is exactly the undocumented step (GAP-03) and therefore the natural place for a defensible contribution; (iv) degrades honestly (metric nDSM without DEM; relative without GSD), satisfying design principle 12; (v) is licence-clean and light enough for a student team [G2][G5]; (vi) exposes calibration tier and uncertainty so evaluators can audit it (GAP-15).

Why not the others: **A** contradicts the evidence on zero-shot nadir depth and would inherit DEM bias as "calibration" — it is the baseline we must beat, not the product. **B** is a strong regressor but not a "pretrained monocular depth backbone" in the PS sense, has an unlicensed reference implementation, and its naive DEM add-back double-counts canopy. **D** is not general (needs shadows and sun metadata; no forest/terrain). **E** is the most interesting research direction but is opaque, untested on RS, needs DSM+DEM co-registered truth only (excluding GAMUS/DFC19), and is vulnerable to DEM-copy shortcuts that would look good at 30 m scale while failing at object scale.

**BACKUP: Concept A — Zero-Shot Backbone + Global Affine Calibration**, retained as (a) the mandatory baseline in every experiment and (b) the operational fallback if fine-tuning cannot be completed in time. C is designed so that with the fine-tuned head removed it *becomes* A plus the datum and tiering machinery — the fallback is a subset, not a rewrite.

**Trade-offs accepted:** a fine-tuning campaign and multi-terrain data preparation (CH/NZ/US DSM−DTM) are on the critical path; terrain-layer accuracy is capped by the DEM (decametre in steep relief); ground-masking is model-dependent; Mode A remains strictly relative.

**Risks remaining:** cross-morphology transfer to Indian cities cannot be trained for with open data (GAP-04); hilly-forest evidence exists only outside India (GAP-05); anchor-count requirements are unknown (GAP-08) and will be measured, not assumed.

Status language: the concept is **proposed, evidence-supported and experimentally testable** — not proven.

---

## 7. Development of the Primary Solution

**Technical identity:** *Two-Layer Calibrated Surface Model (TL-CSM)* pipeline for DepthWizard. The name describes the mechanism: a surface model built from a **height-above-ground layer** (from the image) and a **terrain layer** (from external absolute sources), each calibrated by the source competent for it.

1. **Input handling** — decode PNG/JPG/TIFF; read CRS, affine transform, pixel size, nodata, band count; derive GSD in metres (projected CRS directly; geographic CRS via local metre factors); detect rotation/skew; classify as Mode A (no valid transform/CRS), Mode B (valid). Validate metadata (CRS resolvable, extent plausible); malformed → explicit error, never a silent default (Phase 1 §11).
2. **Image understanding** — radiometric normalization to the training distribution; GSD normalization: resample to the nearest canonical GSD band the model was trained on (e.g., ~0.3 / ~1 / ~3 m), recording the factor; optional lightweight land-cover head (O1) for ground/building/tree/water masks.
3. **Depth/geometry estimation** — fine-tuned DA-V2-Small encoder with an nDSM regression head producing **height above local ground in metres** per pixel at the canonical GSD (Level H). Tiled inference with overlap and feathered blending; because the output is metric, tiles share one scale (no per-tile affine).
4. **Remote-sensing adaptation** — supervised fine-tuning on nadir orthophoto–nDSM pairs spanning GSDs and terrain: GAMUS (0.33 m urban) [E6], DFC19 (1.3 m) [E10], nDSM derived as DSM − DTM from swisstopo (0.5 m; alpine/forest/urban) [F2][G3], LINZ (1 m) [F3][G4], 3DEP + NAIP (1 m; all terrains) [F5]; SynRS3D optional pre-training [E28]. Loss choice and long-tail handling (e.g., classification-regression as in HTC-DC [E4]) are Phase 6/7 decisions; the requirement is metric supervision + GSD conditioning + terrain diversity.
5. **Scale calibration** — hierarchy (§10): Level R (relative), Level H (metric nDSM via learned scale + true GSD), Level T (absolute via terrain layer), Level A (anchor-refined). Each level has entry conditions, a confidence estimate and failure detection.
6. **Metric height / elevation recovery** — nDSM is already metric at Level H. Terrain layer: DEM tiles (default Copernicus GLO-30 [E32]; alternatives SRTM, AW3D30, CartoDEM) reprojected to the image grid, vertical datum transformed to the declared output datum (default EGM2008; see §12), then **ground-masked low-pass reconstruction**: DEM samples are weighted by the fraction of predicted-ground pixels (nDSM < ~1 m, optionally semantic ground) in each DEM cell, and a smooth surface is fitted/interpolated through well-supported cells — so canopy/roof-contaminated DEM cells contribute less. Where support is absent (closed forest, dense CBD), the raw DEM is used and the pixel is flagged. Anchors (Level A) apply a robust (RANSAC/Huber) affine correction — offset (and tilt) to the terrain layer from ground anchors, scale to the object layer from object anchors — with residual statistics reported and a held-out anchor subset kept for checking when N permits [F7].
7. **DSM construction** — DSM = terrain + nDSM (§11), with nodata propagation, water handling and edge-preserving cleanup that is *disclosed*.
8. **Geospatial processing** — outputs written on the input grid (after inverse GSD normalization) with the input CRS/transform, compound vertical CRS metadata, nodata, and sidecar metadata (datum, DEM source/version, anchors used, tier, model hash). §12.
9. **Validation** — built-in harness: co-register to a user-supplied reference (Nuth–Kääb shift), mask, compute ME/RMSE/MAE/NMAD/LE95/r (+ fitted scale), per-class/per-terrain tables, residual raster; Mode A uses affine-invariant metrics only. §14.
10. **3D terrain reconstruction** — heightfield mesh from the DSM with error-bounded simplification (tolerance chosen and reported), LOD tiling for large rasters; mesh-vs-DSM residual computed and shown. §15.
11. **Texture projection** — RGB draped by shared geotransform (identity UV in raster space); no re-registration needed because texture and height are on the same grid.
12. **Interactive visualization** — first-person walk with terrain collision, orbit/fly aerial camera, exaggeration slider defaulting to 1.0 and always displayed, layer toggles (RGB / DSM colour ramp / slope / uncertainty / residual), tier badge.
13. **Height analysis** — point pick returns DSM elevation (datum stated) and nDSM (height above local ground) read from the *rasters*, not the mesh; two-point difference; polygon median height for a building.
14. **Slope analysis** — slope/aspect raster from the DSM (gdaldem-equivalent) with the pixel size in metres; displayed as layer and per-pick value; noise caveat when uncertainty is high.
15. **Failure handling** — §16: every missing input lowers the tier rather than fabricating; every inconsistency raises a flag in the report.
16. **Output generation** — Mode A: rDSM GeoTIFF/PNG (normalized 0–1 or metres-if-user-scale-provided, labelled RELATIVE), mesh/glTF, report. Mode B: nDSM GeoTIFF, DSM GeoTIFF (vertical CRS tagged), uncertainty GeoTIFF, slope GeoTIFF, calibration report (tier, DEM source, datum, anchors, residuals, flags), mesh export.

---

## 8. Non-Georeferenced Mode (PNG/JPG)

- **Reliably estimable:** the relative surface structure — ordering of heights, building/tree/ground layout, relative slopes — via the adapted model's output normalized per image. Because the model is trained in metres, its raw output is metric *only if* the true GSD matched the assumed canonical GSD; without GSD this is unknowable, so the output is re-normalized and labelled relative (GAP-14; constraint 15).
- **Cannot be claimed:** metres, absolute elevation, geographic location, true slope in degrees (slope needs horizontal scale), datum. A user may *optionally* supply a scale hint (known GSD or a known object height); the output is then "user-scaled", still flagged as unverified.
- **Representation:** rDSM as a float raster in [0, 1] (or user-scaled units) with the same pixel grid as the image; an accompanying relative nDSM-like layer; no CRS.
- **Interpretation:** "taller/lower than", "approximately N× as tall as", relative slope steepness; no metres in the UI unless a user scale hint is active (then shown with "unverified scale").
- **Visualization:** the same mesh + texture pipeline with an arbitrary vertical unit and an exaggeration control; measurement tools return ratios/normalized values.
- **Off-nadir:** if strong facades are detected (O5) the output is flagged as perspective-contaminated (GAP-13).

---

## 9. Georeferenced Mode (GeoTIFF)

- **Metadata extracted:** CRS (EPSG or WKT), affine transform (origin, pixel size, rotation), bounds, dimensions, band layout, nodata, any vertical CRS, acquisition/sensor tags if present, RPCs if present (flag only in MVP).
- **Spatial referencing preserved:** all processing in raster space; resampling factors recorded and inverted; outputs written with the exact input transform and CRS; DEM and anchors reprojected *into* the image frame, never the reverse.
- **Metric scale recovered:** horizontal scale from the transform (GSD); vertical scale from the metric nDSM head conditioned on GSD (Level H); optional anchor refinement (Level A).
- **Absolute elevation established:** terrain layer from a DEM (Level T) and/or ground anchors; DSM = terrain + nDSM.
- **Vertical reference handled:** output datum declared (default EGM2008 orthometric-like, because Copernicus is the default DEM and ranks best [E3][E32]); SRTM/AW3D30 (EGM96), CartoDEM (ambiguous — resolved by test against ICESat-2 [E23][F6]), anchors (ICESat-2 ellipsoidal) and any user reference are transformed via geoid grids before use [F13]; the GeoTIFF is tagged with a compound CRS and the report states the datum.
- **Geospatially valid DSM:** correct CRS/transform, declared vertical CRS, nodata, provenance; therefore directly overlayable and comparable in any GIS.
- **Assumptions (explicit):** imagery is orthorectified near-nadir (facades not dominant); GSD within the trained range (else flagged, resampled if possible); a DEM covers the footprint (else Level H only); DEM voids are handled; the declared datum is what the DEM/anchors were converted to.

---

## 10. Scale-Calibration Design

**Quantities (kept distinct throughout):** relative depth/height (model raw, Mode A) → **metric height above ground** (nDSM, Level H) → **absolute elevation** (DSM = terrain + nDSM, Level T/A) → **geospatial DSM raster** (declared datum, CRS).

| Level | Information source | What it contributes | Availability condition | Confidence basis | Failure detection | If missing |
| --- | --- | --- | --- | --- | --- | --- |
| R — Relative | Adapted model, no GSD | Structure only | Always | Model uncertainty | Facade detection; low-texture warning | — |
| H — Metric height above ground | Adapted model + true GSD from transform | Metres above local ground (learned scale; GSD-conditioned) | Valid transform | GSD within training range; TTA variance; terrain class prior from validation tables | GSD out-of-range → flag; extreme heights → flag | Fall back to R |
| T — Absolute (terrain layer) | Coarse DEM (Copernicus default; SRTM/AW3D30/CartoDEM alternatives), datum-transformed, ground-masked low-pass | Low-frequency absolute elevation + datum | DEM covers footprint; datum known | DEM product accuracy by land cover/slope [E19][E21]; ground-support fraction; DEM void fraction | Ground support < threshold → raw-DEM fallback + flag; DEM–model gross inconsistency (e.g., DEM relief ≫ predicted) → flag | Output nDSM only, labelled "no absolute datum" |
| A — Anchored | Sparse anchors: surveyed GCPs, ICESat-2 ATL08 segments, GEDI; user-provided | Offset/tilt correction of terrain layer; scale check of object layer | ≥ 3 usable anchors after outlier rejection (2 gives a fit with zero redundancy — reported as such) | Residual NMAD after fit; anchor source accuracy [F6]; spatial spread | Residual NMAD > anchor σ×k → reject fit, keep T; clustered anchors → tilt disabled | Stay at T |
| (Validation only) D — Dense reference | LiDAR DSM/nDSM | Accuracy assessment | Never used at inference | — | — | — |

**Why not "combine everything by default":** the DEM contributes only low frequency and only where ground is visible; anchors contribute offsets, not structure; the model contributes structure and object scale. Each is used for what evidence says it can do [E9][E19][E21][E32]; the report shows the tier reached.

**Calibration failure detection signals:** anchor residual statistics; DEM-vs-terrain-layer difference distribution; predicted nDSM range vs terrain class expectations; GSD/tile scale consistency across overlaps; ground-support fraction; datum sanity (e.g., mean elevation vs DEM differing by tens of metres indicates an ellipsoid/geoid mix-up [F13]).

**Fallback behaviour:** monotone downgrade R ← H ← T ← A; never upgrade without a source; the tier is printed on outputs and in the UI.

---

## 11. DSM Construction

The model's nDSM is not automatically a DSM. Transformation steps and their rationale:

| Aspect | Treatment | Rationale |
| --- | --- | --- |
| Ground surface | Terrain layer from ground-masked low-pass DEM; nDSM ≈ 0 pixels define ground; DSM = terrain there | DEM is competent only at low frequency [E2]; ground masking limits canopy/roof bias [E19][E21] |
| Buildings | nDSM added to terrain; footprint-consistent heights optionally regularised (median within detected footprint) — disclosed | Long tail and edge blur [E4]; instance consistency matches RMSE-B practice |
| Vegetation | nDSM = canopy height added to terrain (first-surface semantics) ; forest pixels flagged as "canopy top" | DSM is a first-surface model by definition; separate CHM not claimed |
| Bridges / overpasses | Treated as surface (nDSM over water/road) | Consistent with LiDAR DSM semantics |
| Discontinuities / edges | Edge-preserving blending across tiles; no isotropic smoothing across predicted edges | Preserve high-frequency structure the model supplies |
| Holes / nodata | Propagate input nodata; DEM voids interpolated in terrain layer and flagged; no zero-fill | Zero = valid elevation [E2] |
| Noise | Small median/bilateral cleanup *optional and reported*; uncertainty raster kept unfiltered | GAP-17: smoothing must not hide error |
| Water | Predicted ≈ 0 nDSM; terrain layer sets level; optional flattening within water mask (O1) | LiDAR references behave similarly |
| High-frequency preservation | Native-grid output; canonical-GSD resampling inverted with cubic interpolation; no DEM-driven smoothing of the object layer | DEM must never shape the object layer |
| Consistency check | Low-pass(DSM) vs DEM difference statistics reported | Detects terrain-layer failure |

---

## 12. Geospatial Integrity

- **CRS handling:** horizontal CRS read and preserved; geographic CRS inputs processed with local metre GSD (or reprojected to UTM with the reprojection recorded); outputs always carry the input CRS.
- **Affine transform / bounds / pixel size:** untouched; all internal resampling is inverted before writing; bounds identical to input; pixel-centre convention consistent with GDAL.
- **Spatial alignment RGB ↔ DSM:** identical grid by construction → texture projection is exact; DEM/anchors reprojected into this grid with bilinear/cubic (DEM) or point sampling (anchors).
- **Nodata:** explicit nodata value in every output; masks for input nodata, DEM voids, water (optional), image border band (reported).
- **Vertical reference:** default output datum EGM2008 (orthometric-like, EPSG:3855 vertical / compound with horizontal CRS); conversions: SRTM & AW3D30 from EGM96 (≤ 1 m difference in India [F13]); ICESat-2 from ellipsoid via EGM2008 grid (−24…−99 m over India [F13]); CartoDEM datum resolved empirically before use [E23]; user references converted with stated grids; a user may choose a different output datum, in which case all layers are converted consistently.
- **Raster output:** Cloud-Optimized GeoTIFF float32, LZW, nodata, tags for vertical CRS, datum grid version, DEM source/version, model hash, tier, GSD, processing date; sidecar JSON report.
- **Consistency:** nDSM, DSM, uncertainty, slope and residual rasters share one grid; the mesh is built from the DSM raster and its simplification tolerance is stored.

---

## 13. Uncertainty / Confidence

**Included — yes, because the evidence shows error is strongly heterogeneous** (2–3× by morphology [E4][E12]; slope- and forest-dependent DEM error [E21][E19]; long-tail underestimation [E4]) and **no found single-view system exposes it** (GAP-15). An honest system must tell the user *where* the metres are trustworthy.

What is exposed and how it is derived:
- **Per-pixel uncertainty raster (metres):** variance across test-time augmentations (flips/rotations/scales) of the nDSM head, scaled by a class/terrain error prior learned from validation residuals; combined with the terrain-layer uncertainty (DEM product accuracy by land cover/slope, ground-support fraction) and, at Level A, the anchor-fit residual.
- **Calibration quality:** tier reached; DEM source and datum; ground-support statistics; anchor count, residual ME/NMAD, held-out anchor residuals; consistency-check outcomes.
- **Estimated error statement:** expected RMSE/NMAD ranges per terrain class copied from the validation tables of the *same model version* (not invented).
- **Low-confidence regions:** flagged pixels (raw-DEM fallback, facades, borders, out-of-range GSD) as a mask layer.

Contribution to credibility: converts a single number into an auditable error budget; aligns with ASPRS reporting of mean error and separate vegetated accuracy [F7]; makes the visualization honest (uncertainty overlay) rather than uniformly convincing (GAP-17). Uncertainty quality itself is testable via sparsification/AUSE [F16].

---

## 14. Validation Design Connection

Mapping to the Phase 4 blueprint and benchmark structures (§26 there):

| What is measured | Against what | Terrain | Input mode | Reference data | Success / failure definition |
| --- | --- | --- | --- | --- | --- |
| nDSM layer accuracy (Level H) | LiDAR nDSM (DSM − DTM) on image grid | Urban / sparse / hilly / forested strata | Mode B (GSD known) | swisstopo, LINZ, 3DEP+NAIP held-out blocks; GAMUS/DFC19 held-out cities | Success: per-terrain RMSE/MAE/NMAD, RMSE-B, F1-HE improve over the zero-shot baseline (Concept A) and are reported with ME; failure: no improvement or unstable across strata |
| Terrain layer accuracy (Level T) | LiDAR DTM low-passed to DEM scale | Same | Mode B | swissALTI3D, LINZ DEM, 3DEP DTM | Success: ground-masked terrain reduces forest/built-up bias vs raw DEM [E19]; failure: no bias reduction |
| Composed DSM accuracy (Level T/A) | LiDAR DSM (one vertical datum) | Same | Mode B | Same sources | Success: DSM RMSE ≤ nDSM RMSE + terrain error budget, ME within reference tolerance; failure: composition adds error beyond layers |
| Anchor refinement | Held-out anchors + LiDAR | Same | Mode B | ICESat-2 tracks split by track; synthetic anchors from LiDAR | Success: residual falls with N and saturates; failure: no gain or overfit |
| Calibration ablation (C0–C8) | LiDAR | Same | Mode B | Same | Error decomposition: scale vs structure vs datum |
| rDSM quality | Any dense truth after affine alignment | Same | Mode A (metadata stripped from the same tiles) | Same | Affine-invariant metrics, Spearman; must not degrade vs Mode B structure |
| External fixed benchmark | Official DFC19 Track-1 test protocol | Urban | Mode B | US3D | Position relative to published 5.46 m all / 10.69 m bldg [E11] |
| Cross-country generalization | LiDAR | All | Mode B | Train US(+CH), test NZ (and vice-versa) | Degradation ratio reported |
| India assessment | ICESat-2 ATL08 checkpoints; CartoDEM trend | Whatever open Indian imagery covers (LISS-IV; Cartosat if supplied) | Mode B | ATL08 (0.7–4 m), CartoDEM (8 m LE90) | Bounded statements only: "consistent with ICESat-2 within its error"; no metre-level claims |
| Visualization fidelity | DSM raster; known features | All | Both | Same scene | Mesh-vs-DSM RMSE ≤ stated tolerance; projection residual ≤ ~1–2 texels; frame-time percentiles; crash-free; SUS |

Splits: spatial blocks with buffers, city/region hold-outs, footprint-overlap checks across GAMUS/DFC19/GBA/3DEP (Phase 4 §10, §33). Metrics pre-registered before training (Phase 4 must-resolve #7). No expected numbers are stated here.

---

## 15. Visualization Concept

- **Geometry = DSM.** A heightfield mesh is generated from the DSM raster with error-bounded simplification (RTIN/quadtree LOD per prior art [E25]); the tolerance in metres is a visible setting and the resulting mesh-vs-DSM RMSE is displayed. Exaggeration defaults to 1.0 and is shown on screen whenever ≠ 1.
- **Texture = the input RGB**, draped by the shared geotransform; no re-projection ambiguity, hence projection accuracy is a property of the DSM grid, verifiable by back-projecting known features.
- **Navigation:** first-person mode with camera clamped above the mesh (collision) and walk speed scaled to GSD; free aerial orbit/fly mode; bookmarks; both modes read the same scene.
- **Analysis layers (toggle):** RGB; DSM hypsometric ramp; nDSM ramp; slope (degrees) and aspect; uncertainty; flagged regions; residual vs user reference; DEM terrain layer alone (to show what the DEM contributed vs the model).
- **Measurement:** click → elevation (datum stated), height above ground, uncertainty; two-point height difference and horizontal distance; polygon → median building height; profile line → elevation/slope profile. Values come from rasters, not mesh vertices.
- **Validation panel (H9):** load reference raster → co-register → metrics table per class/terrain → residual map overlay.
- **Honesty cues:** tier badge (R/H/T/A), datum label, "RELATIVE" watermark in Mode A, low-confidence hatching.
- **Avoided:** sky/atmosphere effects, procedural buildings, generated facades, water animation, unstated smoothing.
Engine choice (Three.js/Babylon/Cesium/Unity) is deferred to Phase 6; the concept requires only heightfield rendering with LOD, texture draping, camera controls and raster lookup — all commodity [E25].

---

## 16. Failure-Safe Behaviour

| Situation | Behaviour |
| --- | --- |
| No DEM available | Stop at Level H: output metric nDSM labelled "height above local ground; no absolute datum"; DSM not produced; UI shows nDSM |
| GCPs unavailable | Level T; report states "unanchored; terrain from DEM (source, datum, accuracy class)" |
| Metadata incomplete (CRS without vertical; transform without CRS) | Vertical: default declared datum with flag; missing CRS: treat as Mode A with option for user to supply CRS/GSD (flagged "user-supplied") |
| Georeferencing invalid (unresolvable CRS, extent off-Earth, DEM/anchor residual NMAD absurd) | Mode B refused with explicit reason; offered Mode A processing |
| Calibration confidence low (residuals, ground support, GSD out of range) | Keep lower tier; flags in report; uncertainty raster widened; UI warning |
| Poor image quality (heavy JPEG, haze, clouds) | Quality score from image statistics; warning; cloud/nodata mask if detectable; processing continues with flag |
| Model inference fails (OOM, corrupt tile) | Retry with smaller tiles / CPU; if still failing, partial output with nodata and error report — never silent |
| Reference data missing (validation panel) | Validation disabled; report states "not validated"; no accuracy figure shown |
| Raster too large | Tiled inference with overlap; LOD mesh; if beyond limits, user-selected AOI crop with bounds recorded |
| GPU insufficient | Small model runs on CPU (slower, same output); progress indicator; no feature removal |
| Off-nadir / facades detected | Flag: "perspective effects; heights may be biased" |
| Datum sanity failure (mean offset tens of metres vs DEM) | Halt absolute output; message suggests ellipsoid/geoid mismatch |

Principle: **fail honestly** — downgrade the claim, never fabricate metres.

---

## 17. Novelty / Differentiation

| Category | Items |
| --- | --- |
| **Already standard** (must not be claimed) | Pretrained monocular depth backbones; nDSM regression from single nadir images; fine-tuning foundation encoders on LiDAR height; least-squares scale-shift with RANSAC; heightmap→mesh→texture→navigation; slope/hillshade; SRTM/Copernicus use; GeoTIFF/CRS handling; F1-HE/RMSE-B metrics; DEMIX stratification; FABDEM-style DEM de-biasing. [E4][E9][E12][E13][E14][E19][E25][E28][E32] |
| **Combination of existing technologies** | Foundation backbone + metric nDSM head; DEM reprojection + datum transform; ICESat-2 anchors; tiled inference; uncertainty via TTA; standalone packaging of model + viewer. |
| **Proposed technical contribution** | (1) **Explicit two-layer DSM composition** — terrain layer from *ground-masked, datum-transformed* coarse DEM + object layer from an adapted monocular model — as a documented, validated procedure closing GAP-03; (2) **tiered calibration with declared semantics** (R/H/T/A) and failure detection, so every output metre has a traceable source (GAP-02/11); (3) **per-layer validation protocol** (nDSM vs LiDAR nDSM, terrain vs DTM low-pass, DSM vs DSM) with per-terrain stratification and anchor-count curves (GAP-08/09); (4) **uncertainty and calibration-quality exposure** in both raster and viewer for single-view DSMs (GAP-15); (5) a **measurement-faithful visualization** whose mesh error against the DSM is quantified (GAP-17). |
| **Why it matters** | It converts "monocular depth + SRTM" from an unverifiable heuristic into an auditable measurement chain that an ISRO evaluator can interrogate layer by layer; it degrades honestly where India-specific data are missing. |
| **Evidence needed to claim it** | Ablations showing (a) ground-masked terrain reduces forest/built-up bias vs raw DEM add-back [E19 baseline], (b) two-layer DSM RMSE ≈ nDSM RMSE + bounded terrain error, (c) anchor curves, (d) uncertainty AUSE better than random, (e) mesh-vs-DSM residuals within tolerance — all on spatially blocked, multi-terrain LiDAR tests, plus the DFC19 external protocol. |

---

## 18. SIH Compliance Matrix

| SIH Requirement | Proposed Component | How It Satisfies | Validation Evidence |
| --- | --- | --- | --- |
| Accept PNG/JPG | Input handler, Mode A | Decode; relative pipeline | Input test suite |
| Accept GeoTIFF/TIFF | Input handler, Mode B | Metadata parsing, CRS/transform preservation | Metadata round-trip tests |
| rDSM for non-georeferenced | Level R output, relative viewer | Normalized relative surface, labelled | Affine-invariant metrics on metadata-stripped tiles |
| Absolute DSM with metric heights | Levels H→T→A; two-layer composition | nDSM (learned scale) + DEM terrain (datum-transformed) ± anchors | LiDAR DSM tests per terrain; ablations |
| Use of SRTM-class DEM or limited GCPs | Terrain layer; anchor module | DEM low-frequency terrain; anchors refine | C2/C3 experiments |
| Pretrained monocular depth backbone | DA-V2-Small encoder (Apache-2.0) fine-tuned | Backbone retained, adapted | Zero-shot vs fine-tuned ablation |
| Scale-calibration module | Tiered calibration with failure detection | Explicit semantics R/H/T/A | Calibration report; anchor residuals |
| Scene statistics / semantic priors | GSD normalization; ground masking; optional semantic head | Priors used where evidenced | Ablation with/without masking |
| Standard geospatial DSM output | COG GeoTIFF + vertical CRS + report | GIS-valid | GDAL validation; overlay tests |
| Project RGB onto 3D mesh | Heightfield mesh + shared-grid texture | Exact alignment by construction | Projection residuals |
| Rendering engine integration | Viewer (engine chosen Phase 6) | Commodity components | Performance percentiles |
| First-person navigation | Walk mode with collision | — | Clipping incidents; task tests |
| Arbitrary aerial perspectives | Orbit/fly mode | — | Task tests |
| Structural height analysis | Raster-based pick/polygon tools (nDSM + DSM) | Values from rasters, datum stated | Checkpoint comparison (≥ 30) |
| Slope analysis | Slope/aspect rasters + profile tool | Metres-based gradient | Slope RMSE vs reference |
| Upload imagery | UI upload | — | Stability tests |
| Validate against reference datasets | Validation panel (co-registration, metrics, residual map) | Phase 4 blueprint embedded | Reproduced metrics vs offline harness |
| RMSE / MAE / correlation vs LiDAR | Harness metric set | Plus ME/NMAD/LE95, object metrics | Per-terrain tables |
| Stability across urban/sparse/hilly/forested | Multi-terrain training + stratified validation | CH/NZ/US strata | Stratified tables; degradation ratios |
| Software stability | Tiling, CPU fallback, explicit errors | — | Crash-free runs over input suite |
| Standalone deployment | Local packaged app with local inference + bundled DEM for demo AOIs | Offline operation | Clean-machine offline install test |
| Source code + documentation | Repository + technical docs incl. datum/tier semantics | — | Reproducibility metadata |

---

## 19. End-to-End Conceptual Flow

| Stage | Input | Process | Output | Purpose | Failure condition |
| --- | --- | --- | --- | --- | --- |
| INPUT | File | Decode; checksum | Pixel array + raw metadata | Entry | Undecodable → error |
| INPUT TYPE DETECTION | Metadata | CRS/transform validity; GSD derivation; rotation; nodata | Mode A/B; GSD; flags | Route | Invalid CRS → Mode A offer |
| IMAGE / METADATA ANALYSIS | Pixels + metadata | Radiometric normalization; quality score; canonical-GSD resampling; tile plan; facade check | Normalized tiles; factors | Prepare for model | Quality below threshold → warning |
| DEPTH / GEOMETRY ESTIMATION | Tiles | Adapted backbone → metric nDSM (H) or relative (R); TTA variance | nDSM/relative mosaic + uncertainty | Object layer | OOM → smaller tiles/CPU; failure → partial nodata |
| REMOTE-SENSING ADAPTATION | (offline) LiDAR nDSM pairs | Fine-tuning; validation tables | Model version + error priors | Domain gap | Poor validation → fall back to zero-shot baseline (Concept A) |
| SCALE / ELEVATION CALIBRATION | nDSM, DEM, anchors, datum grids | Datum transform; ground-masked terrain; anchor fit; tier decision; checks | Terrain layer; tier; residuals | Absolute elevation | Checks fail → lower tier + flags |
| DSM GENERATION | Terrain + nDSM | Composition; nodata; disclosed cleanup; slope | DSM, nDSM, slope, uncertainty rasters | Deliverable | Composition inconsistency → flag |
| GEOSPATIAL VALIDATION | Rasters (+ user reference) | CRS/transform round-trip; datum tag; optional co-registration + metrics | Valid GeoTIFFs; report; residual map | GIS validity; H9 | Missing reference → "not validated" |
| 3D TERRAIN GENERATION | DSM | Heightfield mesh, LOD, tolerance; mesh-vs-DSM RMSE | Mesh tiles | Geometry | Too large → AOI |
| RGB TEXTURE PROJECTION | RGB + mesh | Shared-grid UV | Textured scene | Fidelity | Grid mismatch (should not occur) → abort |
| INTERACTIVE ANALYSIS | Scene + rasters | Navigation; picks; profiles; layers | Measurements with datum/tier | H7/H8 | — |
| DSM / 3D / REPORT OUTPUT | All | Write COGs, mesh export, JSON/PDF report | Deliverables | H6/H12 | Write failure → error |

---

## 20. Human-Readable Solution Story

One satellite photo cannot tell you how high the ground is above sea level — but it *can* tell you how tall the things standing on the ground are, once a depth model has been taught what rooftops and treetops look like from above. Coarse free elevation maps like SRTM or Copernicus tell you roughly how high the ground is, but at 30-metre blobs that cannot see individual buildings, and they are slightly too high wherever forests and roofs push the radar up.

So we split the job. The picture gives us the *object layer*: a metre-accurate map of height above local ground, produced by a well-known open depth model that we fine-tune on aerial photographs paired with laser-scanned heights from Switzerland, New Zealand and the USA — flat cities, hills, forests and open country. The coarse elevation map gives us the *terrain layer*: the gentle shape of the ground, which we read only where our model says the ground is visible, so tree and roof heights are not counted twice. We convert everything to one declared height reference (India's sea-level reference differs from GPS ellipsoid heights by 24 to 99 metres, so this step is not optional). If a few known elevation points exist — surveyed control points or NASA's ICESat-2 laser footprints — we use them to correct the offset and report the residual; the points used are never the points we test on.

Adding the two layers gives a Digital Surface Model in real metres, saved as a GeoTIFF any GIS can open, together with a map of how uncertain each pixel is and a report saying which calibration level was reached. If a photo has no coordinates, we say so and deliver only relative heights — no invented metres. The 3D flythrough is built from exactly that surface, textured with the original image, with a "how much the mesh differs from the data" number on screen and tools that read heights and slopes from the data rather than from the pretty mesh.

---

## 21. Technical Elevator Pitch

DepthWizard TL-CSM turns a single nadir RGB image into a Digital Surface Model by separating what the image can measure from what it cannot. A pretrained monocular depth backbone (Depth Anything V2-Small, Apache-2.0) is fine-tuned with LiDAR-derived height-above-ground supervision across urban, sparse, hilly and forested terrain from open Swiss, New Zealand, US and GAMUS data, conditioned on ground sample distance, so it outputs metric heights above local ground rather than scale-ambiguous relative depth. For georeferenced inputs, a terrain layer is derived from a coarse DEM (Copernicus/SRTM/CartoDEM) transformed to a declared vertical datum and sampled where the model sees bare ground, then optionally refined with sparse anchors (GCPs or ICESat-2) whose residuals are reported. DSM = terrain + height-above-ground, written as a GeoTIFF with CRS, vertical CRS, nodata, uncertainty raster and a calibration-tier report; non-georeferenced inputs yield an explicitly relative rDSM. Validation follows a spatially blocked, per-terrain protocol against LiDAR DSMs with ME/RMSE/MAE/NMAD/LE95/correlation and object-level metrics, plus the DFC2019 Track-1 external protocol. The 3D flythrough renders exactly this surface with quantified mesh error and raster-based height and slope tools, packaged as a standalone application.

---

## 22. What the System Does NOT Claim

- Reconstruction of hidden surfaces (under canopy, behind/under overhangs, vertical facades) — first-surface only.
- Metric elevation without a DEM or anchors; metric anything without a known GSD (Mode A is relative by definition).
- Terrain detail finer than the coarse DEM's resolution from the DEM; sub-DEM terrain relief comes from the model and carries model error.
- LiDAR-level accuracy; accuracy is whatever the per-terrain validation tables show for the released model version, and is bounded by DEM error in steep relief.
- Ground truth from a coarse DEM; DEMs are calibration inputs, not references.
- Unlimited geographic generalization; performance on Indian morphology is *inferred* from cross-country tests and bounded by sparse ICESat-2 checks unless Indian LiDAR-grade references and Cartosat imagery become available.
- Height of individual small objects near the GSD limit; tall-building heights without underestimation risk.
- A bare-earth DTM (unless a future component is validated for it); the terrain layer is a DEM-derived low-frequency surface, not a certified DTM.
- Correct heights for strongly off-nadir imagery.
- That the visualization is more accurate than the DSM it renders.

---

## 23. Technical Risk Register

| Risk | Cause | Impact | Probability | Mitigation Concept | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Scale failure of object layer | GSD outside training range; morphology shift | Wrong metres, biased buildings | Medium | Multi-GSD training; GSD flag; anchor scale check; uncertainty | Medium (Indian morphology) |
| Domain shift (sensor/radiometry) | Aerial-trained vs satellite MX | Accuracy drop | Medium-High | Include satellite-like data (DFC19 WV-3, SynRS3D); radiometric normalization; sensor-shift test (NAIP→LISS-IV) | Medium |
| Terrain failure on steep relief | DEM slope error; co-registration | Decametre terrain error | High in mountains | Slope-dependent uncertainty; Nuth–Kääb shift on DEM; anchors | High (bounded by DEM) |
| Vegetation | Canopy occlusion; no ground support; DEM canopy bias | Forest DSM bias | High | Ground-masked terrain; raw-DEM fallback flagged; forest-specific validation | Medium-High |
| Geospatial mismatch | Datum/CRS errors | Tens of metres bias | Medium | Datum pipeline; sanity checks; tagged outputs | Low |
| Calibration data availability | No DEM tiles offline / no anchors | Tier H only | Low (DEMs global) | Bundle DEM for demo AOIs; user-supplied DEM/anchors | Low |
| Computation | Large rasters, GPU absent | Slow demo | Medium | Small model; tiling; CPU fallback; AOI | Low-Medium |
| Resolution mismatch at evaluation | Cartosat 0.25/1.13 m vs training | Degraded output | Medium | Canonical-GSD resampling; report | Medium |
| Visualization performance | Full-res meshes | Frame drops | Medium | LOD/RTIN; tolerance control | Low |
| Deployment | Packaging inference + viewer offline | Fails on evaluator machine | Medium | Clean-machine tests; minimal dependencies | Low-Medium |
| Training-data leakage | Overlapping footprints across datasets | Inflated numbers | Medium | Footprint intersection checks; blocked splits | Low |
| Long-tail underestimation | Height imbalance | High-rise errors | High | Long-tail-aware loss; report height-bin errors | Medium-High |
| Ground-mask circularity | Model mislabels canopy as ground | Terrain contaminated | Medium | Conservative threshold; semantic head (O1); support-fraction flag | Medium |
| Licence | Backbone/data terms | Cannot ship | Low | Apache-2.0 Small; CC0/CC BY/OGD data; no FABDEM in product | Low |

---

## 24. MVP vs Advanced System

**MVP REQUIRED FOR SIH**
- Both input modes with detection and honest labelling (R/H/T).
- Fine-tuned DA-V2-Small nDSM head on GAMUS + at least one multi-terrain LiDAR source (CH or NZ or US) with GSD normalization; zero-shot baseline retained.
- Terrain layer from Copernicus/SRTM with datum transform; ground-masked low-pass; DSM composition; nodata; COG outputs with vertical CRS tag; calibration report.
- Tiled inference; CPU fallback.
- Validation harness: co-registration, masks, metric set, per-terrain/per-class tables, residual raster; DFC19 protocol run.
- Viewer: heightfield mesh with tolerance disclosure, RGB drape, first-person + orbit, exaggeration display, height/slope picks from rasters, layer toggles (RGB/DSM/slope), reference-comparison panel.
- Standalone packaged build tested offline on a clean machine.

**ADVANCED / DIFFERENTIATING**
- Level A anchor refinement with ICESat-2 ingestion and held-out anchor checking; anchor-count curves.
- Per-pixel uncertainty raster (TTA + priors) and viewer overlay; AUSE evaluation.
- Semantic head (ground/building/tree/water/road) for masking and class-wise error tables.
- Cross-country generalization experiments (US ↔ CH ↔ NZ) and India ICESat-2-bounded assessment on open imagery.
- Long-tail-aware loss; building-instance regularisation; height-bin reporting.

**OPTIONAL**
- LoD1 extrusion of buildings; profile tool; change-mask handling; off-nadir facade warning; multi-GSD ensembling; alternative DEM sources (AW3D30, CartoDEM with datum test).

---

## 25. Time / Resource Feasibility

- **Hardest components:** (1) preparing co-registered, datum-consistent training pairs from national LiDAR (DSM − DTM, orthophoto date matching, tiling) — data engineering, not ML; (2) making the ground-masked terrain layer robust; (3) the validation harness done correctly (co-registration, masks, blocked splits).
- **Most experimentation:** loss/long-tail handling for the nDSM head; ground-support thresholds; anchor fitting robustness; mesh tolerance vs frame rate.
- **External dependencies:** DA-V2-Small weights and training code [G2]; DEM tiles (Copernicus/SRTM); geoid grids (PROJ); GAMUS; one or more national LiDAR portals [F2–F5]; ICESat-2 via Earthdata (advanced).
- **Data bottlenecks:** national LiDAR downloads are TB-scale nationally — AOI subsets (a few hundred km² per terrain class) are sufficient and downloadable; GAMUS is GB-scale; no Indian HR imagery unless supplied.
- **GPU requirements:** fine-tuning a 24.8 M-parameter model on ~10⁴ tiles is feasible on one consumer/cloud GPU in hours-to-a-day scale (Depth Any Canopy: < $1.30 compute [G5]); inference feasible on CPU for demo-sized scenes.
- **Implementation bottlenecks:** GeoTIFF/datum plumbing correctness; tiled blending; packaging a Python inference stack with a web/desktop viewer offline.
- **High-risk features:** Level A anchors (evidence-free requirements), semantic head, cross-country experiments — all scoped as Advanced so the MVP does not depend on them.
- **Assessment:** the MVP is realistically buildable by a student team within a hackathon development window; the Advanced scope is achievable in part; nothing in the MVP relies on unavailable or priced data.

---

## 26. Build / Buy / Use-Existing Analysis

| Component | Decision | Why |
| --- | --- | --- |
| Monocular depth backbone | USE PRETRAINED + FINE-TUNE (DA-V2-Small) | PS mandates a pretrained backbone; Apache-2.0; training code exists [G2]; fine-tuning evidence [E14][E16] |
| nDSM head / loss | BUILD (small) | Task-specific; long-tail handling design choice [E4] |
| Training data preparation (DSM−DTM, tiling, GSD) | BUILD on USE EXISTING LIBRARY (GDAL/rasterio) | Standard geospatial tooling |
| Training datasets | USE EXTERNAL DATA (GAMUS CC-BY-4.0; swisstopo OGD; LINZ CC BY 4.0; 3DEP/NAIP public domain; DFC19) | Licence-clean, multi-terrain [E6][F2][F3][F5] |
| Coarse DEM | USE EXTERNAL DATA (Copernicus GLO-30 default; SRTM/AW3D30/CartoDEM options) | Best-ranked open DEM [E32]; global |
| Datum transformation | USE EXISTING LIBRARY (PROJ/GDAL geoid grids) | Standard [F13] |
| Ground-masked terrain layer | BUILD | Core contribution; no library does it |
| Anchor fitting (robust affine) | BUILD on standard estimators | Simple; must be inspectable |
| Anchor sources | USE EXTERNAL DATA (ICESat-2 ATL08; user GCPs) | Only India-capable absolute anchors [F6] |
| Tiled inference / blending | BUILD (thin) | Standard pattern |
| Validation harness | BUILD on USE EXISTING LIBRARY (rasterio/xdem-style co-registration) | Must implement Phase 4 blueprint |
| Mesh generation / LOD | USE OPEN-SOURCE (RTIN/Martini-class, quantized-mesh or engine terrain) | Commodity [E25] |
| Rendering engine | USE OPEN-SOURCE (choice Phase 6) | Commodity [E25] |
| Slope/aspect | USE EXISTING LIBRARY (GDAL DEM tools) | Standard |
| Packaging | USE EXISTING tooling (Phase 6) | Standard |
| Uncertainty (TTA + priors) | BUILD (thin) | Simple, explainable |
| Semantic head (optional) | BUILD on GAMUS labels | Available labels [E5] |

---

## 27. Final Solution Specification

**Problem.** Produce an rDSM (PNG/JPG) or an absolute metric DSM (GeoTIFF) from one nadir RGB image using a pretrained monocular depth backbone, a coarse DEM and/or sparse GCPs, output it in a valid geospatial format, validate it against LiDAR across four terrain types, and render it as a navigable textured 3D scene in a standalone application.

**Core insight.** A DSM is two physically different layers with two different competent sources: the *image* can measure height above local ground once the backbone is adapted with LiDAR nDSM supervision (this is what the field's truth and results actually support), while only an *external absolute source* can supply low-frequency terrain and a datum — and only at its own resolution and bias. Calibrate each layer with its competent source, compose explicitly, declare the datum and tier, expose uncertainty, and degrade honestly.

**Primary concept.** Two-Layer Calibrated Surface Model (TL-CSM).

**Inputs.** PNG/JPG (Mode A); GeoTIFF with CRS/transform (Mode B); optional DEM tiles (bundled/user), optional anchors (GCP file / ICESat-2), optional reference DSM for validation, optional user scale hint (Mode A).

**Processing stages.** Input detection → metadata/GSD analysis → canonical-GSD tiling → adapted backbone → metric nDSM (H) / relative (R) → datum-transformed, ground-masked terrain layer (T) → anchor refinement (A) → DSM composition → geospatial outputs + report → mesh/LOD → texture → viewer/analysis → validation panel.

**Calibration strategy.** Tiers R/H/T/A with entry conditions, confidence, failure detection and monotone downgrade; anchors partitioned from checkpoints; datum declared (EGM2008 default); DEM used only for low frequency where ground is visible.

**DSM generation.** DSM = terrain + nDSM on the input grid; first-surface semantics; nodata propagation; disclosed cleanup; slope derived from DSM; consistency check vs DEM.

**Geospatial strategy.** CRS/transform preserved; compound vertical CRS tagging; PROJ geoid grids; COG outputs; provenance in tags and report.

**Validation strategy.** Phase 4 blueprint: spatially blocked multi-terrain LiDAR tests (CH/NZ/US), per-layer and composed metrics (ME/RMSE/MAE/NMAD/LE95/r + RMSE-B/F1-HE + height bins), calibration ablations C0–C8, DFC19 external protocol, cross-country degradation, India ICESat-2-bounded assessment, Mode-A affine-invariant scoring, visualization metrics (mesh residual, projection residual, performance, stability, SUS).

**Visualization concept.** Heightfield mesh from the DSM with disclosed tolerance; RGB draped on the shared grid; first-person + aerial navigation; raster-based height/slope/profile tools; layer toggles incl. uncertainty and terrain-only; tier/datum badges; no embellishment.

**Failure handling.** Tier downgrade with flags; explicit refusals for invalid georeferencing; CPU/tiling fallbacks; "not validated" when no reference; datum sanity checks.

**Differentiation.** Documented, validated two-layer composition with ground-masked DEM terrain; tiered calibration semantics; per-layer validation and anchor curves; uncertainty exposure; measurement-faithful visualization — versus prior art that stops at nDSM, uses zero-shot depth, or leaves the nDSM→DSM step undocumented.

**MVP.** Modes A/B; fine-tuned Small backbone (GAMUS + one multi-terrain LiDAR source); Copernicus/SRTM terrain layer with datum transform and ground masking; DSM/nDSM/slope COGs + report; harness incl. DFC19 protocol; faithful viewer with navigation and raster tools; standalone offline build.

**Advanced.** Anchors (ICESat-2/GCP) with held-out checking; uncertainty raster + AUSE; semantic head; cross-country and India assessments; long-tail loss and instance regularisation.

**Major risks.** Indian morphology transfer; steep-relief terrain error bounded by DEM; forest ground support; long-tail high-rises; training-data engineering time; packaging.

**Feasibility.** Buildable: Apache-licensed 24.8 M-parameter backbone with shipped training code; open multi-terrain LiDAR with DTMs; global DEMs; commodity visualization; MVP independent of priced or unavailable data.

---

## PHASE 5 COMPLETE

**1. Final Proposed Solution Name:** DepthWizard TL-CSM — Two-Layer Calibrated Surface Model.

**2. One-Sentence Definition:** A single-view RGB-to-DSM pipeline that fine-tunes a pretrained monocular depth backbone to predict metric height above ground, composes it with a datum-transformed, ground-masked coarse-DEM terrain layer optionally refined by sparse anchors, and delivers a geospatially valid DSM with declared calibration tier and uncertainty, rendered faithfully in a standalone 3D flythrough.

**3. Core Technical Insight:** The image is competent for object height above ground (given LiDAR-supervised adaptation); only external absolute sources are competent for terrain and datum, and only at low frequency — so calibrate each layer separately, compose explicitly, and never let one source claim what it cannot measure.

**4. Why This Addresses the Real Problem:** It neutralises the documented zero-shot nadir scale failure (GAP-01) by adaptation; it uses the coarse DEM for exactly what it can do and no more (GAP-02); it makes the undocumented nDSM→absolute-DSM step explicit and testable (GAP-03); it treats the vertical datum as first-order (GAP-11; 24–99 m in India); it exposes uncertainty (GAP-15); it degrades honestly where data are absent (Mode A, no DEM, India) instead of fabricating metres.

**5. Top 10 Differentiating Technical Contributions:** (1) explicit two-layer DSM composition; (2) ground-masked low-pass terrain from coarse DEM limiting canopy/roof double-counting; (3) tiered calibration semantics R/H/T/A with failure detection; (4) GSD-conditioned metric nDSM head on a PS-compliant open backbone; (5) multi-terrain LiDAR-derived training (CH/NZ/US + GAMUS); (6) end-to-end vertical-datum pipeline with tagged outputs; (7) sparse-anchor refinement with anchor/checkpoint partitioning and residual reporting (ICESat-2-capable for India); (8) per-pixel uncertainty + calibration-quality report; (9) per-layer, per-terrain validation protocol with calibration ablations and external DFC19 protocol; (10) measurement-faithful visualization with quantified mesh error and raster-based analysis.

**6. Top 10 Technical Risks:** Indian morphology transfer; steep-relief terrain error bounded by DEM; forest ground support; long-tail high-rise underestimation; sensor/radiometric shift to satellite MX; GSD mismatch at evaluation; ground-mask circularity; training-data engineering time; offline packaging of inference + viewer; hidden dataset footprint overlap.

**7. MVP Scope:** §24 MVP list.

**8. Advanced Scope:** §24 Advanced list.

**9. SIH Requirement Coverage:** all explicit requirements mapped in §18; Mode B absolute DSM satisfied at Level T/A with declared semantics; Mode A relative by design.

**10. Validation Strategy Summary:** spatially blocked, per-terrain LiDAR tests in CH/NZ/US with per-layer and composed metrics; calibration ablations; DFC19 external protocol; cross-country degradation; India ICESat-2-bounded assessment; Mode-A affine-invariant scoring; visualization/deployment tests (mesh residual, projection residual, performance percentiles, crash-free runs, clean-machine install, SUS).

**11. What We Must Prove Experimentally:** fine-tuned nDSM beats zero-shot across all four terrains on blocked splits; ground-masked terrain reduces forest/built-up bias vs raw DEM add-back; composed DSM error ≈ layer errors (no composition penalty); anchor residual decreases with N and the fit generalises to held-out anchors; uncertainty ranks errors (AUSE < random); mesh-vs-DSM residual within tolerance; standalone offline operation.

**12. What We Must NOT Claim:** hidden-surface reconstruction; metres without DEM/anchors or without GSD; sub-DEM terrain detail from the DEM; LiDAR-level accuracy; DEM as ground truth; unlimited generalization to India; a certified DTM; correctness on strongly off-nadir imagery; novelty of any individual standard component.

**13. What Phase 6 Must Architect:** module boundaries and data contracts for input detection, tiling/inference, calibration tiers, terrain layer, anchor module, composition, geospatial writer, validation harness; model training/data-preparation pipeline (datasets, splits, GSD bands, losses); DEM/geoid-grid provisioning for offline use; report schema; viewer engine selection against the §15 requirements, mesh/LOD strategy, raster-lookup design; packaging strategy for standalone offline deployment; experiment plan implementing §14; licence manifest.

READY FOR PHASE 6 — SYSTEM ARCHITECTURE
