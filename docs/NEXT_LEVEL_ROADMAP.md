# DepthWizard — Next-Level Roadmap (review + recommendations)

Date: 2026-09-27. Scope: a report only, with nothing implemented. Inputs:
- the current codebase and its measured results (`docs/validation_results.md`, `models/da-v2-small-ndsm/1.0.0/training_report.json`);
- the team's external research note *"DepthWizard: Technical Research and Development Roadmap"*, written before fine-tuning;
- the SIH 26175 problem statement and the ISRO audience.

Items marked **[verify]** are facts about external data or licences that must be checked before relying on them.

---

## 1. Where the project stands today

| Area | Status |
|---|---|
| Mode A (PNG/JPG/TIFF → relative DSM) | Working, with tiled refinement for large images |
| Mode B (GeoTIFF → metric DSM / terrain / nDSM, EGM2008) | Working. Uses the **fine-tuned metric nDSM model** + Copernicus DEM |
| Fine-tuned model | DA-V2 Small, 8k iterations on Colab T4. On held-out Swiss regions: nDSM RMSE **3.86 m** vs 6.74 m for oracle-scaled zero-shot, r 0.875 |
| End-to-end accuracy (held-out test tiles) | DSM −14…−40 % and terrain −54…−60 % RMSE vs the raw Copernicus DEM |
| 3D viewer, measurement, in-app validation | Working (orbit / walk / drone, LoD-1, server-sampled measurement, DEM-baseline comparison) |
| **Indian data** | **None yet.** All training and testing is Swiss. **This is the biggest gap for SIH / ISRO judging** |

---

## 2. Review of the research note

The note is careful and mostly right. It was written *before* fine-tuning, so several of its headline concerns have since been addressed.

### 2.1 Points now resolved

| Research-note concern | Status now |
|---|---|
| "The intended metric nDSM head has not been trained" | **Resolved.** Trained, installed, and measured on region-blocked held-out data. |
| "Adding predicted nDSM to a DSM-like DEM (Copernicus) risks double-counting object height" | **Resolved by design.** DSM = DEM + *high-pass*(nDSM). The DEM's ≥ 30 m content (which already contains the smeared objects) is kept once, only sub-30 m model structure is added, and terrain = DSM − nDSM. The alternative "DEM terrain + full nDSM" was measured and lost (validation tiles: DSM 5.71 vs 6.42 m). |
| "A global affine fit to a 30 m DEM cannot establish building heights" | Agreed, and this is why the zero-shot path is now only a fallback. |
| "Random patch splits leak" | Training used **region-blocked** splits, with the demo test tiles plus a 5 km buffer excluded. |
| "72 tests show software behaviour, not accuracy" | Agreed. Accuracy now lives in reproducible scripts (`validate_demo.py`, `select_metric_composition.py`) with LiDAR references. |
| Fabricated or unsupported accuracy claims | Removed earlier (the old "1.84 m RMSE" dossier claim and the fake metric checkpoint). |

### 2.2 Points still valid (adopt)

1. **Uncertainty is missing.** Every height is shown as a single number. The note's "no uncalibrated confidence %" rule is right.
2. **Anchor semantics are mixed.** The anchor fit combines ground and roof points in one model. The note's rule is correct: ground GCPs constrain datum/terrain only, and roof + local ground pairs constrain height scale. Measured symptom: anchors did not help the fine-tuned model and slightly hurt terrain on one tile.
3. **The RGB-guided edge snapping** (viewer only) can imprint road markings or shadows as relief. Keep it display-only, add an "off" toggle, and never use it in measured rasters. *(Currently true: it is display-only.)*
4. **Building-level outputs beat prettier façades.** A per-building table (median / p90 height, ground, volume, interval) is the highest-value urban feature.
5. **An input "preflight" / geospatial integrity inspector** (CRS, GSD vs model training range, vertical datum state, acquisition date, DEM overlap, off-nadir warning) is cheap and wins trust.
6. **Exports:** Cloud-Optimized GeoTIFF, GeoPackage buildings, GLB. These are standard and quickly demonstrable in QGIS.
7. **Tier G** ("georeferenced but unscaled") and **Tier V** ("validated on this scene") are useful extra labels.
8. **Honest disaster claims:** a water-level slider is inundation *screening*, not hydraulic flood modelling.

### 2.3 Points where we disagree or would refine

| Note says | Our view |
|---|---|
| "One global calibration per scene, never independent per-tile scale" | Per-tile scaling was measured and was safe (it never beat or lost to the DEM badly, and a spectral guard is included). It is now irrelevant: the metric model needs no scale fit. |
| "Semantic head + heteroscedastic head + ensemble" as Tier-1 | The semantic head and uncertainty are worth it. An ensemble doubles CPU inference time and should be the last option. |
| Heavy emphasis on RPC / raw-satellite ingestion | Valuable for real ISRO data (Cartosat L1 products ship with RPCs), but it is a big job. Do it after the Indian orthoimagery path works. |
| "Indian demo = CartoDEM terrain only" | We can do better (section 3): weak Indian height labels exist, and the team can collect its own ground truth. |

---

## 3. Indian data plan (the critical path for SIH)

The judges will ask **"does it work on India?"** Today the honest answer is "unmeasured". The goal is to turn that into measured Indian numbers, even if coarse.

### 3.1 Indian sources to use

| Need | Source | Notes |
|---|---|---|
| Indian terrain DEM | **CartoDEM v3 R1 (30 m, Cartosat-1 stereo)** from Bhuvan / NRSC Open Data Archive | Official ISRO product: use it as the *default DEM inside India* (replacing Copernicus there) and say so. Vertical reference must be verified **[verify: EGM96 vs EGM2008 per product README]** |
| High-res Indian imagery | **Bhoonidhi (NRSC)**: Cartosat-2/2E/3 PAN/MX (0.3–1 m) | Mostly priced / restricted; free archive categories vary **[verify]**. For an ISRO-facing SIH team, **formally request a few sample scenes via the SIH / nodal-centre contacts**; this is the single most valuable ask |
| Free medium-res Indian imagery | Resourcesat LISS-IV (5.8 m), Sentinel-2 (10 m) | Too coarse for buildings. Useful for terrain / landslide / flood context and a "GSD out of range" demo |
| Building footprints (India) | **Google Open Buildings v3** (polygons, South Asia), OSM | Footprints for the building-intelligence layer |
| **Building heights (India, weak labels)** | **Google Open Buildings 2.5D Temporal** (building presence + height rasters, ~4 m, 2016–2023, covers India) **[verify coverage/licence]**; GHS-BUILT-H (100 m); Global Building Height Map | Not ground truth, but city-scale sanity checks and *weak-label* domain adaptation for Indian morphology |
| Independent Indian elevation checkpoints | **ICESat-2 ATL08** (terrain + canopy heights along tracks) and **GEDI L2A** (canopy RH98, ≤ 51.6° N, so all of India) | Free, global, independent of the DEM. The best way to report *measured* Indian terrain / canopy error without LiDAR |
| Indian building heights (team-collected) | **Your own survey**: 30–60 buildings on campus or in the city, measured with a laser distance meter, clinometer, or a drone's barometric/RTK altitude; plus OSM `building:levels` × ~3 m | Cheap and fast, and ASPRS asks for ≥ 30 checkpoints. "We measured 40 Indian buildings ourselves" is a strong judge story |
| Survey of India products | SoI open-series maps, and SVAMITVA drone orthophotos (≈ 5 cm, villages) **[verify public access]** | If obtainable, drone orthophotos with SoI heights would be an excellent Indian benchmark |

### 3.2 Indian demo AOIs (pick 3–4 covering the PS terrain classes)

| AOI | Why | Class |
|---|---|---|
| **Wayanad (Mundakkai–Chooralmala), Kerala** | 30 Jul 2024 debris-flow disaster; hilly, forested, before/after imagery exists | Hilly / forest / disaster |
| **Joshimath, Uttarakhand** | 2023 land subsidence; Himalayan town on slopes | Mountain / disaster |
| **Mumbai (Dharavi + BKC) or Bengaluru** | Dense informal low-rise next to high-rise: the hardest urban case | Dense urban |
| **Chennai / Assam (Brahmaputra floodplain)** | Flood-prone flat terrain for inundation screening | Flat / flood |
| **Your college campus** | Where your self-surveyed ground truth is | Validation |

### 3.3 Making the model work on India (domain adaptation, no Indian LiDAR needed)

1. **Train on more than Switzerland:** add USGS 3DEP LiDAR + NAIP (0.6 m, varied US morphology) and LINZ New Zealand. Diversity of roofs, climate and density matters more than more Swiss tiles.
2. **Sensor/GSD augmentation to Cartosat-like data:**
   - 0.5–1.0 m GSD jitter;
   - panchromatic-sharpened colour statistics;
   - haze and illumination changes;
   - JPEG artefacts.
3. **Weak supervision on India:** a small loss term against Open Buildings 2.5D heights (downsampled, low weight), plus ICESat-2 canopy/terrain points as sparse metric constraints. This is the note's "Indian domain adaptation without Indian LiDAR" research item, and it is feasible.
4. **Self-training:** run the model on unlabeled Indian scenes, keep high-confidence (low-uncertainty) pixels as pseudo-labels, and fine-tune briefly.
5. **Report per-domain numbers:** Swiss held-out, US held-out, and Indian checkpoints (ICESat-2 + self-surveyed buildings).

**Deliverable claim for SIH:** "Validated against LiDAR on unseen Swiss and US regions; on India, measured against ICESat-2 tracks and N self-surveyed buildings: X m."

---

## 4. Model improvements (ordered by value per effort)

| # | Improvement | Why | Effort | Notes |
|---|---|---|---|---|
| M1 | **Longer training + more data (US 3DEP/NAIP, NZ)** | Val RMSE was still falling at 8k iterations (4.92 → 4.25 m); geographic diversity is needed for India | Low–Med | Same notebook; add data sources; 20–30k iterations on an L4/A100 or several T4 sessions |
| M2 | **Uncertainty head** (Gaussian NLL, predicts σ per pixel) + 4× flip/rotate TTA | Enables intervals ("18 m, 90 %: 13–24 m") | Med | Calibrate the intervals on held-out regions; report coverage |
| M3 | **Semantic head** (ground / building / tree / water) | Better LoD-1 footprints, tree-vs-building separation, and a ground mask for terrain | Med | Labels from swissTLM3D / OSM / Open Buildings; the architecture scaffold already exists in `ml/train/model.py` |
| M4 | **Building-weighted + edge loss** | The full-image loss is dominated by ground; buildings/trees RMSE (5.5 m) is the weak spot | Low | Weight pixels with nDSM > 2.5 m |
| M5 | **GSD conditioning** | Makes 0.3 m, 0.5 m and 1 m inputs behave consistently | Med | Start with canonical resampling (already done at 0.5 m); add an explicit GSD embedding only if an ablation shows a gain |
| M6 | **Test-time speed**: ONNX Runtime / FP16, tile batching | CPU run is now ~30 s for 1 km² at 2 m | Low–Med | Useful for a live demo on judges' laptops |
| M7 | Larger backbone (DA-V2 Base) | Likely more accuracy | Med | Base/Large weights are **CC-BY-NC**: fine for a hackathon demo, but flag the licence; keep Small as the distributable default |

---

## 5. Feature ideas (with our creative additions)

### Tier 1: highest judge impact, reasonable effort

1. **India mode.**
   - Auto-detect an Indian AOI and switch the DEM to CartoDEM.
   - Show Indian districts / place names.
   - Add an ISRO product-provenance card (sensor, date, GSD).
   - Add Hindi (+ one regional-language) UI strings.
2. **Uncertainty everywhere.**
   - Per-pixel σ layer, and interval bands in the point readout and measurements.
   - An "uncertainty" 3D shader (colour by σ) and quality masks.
   - Values rounded to their uncertainty.
3. **Building Intelligence panel.** Click a building to see:
   - footprint (Open Buildings / OSM / predicted);
   - median and p90 height ± interval;
   - ground elevation and approximate floors (range only);
   - volume.

   The whole city exports as a GeoPackage / CSV table. Add filters such as "show buildings taller than 30 m".
4. **Disaster toolkit (screening, clearly labelled):**
   - **Inundation screening:** a water-level slider over the terrain layer shows flooded buildings, their count, and exposed population (with a WorldPop / Census raster **[verify licence]**).
   - **Landslide / slope hazard:** slope > 30° mask with the drainage direction; the Wayanad / Joshimath demo.
   - **Before/after change:** the same AOI on two dates → ΔnDSM with significance from the uncertainty layer (collapsed buildings, debris deposits, cut/fill volume in m³).
   - **Line of sight / viewshed:** from a clicked point (relay tower or observation post). Useful for disaster communications and the PS's reconnaissance angle.
   - **Evacuation-route profile:** draw a line and get the elevation profile, max slope, and whether it goes below the flood level.
5. **Geospatial preflight card** before processing:
   - CRS, GSD vs the model's training range, vertical datum state (known / assumed / unknown), DEM overlap %, acquisition date;
   - off-nadir / RPC warnings, with a refusal or downgrade when a metric claim is not supported.
6. **Standard exports:** COG DSM/DTM/nDSM/σ, GeoPackage buildings, GLB city model (opens in Blender or Windows 3D Viewer), plus a one-click **PDF / HTML "Scene Report"** (maps, metrics, provenance), which gives judges a tangible artefact.

### Tier 2: strong differentiation

7. **Sparse-truth calibration studio.**
   - Import ICESat-2 / GEDI tracks over the scene, or type measured building heights.
   - DepthWizard shows the residuals, then fits a *separate* ground-offset and object-scale model (following the note's anchor rules).
   - A **calibration-support map** shows where the fit is trustworthy.
8. **Cesium / 3D Tiles output**, so the city streams in any web GIS, and a QGIS plugin (drag-and-drop a GeoTIFF, get DSM layers back).
9. **Shadow-based height check (creative, physics-based).** Sun elevation from image metadata (or estimated), plus shadow length from a shadow mask, gives a building height independent of the neural net. Use it as:
   - a sanity check per building;
   - an extra weak label on Indian imagery.

   It is a classic remote-sensing method and very persuasive to ISRO scientists.
10. **Multi-sensor fusion option.** If a Cartosat stereo pair or a Sentinel-1 InSAR DEM is available, fuse it as a better "terrain" layer. The architecture already accepts any user DEM.
11. **Batch / area mode:** process a folder or a larger AOI as tiles with a job queue, resume, and a mosaic, for district-scale disaster response.

### Tier 3: research (mention in the pitch as the roadmap)

12. **Indian RGB–LiDAR benchmark request** to ISRO/NRSC (10–20 small sites). This is the most valuable scientific asset.
13. Roof-type / roof-plane estimation at < 0.5 m.
14. Joint canopy-height product for forest-carbon applications (GEDI-calibrated).
15. Off-nadir correction using RPC + footprints.

### Do not build (agree with the note)

- A hydraulic flood simulator.
- Automatic earthquake-damage verdicts.
- Floor counts stated as fact.
- Generated façades presented as real.
- "Confidence %" without calibration.

---

## 6. Validation upgrades

1. **An evidence pack** that one command regenerates:
   - fixed split files with tile IDs and hashes;
   - baselines (Copernicus / CartoDEM only, zero-shot, fine-tuned);
   - per-class and per-slope metrics;
   - residual COGs;
   - a runtime profile.
2. **Building-level metrics** (matched-building median-height MAE / RMSE, bias by height class) next to pixel metrics.
3. **Uncertainty metrics:** interval coverage (50 / 80 / 90 %) and sparsification (AUSE).
4. **Indian checkpoints:** ICESat-2 ATL08 terrain / canopy RMSE and self-surveyed building-height MAE.
5. **Strict separation:** anchors ≠ checkpoints ≠ training tiles; the tool refuses to validate against the calibration DEM.
6. **Bootstrap confidence intervals** on every reported RMSE; with only two test tiles, the uncertainty of the metric itself matters.

---

## 7. Engineering / product polish

- **Distribution:**
  - one installer (PyInstaller or a portable Python + prebuilt frontend) including the fine-tuned weights;
  - a first-run check;
  - fully offline (no web fonts / CDN).
- **GPU auto-use** with an FP16 path; show the device and timings in the UI.
- **Job queue** with cancel / resume; the current design has one worker and no cancel.
- **Model versioning in the UI:** show the model card (training data, test RMSE, licence) behind an "i" icon.
- **Demo resilience:** cached precomputed results for the demo AOIs so a live demo never waits 30 s.
- **Accessibility / localisation:** English + Hindi; high-contrast mode.

---

## 8. Suggested 3-week plan (SIH-oriented)

| Week | Goals |
|---|---|
| **1: India data + model v2** | Download CartoDEM for 3 Indian AOIs. Obtain imagery (a Bhoonidhi request, plus whatever is legally usable). Download ICESat-2 ATL08 and Open Buildings footprints / 2.5D heights for those AOIs. Survey 30–40 campus buildings. Retrain with Swiss + US 3DEP/NAIP + building-weighted loss + an uncertainty head (Colab) |
| **2: Indian validation + key features** | Default to CartoDEM in India. Add the preflight card, uncertainty layer and intervals. Build the Building Intelligence panel and GeoPackage/COG/GLB export. Measure Indian numbers (ICESat-2, self-survey, Open Buildings consistency) |
| **3: Disaster story + polish** | Inundation screening and the slope-hazard demo on Wayanad / Assam. Before/after change on Wayanad 2024. Scene-report PDF, installer, cached demo, pitch deck updated with measured Swiss + US + Indian numbers |

---

## 9. What to say to ISRO judges (bounded, defensible claims)

- **Mode A:** "Relative surface structure from a non-georeferenced image, clearly unitless."
- **Accuracy:** "A fine-tuned single-image height model that predicts height above ground in metres. It was validated against airborne LiDAR on regions it never saw: 3.9 m RMSE for height, and a 30–60 % error reduction versus the free 30 m DEM."
- **Geodesy:** "A datum-safe fusion with national DEMs (CartoDEM in India): terrain, DSM and nDSM in EGM2008, with geoid grids verified offline."
- **India:** "Measured on India with ICESat-2 tracks and N surveyed buildings." *(Once done; never before.)*
- **Honesty and 3D:** "Every number in the 3D view comes from the raster and carries its tier, datum and (after M2) an uncertainty interval."
- **Limits:** "Not a replacement for stereo / LiDAR. It is a fast, offline, first-look product for disaster response when those are unavailable."

---

## 10. Priority summary

1. **Indian data and Indian measured numbers** (section 3). Without them the SIH story has a hole.
2. **Model v2**: more diverse training, a building-weighted loss, and an uncertainty head.
3. **Uncertainty + preflight + building panel.** These carry the trust and usefulness story.
4. **Disaster toolkit** on real Indian events (Wayanad, Joshimath, Assam), with honest "screening" labels.
5. **Exports, installer, scene report, cached demo.** Polish for judging day.
