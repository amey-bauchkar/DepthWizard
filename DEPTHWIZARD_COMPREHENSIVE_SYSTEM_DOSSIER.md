# DepthWizard — System Dossier (as built)

**Problem Statement:** SIH 26175 — Single-View Height Estimation and 3D Flythrough · **Organisation:** ISRO, Department of Space · **Theme:** Disaster Management · **Category:** Software
**Status date:** 2026-09-26. Every number in this document is measured by a script in this repository, and the script is named next to the number. The research and design record is in `SIH26175-Phase1…10`. Where the build differs from that plan, this dossier describes the build.

---

## 1. Compliance with the problem statement

| PS requirement | Implementation | Where |
| :--- | :--- | :--- |
| Non-georeferenced RGB (PNG / JPG) → relative DSM | **Mode A**: PNG, JPEG and TIFF without CRS (8/16-bit, float) → Depth Anything V2 Small → percentile-normalised `rdsm.tif` (unitless 0–1, tier R, no CRS claimed). Large images are refined with native-resolution tiles. | `backend/jobs/pipeline.py`, `core/ingest/ingest.py` |
| Georeferenced RGB (GeoTIFF) → absolute metric DSM | **Mode B**: CRS + geotransform read, GSD computed, geographic inputs reprojected to local UTM; DSM in metres on EGM2008 (`dsm.tif`) + terrain, nDSM, slope, aspect, flags, input DEM on the job grid | `backend/jobs/pipeline_b.py`, `core/ingest/geotiff.py` |
| Pre-trained monocular depth backbone | Depth Anything V2 Small (ViT-S/14 + DPT, 24.8 M params, Apache-2.0), vendored code, SHA-256-verified weights, CPU or CUDA. **Fine-tuned** by us on swisstopo LiDAR to output height above ground in metres (`da-v2-small-ndsm`); the zero-shot model remains for Mode A and as the fallback | `core/inference/predictor.py`, `ml/registry/`, `notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb` |
| Scale calibration with low-res DEM or few GCPs | Copernicus GLO-30 (bundled for the demo areas) or a user DEM (EGM2008 / EGM96 / ellipsoidal) → **DEM-preserving tiled detail fusion**: DEM ≥ 30 m + fine-tuned model heights < 30 m (tier T); anchors CSV → detail gain / significant offset (tier A); no DEM → metric heights only (tier H) | `core/calib/fusion.py`, `core/calib/terrain.py`, `core/calib/anchors.py` |
| DSM in a standard geospatial format | Float32 GeoTIFF, LZW, CRS + transform + nodata + provenance tags (`UNITS`, `METRIC`, `CALIBRATION_TIER`, `VERTICAL_CRS`, `MODEL`, `MODEL_SHA256`, `METHOD`) | `core/geo/raster_io.py` |
| Texture projected on a 3D mesh in a rendering engine | Three.js heightfield (regular grid or adaptive RTIN), aerial texture drape with anisotropic filtering, anti-smear facade shader, heat-map + contour and blueprint shaders, LoD-1 extruded buildings | `frontend/src/viewer.ts`, `lod1.ts`, `martini.ts` |
| First-person navigation, heights and slopes from any viewpoint | Orbit, 🚶 walk (pointer lock, WASD, sprint, terrain-following at 1.7 m eye height, LoD-1 wall collision), 🎬 drone orbit, preset views; heading-aware north arrow, live scale bar. Clicks are sampled **server-side** from the GeoTIFFs (elevation, terrain, nDSM, slope, aspect, flags); two-point measure gives distance, ΔZ and grade | `frontend/src/viewer.ts`, `backend/jobs/query.py` |
| Upload imagery, visualise, validate against reference | Browser UI: upload or one-click demo, optional DEM / anchors; validation panel for any reference raster (DSM / DTM / nDSM, any vertical CRS) | `frontend/`, `core/validate/harness.py` |
| Standalone deployment | Single local server (FastAPI serves API + built viewer), `run_server.py` / `start_depthwizard.bat` / `start_depthwizard.sh`, `scripts/doctor.py` environment check, fully offline after setup. **Every result also exports as a standalone offline 3D scene**: one HTML file (viewer + heightfields + texture + LoD-1 + provenance) that runs by double-click with no server, Python or internet — the interactive session needs no live backend | repository root, `core/export/scene.py`, `frontend/src/standalone.ts` |

## 2. Mode B method (TL-CSM v1.0)

1. **Ingest.** Read the RGB bands (non-uint8 data is percentile-stretched) and the CRS. The GSD comes from the transform. A geographic CRS is reprojected to the local UTM zone before any metric step (Phase 8 C-4).
2. **Whole-image inference** (Depth Anything V2 Small, 518 px short side) gives the relative structure (`relative.tif`, tier R). A morphological ground filter on it gives the ground mask.
3. **DEM.** Bundled Copernicus GLO-30 tiles are found by filename. A user DEM can be uploaded instead. The DEM is resampled bilinearly to the job grid and converted to EGM2008 with the **C-1 datum guard**: offline PROJ, required geoid grids present, "ballpark" pipelines refused, known-point self-test.
4. **Tiled inference.** The image is resampled to ~0.5 m and cut into overlapping 518 px tiles (25 % overlap). At most 64 tiles are used; the inference GSD is coarsened automatically to stay within that. Each tile is predicted independently by the **fine-tuned metric nDSM model** when installed (output: metres above ground), otherwise by the zero-shot model.
5. **Detail fusion.** *Fine-tuned model:* tiles are feather-blended into an nDSM in metres; DSM = DEM + high-pass(nDSM) below one DEM posting (gain 1, since the scale is learned); terrain = DSM − nDSM. This composition was chosen over "DEM terrain + nDSM" on six validation-region tiles (DSM 5.71 vs 6.42 m, terrain 4.10 vs 5.69 m; `docs/metric_composition_selection.json`). *Zero-shot fallback:* for each tile, the least-squares gain between the model's band-pass and the DEM's band-pass in 30–120 m (one to four DEM postings) gives metres per relative unit. Negative gains are clamped to zero. A spectral guard caps tiles whose fine-scale content is noise-like. The gain multiplies the model's high-pass (< 30 m). Tiles are feather-blended and the result is re-high-passed on the job grid. **DSM = DEM + detail**, so at the DEM's own resolution the DSM equals the DEM.
6. **Terrain.** Fine-tuned model: `terrain = DSM − predicted nDSM`. Zero-shot fallback: `terrain = min(ground-weighted normalized-convolution DEM reconstruction [+ tier-A offset], morphological ground of the DSM, DSM)`. In both cases **nDSM = DSM − terrain** (≥ 0), so `dsm = terrain + ndsm` holds exactly.
7. **Anchors (tier A).** Ground anchors give a robust median terrain offset with blunder flags and a hold-out (Phase 8 C-3). All anchors fit `z = DEM + c + k·detail` by Tukey IRLS. The offset `c` is kept only if it is significant (|c| > 2 SE), because a point anchor cannot separate a datum error from sub-cell mixing. The fit is accepted only if its **leave-one-out** RMSE beats tier T.
8. **Quality and tier** (`core/calib/tier.py`). Tier T is at most LIMITED: the model's accuracy was measured on other regions, not on the user's scene. Without a DEM (or a safe datum) the fine-tuned model gives **tier H**: metric heights above ground, no absolute elevation. Without the fine-tuned model, missing DEM / unsafe datum / a DEM–terrain offset beyond the sanity threshold give tier R with WARNING / INVALID and the reason.
9. **Derivatives.** Slope and aspect (Horn 3×3), hillshade, a per-pixel flag raster (border, raw-DEM fallback, DEM void, nodata, no detail, low ground support), viewer heightfields (area-average downsampling, residual reported), and LoD-1 blocks (spectral + nDSM rules, watershed instance separation, RDP simplification; for visualisation only).

**Why:** see `docs/validation_results.md`. Scaling a zero-shot object layer against the DEM residual, and replacing the DEM by terrain + objects (the earlier design), was **worse than the raw DEM** against LiDAR on every tile (e.g. Zürich 2 m DSM RMSE 10.77 m vs 9.00 m). Keeping the DEM and adding only calibrated sub-posting detail is never worse and measurably better. Tiling matters: one down-scaled pass of the model correlated 0.06 with LiDAR building detail on the 0.5 m Zürich tile; native-resolution tiles reached 0.50–0.54.

## 3. Mode A method

Whole-image prediction → robust 1–99 % normalisation → `rdsm.tif` (tier R, `METRIC=false`, no CRS). For images larger than ~1.3× the model input, the same fusion routine adds native-resolution tile detail to the whole-image prediction, with the whole-image prediction as the base. That keeps a single consistent low-frequency frame and sharpens detail. The output is unitless by construction and every label says so.

## 4. Measured accuracy

### 4.1 Fine-tuned model on held-out regions (height above ground)

Colab T4, 8000 iterations; 108 training / 16 validation / 20 test swisstopo 1 km tiles, split **by region**. The DepthWizard test tiles and everything within 5 km of them were never downloaded. Source: `models/da-v2-small-ndsm/1.0.0/training_report.json`.

| Test regions: Basel, Lugano, Davos, Thurgau, Jura | RMSE | MAE | ME | r | buildings / trees (≥ 2.5 m) RMSE |
|---|---|---|---|---|---|
| **Fine-tuned nDSM model** | **3.86 m** | **1.90 m** | −0.10 m | **0.875** | **5.52 m** |
| Zero-shot + oracle per-tile affine fit (its best case) | 6.74 m | 4.85 m | 0.00 m | 0.528 | 9.45 m |

### 4.2 End-to-end DSM / terrain on the DepthWizard test tiles

Source: `scripts/validate_demo.py` → `docs/validation_results.md`. References: swissSURFACE3D (DSM) / swissALTI3D (DTM) 0.5 m airborne LiDAR, area-averaged to the job grid, LN02 → EGM2008. RMSE in metres, no anchors (tier T).

| Tile | Layer | Copernicus alone | Zero-shot fusion | **Fine-tuned model** |
|---|---|---|---|---|
| Zürich urban 0.5 m | DSM | 9.25 | 8.91 | **5.53** (−40 %) |
| Zürich urban 0.5 m | terrain vs DTM | 8.00 | 6.11 | **3.17** (−60 %) |
| Zürich urban 2 m | DSM | 9.00 | 8.89 | **6.34** (−30 %) |
| Zürich urban 2 m | terrain vs DTM | 7.97 | 6.23 | **3.25** (−59 %) |
| Emmental rural / forest 2 m | DSM | 7.45 | 7.18 | **6.41** (−14 %) |
| Emmental rural / forest 2 m | terrain vs DTM | 11.49 | 10.35 | **5.30** (−54 %) |

Honest reading:

* The remaining DSM error is dominated by the 30 m DEM at the ≥ 30 m scale and by building edges.
* Training and testing are Swiss only; accuracy on Indian imagery or other sensors is unmeasured.
* Simulated anchors (sampled from the same LiDAR, with a 15 m exclusion radius) add nothing measurable with the fine-tuned model:
  * the leave-one-out test rejected the anchor fit on 2 of 3 tiles;
  * on the 0.5 m tile it was accepted but worsened the terrain from 3.17 to 4.04 m.

### 4.3 India: Sikkim vs NASA ICESat-2 (independent satellite laser checkpoints)

Six 1.2 km scenes of Maxar WorldView 0.5 m imagery (Maxar Open Data Program, CC BY-NC 4.0) are scored against 1,114 ICESat-2 20 m segments. The heights are converted from ellipsoidal to EGM2008 with the C-1 datum guard. No Indian data was used for training or tuning. Pooled RMSE in metres (`docs/validation_india.md`):

| vs ICESat-2 | Copernicus alone | Zero-shot | Fine-tuned |
|---|---|---|---|
| Terrain vs ground | 10.51 | 10.25 | **8.54** |
| DSM vs top of surface | 11.81 | 11.03 | **11.10** |
| Height above ground | – | 11.38 | **9.20** |

Per site, the fine-tuned model improves height above ground and DSM on 5 of 6 sites and terrain on 4 of 6. It is worse on the two forested-valley sites, where Copernicus already lies within ~2 m of the ground under forest.

Two fixes were tested and rejected because they did not improve track-wise cross-validation:
- a DEM-only estimate of how much object height the DEM contains;
- an ICESat-2-anchored estimate of the same quantity.

The limitation is documented rather than tuned away.

The in-app validation panel reproduces these comparisons for any job. It shows the product and the *input DEM alone* on the same mask, and it reports co-registration (grid search with sub-pixel refinement, ≤ 4 px), mask accounting, datum handling, slope / object / height strata, an oracle affine diagnostic and a residual map.

## 5. Visualisation and interaction

* **Mesh:** heightfield from the DSM, terrain, nDSM or relative layer at up to 768² vertices. Downsampling is area-averaged and the residual RMSE is shown. Optional adaptive RTIN (Martini): the tolerance is in metres, or in scene units for relative layers. Display-only smoothing and RGB-guided edge snapping never change the GeoTIFFs.
* **LoD-1 city:** extruded blocks on the terrain layer with roof texture from the aerial image and height-classed facades. Blocks are offered only over absolute layers (terrain / DSM), where their base elevations are meaningful.
* **Navigation:** orbit (damped), preset nadir / oblique / horizon views, drone orbit, first-person walk with terrain following and wall collision.
* **HUD:** state, tier + datum, quality, vertical exaggeration ("true scale" at 1.0×), z-range, live scale bar measured at the view centre, north arrow that rotates with the camera.
* **Truthfulness:** every number shown comes from the server rasters, never from the mesh. Point readings carry the measured typical error (±, with its source), and an input check card states the expected tier and accuracy before a job runs.
* **Exports:** the offline 3D scene (single HTML; readings are cell means of the full-resolution rasters, not the display mesh) and a GIS package (COG rasters with QGIS styles, buildings GeoPackage with an embedded style, GeoJSON / CSV, textured GLB validated with the Khronos glTF validator, STAC 1.0 item, provenance, README). The GeoPackage and GLB writers have no dependencies (`core/export/gpkg.py`, `glb.py`).
* **Before / after change screening** (`core/change/detect.py`): two results of the same place are reprojected to one grid and co-registered by phase correlation. The difference of the nDSMs is thresholded at 3 × the pair's own robust noise (NMAD), with 3 m tolerance to building lean. Buildings are flagged as major height loss (≥ half the height gone), loss or gain. The Islahiye (Türkiye, 2023 earthquake) demo is described in `docs/change_screening.md`: 37 of 1,432 buildings flagged; a random audit found 17 of 24 collapsed, 3–4 uncertain and 3–4 standing, so precision is about 71–88 %. Recall is not measured.
* **Building outlines** (`core/terrain/footprints.py`): open footprints (bundled Microsoft ML footprints for the demo areas, or an uploaded GeoJSON), co-registered to the image by FFT; heights from the nDSM. Detection with an object filter (`building_filter.py`, leave-one-scene-out validated) is the fallback. Measured in `docs/building_detection_validation.md`: the old detector's rural Sikkim blocks were 97 % trees / rock; building heights on Zürich have RMSE 3.3 m vs LiDAR; small rural houses in Sikkim are read too low.

## 6. Engineering

* **Jobs:** filesystem state (`data/jobs/<id>/job.json`), one worker thread, stage timings recorded, jobs interrupted by a restart are marked FAILED (never resumed silently). Structured JSON logs per job.
* **Configuration:** `configs/default.yaml` → `DW_CONFIG` file → `DW_<SECTION>_<KEY>` environment variables. The config hash is recorded per job.
* **Errors:** a stable error taxonomy (`backend/errors.py`) with user-facing messages. Upload types are checked by magic bytes, not extension.
* **Tests:** `python -m pytest` covers unit tests (ingest, preprocessing / registry / rDSM, calibration and fusion, LoD-1, guided filter, training-scaffold modules, metric-model composition and tier H with a deterministic stub), API tests (Mode A, Mode B end-to-end with synthetic GeoTIFFs, validation, anchors / tier A, no-DEM honesty, TIFF input, delete), Phase 8 geodetic regressions, and real-model runs on the demo JPEG and GeoTIFF.
* **Performance** (CPU, measured in the UI): Zürich 2 m GeoTIFF ≈ 20–30 s end to end (25 inference tiles); Zürich 0.5 m 2000 × 2000 ≈ 45–55 s. Fine-tuning: ≈ 2–3 h on a free Colab T4, including data preparation.

## 7. Known limitations and next steps

1. **More / broader training.** Validation RMSE was still falling at 8000 iterations (4.92 → 4.25 m); a longer run, a larger backbone (ViT-B, CC-BY-NC) and non-Swiss data (Indian imagery with an independent reference) are the next accuracy steps.
2. **Indian validation sites.** Evaluate on Cartosat or other Indian imagery against an independent reference.
3. **Learned building segmentation** for areas without footprint data (the rule-based fallback reaches only 11–16 % area precision in rural forested hills), and **Indian building heights**: the model under-reads small rural houses (66 % of known buildings on Teesta east read below 2 m).
5. **Recall of the change screening** against an official damage grading (Copernicus EMS / UNOSAT) for the Islahiye demo.
4. **Better DEM products** such as FABDEM or NASADEM as selectable baselines. They can already be uploaded as a user DEM with their vertical CRS.
