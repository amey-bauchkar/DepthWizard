# DepthWizard — claims register

Every claim made about DepthWizard in the PPT, the pitch or a demo must be listed here, with the number, the
conditions it holds under, and the file or script that proves it. **If a claim is not in this file, do not say it.**
When a number changes, update it here first, then in the slides.

Last checked: 2026-09-30 (model `da-v2-small-ndsm` 2.0.0, fusion g = 1, f = 0.75).

## 1. Speed and hardware

| Claim | Number and conditions | Evidence |
|---|---|---|
| ~30 s per scene on a laptop | 29.3 s and 29.4 s for Teesta west, 2400 × 2400 px at 0.5 m (1.2 × 1.2 km), in the app server. Tile inference ~22 s, post-processing ~6 s. | Measured 2026-09-30 on an Intel Core 5 120U (10 cores), CPU-only PyTorch. Stage times are in each job's `job.json` (`stages_ms`). |
| No GPU needed | All numbers above are CPU-only. | Same run. |
| Models are loaded before the first job | Model load and warm-up (~2.5–5 s) run when the server starts, not on the first job. | `JobManager.warm_up_in_background`, `backend/main.py` lifespan. |
| Large scenes work, but slowly | 6000 × 6000 px at 0.6 m (~3.6 × 3.6 km): ~25 min on CPU, 361 tiles, 6.6 GB RAM peak. Measured before the post-processing speed-ups of 2026-09-30. | Earlier big-scene run; limits in `backend/config/settings.py` (`max_image_dim` 6400, `max_tiles` 400). |
| Speed-ups did not change results | Every output of a Teesta west job was identical before and after: raster values, preview pixels, heightfield bytes and JSON. | Job-to-job comparison on 2026-09-30. `tests/unit/test_previews.py` (colour ramp is byte-identical); `tests/unit/test_hazard_live.py`. |

Say "about 30 s for a 1.2 × 1.2 km scene on a laptop CPU". Do **not** say "any image in 45 s".

## 2. Height accuracy (measured, never assumed)

| Claim | Number and conditions | Evidence |
|---|---|---|
| Surface (DSM) better than the 30 m DEM (Zürich test tile, 0.6 m) | RMSE vs LiDAR 9.21 m → 5.16 m (−44 %) | `data/resolution/scores.json` (`ch_2682-1247@0.6`), `scripts/validate_resolution.py` |
| Ground (terrain) better than the 30 m DEM (Zürich, 0.6 m) | RMSE vs LiDAR 8.03 m → 2.90 m (−64 %) | Same |
| Other test tiles, 0.6 m | Emmental DSM 7.64 → 6.07 m, terrain 11.50 → 5.75 m. State College DSM 6.06 → 4.36 m, terrain 4.17 → 3.86 m. Las Cruces DSM 1.85 → 1.46 m, **terrain 0.99 → 1.35 m (worse)**. | Same |
| Works from 0.35 to 10 m pixel size | The DSM beats the DEM at every tested resolution (0.35, 0.6, 1, 2.5, 5, 10 m) on all 4 test tiles. The terrain beats the DEM except Las Cruces at 0.6–2.5 m. | Same (gain table: g = 1 chosen on validation tiles at every resolution) |
| India: checked against NASA ICESat-2 | 6 Sikkim scenes, 1114 laser checkpoints. Ground RMSE 7.98 m (DEM) → 7.28 m (−9 %). Surface RMSE 10.98 m → 10.75 m. **Teesta east ground gets worse: 3.28 → 3.99 m.** | `data/dem_trust_2.0.0/scores.json` (`india`, key `1.0,0.75`), `scripts/select_dem_trust.py`. Constant: `MEASURED_DEM_RMSE` in `backend/jobs/pipeline_b.py`. |
| Height-above-ground model (held-out regions) | v2: Swiss RMSE 3.83 m, US RMSE 3.26 m. Objects ≥ 2.5 m: ~5.4 m. | `models/da-v2-small-ndsm/2.0.0/model_card.json` (`validation`) |
| Building heights | Per-building median vs LiDAR (OSM footprints). Test tiles: Zürich 3.09 m RMSE, State College 3.42 m. Validation mean 2.46 m (was 3.10 m with dsm − terrain). | `scripts/validate_building_heights.py` (numbers in its header), `BUILDING_HEIGHT_SOURCE` in `backend/jobs/pipeline_b.py` |
| Tested on GAMUS (ISRO-recommended dataset) | 90 test tiles. RMSE 5.68 m, MAE 3.21 m, r 0.755 vs zero-shot with oracle scale 5.88 m / 4.17 m / 0.679. **Worse on object pixels (8.02 vs 7.22 m) and on the 10-tile validation split (5.93 vs 4.99 m).** Wins DC and NYC, loses PHL. | `data/gamus/results_test_gsd0.25.json`, `results_val_*.json`, `scripts/validate_gamus.py` |
| PNG / JPG (no coordinates) | Shape only, no sea-level heights. Zürich 0.5 m: Pearson r 0.852 vs the LiDAR DSM. | `docs/validation_mode_a.md`, `scripts/validate_mode_a.py` |

Note: `docs/validation_india.md` is an older run (ground 7.61 m, different settings). The current numbers are in
`data/dem_trust_2.0.0/scores.json`.

## 3. Inputs and outputs

| Claim | Evidence |
|---|---|
| Inputs: GeoTIFF, PNG/JPG, Cartosat / LISS-IV / PAN product zips; oversized images are downsampled, not rejected | `core/ingest/geotiff.py`, `core/ingest/ingest.py`, `core/ingest/isro.py` |
| DEMs: CartoDEM v3 R1 (ISRO/NRSC) when installed, Copernicus GLO-30, or a user-uploaded DEM | `core/calib/dem.py` (`select_dem`, `discover_dem`) |
| All heights converted to one sea-level reference (EGM2008); geoid vs ellipsoidal heights detected automatically | `core/calib/dem.py` (`resolve_auto_datum`), `core/geo/vertical.py` |
| Outputs: GeoTIFF (DSM, terrain, height above ground, slope, aspect, quality flags); buildings as GeoJSON/CSV; GIS package zip; PDF report; offline 3D HTML | `backend/api/routes.py`, `core/export/` |
| Every result has a quality tier (R / T / A / H) and quality flags | `core/calib/tier.py`, `flags.tif` |
| In-app validation against a LiDAR / reference raster or checkpoint CSV (RMSE, MAE, r) | `/api/jobs/{id}/validate`, `core/validate/` |

## 4. Disaster features (screening, not simulation)

| Feature | What it does | How it was checked |
|---|---|---|
| Flood | River-rise model (height above nearest drainage) or still water level. Depth per building, with a likely / possible range from the measured terrain error. | Two real floods (Teesta GLOF 2023, Sunkoshi 2024): 49–57 % of the "likely" area and 23–44 % of the "possible" area flooded; 94–97 % of the "unlikely" area stayed dry. `FLOOD_BAND_OBSERVED` in `core/disaster/flood.py`, `docs/hazard_validation.md`. |
| Live flood / slope slider | The map follows the slider in the browser; the numbers during a drag equal the full run's. | `tests/unit/test_hazard_live.py`. A live update takes ~0.1–0.4 s, a full run ~0.5 s (Teesta west, 2026-09-30). |
| Landslide hazard | IS 14496-2 factor rating. Optional rainfall trigger (Open-Meteo) and new-scar detection (Sentinel-2): both need internet. | `core/disaster/landslide.py` |
| Helicopter landing sites | Open-ground sites (not rooftops) with slope, obstacle and approach-path checks (FM 3-21.38 rules), each with a confidence score. | 19 real helipads (OSM, Maxar India/Nepal imagery): 9 of 19 found at some pad size; hit rate 26–37 %, 6–9× better than random. `docs/helipad_validation.md`. |
| Road access | Roads cut by flood or landslide; settlements cut off; rough population estimate (4.9 people per household). | `core/disaster/roads.py`, `core/disaster/settlements.py`, `HOUSEHOLD_SIZE` in `core/screening_params.py` |
| Before / after damage | Flags buildings that lost height between two dates. | Islahiye 2023 earthquake demo: precision ~71–88 %; recall **not measured**. `docs/change_screening.md` |
| Slope accessibility | Ground at or below a chosen slope, buildings excluded. | `core/disaster/accessibility.py` |
| Height confidence map | Per-pixel 80 % interval from 4 flipped model runs, calibrated on Swiss validation tiles (~2 min on CPU). | `core/inference/confidence.py`, `confidence_calibration.json` |
| One-click PDF report | Map plus flood, landslide, landing-site and road summaries. | `core/export/report.py` |

All hazard outputs are elevation-based screening. They are **not** hydraulic simulations or certified hazard maps; the app says so in every result.

## 5. Offline, data and security

| Claim | Conditions | Evidence |
|---|---|---|
| Core processing runs offline | Needs the area's DEM on disk (bundled Copernicus tiles for the demo areas, installed CartoDEM, or an uploaded DEM). | `assets/dem`, `core/calib/dem.py` |
| Needs internet (optional) | Bhuvan layers, Sentinel-2 scars, Open-Meteo rainfall, and fetching new footprints, roads or rivers. | `core/geo/bhuvan.py`, `core/geo/sentinel2.py`, `scripts/fetch_*.py` |
| Images never leave the machine | The server binds to 127.0.0.1; no image is uploaded anywhere. | `configs/default.yaml` (`server.host`) |
| Model integrity | SHA-256 check of the model weights. | `ml/registry/registry.py` (`verify_hash`) |
| Input limits | Upload size cap (600 MB), zip uncompressed cap (3000 MB), job-id validation. | `backend/config/settings.py`, `backend/jobs/manager.py` |

## 6. Engineering

| Claim | Evidence |
|---|---|
| 230 automated tests pass | `python -m pytest tests` (2026-09-30) |
| One-click Windows app (one-folder package, ~1 GB zip) with a built-in self-test | `packaging/launcher.py --selftest`, `scripts/package_win.py`. **The zip in `dist/` predates the 2026-09-30 changes: rebuild it before handing it out.** |
| 3D view in the browser: orbit, walk and fly-around modes, LoD-1 buildings; redraws only when something changes | `frontend/src/viewer.ts` |

## 7. Cost and licences

| Claim | Evidence / condition |
|---|---|
| No licence fees for the software | Open-source stack. Depth Anything V2 **Small** is Apache-2.0 (Base/Large are not; do not swap them in). |
| Fine-tuned model training data | swisstopo open data (SWISSIMAGE, swissSURFACE3D, swissALTI3D), USDA NAIP and USGS 3DEP (public domain). `models/da-v2-small-ndsm/2.0.0/model_card.json` |
| Demo imagery is **non-commercial** | Maxar Open Data (CC BY-NC 4.0). Footprints: Microsoft (ODbL). OpenStreetMap (ODbL). Credit them on the references slide. |

## 8. Known limitations (say these before a judge finds them)

- **No Indian training data.** Small rural houses are under-read. Teesta west: 120 of 306 buildings (39 %) read below 2.2 m are flagged "height not resolved" (a lower bound); the median of the resolved buildings is 2.8 m.
- **The India gain is modest:** ground error 7.98 → 7.28 m against ICESat-2, and Teesta east gets worse.
- **Flat arid ground can get worse** than the DEM (Las Cruces terrain 0.99 → 1.35 m).
- **GAMUS is mixed:** better overall on the test split, worse on object pixels and on the validation split.
- **PNG/JPG gives shape, not sea-level heights.** Absolute elevation needs a GeoTIFF plus a DEM.
- **Large scenes are slow on CPU** (minutes, not seconds).
- **Hazard maps are screening**, not simulations; damage-detection recall is not measured.
- **The model card's `known_limitations` still says "Switzerland only"**; v2 is trained on Swiss + US data (fix the card text).

## 9. Do not claim (removed from the slides, and why)

| Removed claim | Why |
|---|---|
| "Any image in under 45 s" / "under 60 s" | True only for ~1.2 × 1.2 km scenes; large scenes take minutes. |
| "60 FPS" | Never measured. |
| "Tiny 8 MB file" | Size never measured; depends on the scene. |
| "Flat rooftop landing zones" | Sites are on open ground, not rooftops. |
| "Verified by NASA" | NASA ICESat-2 data is used as our checkpoints; NASA verified nothing. |
| "Eliminates AI hallucinations" | Unprovable. Say "every number is read from georeferenced rasters, with a measured error". |
| "Replaces LiDAR surveys" | It is screening while LiDAR is unavailable, and less accurate than LiDAR. |
| "100 % offline" | Optional layers need internet; offline needs the area's DEM on disk. |
| "Passes all 205 tests" | Now 230; quote the current count. |
| "Saves crores" / "₹47.44 lakh crore portfolio" / "$20B market by 2030" | No source in this project. |
| Chart values "72 hours", "₹2.5 lakh", "8.00 m → 3.17 m", "99.9 % faster", "100 % cost reduction", "−60 %" | Not measured by us. Use section 2 numbers. |
| "Plugs directly into government command systems" / "exports to Bhuvan" | Outputs are standard GeoTIFF/GeoJSON; Bhuvan layers are only overlaid in the app. |
| "4-page emergency plan" | The report is a summary PDF; its page count depends on which screenings were run. |
| "Ready for immediate use by SDMA / NDRF" | Say "ready for a pilot". |
| OpenCV, Safetensors, PostgreSQL, React, Docker, AsyncIO, JSON Schema, "magic-byte validation" | Not used in the code. |

## How to re-check the numbers

```
python -m pytest tests
python scripts/validate_resolution.py
python scripts/select_dem_trust.py
python scripts/validate_building_heights.py
python scripts/validate_gamus.py --split test --gsd 0.25
python scripts/helipad_study.py
python scripts/validate_flood.py
```
