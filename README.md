# DepthWizard — Single-View Height Estimation and 3D Flythrough

**Problem Statement:** SIH 26175 · **Organisation:** Indian Space Research Organisation (ISRO), Department of Space · **Theme:** Disaster Management · **Category:** Software

DepthWizard turns a single optical RGB image into an elevation model and an interactive 3D scene:

* **Non-georeferenced PNG / JPG / plain TIFF → Mode A:** a relative DSM (`rdsm.tif`, unitless 0–1, tier R).
* **Georeferenced GeoTIFF → Mode B:** an absolute DSM in metres on a declared vertical datum (`dsm.tif`, EGM2008), plus terrain, nDSM, slope, aspect and quality flags — all GeoTIFF.
* **3D flythrough** (Three.js): orbit, first-person walk (WASD, terrain-following, building collision), automatic drone orbit, LoD-1 building blocks, heat-map / contour shaders, server-side point & distance / height / slope measurement.
* **In-app validation** against LiDAR or any reference raster: ME, RMSE, MAE, NMAD, LE90/95, Pearson r, Spearman ρ, slope / object / height strata, residual map — shown next to the *input DEM alone* on the same pixels.

Everything runs offline on a CPU laptop (CUDA used automatically when present).

---

## Measured accuracy (swisstopo airborne LiDAR, reproducible)

`python scripts/validate_demo.py` runs the bundled GeoTIFF tiles through the real pipeline and compares with swissSURFACE3D (DSM) and swissALTI3D (DTM) 0.5 m LiDAR. **Baseline** = the free Copernicus GLO-30 DEM resampled to the same grid, i.e. what you would have without DepthWizard. RMSE in metres, lower is better (full tables, NMAD / LE90 / r and strata: [`docs/validation_results.md`](docs/validation_results.md)).

| Tile (1 km²) | Layer vs LiDAR | Copernicus GLO-30 alone | Zero-shot model | **Fine-tuned model** |
|---|---|---|---|---|
| Zürich urban, 0.5 m | DSM | 9.25 | 8.91 | **5.53** (−40 %) |
| Zürich urban, 0.5 m | terrain | 8.00 | 6.11 | **3.17** (−60 %) |
| Zürich urban, 2 m | DSM | 9.00 | 8.89 | **6.34** (−30 %) |
| Zürich urban, 2 m | terrain | 7.97 | 6.23 | **3.25** (−59 %) |
| Emmental rural / forest / hilly, 2 m | DSM | 7.45 | 7.18 | **6.41** (−14 %) |
| Emmental rural / forest / hilly, 2 m | terrain | 11.49 | 10.35 | **5.30** (−54 %) |

No anchors are used in this table (tier T). These two tiles, and everything within 5 km of them, were **never used** for training, validation or any design choice, so they are a held-out test.

**The fine-tuned model.** Depth Anything V2 Small was fine-tuned on a Colab T4 (8000 iterations) to predict height above ground in metres from 0.5 m RGB tiles. The training data were 108 swisstopo 1 km tiles, with 16 for validation; the target was LiDAR nDSM = swissSURFACE3D − swissALTI3D. On 20 tiles from 5 held-out regions (Basel, Lugano, Davos, Thurgau, Jura) its height-above-ground error is:

| Held-out regions, nDSM vs LiDAR | RMSE | MAE | r | buildings / trees RMSE |
|---|---|---|---|---|
| **Fine-tuned** | **3.86 m** | **1.90 m** | **0.875** | **5.52 m** |
| Zero-shot, oracle-scaled (its best case) | 6.74 m | 4.85 m | 0.528 | 9.45 m |

Source: `models/da-v2-small-ndsm/1.0.0/training_report.json`. How the model is combined with the DEM (`fusion.metric_composition`) was chosen on 6 validation-region tiles (Aarau, Fribourg). DSM = DEM + high-pass(model nDSM) scored DSM 5.71 m and terrain 4.10 m, against 7.20 m and 6.90 m for the DEM alone ([`docs/metric_composition_selection.json`](docs/metric_composition_selection.json), `scripts/select_metric_composition.py`).

Limits, stated plainly:
* Training and testing are Swiss only (swisstopo is the free LiDAR + orthophoto source). Accuracy on Indian imagery or other sensors is **not yet measured**.
* Only two test tiles have a bundled DEM.
* With the fine-tuned model, simulated anchors added nothing measurable. The anchor fit was rejected on 2 of 3 tiles, and on the 0.5 m tile it slightly worsened the terrain (3.17 → 4.04 m). The anchors are simulated from the same LiDAR, with pixels within 15 m excluded from the metrics.

## Measured on India (Sikkim, NASA ICESat-2 checkpoints)

No public airborne LiDAR exists for Indian sites, so DepthWizard is validated there against **independent satellite laser altimetry**: NASA ICESat-2 ground and canopy heights. That's 1,114 20 m segments, 2018–2025, fetched without any login through the public SlideRule service.

The imagery is **Maxar WorldView at 0.5 m** (Maxar Open Data Program, Sikkim flood event, CC BY-NC 4.0). The six sites are 1.2 km scenes: Namchi town, the Chungthang valley and dam, two Teesta-valley hillside sites, a steep forest, and a North Sikkim alpine / glacial area. No Indian data was used for training or tuning. Pooled RMSE in metres ([`docs/validation_india.md`](docs/validation_india.md), `scripts/fetch_india_demo.py` + `scripts/validate_india.py`):

| vs ICESat-2 (fine-tuned model) | Copernicus alone | DepthWizard on Copernicus | CartoDEM alone | **DepthWizard on CartoDEM** |
|---|---|---|---|---|
| Terrain vs ground | 10.51 | 8.54 | 7.98 | **7.61** |
| DSM vs top of surface | 11.81 | 11.10 | 11.92 | **10.72** |
| Height above ground (canopy / buildings) | – | 9.20 | – | **9.20** |

**CartoDEM** (ISRO/NRSC, from Bhoonidhi) is used automatically when its tile is in `assets/dem/cartodem/`:
- **Datum:** the app detected its heights as **ellipsoidal** on all six sites. The CartoDEM − Copernicus offset of −25…−47 m matches the local geoid undulation of −30…−44 m. The heights are converted to EGM2008; without that check they would be ~40 m off.
- **Per site the result is mixed:** CartoDEM is much better on steep forest (Chungthang west: 19.7 → 12.8 m) and worse on alpine terrain (1.7 → 5.9 m).
- **Records:** the Copernicus-based run is kept in `docs/validation_india_copernicus.md`.

The zero-shot model on Copernicus, for reference: terrain 10.25, DSM 11.03, height above ground 11.38.

Site by site, the fine-tuned model improves the height above ground on 5 of 6 sites and the DSM on 5 of 6. The terrain improves on 4 of 6. It is worse on the two forested-valley sites (Namchi, Chungthang), where Copernicus already lies close to the ground under the forest.

Errors are dominated by very steep Himalayan slopes, where the 30 m DEM itself is off by 15–20 m. These numbers are the model's first measurement on Indian imagery; it was trained only on Swiss data.

In the app, the six Sikkim scenes are one-click demos. **Validate vs checkpoints** runs the same comparison for any job, using the bundled ICESat-2 data or an uploaded CSV (`id,lon,lat,h_ground[,h_canopy]`).

## Building intelligence and CartoDEM

* **Building panel** (Mode B, panel 5):
  * **Per building:** median height with the model's measured typical error (±5.5 m = RMSE for objects on held-out LiDAR), the roof-height spread, ground and roof elevation, footprint, volume, and an *approximate* floor range. Also location.
  * **Scene summary:** building counts per height class, total footprint and total volume.
  * **Interaction:** filter by height and footprint; click a row to fly the 3D view to that building, or click a building in 3D to select it.
  * **Export:** GeoJSON (opens in QGIS) or CSV (`/api/jobs/{id}/buildings[.geojson|.csv]`).
  * **Limit:** footprints come from a rule-based RGB + nDSM mask, so on forested hills some tree clusters are counted as buildings.
* **CartoDEM (ISRO/NRSC)** is preferred over Copernicus in India. Put the Bhuvan tiles into `assets/dem/cartodem/` (see its README).
  * Whether its heights are geoid or ellipsoidal is decided automatically, by comparison with Copernicus.
  * If neither fits, CartoDEM is refused for that scene and the reason is reported, rather than risking a silent 40–90 m datum error.

## How Mode B produces metres

```
GeoTIFF ──► ingest (CRS, GSD; geographic → local UTM)          Copernicus GLO-30 / user DEM ──► datum guard ──► DEM on job grid (EGM2008)
   │                                                                                                  │
   ├─► Depth Anything V2 Small (zero-shot), whole image ──► relative structure (tier R preview)       │
   │                                                                                                  ▼
   └─► overlapping 518 px tiles at ~0.5 m ──► fine-tuned nDSM model: height above ground (m) ──► high-pass < 30 m
                                              (no model installed: zero-shot tiles, per-tile scale fitted to the DEM band)
                        DSM = DEM + detail   (DEM cell means preserved: never worse than the DEM at its own resolution)
                        terrain = DSM − predicted heights · nDSM · slope / aspect · flags · LoD-1 blocks
optional anchors CSV ──► tier A: detail gain (+ offset only if statistically significant), accepted only if leave-one-out error improves
no DEM for the area  ──► tier H: metric heights above ground only (fine-tuned model), no absolute elevation
```

Why this design: the DEM is correct for everything larger than one posting (30 m) but cannot see buildings or tree crowns; the model sees those in metres but has no datum. DepthWizard keeps the DEM for ≥ 30 m and adds the model's < 30 m structure, and derives the ground as surface minus predicted heights. Running the model on native-resolution tiles instead of one down-scaled pass matters: even zero-shot, the correlation with LiDAR building detail rose from 0.06 to 0.50–0.54 on the Zürich tile. Without the fine-tuned model the app falls back to zero-shot tiles with a per-tile scale fitted against the DEM band 30–120 m.

**Tiers and quality** are attached to every raster, API response and HUD: **R** relative · **H** metric heights above ground (fine-tuned model, no DEM) · **T** DEM + model detail (quality ≤ LIMITED: validated on other regions, not on the user's scene) · **A** anchor-refined. **Geodetic safety (Phase 8):** vertical transforms run offline with bundled EGM96 / EGM2008 / LN02 grids and refuse PROJ "ballpark" (zero-correction) pipelines (C-1). Geographic inputs are reprojected to UTM before any metric operation (C-4). Co-registration uses Horn gradients as correlation (C-2), and anchors use median / RANSAC with blunder flags (C-3).

## Fine-tuning (reproduce or improve the model)

1. Open [`notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb`](notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb) in Google Colab, choose a T4 GPU runtime, then Run all (about 2–3 h). The notebook:
   * downloads the swisstopo tiles itself;
   * excludes the DepthWizard test tiles;
   * splits train / validation / test by region;
   * evaluates on the held-out regions;
   * exports `depthwizard_ndsm_model.zip`.
2. Run `python scripts/install_finetuned_model.py depthwizard_ndsm_model.zip`. This verifies the SHA-256 and that the weights differ from the baseline, writes a model card with the measured numbers, and registers the model.
3. Restart the server. Mode B then uses the model automatically; set `DW_MODEL_METRIC_ENABLED=false` to compare with zero-shot. Re-measure with `python scripts/validate_demo.py`.

The notebook is generated from `notebooks/build_finetune_notebook.py`. The weights (99 MB) are git-ignored, so ship `depthwizard_ndsm_model.zip` alongside the repository.

---

## Quick start (Windows / Linux / macOS)

Prerequisites: **Python 3.12+** (tested on 3.12.14 and 3.14.0) and **Node.js 20+** (only to build the frontend once). About 1 GB of disk.

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements/dev.txt
python scripts/fetch_model.py     # Depth Anything V2 Small, 99 MB, Apache-2.0, SHA-256 verified
python scripts/install_finetuned_model.py depthwizard_ndsm_model.zip   # fine-tuned nDSM model (optional but recommended)
python scripts/doctor.py          # environment + geoid grids + DEM tiles + weights check
cd frontend && npm ci && npm run build && cd ..
python run_server.py              # opens http://127.0.0.1:8000
```

After setup, **`start_depthwizard.bat`** (Windows) or **`./start_depthwizard.sh`** starts everything with one click. It builds the frontend if needed and opens the browser.

* Development: `cd frontend && npm run dev` (Vite on :5173, proxies the API to :8000).
* Tests: `python -m pytest` (unit, API, Mode B end-to-end, Phase 8 geodetic regressions, real-model runs on the demo tiles).
* Accuracy: `python scripts/validate_demo.py` regenerates `docs/validation_results.{md,json}`.

## Suggested 5-minute demo

1. Click **Zürich City 2m** (one-click benchmark) → **Generate Surface**. About 30 s on a CPU. The result card shows the tier, datum EGM2008, that heights come from the fine-tuned model, and why the quality is what it is.
2. Click the DSM image → server-sampled elevation, terrain, nDSM, slope with position in LV95 and WGS84. **Measure (2 clicks)** → distance, ΔZ, grade.
3. **Run validation** (bundled swissSURFACE3D, LN02 is pre-selected) → metrics next to "input DEM alone", with a residual map.
4. **Open 3D View** → orbit, 🎬 Drone Fly, 🚶 Walk (WASD, Shift, mouse; walls block you), 🌈 Heatmap, Adaptive RTIN mesh, layer switch (LoD-1 city / DSM / terrain / nDSM / relative).
5. Load **Emmental Ridge** for forest and hills, and **Zürich Photo HD** (JPEG) for Mode A.

---

## API

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | status, model name / hash / device |
| `GET` | `/api/system` | versions (GDAL, PROJ, torch), configuration, geoid grids, DEM tiles, mode semantics |
| `GET` | `/api/demo` | bundled demo inputs and LiDAR references |
| `POST` | `/api/jobs` | create a job: multipart `file` (+ optional `dem` GeoTIFF with `dem_vertical_crs`, `anchors` CSV) |
| `POST` | `/api/jobs/{id}/run` | run the pipeline (UPLOADED → PREPROCESSING → INFERENCE → CALIBRATION / RASTERIZING → READY) |
| `GET` | `/api/jobs/{id}` · `/result` · `/metadata` | status and stage timings · result manifest (mode, tier, quality, datum, layers) · provenance |
| `GET` | `/api/jobs/{id}/artifact/{name}` | `dsm.tif`, `terrain.tif`, `ndsm.tif`, `dem.tif`, `slope.tif`, `aspect.tif`, `flags.tif`, `relative.tif`, `rdsm.tif`, `buildings.json`, heightfields, previews, `log.jsonl` |
| `GET` | `/api/jobs/{id}/sample?x=&y=&crs=pixel\|job\|wgs84` | server-side sampling of every layer at a point |
| `POST` | `/api/jobs/{id}/measure` | distance, ΔZ and grade between points |
| `POST` | `/api/jobs/{id}/validate` · `GET …/validation` | validate against an uploaded or bundled reference (dsm / dtm / ndsm, any vertical CRS) · history |
| `DELETE` | `/api/jobs/{id}` | delete a job |

Anchors CSV: `id,x,y,z,type[,sigma]` with optional `# crs=EPSG:xxxx` and `# vcrs=EGM2008|EGM96|ellipsoidal|EPSG:code` comment lines (`type` = `ground` or `object`). Minimum 5 anchors.

## Repository layout

* `backend/` — FastAPI app, job manager (single worker, filesystem jobs), Mode A / Mode B pipelines, server-side sampling and validation.
* `core/` — ingest, geodesy (datum guard, reprojection), inference wrapper, calibration (`calib/fusion.py`: tiled inference + detail fusion; `terrain.py`; `anchors.py`; `tier.py`), DSM derivatives, heightfields, LoD-1, validation harness.
* `ml/registry/` — model registry and the vendored Depth Anything V2 code (Apache-2.0). `notebooks/` — the Colab fine-tuning notebook (and its generator). `ml/train/` — an older training scaffold (the notebook is the tested path).
* `models/` — `da-v2-small-baseline` (zero-shot, fetched) and `da-v2-small-ndsm` (fine-tuned; model card + training report committed, weights installed from the zip).
* `frontend/` — TypeScript + Three.js viewer (Vite).
* `assets/` — demo tiles and LiDAR references (© swisstopo, OGD), Copernicus GLO-30 tiles (© DLR / Airbus, distributed by ESA), PROJ geoid grids.
* `configs/default.yaml` — every parameter (override with `DW_CONFIG=<file>` or `DW_<SECTION>_<KEY>` environment variables).
* `SIH26175-Phase*.md` — the research and design record (Phases 1–10). `DEPTHWIZARD_COMPREHENSIVE_SYSTEM_DOSSIER.md` — the technical description of the built system.

Licences: Depth Anything V2 Small weights and code are Apache-2.0. swisstopo data is Open Government Data (attribution "© swisstopo"). Copernicus DEM GLO-30 is © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
