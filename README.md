# DepthWizard — Single-View Height Estimation and 3D Flythrough

**Problem Statement:** SIH26175 · **Organization:** Indian Space Research Organisation (ISRO) — Department of Space  
**Theme:** Disaster Management / Software · **Build Status:** Complete System (Mode A + Mode B, 72 Passing Tests)

---

## Overview

DepthWizard turns a single nadir optical RGB satellite or aerial image into a metric Digital Surface Model (DSM) and an interactive, real-time 3D navigable environment.

```
                          ┌──────────────────────────┐
                          │ Single RGB Remote Sensing│
                          │      Image Ingest        │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
[ Non-Georeferenced (Mode A) ]                            [ Georeferenced (Mode B) ]
• PNG / JPG / WebP                                        • GeoTIFF (CRS, Affine Transform)
• Relative Depth via Depth Anything V2                    • Ground Sample Distance (GSD) calculation
• Unitless [0, 1] rDSM raster (`rdsm.tif`)                • Local UTM Reprojection
• Calibration Tier: R (Relative)                          • Copernicus GLO-30 DEM baseline + GCP Anchors
• 3D Heightfield & Orbit Flythrough                       • Absolute Metric Elevation (AMSL) in metres
                                                          • Calibration Tiers: T (DEM) / A (Anchors)
                                                          • LoD-1 3D Digital Twin City Extrusions
                                                          • In-App LiDAR Validation Suite
```

### Core Capabilities
1. **Mode A (Relative Surface / Tier R):** Ingests non-georeferenced images, executes zero-shot relative depth estimation via Depth Anything V2 Small (ViT-S), and outputs unitless normalized $[0, 1]$ 32-bit float rasters with interactive 3D terrain exploration.
2. **Mode B (Metric Absolute DSM / Tiers T & A):** Ingests georeferenced GeoTIFFs, parses spatial metadata, converts relative depth into real-world elevation in metres (AMSL) using Copernicus 30m DEM baselines and sparse Ground Control Point (GCP) anchors with RANSAC robust fitting.
3. **Geoid & Datum Guard:** Explicitly addresses India's 24m–99m geoid-ellipsoid undulation gap via offline PROJ grids (EGM96 / EGM2008), rejecting uncalibrated "ballpark" approximations.
4. **Interactive 3D WebGL Engine:** Dynamic Three.js heightfield streaming with $16\times$ anisotropic texture drape, vertical exaggeration slider ($0.5\times - 5.0\times$), True North compass HUD, and scale indicators.
5. **🚶 Walk Mode (First-Person Flythrough):** Ground-level exploration using `W/A/S/D` controls and pointer-lock camera with real-time terrain collision detection.
6. **🏢 3D Digital Twin City (LoD-1):** Automated building footprint extraction from nDSM with 3D prism extrusions, height-classified facade textures, and rooftop silhouette outlines.
7. **Server-Authoritative Measurement:** Raycast elevation sampling returning real-world coordinates, terrain elevation, $\Delta Z$ structural height, horizontal distance, and surface slope angles.
8. **Automated LiDAR Validation Benchmark:** In-app scientific verification against LiDAR reference datasets (computing RMSE, MAE, NMAD, LE90, Pearson $r$, Spearman $\rho$, and stratified slope/height error distributions).

---

## Quick Start (Windows / Linux)

**Prerequisites:** Python **3.12**, Node.js **20+ LTS**, ~1 GB disk space. Runs completely offline on standard CPU (CUDA GPU supported automatically if available).

```bash
# 1. Setup Python Environment
python -m venv .venv                      # Python 3.12 interpreter
.venv/Scripts/activate                    # Linux: source .venv/bin/activate
pip install -r requirements/dev.txt       # Core dependencies + test framework

# 2. Fetch Verified Local Model Weights (99 MB, Apache-2.0, SHA-256 verified)
python scripts/fetch_model.py

# 3. Verify Environment Health & Proj Grids
python scripts/doctor.py

# 4. Build Interactive Frontend
cd frontend && npm ci && npm run build && cd ..

# 5. Launch Standalone Web Application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000 in your browser
```

* **Development Mode:** `cd frontend && npm run dev` (Vite on `:5173`, proxies API requests to `:8000`).
* **Test Suite:** `python -m pytest -q` (**72 tests passing**: unit, API, Mode B, geodetic regressions, and real neural inference on demo tiles).

---

## API Reference

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Application status, model version/hash, device info (CPU/CUDA) |
| `GET` | `/api/system` | Runtime versions (GDAL, PROJ, PyTorch), Mode A/B semantics, geoid grid status |
| `GET` | `/api/demo` | List bundled demo tiles (Zürich urban, Emmental rural/hilly) and LiDAR references |
| `POST` | `/api/jobs` | Create job (multipart: `file` image, optional `dem` GeoTIFF, optional `anchors` CSV) |
| `POST` | `/api/jobs/{id}/run` | Execute pipeline (state machine: `UPLOADED` → `INFERENCE` → `READY`) |
| `GET` | `/api/jobs/{id}` | Job progress, stage timing breakdown (`stages_ms`), and status |
| `GET` | `/api/jobs/{id}/result` | Result manifest (mode, metric flag, calibration tier R/T/A, raster bounds) |
| `GET` | `/api/jobs/{id}/metadata` | Provenance records, sensor GSD, vertical CRS, and heightfield metadata |
| `GET` | `/api/jobs/{id}/artifact/{name}` | Download artefacts (`dsm.tif`, `rdsm.tif`, `terrain.tif`, `ndsm.tif`, `buildings.json`, `heightfield.f32`, `texture.jpg`) |
| `GET` | `/api/jobs/{id}/sample` | Server-authoritative coordinate query: elevation, slope, and provenance |
| `POST` | `/api/jobs/{id}/measure` | Server-authoritative multi-point analysis: $\Delta Z$ height, ground distance, slope |
| `POST` | `/api/jobs/{id}/validate` | Run scientific accuracy validation against uploaded or bundled LiDAR reference |
| `GET` | `/api/jobs/{id}/validation` | Retrieve validation report (RMSE, MAE, NMAD, LE90, Pearson $r$) |
| `DELETE` | `/api/jobs/{id}` | Delete job directory and cached artefacts |

---

## Provenance & Calibration Tiers

DepthWizard enforces strict scientific honesty so evaluators know exactly how each meter was produced:
* **Tier R (Relative):** Non-georeferenced images. Surface is structurally consistent but unitless ($[0, 1]$).
* **Tier H (Height-Above-Ground):** Relative depth mapped to metric object heights via supervised nDSM.
* **Tier T (Terrain-Anchored):** Absolute metric DSM derived from low-frequency DEM ground baseline + high-frequency object structure.
* **Tier A (Anchor-Calibrated):** Metric DSM with scale and shift refined by sparse Ground Control Points (GCPs) via RANSAC linear fitting.

---

## Geospatial Safety Guards (Phase 8 Design Corrections)

* **C-1 (`core/geo/vertical.py`):** Offline PROJ vertical datum guard. Rejects "ballpark" zero-grid conversions; verifies EGM96/EGM2008 geoid grids before claiming absolute elevation.
* **C-2 (`core/validate/coregister.py`):** Iterative sub-pixel co-registration using Horn gradient correlation with reference smoothing, recovering spatial shifts prior to metric evaluation.
* **C-3 (`core/calib/anchors.py`):** Robust anchor recovery via median offset and RANSAC scale ($N \ge 5$), rejecting outliers with NMAD blunder detection.
* **C-4 (`core/geo/reproject.py`):** Automated geographic-to-local UTM reprojection before any metric operation to prevent latitude-dependent metric distortion.

---

## Repository Layout

* `backend/` — FastAPI application, async job manager, API routes, logging, and error handling.
* `core/` — Geospatial processing, vertical datum management, Depth Anything inference wrapper, multi-tier calibration, terrain rasterization, LoD-1 vector extraction, and LiDAR validation.
* `ml/registry/` — Local model registry and vendored Depth Anything V2 implementation (Apache-2.0).
* `models/` — Weights directory with SHA-256 verification and model cards.
* `frontend/` — Production Three.js 3D WebGL viewer, Walk mode, HUD, and analysis controls (TypeScript + Vite).
* `assets/demo/` — Open demo tiles (Zürich urban, Emmental rural/hilly, © swisstopo OGD).
* `tests/` — Automated test suite (**72 passing tests** across unit, API, Mode B, and regression suites).
* `configs/default.yaml` — All configurable parameters for ingest, inference, calibration, and rendering.
