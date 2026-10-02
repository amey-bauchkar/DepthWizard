# DepthWizard — Single-View Height Estimation and 3D Flythrough

> **Smart India Hackathon 2024 · Problem Statement SIH 26175**  
> **Organisation:** Indian Space Research Organisation (ISRO), Department of Space  
> **Theme:** Disaster Management · **Category:** Software  
> **Team:** Amey Bauchkar · Tanmay Padwal · Soham Jadhav

---

## ⚡ Executive Summary for Judges (60-Second Read)

| Key Aspect | Summary |
| :--- | :--- |
| **The Challenge** | During rapid-onset disasters (floods, landslides, earthquakes), emergency responders urgently require 3D terrain elevation and structural height data. Conventional multi-view stereo satellite photogrammetry and airborne LiDAR take days to schedule, require multi-angle passes, and cost millions. |
| **Our Solution** | **DepthWizard** reconstructs high-resolution, geodetically calibrated 3D elevation models (DSM, DTM, nDSM) and interactive 3D flythroughs from a **single monocular optical RGB satellite or aerial image** in ~25–30 seconds on a standard commodity CPU laptop (zero GPU mandate). |
| **Core Innovation** | **Dual-Frequency Frequency-Split Fusion:** Large-scale regional vertical datum (Copernicus GLO-30 / ISRO CartoDEM on EGM2008 geoid) is preserved for low-frequency topography ($\ge 30\text{ m}$), while a fine-tuned Vision Transformer (Depth Anything V2 Small) provides high-frequency metric structural and vegetation heights ($< 30\text{ m}$). |
| **Accuracy (Verified)** | **Sub-4 m RMSE** on held-out benchmark LiDAR (**3.69 m** pooled RMSE vs swissSURFACE3D & US 3DEP; **7.61 m** on steep Himalayan terrain vs NASA ICESat-2 with CartoDEM). Reduces terrain baseline error by **40% to 60%**. |
| **Actionable Screenings** | Instant disaster planning modules: 1-click **Flood inundation** (HAND model), **Landslide susceptibility** (BIS IS 14496 + live rainfall), **Helicopter Landing Zones** (US Army FM 3-21.38), **Cut-off Road Isolation**, and **Building Collapse Detection** (pre/post event). |
| **Zero-Footprint Delivery** | Generates a 1-file standalone **Offline 3D Scene (`.html`, 7–10 MB)** that opens with zero dependencies in any browser without Python, Node, or Internet access — fulfilling the strict tactical field deployment requirement. |

---

## 🚀 Key System Architecture & Modes

DepthWizard operates in two flexible pipelines:

```
                      ┌────────────────────────────────────────┐
                      │    Input Optical RGB Imagery (Single)   │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ Mode A: Relative ]                           [ Mode B: Metric & Geodetic ]
   (PNG / JPG / Unreferenced TIFF)                     (Georeferenced GeoTIFF / Cartosat)
                  │                                               │
    Depth Anything V2 (ViT-S/14)                                  ├─► Ingest (CRS, UTM reprojection)
   Native-resolution inference (0.5m)                             ├─► ISRO CartoDEM / Copernicus GLO-30
                  │                                               │   (Automatic Ellipsoid/Geoid Datum Guard)
                  ▼                                               ├─► Fine-Tuned Metric nDSM v2.0.0 (ViT-S)
    Relative Surface Mesh (rdsm.tif)                              │   (Overlapping 518px sliding-window)
       Unitless (0 - 1) Preview                                   ├─► High-Pass Frequency Fusion
       1.8s instant turnaround                                    │   (Zero-bias metric DSM = DEM + nDSM_detail)
                                                                  ├─► Bare-Earth DTM = DSM - Predicted Heights
                                                                  ▼
                                                   Geodetic Products (EGM2008 Datum):
                                                   • Absolute DSM & DTM (GeoTIFF)
                                                   • Normalized DSM (nDSM)
                                                   • Slope & Aspect Derivatives
                                                   • LoD-1 3D Building Blocks & Statistics
                                                   • Confidence & Geodetic Quality Flags
```

### 1. Mode A: Instant Relative 3D (Non-Georeferenced)
* **Input:** Any drone photo, aerial JPEG/PNG, or unreferenced TIFF.
* **Output:** Relative DSM (`rdsm.tif`), interactive 3D terrain mesh.
* **Speed:** ~1.8 seconds.
* **Correlation:** Pearson $r = 0.84$ with real LiDAR surface structures on 0.5 m imagery.

### 2. Mode B: Metric Geodetic Reconstruction (GeoTIFF / Satellite Products)
* **Input:** Maxar WorldView, Cartosat-1/2/3, Resourcesat, or aerial GeoTIFF.
* **Vertical Datum Guard:** Automated datum harmonization ensuring all outputs strictly align to **EGM2008 orthometric elevation**. Automatically resolves ellipsoidal vs geoidal offsets in CartoDEM to prevent silent 40–90 m vertical datum shifts.
* **Fine-Tuned Metric Model (v2.0.0):** Depth Anything V2 Small (24.8M parameters, DINOv2 ViT-S/14 backbone) fine-tuned on multi-region airborne LiDAR (swisstopo and USGS 3DEP) on 0.5 m RGB tiles to directly predict height above ground in true metres.
* **Pre-bundled Model:** The model weights (`depth_anything_v2_ndsm_s.pth`, 94.6 MB) are committed directly in the repository—no external downloads or API keys needed.

---

## 📊 Measured Accuracy & Validation

Every metric is reproducible via automated scripts on held-out benchmarks:

### 1. Held-Out Airborne LiDAR (swisstopo & USGS 3DEP)
*Evaluated on 41 held-out test regions completely isolated from training.*

| Benchmark Dataset | Baseline DEM Alone (RMSE) | DepthWizard v2.0.0 (RMSE) | Error Reduction |
| :--- | :---: | :---: | :---: |
| **Zürich Urban (0.5 m)** | 9.25 m | **5.53 m** (DSM) / **3.17 m** (DTM) | **−40% to −60%** |
| **Emmental Forest/Hilly (2 m)** | 11.49 m | **5.30 m** (DTM) | **−54%** |
| **USGS 3DEP (US Multi-Site)** | 6.25 m | **3.26 m** (nDSM) | **−49%** |
| **All Held-Out Regions Pooled** | 6.19 m | **3.69 m** (nDSM) | **−40%** |

### 2. Himalayan Disaster Zone Validation (Sikkim, India vs NASA ICESat-2)
*Evaluated against 1,114 NASA ICESat-2 satellite photon laser altimetry checkpoints across 6 rugged 1.2 km scenes (Namchi, Chungthang dam, Teesta Valley).*

| Validation Reference | ISRO CartoDEM Alone | DepthWizard on CartoDEM | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Bare-Earth Terrain** | 7.98 m | **7.61 m** | **+0.37 m gain** |
| **Top-of-Surface DSM** | 11.92 m | **10.72 m** | **+1.20 m gain** |
| **Height Above Ground (Canopy/Buildings)** | — | **9.20 m RMSE** | Consistent across extreme terrain |

---

## 🛡️ Disaster Management Capabilities

DepthWizard is purpose-built for emergency operations in disaster-hit regions:

* 🌊 **Flood Inundation Screening (HAND):**
  Height Above Nearest Drainage (HAND) hydrological model integrating OpenStreetMap river networks. Provides live water-depth sliders and building inundation counts. Validated against Sentinel-2 observations of the Sikkim Teesta GLOF (Oct 2023) and Nepal Sunkoshi flood (Sep 2024).
* ⛰️ **Landslide Susceptibility Screening:**
  Implements Bureau of Indian Standards (BIS IS 14496 Part 2) slope-stability criteria combined with real-time precipitation from Open-Meteo.
* 🚁 **Helicopter Landing Zones (HLZ):**
  Identifies obstacle-free landing sites conforming to US Army FM 3-21.38 tactical criteria (slope $\le 8^\circ$, minimum 25 m clearance diameter). Successfully localized 9 of 19 ground-truth helipads in Sikkim and Nepal (6–9× better than random).
* 🚧 **Road Network Cut-off Analysis:**
  Overlays OSM road vectors against hazard footprints to identify blocked transport arteries and isolated settlements with population estimates.
* 🏚️ **Pre/Post Disaster Change Screening:**
  Bitemporal co-registration and height-differencing for earthquake and conflict zones (demonstrated on the Feb 2023 Türkiye earthquake in Islahiye with 71–88% precision on collapsed structures).
* 📄 **1-Click Damage Assessment Report:**
  Instantly compiles a standardized PDF report containing operational metrics, coordinate footprints, hazard maps, and high-priority action lists.

---

## 🎮 Interactive 3D Flythrough & Viewer

Built with **Three.js** and styled with an accessible, high-contrast user interface:
* **Orbit Mode:** Smooth 360° rotational inspection with dynamic lighting and shadow contours.
* **Drone Camera:** Automated circular flythrough path for presentations and aerial appraisals.
* **First-Person Walk:** WASD keyboard navigation with terrain-following and building collision detection.
* **Realistic Vertical Scale:** Default height scaling tuned to `0.02` for true-to-life elevation visualization without unnatural vertical exaggerations.
* **Interactive Tooling:** 2-click ΔZ elevation, distance, and slope measurements; dynamic contour shaders; and LoD-1 extruded building models with clickable metadata cards.

---

## 📦 Deliverables & Export Formats

1. **Offline 3D Scene (`.html`):** Single-file self-contained bundle (7–10 MB). Contains full Three.js engine, terrain mesh, textures, and measurements. Runs on any computer with a web browser without internet or Python.
2. **GIS Production Package (`.zip`):**
   * Cloud-Optimized GeoTIFFs (DSM, DTM, nDSM, slope, aspect, flags) with embedded QGIS symbology.
   * LoD-1 3D Buildings in GeoPackage (`.gpkg`), GeoJSON, and CSV formats.
   * Khronos-compliant 3D model in textured GLB format (`.glb`).
   * STAC 1.0 item metadata with provenance log.
3. **Executable Standalone:** One-click packaging with PyInstaller (`DepthWizard.exe`) for offline Windows field stations.

---

## 🛠️ Quick Start (Running Locally)

### Prerequisites
* **Python 3.12+** (tested on 3.12 and 3.14)
* **Node.js 20+** (only needed to build the web frontend once)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amey-bauchkar/DepthWizard.git
cd DepthWizard

# 2. Set up Python virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements/dev.txt

# 4. Build the modern frontend bundle
cd frontend
npm ci
npm run build
cd ..

# 5. Launch DepthWizard
python run_server.py
```
DepthWizard will start on **`http://127.0.0.1:8000`** and automatically open in your default browser.

*(On Windows, you can simply double-click **`start_depthwizard.bat`** after initial install).*

---

## 🐳 Docker & Cloud Deployment (Azure / Linux)

DepthWizard is fully containerized and runs on any cloud host:

```bash
# Build the Docker image
docker build -t depthwizard:latest .

# Run the container
docker run -d --name depthwizard --restart always -p 80:7860 depthwizard:latest
```
Access the application at `http://<your-server-ip>/`. The API health check is available at `/health`.

---

## 📂 Repository Structure

```
DepthWizard/
├── backend/                  # FastAPI web server, job pipeline, API endpoints
├── core/                     # Core computational algorithms
│   ├── calib/                # Frequency-split detail fusion & anchor fitting
│   ├── disaster/             # Flood (HAND), landslide, road cut-off, helipads
│   ├── dsm/                  # Surface derivation, slope, aspect, geodetic checks
│   ├── export/               # Standalone HTML scene, GeoPackage, GLB, GIS package
│   ├── ingest/               # Coordinate reference system & GeoTIFF ingest
│   └── terrain/              # DTM derivation, LoD-1 buildings, height stats
├── frontend/                 # Three.js 3D viewer & responsive UI (Vite + TypeScript)
├── models/                   # Registered model weights
│   └── da-v2-small-ndsm/
│       └── 2.0.0/            # Fine-tuned nDSM model weights & model card
├── assets/                   # Bundled demo tiles, geoid grids, and reference LiDAR
├── docs/                     # Validation reports and mathematical audits
├── scripts/                  # Automated validation, benchmark, and deployment tools
└── run_server.py             # Main entry point
```

---

## 👥 Contributors

* **Amey Bauchkar** ([@amey-bauchkar](https://github.com/amey-bauchkar))
* **Tanmay Padwal** ([@Tomeseto](https://github.com/Tomeseto))
* **Soham Jadhav** ([@ANTHOSJ](https://github.com/ANTHOSJ))

---

## 📜 Licences & Acknowledgements

* **Depth Anything V2:** Apache-2.0 Licence.
* **swisstopo Airborne LiDAR:** Open Government Data (OGD), Federal Office of Topography swisstopo.
* **Copernicus DEM GLO-30:** © DLR e.V. and Airbus Defence and Space GmbH, provided by ESA.
* **Maxar Open Data:** CC BY-NC 4.0 (Maxar Technologies Open Data Program).
* **NASA ICESat-2:** SlideRule Earth API (University of Washington / NASA).
* **CartoDEM:** National Remote Sensing Centre (NRSC) / ISRO, Department of Space.
