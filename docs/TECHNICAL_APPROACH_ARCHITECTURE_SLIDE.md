# DepthWizard — Technical Approach Slide Architecture

## 1. Executive Summary for Slide
**"From Single 2D Satellite/Aerial Image to Calibrated 3D Metric Terrain & LoD-1 Cityscapes"**
DepthWizard resolves the fundamental limitation of monocular depth estimation (scale ambiguity and datum disconnect) via **TL-CSM (Two-Level Calibrated Surface Model)**: keeping the verified low frequencies of coarse DEMs (≥ 30 m) while injecting learned high-frequency metric height detail (< 30 m) from a fine-tuned ViT backbone.

---

## 2. Technical Approach Flowchart (Mermaid)

```mermaid
flowchart TD
    %% Global Styling
    classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef geodeticStyle fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef mlStyle fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef fusionStyle fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef derivStyle fill:#4c1d95,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef outputStyle fill:#831843,stroke:#f472b6,stroke-width:2px,color:#f8fafc;

    %% 1. INPUT LAYER
    subgraph S1 [" 1. MULTI-SOURCE INPUTS "]
        direction TB
        IN_RGB["<b>Single Optical RGB</b><br/>Maxar / Cartosat / Drone<br/><i>(GeoTIFF or JPEG/PNG)</i>"]:::inputStyle
        IN_DEM["<b>Coarse Base DEM (30m)</b><br/>Copernicus GLO-30 / CartoDEM<br/><i>(Bundled / Auto-fetched)</i>"]:::inputStyle
        IN_AUX["<b>Optional Constraints</b><br/>• GCP / Laser Anchors (CSV)<br/>• Building Footprints (GeoJSON)"]:::inputStyle
    end

    %% 2. GEODETIC PREPROCESSING
    subgraph S2 [" 2. GEODETIC INTEGRITY & PREPROCESSING "]
        direction TB
        GEO_CRS["<b>CRS Normalization</b><br/>Geographic to Local UTM<br/>GSD Calculation"]:::geodeticStyle
        GEO_DATUM["<b>Geoid / Datum Guard (C-1)</b><br/>Auto Ellipsoid vs Geoid Detection<br/>EGM2008 Transformation"]:::geodeticStyle
    end

    %% 3. DUAL-SCALE ML DEPTH CORE
    subgraph S3 [" 3. DUAL-SCALE MONOCULAR AI ENGINE "]
        direction TB
        ML_MACRO["<b>Global Macro Pass (518px)</b><br/>Depth Anything V2 Small<br/>Relative Structural Coherence"]:::mlStyle
        ML_TILED["<b>Native-Res Overlapping Tiling</b><br/>518px Tiles (25% Overlap)<br/>Sub-posting Ground Sampling"]:::mlStyle
        ML_FT["<b>Fine-Tuned Metric Model</b><br/>ViT-S/14 Trained on LiDAR<br/>Predicts nDSM Metres (0.5m GSD)"]:::mlStyle
    end

    %% 4. MULTISCALE FREQUENCY FUSION (TL-CSM)
    subgraph S4 [" 4. MULTISCALE FREQUENCY FUSION (TL-CSM) "]
        direction TB
        FUS_SPLIT["<b>Frequency Band Splitting</b><br/>Low-Pass DEM (≥ 30m Base)<br/>High-Pass Model (< 30m Detail)"]:::fusionStyle
        FUS_BLEND["<b>Feathered Spatial Fusion</b><br/>DSM = DEM + Detail<br/><i>(DEM cell-means preserved)</i>"]:::fusionStyle
        FUS_TIER["<b>Dynamic Quality & Tier Tagging</b><br/>Tier R (Relative) | Tier H (Metric nDSM)<br/>Tier T (DEM Fusion) | Tier A (Anchored)"]:::fusionStyle
    end

    %% 5. GEOMORPHIC DERIVATIVES & 3D MODELING
    subgraph S5 [" 5. SURFACE EXTRACTION & VECTORIZATION "]
        direction TB
        DER_SURF["<b>Calibrated Rasters</b><br/>• DSM (Surface Elevation)<br/>• DTM (Ground Terrain = DSM - nDSM)<br/>• Horn 3x3 Slope & Aspect Rasters"]:::derivStyle
        DER_LOD1["<b>LoD-1 3D City Synthesis</b><br/>Footprint Snapping + Watershed<br/>Median Roof Heights & Volumes"]:::derivStyle
    end

    %% 6. DEPLOYMENT & DELIVERY
    subgraph S6 [" 6. EXPLOITATION & FIELD DEPLOYMENT "]
        direction TB
        OUT_WEB["<b>Interactive 3D Explorer</b><br/>Three.js + Adaptive RTIN Mesh<br/>WASD Walk (Collision) + Drone Fly"]:::outputStyle
        OUT_HTML["<b>Offline Single-File Scene</b><br/>7–11 MB Self-Contained HTML<br/>Zero Server / Zero Python needed"]:::outputStyle
        OUT_GIS["<b>Production GIS Package</b><br/>Cloud-Optimized GeoTIFFs (COG)<br/>GeoPackage + Khronos GLB + STAC 1.0"]:::outputStyle
        OUT_ANALYTICS["<b>Operational Tooling</b><br/>LiDAR / ICESat-2 Validation Engine<br/>Disaster Change / Collapse Screening"]:::outputStyle
    end

    %% PIPELINE CONNECTIONS
    IN_RGB --> GEO_CRS
    IN_DEM --> GEO_DATUM
    
    GEO_CRS --> ML_MACRO
    GEO_CRS --> ML_TILED
    ML_TILED --> ML_FT
    
    GEO_DATUM --> FUS_SPLIT
    ML_FT --> FUS_SPLIT
    ML_MACRO -.->|Fallback Scale| FUS_SPLIT
    IN_AUX -.->|Tukey IRLS RANSAC| FUS_BLEND

    FUS_SPLIT --> FUS_BLEND
    FUS_BLEND --> FUS_TIER
    
    FUS_TIER --> DER_SURF
    FUS_TIER --> DER_LOD1
    IN_AUX -.->|Vector Outlines| DER_LOD1

    DER_SURF --> OUT_WEB
    DER_SURF --> OUT_GIS
    DER_LOD1 --> OUT_WEB
    DER_LOD1 --> OUT_GIS
    OUT_WEB --> OUT_HTML
    DER_SURF --> OUT_ANALYTICS
```

---

## 3. High-Impact Slide Architecture (Horizontal Bento Grid)

For presentation slides (16:9), a 5-column pipeline with clear icons is most effective:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       DEPTHWIZARD TECHNICAL ARCHITECTURE                                        │
│               End-to-End Pipeline: From 2D Monocular Satellite Imagery to Sub-5m 3D Terrain                    │
├───────────────┬─────────────────┬───────────────────┬──────────────────────┬────────────────────────────────────┤
│ 1. INGEST &   │ 2. DUAL-SCALE   │ 3. MULTISCALE     │ 4. GEOMORPHIC &      │ 5. INTERACTIVE 3D &                │
│    GEODESY    │    AI INFERENCE │    DETAIL FUSION  │    VECTOR PRODUCTS   │    FIELD DELIVERY                  │
├───────────────┼─────────────────┼───────────────────┼──────────────────────┼────────────────────────────────────┤
│ • Single RGB  │ • Depth Anything│ • TL-CSM Frequency│ • DSM = DEM + Detail │ • Three.js Web Explorer            │
│   (GeoTIFF/   │   V2 Backbone   │   Decomposition   │   (Preserves DEM     │   - Adaptive RTIN (Martini) Mesh   │
│   JPG/PNG)    │   (ViT-S/14)    │ • Base (≥30m):    │    mean posting)     │   - 1.7m First-Person WASD Walk    │
│ • Coarse DEM  │ • Global Pass:  │   Copernicus DEM  │ • DTM = DSM − nDSM   │   - Drone Orbit & Anti-Smear Shade │
│   (Copernicus │   518px Macro   │ • Detail (<30m):  │   (Exact Identity)   │ • Standalone Offline 3D Scene      │
│   GLO-30 /    │   Structure     │   nDSM Tile Model │ • Terrain Derivatives│   - Single 7-11 MB HTML File       │
│   CartoDEM)   │ • Native Tiling:│ • Feather Blending│   (Slope, Aspect,    │   - Runs offline without Python/DB │
│ • Geodetic    │   Overlapping   │ • Tier Tagging:   │    Hillshade, Flags) │ • Standard GIS Export (.zip)       │
│   Datum Guard │   518px at ~0.5m│   - Tier R (Rel)  │ • LoD-1 3D City      │   - Cloud-Optimized GeoTIFFs (COG) │
│   (Auto EGM   │ • Fine-Tuned    │   - Tier H (nDSM) │   - Extruded blocks  │   - GeoPackage, GLB, STAC 1.0 Item │
│   2008 / UTM  │   Metric Model  │   - Tier T (DEM)  │   - Median roof ht   │ • In-App Validation & Change       │
│   conversion) │   (LiDAR nDSM)  │   - Tier A (GCP)  │   - Volume / floors  │   - ICESat-2 / LiDAR Ground Truth  │
│               │                 │                   │                      │   - Building Collapse Screening    │
└───────────────┴─────────────────┴───────────────────┴──────────────────────┴────────────────────────────────────┘
```

---

## 4. Key Talking Points for Evaluators / Jury

1. **Why it beats standard Monocular Depth:** Standard models suffer from scale ambiguity and arbitrary units. DepthWizard's fine-tuned model predicts actual metric heights above ground ($nDSM$), and multiscale fusion anchors it to geodetic reality.
2. **DEM-preserving fusion:** By high-pass filtering the ML detail ($<30\text{ m}$) onto the coarse DEM ($ \ge 30\text{ m}$), DepthWizard adds **no bias at the DEM's own scale** (30 m cell means of DSM − DEM average 0.00 m; single cells still differ by a few metres). Measured, not guaranteed: the DSM beat the DEM on all 3 Swiss test tiles and on 5 of 6 Sikkim sites.
3. **Geodetic Safety (C-1 Guard):** Eliminates silent 30–90 m vertical errors by auto-detecting ellipsoidal vs orthometric datums (CartoDEM/Copernicus) and transforming strictly to EGM2008 with zero "ballpark" approximations.
4. **Zero-Backend Field Deployment:** Solves the disaster/tactical communication bottleneck by exporting a **single self-contained HTML 3D scene (7–11 MB)** that boots instantly on any laptop/tablet without internet or local servers.
