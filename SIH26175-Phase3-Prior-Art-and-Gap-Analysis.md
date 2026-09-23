# Phase 3 — Existing Solutions, Prior Art & Technical Gap Analysis

SIH26175 DepthWizard · 2026-09-20

Scope note: Phase 3 answers one question — *how close is existing technology and research to solving SIH26175 end-to-end?* It does not design our solution and does not begin Phase 4. Every important claim carries an evidence tag `[E#]` resolved in Section 27 (Evidence Ledger). Claims inherited from Phase 1/2 were re-verified against primary sources; where the inherited claim was wrong, this report says so explicitly rather than carrying it forward.

**Corrections to the Phase 1/2 inputs, established during this phase (read first):**

| Inherited claim | Status after verification | Evidence |
| --- | --- | --- |
| "SRTM heights are referenced to the WGS84 ellipsoid" (Phase 2 §10) | **WRONG.** The NASA/USGS SRTM V3 User Guide states elevation is "in meters as referenced to the WGS84/EGM96 geoid." SRTM is geoid (orthometric-like), not ellipsoidal. | [E2] |
| "Final evaluation will use ISRO's own RGB optical satellite imagery — confirmed directly from the repository text" (Phase 2 §7; Phase 1 finding #7) | **NOT FOUND.** Neither README commit (2026-07-29, 2026-08-29) of `IMG-PROCESS-SAC/SIH2026` contains any statement about evaluation imagery source. The README says only "single-view optical satellite imagery (PNG, JPG, or TIFF)". Treat the ISRO-imagery assumption as *reasonable inference*, not a verified fact. | [E1] |
| "THE Benchmark" attributed to "Srivastava et al." (Phase 2 §9) | **Misattributed.** arXiv 2112.14985 is Xiong, Huang, Hu & Zhu (TUM), 2021. | [E7] |
| Patent US 11361196 "Object height estimation from monocular images" presented as directly relevant | **Not remote sensing.** Assignee is Zoox (autonomous vehicles); it covers street-level object height from crops. The relevant patent is Intermap's US 12056888 (building heights from mono satellite imagery). | [E33] |
| GAMUS "GSD not stated"; "five cities" | GSD **is** stated: 0.33 m. The paper describes 11,507 tiles from five cities, but the **Hugging Face release contains 8,724 tiles from three cities only (Philadelphia, NYC, Washington DC)**, as `.h5` arrays with no georeferencing. | [E5][E6] |
| "All methods degrade sharply on unseen cities" (HTC-DC Net) | **Partially true.** São Paulo (RMSE 9.30 m) and Guangzhou (building RMSE 11.84 m) degrade; **Los Angeles does not** (3.38 m, better than the seen-city 4.49 m). Degradation tracks urban-morphology shift, not "unseen" per se. | [E4] |
| SynRS3D GSD "0.05–1 m" | Paper states 0.09–1 m. | [E28] |
| Links given for Amirkolaee & Arefi (2301.04581), UniDepth (2507.02148), Marigold (2409.04086) | All three arXiv IDs resolve to different papers. 2301.04581 is Mao et al. 2023 (itself relevant single-view building-3D prior art). | [E8] |
| "DA-V2 project page states plainly that NYU/KITTI-trained models fail to generalize" | **Not on the project page.** The DA-V2 metric README says the metric heads are fine-tuned on Hypersim/Virtual-KITTI and makes no generalization-failure statement. The generalization point is argued in the paper, not the repo. | [E26] |

---

## 1. Executive Prior-Art Summary

**The honest one-paragraph answer.** Every individual stage of SIH26175 has prior art, and several stages are commodity engineering. The *elevation half* has a mature, decade-old research lineage under the name **monocular height estimation (MHE)** — single nadir optical image → height-above-ground (nDSM) — with public benchmarks, an operational global product (GlobalBuildingAtlas, 3 m, RMSE 5.5 m) and a canopy-height analogue at 1 m (Tolan et al.). The *visualization half* — heightmap → textured mesh → first-person / orbit navigation — is fully established in Cesium, deck.gl, Qgis2threejs, Unity and Babylon.js. **What has not been demonstrated end-to-end by any single system found** is the specific chain the PS asks for: a *general-purpose pretrained monocular depth backbone* (not a purpose-trained MHE network), applied zero-shot or lightly adapted to *arbitrary* nadir RGB, converted into an *absolute, datum-referenced DSM* via *coarse DEM or sparse GCP calibration*, validated with RMSE/MAE/correlation across *urban, sparse, hilly and forested* terrain, and shipped as a *standalone* interactive product. The evidence indicates that the hardest links in that chain are (a) zero-shot foundation-model depth is measurably unreliable on nadir imagery — δ₁ ≈ 0 % for most metric models on nadir UAV views, "severe scale drift" on DFC2019 satellite scenes — so "pretrained backbone + calibration" without domain adaptation is unsupported by any positive evidence found; (b) a 30 m DEM cannot supply building-scale vertical reference and is itself biased upward by several metres in forest; (c) no dataset found pairs RGB with LiDAR-grade *absolute* DSM across all four PS terrain types; and (d) published MHE accuracy on the official spatially-held-out DFC19 test set (RMSE ≈ 5.5 m all pixels, ≈ 10.7 m on buildings) is 2–3× worse than the same task on random patch splits (≈ 2.1 m), showing that most headline numbers in the field are inflated by spatial leakage. Nothing in the PS is *conceptually* novel; the remaining difficulty is *quantitative and integrative*.

**Maturity map (descriptive, not a ranking):**

| Stage | Maturity | One-line evidence |
| --- | --- | --- |
| Relative monocular depth (natural images) | Mature | DA-V2/Metric3D/UniDepth/MoGe families; [E26][E15] |
| MHE on nadir imagery, in-distribution | Mature research, operational at 3 m | HTC-DC Net RMSE 1.3–4.5 m in-distribution; GlobalBuildingAtlas RMSE 5.5 m global [E4][E12] |
| Cross-region MHE generalization | Open | GBH São Paulo/Guangzhou 9–12 m; GBA South America 8.9 m [E4][E12] |
| Zero-shot foundation depth on nadir imagery | Negative evidence | AerialMetric δ₁≈0 %; Sat3R zero-shot DA2 MAE 4.59 m with scale drift; Depth Any Canopy zero-shot MAE 3× worse than fine-tuned [E15][E16][E14] |
| Scale-shift alignment from sparse metric anchors | Mature (camera frame) | Wofk et al. 150 points; Prior Depth Anything / Marigold-DC [E9][E27] |
| Coarse DEM → building-scale DSM | Not demonstrated | 30 m DEM lacks the signal; FABDEM shows the DEM itself is biased 1.6–5.2 m by buildings/forest [E19] |
| nDSM + terrain → absolute georeferenced DSM from one image | Not found as a certified pipeline | No source closes this loop end-to-end [E4][E12][E16] |
| Heightmap → textured mesh → navigation | Commodity | Cesium, deck.gl TerrainLayer, Qgis2threejs, Unity, Babylon [E25] |
| Validation across four terrain types with LiDAR reference | No public benchmark exists | Datasets are urban/suburban; GeoNRW is the closest with a forest class [E28][E29] |

---

## 2. SIH26175 Requirement Decomposition

| # | SIH Requirement | Existing Technology | Research Maturity | Existing Implementations | Remaining Difficulty | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | RGB ingestion (PNG/JPG/TIFF) | Image codecs; GDAL/rasterio for TIFF | Established | GDAL, rasterio, Pillow | None technical; only robustness to malformed metadata | [E25] (GDAL is the base of every tool cited) |
| 2 | Remote-sensing preprocessing (tiling, normalization, pansharpening awareness) | Standard RS practice; tiling with overlap | Established | Every MHE codebase crops to 256–1024 px patches | Seam artefacts at tile borders; models trained on 0.09–3 m GSD assume a GSD range | [E4][E28] |
| 3 | Monocular depth (pretrained backbone) | DA-V2, DA3, Metric3D v2, UniDepth v2, MoGe-2, Depth Pro, Marigold | Mature for natural images | Open weights; licences vary (DA-V2 Base/Large are CC-BY-NC-4.0) | Nadir domain is out-of-distribution; see §13 | [E26][E15] |
| 4 | Relative-depth estimation on nadir RGB | Same backbones, zero-shot | Weak positive evidence only | Sat3R, AerialMetric report zero-shot baselines | Blurred boundaries, scale drift; ordinal correctness not quantified for satellites | [E15][E16] |
| 5 | Domain adaptation (natural → nadir) | Fine-tuning on RGB–nDSM pairs; synthetic-to-real UDA (RS3DAda); RPC-aware pseudo-labels (Sat3R) | Active research | SynRS3D/RS3DAda, Depth Any Canopy, Sat3R, Tolan et al. | Requires RS height data; hilly/forest coverage lacking | [E28][E14][E16][E13] |
| 6 | Scale recovery (relative → metric) | LS scale-shift in inverse-depth; RANSAC; dense scale maps; prior-conditioned foundation models | Mature in camera frame | Wofk et al.; Prior Depth Anything; Marigold-DC; PromptDA | Camera-frame formulation does not map 1:1 to ortho geometry; anchor density needed for terrain scale is undocumented | [E9][E27] |
| 7 | Metric depth (direct) | Metric3D v2, UniDepth, Depth Pro, MoGe-2 use camera intrinsics / canonical camera | Mature for perspective cameras | Open weights | Orthorectified imagery has no meaningful focal length; zero-shot metric fails on nadir (δ₁ ≈ 0 %) | [E15] |
| 8 | Height estimation (above ground, nDSM) | MHE networks (IM2HEIGHT → HTC-DC Net → TSE-Net); DINOv2+DPT heads | Mature research | HTC-DC-Net (no licence), tse-net, SynRS3D, GlobalBuildingAtlas | Long-tail underestimation of tall objects; cross-region drop | [E4][E12][E28] |
| 9 | Absolute elevation | nDSM + DTM; or direct DSM regression (GeoNRW-style first-return DSM) | Partially demonstrated | No single-image system outputs datum-referenced absolute DSM in found sources | Requires an external terrain source; datum reconciliation | [E2][E3][E29] |
| 10 | DEM-assisted calibration (SRTM/Copernicus) | DEM as low-frequency anchor; DEM-conditioned refinement | Used for terrain, not for building-scale | Panagiotou 2020 / Madani 2025 regress *to* SRTM/AW3D30 (relative only) | 30 m posting; DEMs are DSM-like with forest/building bias; only low-frequency scale can be recovered | [E2][E17][E18][E19] |
| 11 | GCP-assisted calibration | Photogrammetric GCP practice; sparse-anchor alignment | Mature in adjacent fields | None specific to monocular RS depth found | No paper found calibrates monocular RS depth with GCPs; count/distribution requirements unknown | [E9] (negative finding) |
| 12 | DSM generation (raster) | Trivial once heights known; DSM ≠ nDSM ≠ DTM | Established | Every MHE pipeline outputs nDSM rasters | Most "DSM" outputs in the field are actually nDSM | [E4][E5][E28] |
| 13 | GeoTIFF / geospatial output | GDAL GeoTIFF, COG; vertical CRS tagging | Established | GDAL | Vertical datum rarely tagged in practice | [E2][E3] |
| 14 | CRS preservation | GDAL/PROJ | Established | GDAL | Reprojection cost/resampling artefacts | [E25] |
| 15 | Vertical-reference handling | EPSG vertical CRS (EGM96 = 5773, EGM2008 = 3855); PROJ geoid grids | Established but often ignored | GDAL/PROJ | SRTM = EGM96, Copernicus = EGM2008, AW3D30 = EGM96, CartoDEM/CartoDSM = "WGS-84" (ellipsoid vs geoid not clarified) | [E2][E3][E23] |
| 16 | Reference validation (RMSE/MAE/corr.) | ASPRS/ISPRS practice; Höhle & Höhle robust stats; DEMIX terrain-stratified criteria | Established methodology | xdem, DEMIX tools | LiDAR reference absent for India; coarse-reference validation protocol not found | [E32][E23] |
| 17 | Terrain reconstruction (heightfield) | Regular grid mesh; RTIN (MARTINI); Delatin; quadtree LOD (Cesium) | Commodity | deck.gl TerrainLayer, Cesium, cesium-terrain-builder | Noise → spikes; needs error-bounded simplification | [E25] |
| 18 | Texture projection | UV mapping of ortho onto heightfield | Commodity | All engines | Stretching on steep discontinuities | [E25] |
| 19 | 3D mesh generation | Same as 17 | Commodity | Qgis2threejs glTF export | Size limits in browser | [E25] |
| 20 | First-person navigation | Game-engine controllers; Cesium for Unity/Unreal | Commodity | Unity, Unreal, Three.js PointerLockControls | Collision with noisy mesh | [E25] |
| 21 | Aerial camera navigation | Orbit/fly controls | Commodity | Cesium, Three.js OrbitControls | None | [E25] |
| 22 | Height analysis (pick → metres) | Raster lookup; local ground estimation for nDSM | Established | Qgis2threejs measure tool, GIS | Distinguishing DSM vs height-above-ground in UI | [E25] |
| 23 | Slope analysis | gdaldem slope/aspect; shader gradients | Established | GDAL | Slope from a noisy predicted DSM is noise-amplifying | [E32] (DEMIX uses slope as a criterion) |
| 24 | Interactive visualization | WebGL/WebGPU engines | Commodity | Three.js, Babylon.js, Cesium | Full-res satellite tiles need LOD | [E25] |
| 25 | Standalone deployment | Electron/PyInstaller/Unity builds; static HTML export | Established | Qgis2threejs standalone HTML; Unity builds | Bundling a GPU inference stack with a viewer | [E25] |

---

## 3. Existing End-to-End Systems

Systems attempting a substantial part of RGB → depth/height → DSM → 3D. Classification key: **DIRECT** (single nadir/near-nadir optical image → height/elevation raster, on remote-sensing imagery), **PARTIAL** (multi-view, or non-RS, or stops before DSM/3D), **RELATED**, **NOT DIRECTLY RELEVANT**.

### 3.1 GlobalBuildingAtlas (TUM, 2025) — DIRECT PRIOR ART (elevation half)
Name: GlobalBuildingAtlas · Authors: Zhu, Chen, Zhang, Shi et al. (TUM) · Year: 2025 (ESSD 17) · Input: single PlanetScope surface-reflectance image, RGB+NIR, ~3 m · Platform: PlanetScope · Geometry: near-nadir ortho · Output: building nDSM → building-instance heights → LoD1 3D models · Relative/metric: metric (trained) · Type: height above ground / building height · Calibration: none needed (supervised regression) · Reference: airborne LiDAR nDSM from 168 city-scale regions (NA 39, EU 109, Oceania 17; none in Africa) · Evaluation: RMSE global 5.5 m; Oceania 1.5, Europe 4.1, N. America 5.3, **Asia 5.9**, S. America 8.9 · Geospatial: yes (global tiles) · Visualization: LoD1 models, no interactive viewer shipped · Deployment: dataset · Open source: code (HTC-DC Net) + data · Licence: ODbL (data) · Limitations (authors'): under-estimates high-rises in Asian/South-American cities; Africa never trained/validated. [E12]

### 3.2 HTC-DC Net (TUM, 2023) — DIRECT PRIOR ART (elevation half)
Input: single RS image (DFC19 WorldView-3 1.3 m; GBH PlanetScope 3 m; Vaihingen IRRG 0.09 m) · Output: nDSM · Metric: trained · Reference: LiDAR nDSM · Evaluation: RMSE 2.12 (DFC19 random split), 4.49 (GBH seen), 3.38/9.30 (LA/São Paulo held-out), 1.30 (Vaihingen) · Geospatial: patches only · Visualization: none · Open source: code + weights at zhu-xlab/HTC-DC-Net · **Licence: none declared in repository** (all-rights-reserved by default) · Limitations: 2–3× degradation on cities with different morphology; 57 % of pixels < 1 m → long-tail underestimation. [E4]

### 3.3 Monocular Geocentric Pose (JHU/APL, 2020–2021) — DIRECT PRIOR ART (oblique variant)
Christie, Foster, Hagstrom, Hager, Brown; CVPR EarthVision 2021 · Input: single **oblique** WorldView-2/3 image · Output: above-ground height + per-pixel flow to orthorectify · Metric: trained · Reference: LiDAR (and MVS shown comparable) · Datasets: US3D JAX/OMA/ATL + San Fernando (ARG) · Evaluation on the **official DFC19 Track-1 test set**: all-pixel RMSE 5.46 m, buildings 10.69 m (prior SOTA Kunwar 9.26 / 19.65) · Open source: pubgeo/monocular-geocentric-pose · Limitation: ARG generalization poor; R² poor for rare tall structures. [E11]

### 3.4 DFC2019 Track 1 "Single-view semantic 3D" (IEEE GRSS / JHU-APL, 2019) — DIRECT PRIOR ART (benchmark)
Input: one *unrectified* WorldView-3 image · Output: semantic label + above-ground height (m) · Metric: mIoU-3 (correct class AND |Δh| < 1 m) · Winner: Kunwar (NestAI), U-Net ensemble, mIoU-3 0.5571 · Data: US3D, Jacksonville + Omaha, LiDAR reference · Licence: repo MIT; data via IEEE DataPort. [E10][E11]

### 3.5 Tolan et al. / Meta–WRI Canopy Height (2024) — DIRECT PRIOR ART (vegetation)
Input: single Maxar ~0.5 m RGB · Backbone: DINOv2 self-supervised + DPT-style decoder · Output: canopy height (m, above ground) · Reference: NEON/USA aerial LiDAR · Evaluation: MAE 2.8 m, ME 0.6 m · Deployment: global 1 m map on AWS/GEE · Relevance: shows a foundation-model encoder + LiDAR supervision yields metric vegetation height from one nadir image — the "forested" terrain analogue. [E13]

### 3.6 Depth Any Canopy (ECCVW 2024) — DIRECT PRIOR ART (fine-tuned DA-V2 on nadir)
Fine-tunes Depth Anything V2 for canopy height on NEON 0.1 m RGB / 1 m LiDAR CHM · Zero-shot DA-V2-S MAE 0.4116 (normalized) → 0.1410 after fine-tuning; authors: "it is necessary to adapt the model to this new unseen task" · Open source. [E14]

### 3.7 Sat3R (2026) — PARTIAL PRIOR ART (multi-view fusion)
Fine-tunes DA-V2 with RPC-derived pseudo depth (SiLog loss, max depth 150 m) → per-view metric depth → **multi-view fusion** → RPC back-projection → DSM (p90 rasterization) · DFC2019 six scenes: zero-shot DA2 MAE 4.59 m, DA3 4.37 m, Sat3R 2.82 m, optimization-based SatDN 1.44 m · Authors: zero-shot baselines "suffer from severe scale drift and blurry boundaries" · Multi-view → PARTIAL, but the zero-shot numbers are the best direct evidence found of foundation-model behaviour on satellite imagery. [E16]

### 3.8 Panagiotou et al. 2020 "Generating Elevation Surface from a Single RGB Remotely Sensed Image" — PARTIAL PRIOR ART
cGAN (U-Net + PatchGAN) · Input: Sentinel-2 RGB · Target: ALOS World 3D elevation · Output: **relative within each image** (repo README) · No metric accuracy reported in README · MIT. Despite the title, this is a terrain-texture-to-relative-elevation translator, not a metric DSM system. [E17]

### 3.9 Madani et al. 2025 (Monash Indonesia) "DEM estimation from RGB satellite imagery using generative DL" — PARTIAL PRIOR ART
pix2pix · Landsat-5/7 30 m RGB → SRTM · 12,357 global samples · Output normalized to [-1, 1]; RMSE 0.467 (normalized), SSIM 0.21 · Authors: underperforms in lowland/residential areas. Illustrates that "RGB → DEM" GANs learn relative terrain texture, not elevation. [E18]

### 3.10 Maxar Precision3D (formerly Vricon) — RELATED (multi-view stereo, commercial)
50 cm DSM/DTM/point cloud; 3 m SE90/LE90 absolute, no GCPs; **multi-view stereo from many passes** — the same end-user product via a different, more accurate technique. [E30]

### 3.11 Google Open Buildings 2.5D Temporal (2023–24) — PARTIAL
Sentinel-2 10 m *time series* → building presence/height at 4 m effective resolution; height MAE 1.5 m against Global-North ground truth; authors warn of possible large errors on tall buildings in the Global South. Multi-frame → PARTIAL. [E31]

### 3.12 ISRO CartoDEM / CartoDSM (NRSC) — RELATED (stereo, government)
Cartosat-1 along-track **stereo** (2.5 m) → CartoDEM 30 m (free, Bhuvan) and 2.5 m CartoDSM · Vertical 8 m LE90, horizontal 15 m CE90 · Datum stated "WGS-84 (horizontal & vertical)" · Known defects: "sinks and spikes in hill shadows, snow areas", ~1 % gaps. This is the evaluator's home-grown reference point for what a *stereo* pipeline achieves over India. [E23]

### 3.13 Blackshark.ai SYNTH3D — RELATED (commercial, claims unverified)
Semantic 3D globe (terrain, buildings "with accurate heights", vegetation) derived by ML from 2D satellite/aerial imagery; streamed via Eagle 3D. No public accuracy figures or method description found → **UNVERIFIED** as to single-view monocular height. [E35]

### 3.14 Intermap patent US 12056888 (2021) — RELATED (commercial IP)
"Calculating building heights from mono imagery": orthorectify against a DTM, detect roof edges, use building lean/shadow geometry. Confirms the mono-image building-height problem is commercially patented territory. [E33]

### 3.15 Qgis2threejs / deck.gl TerrainLayer / Cesium for Unity & Unreal — DIRECT PRIOR ART (visualization half)
Qgis2threejs: QGIS DEM + imagery → Three.js web scene, standalone HTML and glTF export, GPL-3.0, actively maintained. deck.gl TerrainLayer: heightmap image + texture → mesh via MARTINI with an error-tolerance parameter (default 4 m). Cesium for Unity / Unreal: Apache-2.0, streams terrain + imagery into game engines with first-person controllers. Unity Terrain: RAW 16-bit heightmap import. Babylon.js: `CreateGroundFromHeightMap`. All are working, documented systems for the PS's visualization chain. [E25]

### 3.16 Others found (brief)
- **DFC2023 Track 2** (optical + SAR → nDSM, 1,773 images, 12 cities, 5 continents) — PARTIAL (uses SAR). [E34]
- **Mao et al. 2023** (2301.04581) "Elevation Estimation-Driven Building 3D Reconstruction from Single-View RS Imagery" — DIRECT (single-view building 3D). [E8]
- **Mahmud et al. CVPR 2020** boundary-aware 3D building reconstruction from a single overhead image — DIRECT (cited with DFC19 numbers in [E11]).
- **IM2ELEVATION (2020, Dublin)**, **Amirkolaee & Arefi (2019)**, **IMG2nDSM (2021)**, **IM2HEIGHT (2018)** — DIRECT lineage, aerial nDSM. [E36]
- **SynRS3D / RS3DAda (NeurIPS 2024)** — DIRECT (synthetic training + UDA for nDSM). [E28]
- **UseGeo (ISPRS)** and **AerialMetric (2026)** — RELATED benchmarks for monocular depth on UAV imagery with LiDAR/RTK reference. [E38][E15]
- **Prior Depth Anything (ICLR 2026), Marigold-DC (ICCV 2025), PromptDA** — RELATED: fuse sparse/coarse metric priors with relative foundation depth. [E27]

---

## 4. Direct Prior-Art Search

Formulations searched (this phase and Phase 2 combined): "single RGB image DSM", "single image DSM reconstruction", "single-view DSM", "single-view surface reconstruction remote sensing", "single satellite image DSM", "single aerial image DSM", "single image elevation estimation", "single image height estimation remote sensing", "monocular DSM generation", "monocular remote sensing DSM", "monocular satellite depth", "monocular aerial depth", "RGB satellite image height estimation", "single image building height estimation", "single-view 3D terrain reconstruction", "satellite image 3D reconstruction", "optical image to DSM", "RGB to surface model", "remote sensing monocular depth", "single-image surface reconstruction aerial imagery", plus synonyms: "geocentric pose", "nDSM estimation", "canopy height from RGB", "elevation surface generation", "DEM estimation from RGB satellite imagery", "Depth Anything remote sensing", "RPC-aware depth".

**Convergence finding.** The problem lives under four names that do not fully cross-cite: (1) **monocular height estimation / nDSM estimation** (TGRS/ISPRS lineage, [E4][E7][E36]); (2) **geocentric pose** (JHU-APL, oblique, [E11]); (3) **canopy height mapping** (ecology/RS, [E13][E14]); (4) **image-to-DEM translation** (GAN lineage, [E17][E18]). A fifth, very recent thread adapts **depth foundation models** to satellite geometry ([E16], [E15]). Phase 2 found only lineage (1). The "single RGB → *absolute* DSM with SRTM/GCP calibration" formulation returned no paper or system; the closest are Sat3R (multi-view, RPC scale) and the GAN DEM papers (relative only).

---

## 5. Prior-Art Feature Matrix

Values: YES / NO / PARTIAL / UNCLEAR. "Absolute Elevation" = datum-referenced height; "DSM" = surface elevation raster (nDSM counts as PARTIAL).

| System / Paper | RGB | Single View | Remote Sensing | Relative Depth | Metric Depth | Height | Absolute Elevation | DSM | DEM Fusion | GCP | Geo Metadata | 3D Mesh | Interactive | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HTC-DC Net [E4] | YES (IRRG on Vaihingen) | YES | YES | NO | NO | YES (nDSM, m) | NO | PARTIAL (nDSM) | NO | NO | NO (patches) | NO | NO | YES (LiDAR RMSE) |
| GlobalBuildingAtlas [E12] | YES (+NIR) | YES | YES | NO | NO | YES (m) | NO | PARTIAL (building nDSM) | NO | NO | YES | PARTIAL (LoD1) | NO | YES (LiDAR, per continent) |
| Geocentric Pose [E11] | YES | YES (oblique) | YES | NO | NO | YES (m) | NO | PARTIAL | NO | NO | PARTIAL | NO | NO | YES (official DFC19 test) |
| DFC2019 Track 1 [E10] | YES | YES | YES | NO | NO | YES | NO | PARTIAL | NO | NO | NO | NO | NO | YES (mIoU-3) |
| Tolan et al. canopy [E13] | YES | YES | YES | NO | NO | YES (canopy, m) | NO | PARTIAL (CHM) | NO | NO | YES | NO | NO | YES (LiDAR MAE) |
| Depth Any Canopy [E14] | YES | YES | YES | PARTIAL (DA-V2 base) | NO | YES (normalized) | NO | PARTIAL | NO | NO | NO | NO | NO | YES |
| Sat3R [E16] | YES | NO (multi-view) | YES | YES (DA-V2 base) | YES (RPC-fine-tuned) | NO | YES (RPC altitude) | YES | NO | NO | YES (RPC) | NO | NO | YES (DFC2019 MAE) |
| Panagiotou 2020 [E17] | YES | YES | YES | YES (relative elevation) | NO | NO | NO | PARTIAL (DEM-like) | PARTIAL (AW3D30 as target) | NO | UNCLEAR | NO | NO | UNCLEAR (none in README) |
| Madani 2025 [E18] | YES | YES | YES | YES (normalized) | NO | NO | NO | PARTIAL | PARTIAL (SRTM as target) | NO | NO | NO | NO | PARTIAL (normalized RMSE only) |
| Maxar Precision3D [E30] | YES | NO | YES | NO | YES | YES | YES | YES | NO | NO (none needed) | YES | YES (point cloud) | PARTIAL (via ArcGIS) | YES (3 m SE90) |
| Open Buildings 2.5D [E31] | YES (S2) | NO (time series) | YES | NO | NO | YES | NO | PARTIAL | NO | NO | YES | NO | NO | YES (MAE 1.5 m, Global North) |
| CartoDSM [E23] | PAN | NO (stereo) | YES | NO | YES | NO | YES | YES | NO | YES (GCP-based triangulation) | YES | NO | NO (Bhuvan viewer only) | YES (8 m LE90) |
| SynRS3D / RS3DAda [E28] | YES | YES | YES (synthetic + real) | NO | NO | YES (nDSM) | NO | PARTIAL | NO | NO | NO | NO | NO | YES (11 real datasets) |
| Wofk et al. GA+SML [E9] | YES | YES | NO (indoor VIO) | YES | YES (aligned) | NO | NO | NO | NO | PARTIAL (150 sparse metric points) | NO | NO | NO | YES |
| Prior Depth Anything / Marigold-DC [E27] | YES | YES | NO | YES | YES (with priors) | NO | NO | NO | PARTIAL (low-res depth prior) | PARTIAL (sparse points) | NO | NO | NO | YES (7 datasets, indoor/driving) |
| Qgis2threejs [E25] | YES (texture) | n/a | YES | n/a | n/a | n/a | consumes any DEM | consumes | NO | NO | YES | YES | YES (orbit; walk UNCLEAR) | NO |
| Cesium for Unity/Unreal [E25] | YES | n/a | YES | n/a | n/a | n/a | consumes | consumes (quantized-mesh) | NO | NO | YES | YES | YES (first-person) | NO |
| Blackshark SYNTH3D [E35] | YES | UNCLEAR | YES | UNCLEAR | UNCLEAR | YES (claimed) | UNCLEAR | UNCLEAR | UNCLEAR | UNCLEAR | YES | YES | YES (streamed) | UNCLEAR (none public) |

**Key classifications with evidence:** Sat3R is NOT single-view because per-view depth maps are fused by a multi-view module before RPC back-projection [E16 §3.4]. GlobalBuildingAtlas is DIRECT because heights are "derived exclusively from PlanetScope satellite imagery" by a single-image model [E12]. Panagiotou/Madani are PARTIAL on metric because outputs are normalized/relative by the authors' own statements [E17][E18].

---

## 6. Monocular Depth Model Landscape

The question here is *what the evidence establishes*, not which model to use.

| Model | Original task / training domain | Rel./Metric | Camera assumption | Scale ambiguity | Datasets | Remote-sensing evidence | Documented performance relevant to SIH | Limitations | Compute | Open source / licence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiDaS v3.1 (Intel, 2023) | Relative depth, 12 mixed datasets | Relative (affine-invariant) | None | Per-image scale+shift unknown | Mixed indoor/outdoor/synthetic | None found | Baseline for ZoeDepth | Affine-invariant only | Small–large variants | MIT [Phase 2, not re-verified here] |
| DPT (2021) | Dense prediction transformer head | Relative | None | Per-image | Same as MiDaS | Used as decoder in SynRS3D/RS3DAda, Tolan et al. | The DPT head *architecture* transfers well when retrained with LiDAR supervision [E13][E28] | Needs retraining | ViT-L | MIT |
| ZoeDepth (2023) | Relative pretrain + metric bins fine-tuned NYU/KITTI | Both | Implicit (training cameras) | Metric only in-domain | NYU, KITTI | AerialMetric zero-shot: AbsRel 93–97 %, δ₁ 0 % on aerial oblique [E15] | Catastrophic failure off-domain | Domain-locked metric head | Medium | MIT |
| Depth Anything V2 (2024) | Relative; metric heads fine-tuned on Hypersim (indoor) / Virtual KITTI (outdoor) | Both | None for relative; implicit for metric | Relative: per-image affine | 595 K synthetic + 62 M pseudo-labelled | Zero-shot on DFC2019 satellite: MAE 4.59 m with "severe scale drift" [E16]; zero-shot canopy MAE 3× worse than fine-tuned [E14]; fine-tuning works (Depth Any Canopy, Sat3R) | Best-documented backbone for RS *after* fine-tuning | Blurry boundaries zero-shot | S 24.8 M / B 97.5 M / L 335 M; Giant "coming soon" | **Small: Apache-2.0; Base/Large/Giant: CC-BY-NC-4.0** [E26] |
| Depth Anything 3 (2025–26) | Any-view geometry | Both | Handles multi-view | — | — | Zero-shot DFC2019 MAE 4.37 m [E16]; benchmarked in AerialMetric [E15] | Similar zero-shot behaviour to DA-V2 on satellite | Details **UNVERIFIED** in this phase | — | UNVERIFIED |
| Marigold (2023–24) | Diffusion-based affine-invariant depth from Stable Diffusion | Relative | None | Per-image affine | Synthetic (Hypersim, VKITTI) | None found; Marigold-DC (sparse-point completion) is the relevant offshoot [E27] | Zero-shot completion from extremely sparse points | Slow (diffusion) | SD-scale | Apache-2.0 (repo) — not re-verified |
| Metric3D v2 (2024) | Zero-shot metric depth via canonical camera transform | Metric | **Requires/estimates focal length** | Resolved via camera model | Broad multi-dataset | AerialMetric zero-shot: AbsRel 63–81 %, δ₁ ≈ 0 % [E15] | Camera-model scale recovery is meaningless for orthophotos (no perspective) | Fails off-domain | ViT-L | BSD-2 (not re-verified) |
| UniDepth v1/v2 (2024–25) | Metric depth + self-prompted camera | Metric | Predicts intrinsics | Via camera module | Broad | AerialMetric zero-shot: v1 AbsRel 80–89 %, v2 17–31 %, δ₁ 24–73 % [E15] | v2 is the least-bad zero-shot metric model on oblique aerial | Still poor on nadir | ViT-L | CC-BY-NC-4.0 (not re-verified) |
| MoGe / MoGe-2 (2024–25) | Affine-invariant point maps; MoGe-2 metric | Both | Predicts | — | Broad | AerialMetric zero-shot AbsRel 26–71 %; after aerial fine-tune 10.3 % (City) [E15] | Adaptable | Zero-shot unreliable | ViT-L | MIT (not re-verified) |
| Depth Pro (Apple, 2024) | Metric, sharp boundaries, focal estimation | Metric | Estimates focal | Via focal | Broad | AerialMetric zero-shot AbsRel 92–98 %, δ₁ 0 % [E15] | Fails on aerial | — | Large | Apple licence (not re-verified) |

**What the evidence establishes (not what we should pick):**
1. **Relative structure transfers partially; metric scale does not transfer at all to nadir/aerial views.** Every metric-depth model tested zero-shot on aerial imagery shows δ₁ near 0 % [E15]; on satellite scenes DA-V2/DA-3 show scale drift even with multi-view fusion [E16]. **ESTABLISHED (two independent 2026 sources).**
2. **Fine-tuning a foundation encoder (DINOv2/DA-V2) with LiDAR-derived height supervision produces metric nadir height** — canopy MAE 2.8 m [E13], normalized canopy MAE 0.41 → 0.14 [E14], DFC2019 MAE 4.59 → 2.82 m [E16]. **ESTABLISHED.**
3. **Camera-intrinsics-based metric recovery (Metric3D, UniDepth, Depth Pro) has no geometric basis on orthorectified imagery**, whose effective focal length is infinite. Established geometry; model-behaviour consequence confirmed empirically by [E15]. 
4. **Licensing is not uniform**: DA-V2 Base/Large are non-commercial (CC-BY-NC-4.0); only Small is Apache-2.0 [E26]. HTC-DC Net has no licence [E4].

---

## 7. Remote-Sensing Prior Art

| Work | Platform | Nadir/Oblique | GSD | Image type | Output (precise quantity) | Ground truth | Model | Training data | Evaluation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IM2HEIGHT 2018 [E36] | Aerial | Nadir ortho | ~0.05–0.09 m | RGB/IRRG | **nDSM (height above ground)** | LiDAR nDSM | FCN encoder-decoder | Potsdam/Vaihingen | RMSE | Small towns only |
| Amirkolaee & Arefi 2019 [E36] | Aerial | Nadir | 0.09 m | IRRG | nDSM/DSM | LiDAR | Encoder-decoder + patch merging | ISPRS | RMSE; strong baseline in [E4] | Same |
| IM2ELEVATION 2020 [E36] | Aerial | Nadir | high-res | RGB | DSM / building height | LiDAR 2015 vs optical 2017 (registration issues noted) | Conv-deconv | Dublin | RMSE/MAE | Temporal misregistration harms MAE |
| DFC2019 Track 1 [E10][E11] | Satellite WV-3 | Near-nadir, **unrectified** | ~1.3 m (as used) | Pan/VNIR RGB | **Above-ground height (m)** | LiDAR (80 cm NPS) | U-Net ensembles | US3D JAX/OMA | mIoU-3 (1 m); RMSE 5.5–9.3 m all-pixel on official test set | Two US cities |
| Geocentric Pose 2021 [E11] | Satellite WV-2/3 | **Oblique** | 0.3–0.5 m | RGB | Height above ground + orientation | LiDAR; MVS shown equivalent | ResNet U-Net | US3D + ARG | RMSE 5.46 m all / 10.69 m bldg | ARG poor; rare tall buildings |
| HTC-DC Net 2023 [E4] | Satellite + aerial | Nadir | 0.09 / 1.3 / 3 m | RGB(+) | **nDSM** | LiDAR nDSM | Classification-regression, HTC-AdaBins | DFC19, GBH, Vaihingen | RMSE by pixel/building | Random patch splits on DFC19 & Vaihingen; cross-city drop |
| THE Benchmark 2021 [E7] | Mixed | Nadir | mixed | RGB | nDSM | LiDAR | Transfer-learning study | DFC19, Vaihingen, Potsdam… | Cross-dataset transfer | Confirms transfer difficulty (details not re-read this phase) |
| GlobalBuildingAtlas 2025 [E12] | PlanetScope | Nadir | 3 m | RGB+NIR | **Building height (nDSM → per-instance)** | LiDAR, 168 regions | HTC-DC Net | 187,239 pairs | RMSE 1.5–8.9 m by continent | Under-estimates high-rises; no Africa |
| SynRS3D 2024 [E28] | Synthetic | Nadir | 0.09–1 m | RGB | **nDSM** (exact, synthetic) | Synthetic | DINOv2 + DPT | 69,667 images | MAE/RMSE/δ/F1-HE on 11 real datasets | Sim-to-real gap remains in T.D.1 |
| Tolan et al. 2024 [E13] | Maxar | Nadir | ~0.5 → 1 m | RGB | **Canopy height (m)** | NEON ALS | DINOv2 + DPT | USA | MAE 2.8 m | US-trained |
| Depth Any Canopy 2024 [E14] | NEON aerial / Maxar | Nadir | 0.1 m / 1 m | RGB | Canopy height (normalized) | ALS CHM | DA-V2 fine-tuned | EarthView | MAE 0.13 (norm.) | Normalized units |
| Sat3R 2026 [E16] | WorldView (DFC2019) | Off-nadir multi-view | ~0.3–0.5 m | RGB | **Absolute DSM (RPC altitude)** | DFC2019 LiDAR DSM | DA-V2 fine-tuned | DFC2019 train | MAE 2.82 m | Multi-view required |
| Panagiotou 2020 [E17] | Sentinel-2 | Nadir | 10 m | RGB | **Relative elevation** | AW3D30 | cGAN | GEE export | none in README | Relative only |
| Madani 2025 [E18] | Landsat | Nadir | 30 m | RGB | **Normalized elevation** | SRTM | pix2pix | 12,357 global | RMSE 0.467 (norm.) | Weak in lowlands |
| AerialMetric 2026 [E15] | UAV | Nadir → -45° | cm | RGB | **Metric depth (camera frame)** | RTK + LiDAR/photogrammetry | Foundation models zero-shot / fine-tuned | 52 K real + 16 K synthetic | AbsRel/δ₁ | UAV altitudes 70–300 m only |

**Terminology separation, evidenced:**
- *Relative depth*: what DA-V2/MiDaS output; per-image affine ambiguity [E26][E9].
- *Metric depth*: camera-frame distance in metres; what Metric3D/UniDepth attempt and what Sat3R fine-tunes [E15][E16].
- *Height above ground (nDSM)*: what **every MHE paper and GlobalBuildingAtlas and canopy-height work actually predicts** [E4][E12][E13][E28]. GAMUS and SynRS3D ship nDSM, not DSM [E5][E28].
- *Absolute elevation*: datum-referenced; produced only by stereo/MVS systems (Precision3D, CartoDSM) or by Sat3R via RPC altitude [E30][E23][E16].
- *DSM*: surface elevation raster; GeoNRW's first-return LiDAR raster is a true DSM [E29]; most "DSM" in MHE titles is nDSM.
- *Building height*: per-instance aggregate of nDSM within a footprint (RMSE-B in [E4], GBA LoD1 [E12]).

---

## 8. Scale Calibration Prior Art

The four transformations are different problems and are treated separately.

### 8.1 RELATIVE DEPTH → METRIC DEPTH
| Method | Input reference | Assumptions | Parameters | # reference points | Robustness | Noise sensitivity | Terrain dependency | Limitations | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Global least-squares scale-and-shift (inverse-depth space) | Sparse metric depths (VIO/LiDAR/GCP) | Prediction is affine in inverse depth; single global (s, t) | 2 | Wofk et al.: 150 points suffice for global alignment | Low without RANSAC | Sensitive to outliers; shift term absorbs bias | None modelled | "Global alignment will not adequately resolve metric scale in all [regions]" | [E9] |
| Global + learned dense scale map (SML) | Same + RGB | Residual scale varies spatially and is predictable from image | Per-pixel scale | 150 | Higher | Learned | None modelled | Requires training on target domain | [E9] (up to 30 % iRMSE reduction) |
| RANSAC scale-shift | Same | Outlier fraction < 50 % | 2 | ≥ 3 | High | Robust | None | Still global | Standard practice (Phase 2, corroborated by [E9] discussion) |
| Region-aware scale adaptation (2025) | Sparse measurements | Scale varies by region/segment | Per-region | Sparse | Medium | Medium | Implicit | Camera-frame | [Phase 2 source 2507.14879, title verified only] |
| Prior-conditioned foundation models (Prior Depth Anything, Marigold-DC, PromptDA) | Sparse points or low-res metric depth map | Prior is metric and roughly co-registered | Dense output | Very sparse → dense; low-res prior → super-resolution | Zero-shot across 7 datasets | Designed for noisy priors | Indoor/driving only | Not tested on nadir RS; low-res prior analogous to a coarse DEM but never demonstrated with one | [E27] |
| Direct metric fine-tuning (SiLog / Hypersim-style) | Dense metric GT (or RPC pseudo-GT) | Training and test scale distributions match | Network weights | Full supervision | High in-distribution | — | Learned | Needs RS metric GT; max-depth clamp (150 m in Sat3R) | [E16][E26] |
| Camera-intrinsics metric recovery | Focal length | Perspective projection | 1 (focal) | 0 | — | — | — | **Inapplicable to orthophotos**; fails zero-shot on nadir UAV (δ₁≈0 %) | [E15] |

### 8.2 METRIC DEPTH → HEIGHT
For a true nadir orthographic view, depth-from-sensor ≈ H − z, so height is a sign flip plus offset — trivial in principle. In practice all found MHE systems **skip this step by regressing height directly** [E4][E12][E13]. For off-nadir imagery the mapping needs RPC/ray geometry (Sat3R back-projects along RPC rays [E16]; geocentric-pose regresses the parallax flow explicitly [E11]). No source evaluates "metric depth → height" as a separate error budget on satellite imagery.

### 8.3 HEIGHT (nDSM) → ABSOLUTE ELEVATION
DSM = nDSM + DTM. The reverse operation (adding a terrain surface back) requires a DTM at compatible resolution and datum. Found sources: dataset builders subtract DTM from DSM to *make* nDSM [E4][E5][E28] — no found system adds a coarse public DEM *back* and validates the resulting absolute DSM. Public DEMs are themselves DSM-like (SRTM C-band phase centre sits between ground and canopy top; Copernicus GLO-30 carries 5.15 m forest MAE and 1.61 m built-up MAE before FABDEM correction) [E19][E21], so nDSM + SRTM double-counts canopy/roof height wherever the DEM already includes it. **Not demonstrated end-to-end in any found source.**

### 8.4 ABSOLUTE ELEVATION → GEOSPATIAL DSM
Requires horizontal CRS + geotransform (from the GeoTIFF) and an explicit vertical CRS. Established engineering (GDAL/PROJ; EPSG:5773 EGM96 height, EPSG:3855 EGM2008 height). Failure risk is not technical but procedural: SRTM (EGM96) vs Copernicus (EGM2008) vs LiDAR (national geoid) vs CartoDSM ("WGS-84") [E2][E3][E23].

---

## 9. DEM / SRTM Prior Art

| Product | Source | Posting | Vertical datum | Surface type | Documented accuracy | Forest/building bias | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SRTM v3 (SRTMGL1) | C-band InSAR, Feb 2000; voids filled with ASTER GDEM2/GMTED/NED | 1″ (~30 m) | **WGS84/EGM96 geoid** | Radar phase centre — between ground and canopy top | Spec: 16 m abs / 10 m rel (90 %); ICESat: 0.60 ± 3.46 m (low relief, sparse trees); 3.53 ± 8.04 m (flat, dense vegetation); 5.61 ± 15.68 m (high relief) | Positive bias in vegetation; larger in evergreen than leaf-off deciduous | [E2][E21] |
| Copernicus DEM GLO-30 | TanDEM-X X-band, 2011–15 | 30 m | EGM2008 | DSM | DEMIX: best 1″ global DEM overall | Forest MAE 5.15 m, built-up 1.61 m before correction | [E3][E19][E32] |
| FABDEM | Copernicus GLO-30 with forests & buildings removed (random forest) | 30 m | EGM2008 | DTM-like | Forest MAE 2.88 m, built-up 1.12 m after correction | Residual 2–3 m in forest | [E19] |
| ALOS AW3D30 | PRISM optical stereo | 30 m | EGM96 | DSM | Uuemaa 2020: least slope-sensitive with SRTM/NASADEM | Optical stereo sees canopy top | [E3][E20] |
| ASTER GDEM | Optical stereo | 30 m | EGM96 | DSM | DEMIX: worst of the 1″ set | — | [E32] |
| CartoDEM (ISRO) | Cartosat-1 stereo 2.5 m | 30 m free (2.5 m CartoDSM) | "WGS-84" | DSM | 8 m LE90 vertical, 15 m CE90 | Stereo sees canopy top; sinks/spikes in hill shadows | [E23] |

**Roles found in existing systems:**
- *Supervision target*: SRTM and AW3D30 used as training targets for RGB→terrain GANs — yielding relative outputs only [E17][E18].
- *Orthorectification base*: DTM used to remove terrain distortion before mono building-height measurement (Intermap patent) [E33].
- *Validation reference*: DEMs are validated against LiDAR/ICESat; no found paper validates a *predicted* DSM against SRTM as reference (Phase 2 finding, unchanged).
- *Calibration / scale anchoring of monocular depth*: **not found in any paper**; only the SAC README suggests it [E1].
- *Terrain prior*: FABDEM/Copernicus used as low-frequency terrain in flood modelling; not in monocular height pipelines found.

**TRUE HIGH-RESOLUTION DSM vs INTERPOLATED TERRAIN.** A 30 m DEM resampled to 0.3–3 m contains no information at wavelengths below ~60 m. Any building-scale relief in a product built from "SRTM + monocular relative depth" comes entirely from the monocular model; the DEM can only constrain a very-low-frequency scale/offset. This is a geometric fact (Nyquist), corroborated by the accuracy figures above: the DEM's own error (3–8 m σ in vegetation/relief) exceeds typical building heights in sparse settlements. **ESTABLISHED.**

---

## 10. GCP Prior Art

No paper was found that calibrates *monocular* remote-sensing depth/height with ground control points. The table therefore records the closest evidenced analogues and marks the gap.

| Method | GCP count | GCP type | Distribution | Transformation | Output | Accuracy | Limitation | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Photogrammetric stereo triangulation (Cartosat-1 → CartoDEM) | Sparse per strip | Surveyed 3D points | Along 500 × 27 km strips | Bundle adjustment | Absolute DSM | 8 m LE90 | Stereo, not monocular | [E23] |
| Maxar Precision3D | **0** | — | — | Rigorous sensor model, multi-view | Absolute DSM | 3 m SE90 | Multi-view | [E30] |
| Wofk et al. global alignment | 150 sparse metric depths | VIO 3D points (camera frame) | Random over image | Scale + shift (2 params) | Metric depth | Up to 30 % further iRMSE reduction with dense refinement | Indoor, camera frame | [E9] |
| Marigold-DC / Prior Depth Anything | "extremely sparse" to dense | LiDAR/ToF points | Random | Dense guided completion | Metric depth | SOTA on 7 indoor/driving sets | Not RS | [E27] |
| Single-point anchoring | 1 | e.g. known building height | — | Scale only (shift fixed) | Metric height | UNKNOWN / NOT FOUND for RS | Cannot fix shift and scale jointly | — |
| Two-point calibration | 2 | Two known elevations | — | Scale + shift | Absolute elevation | UNKNOWN / NOT FOUND | Any error in either point propagates globally | — |
| Multi-point affine / regression / robust fit | ≥ 3 | Elevations | Distributed | Affine (or higher-order) with RANSAC | Absolute elevation | UNKNOWN / NOT FOUND for monocular RS | Spatially varying residual (established in camera frame [E9]) | — |

**Calibration vs validation GCPs.** Photogrammetric practice (ASPRS) separates control points used in adjustment from independent checkpoints used for accuracy reporting; DEM accuracy standards (Höhle & Höhle; ASPRS) require independent checkpoints [E32]. No MHE paper found uses GCPs at all — they use dense LiDAR rasters — so the calibration/validation split for a sparse-GCP monocular pipeline is **undocumented** in the literature found.

---

## 11. Dataset / Benchmark Prior Art

| Dataset | RGB | Single-view suitable | DSM | DTM | LiDAR | Building height | Terrain diversity | GSD | CRS / georef | Vertical ref. | Geography | Licence | GT quality | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAMUS (paper) | YES | YES | **NO (nDSM)** | NO | YES-derived | via nDSM | Urban only | 0.33 m | Not shipped | n/a (nDSM) | 5 US cities | CC-BY-4.0 | LiDAR + open city data | [E5] |
| GAMUS (HF release, as downloaded) | YES | YES | NO (nDSM) | NO | derived | via nDSM | Urban only | 0.33 m | **NO — `.h5` arrays, no geotransform** | n/a | **3 cities: PHL, NYC, DC; 8,724 tiles** | CC-BY-4.0 | Same | [E6] |
| DFC2019 / US3D | YES (pan+VNIR) | YES (Track 1) | nDSM (+ DSM in Track 3) | via LiDAR | YES (80 cm NPS) | via footprints | Urban/suburban | ~0.3–1.3 m | Partial (RPC available) | LiDAR datum | Jacksonville, Omaha (+ Atlanta, ARG in geopose ext.) | Open (IEEE DataPort); repo MIT | High | [E10][E11] |
| GBH (TUM) | YES (PlanetScope) | YES | NO (nDSM) | NO | YES-derived | YES | Urban, 19 + 3 cities | 3 m | Not stated | n/a | Global (EU/NA heavy) | Not fully public | LiDAR; Guangzhou heights = floors × 3 m | [E4] |
| GlobalBuildingAtlas training set | YES (+NIR) | YES | NO (nDSM) | NO | YES | YES | Urban, 168 regions | 3 m | YES | n/a | NA/EU/Oceania | ODbL (outputs) | LiDAR | [E12] |
| ISPRS Vaihingen / Potsdam | IRRG / RGB | YES | nDSM | NO | YES | via nDSM | Small-town | 0.09 / 0.05 m | Yes (tiles) | — | Germany | Open (ISPRS) | High | [E4][E5] |
| **GeoNRW** | YES | YES | **YES — first-return LiDAR at 1 m (true DSM)** | NO | YES | NO | **Includes forest class; "mostly urban"** | 1 m | YES | German national (not checked) | North Rhine-Westphalia | DL-DE-BY-2.0 | High | [E29] |
| SynRS3D | YES (synthetic) | YES | NO (nDSM) | NO | Synthetic exact | via land cover | 6 city styles; synthetic vegetation | 0.09–1 m | NO | n/a | Synthetic | Open | Exact but synthetic | [E28] |
| SynRS3D real target sets (Houston, JAX, OMA, GeoNRW-U/R, Potsdam, ATL, ARG, Nagoya, Tokyo, Vaihingen) | YES | YES | mostly nDSM | NO | YES | — | Urban/suburban; GeoNRW-Rural closest to "sparse" | 0.09–1 m | varies | varies | US, DE, JP, AR | varies | LiDAR | [E28] |
| DFC2023 Track 2 | YES (GaoFen) + SAR | PARTIAL (optical+SAR) | nDSM | NO | YES-derived | YES (125,153) | Urban, 12 cities, 5 continents | ~0.5–0.8 m (not verified) | Not stated | n/a | Global | Contest terms | LiDAR-derived | [E34] |
| NEON / EarthView (canopy) | YES | YES | CHM (nDSM for trees) | NO | YES | NO | **Forested** | 0.1 m / 1 m | YES | — | USA | Open | ALS | [E14] |
| UseGeo (ISPRS) | YES (UAV) | YES | YES (LiDAR) | — | YES | NO | Rural/hilly UAV sites | cm | YES | — | Italy | Open (ISPRS) | High | [E38] |
| AerialMetric | YES (UAV) | YES | Depth, not DSM | — | YES/RTK | NO | City/Rural/Natural | cm | NO | camera frame | China + web | UNVERIFIED | Mixed | [E15] |
| SRTM / Copernicus / AW3D30 / CartoDEM | NO | — | DSM-like at 30 m | NO (FABDEM approximates) | NO | NO | Global | 30 m | YES | EGM96 / EGM2008 / EGM96 / WGS-84 | Global / India | Public | 3–16 m σ | [E2][E3][E19][E23] |

**Landscape conclusions (not a strategy):**
1. **No public dataset found pairs RGB with LiDAR-grade *absolute* DSM across urban + sparse + hilly + forested terrain.** GeoNRW is the only one shipping a true DSM (first return) with a forest class; NEON covers forest but as canopy height. [E29][E14]
2. The organiser-recommended GAMUS, as actually downloadable, is three US cities, nDSM only, non-georeferenced → it supports *relative structure* and *metric height-above-ground* training, but **cannot itself support SRTM-anchored absolute calibration** (no coordinates). This sharpens Phase 2's "direct tension" into a verified data fact. [E1][E6]
3. Nothing found is in or near India; the only Indian elevation references found are ISRO's own CartoDEM/CartoDSM (stereo, 8 m LE90) and global DEMs. [E23]

---

## 12. Validation Prior Art

**How existing work validates:**
| Practice | Where seen | Methodological issue | Evidence |
| --- | --- | --- | --- |
| Pixel-wise RMSE/MAE against LiDAR nDSM resampled to image grid | All MHE papers | MAE/RMSE dominated by ~57 % near-zero ground pixels; SynRS3D authors show a network predicting all zeros or doubling all heights can still score "competitively" | [E4][E28] |
| Split RMSE: all / building / non-building / building-instance | HTC-DC Net, GBA | Good practice; needs footprints | [E4][E12] |
| **Random patch split within the same cities** | HTC-DC Net DFC19 (44,258 patches randomly split) and Vaihingen | Spatial leakage; Kattenborn et al.: random CV overestimates CNN performance by up to 28 %; same DFC19 data scores RMSE 2.12 m (random split) vs 5.46–9.26 m (official spatial test set) | [E4][E11][E22] |
| Held-out cities | GBH (LA/São Paulo/Guangzhou), GBA per-continent, SynRS3D T.D.2 | Best available protocol; shows 2–3× degradation on morphology shift | [E4][E12][E28] |
| mIoU-3 (class correct AND |Δh| < 1 m) | DFC2019 Track 1 | Couples semantic and height; threshold hides bias sign | [E10] |
| R² / correlation | Geocentric pose supplement | Authors note RMSE can be low where buildings are low while R² is poor for rare tall structures — correlation and RMSE disagree | [E11] |
| F1-HE (object-focused) | SynRS3D | Proposed precisely because MAE/RMSE/δ hide object-height failures | [E28] |
| Robust DEM statistics (median, NMAD, LE95) | Höhle & Höhle 2009; xdem; DEM literature | Recommended because DEM errors are non-Gaussian and outlier-heavy; not adopted in MHE papers | [E32] |
| Terrain-stratified ranking (slope, roughness, land cover) | DEMIX wine contest | Directly analogous to the PS's four-terrain requirement; never applied to monocular DSMs in found sources | [E32] |
| Reference-vs-reference bias (LiDAR vs MVS supervision) | Geocentric pose Tab. 9 | Reference choice shifts RMSE by ~1 m | [E11] |
| Temporal mismatch of reference | IM2ELEVATION (LiDAR 2015 vs imagery 2017) | Registration/temporal change corrupts MAE | [E36] |
| Vertical-datum consistency | DEM validation literature | Rarely stated in MHE papers because nDSM is datum-free; becomes mandatory for absolute DSM | [E2][E3] |
| Nodata / water handling | SRTM guide (water-masked voids, −32768); cesium-terrain-builder | Zeros vs nodata confusion documented | [E2][Phase 2 §10] |

**Methodological problems identified in existing research:** (i) spatial leakage from random patch splits; (ii) ground-pixel dominance of MAE/RMSE; (iii) nDSM evaluated but "DSM" claimed; (iv) no terrain-stratified reporting outside DEM intercomparison; (v) correlation reported without bias; (vi) no coarse-reference (SRTM-grade) validation protocol exists — validating a 0.3–3 m DSM against a 30 m DEM with 3–8 m σ would measure the reference's error, not the prediction's [E2][E21].

---

## 13. Remote-Sensing Domain Gap

| Axis | ESTABLISHED FACT | REASONABLE INFERENCE | Evidence |
| --- | --- | --- | --- |
| Viewpoint / nadir geometry | Zero-shot metric depth models fail at nadir (−90°) and oblique (−45°) UAV views; δ₁ ≈ 0 % for ZoeDepth, Depth Pro, Metric3D v2, UniDepth v1 | Failure is worse on orthorectified satellite imagery, where perspective cues are removed entirely | [E15] |
| Scale / GSD | Accuracy drops "precipitously" from 80 m to 120 m altitude for zero-shot models; MHE models are GSD-specific (0.09 vs 1.3 vs 3 m give RMSE 1.3 / 2.1 / 4.5 m) | Models trained at one GSD will not transfer to Cartosat-3 1.13 m MX or 0.25 m pansharpened without retraining | [E15][E4][E24] |
| Metric scale | Zero-shot DA-V2 on DFC2019 shows "severe scale drift"; canopy zero-shot MAE 3× worse than fine-tuned | Relative ordering may survive; absolute scale does not | [E16][E14] |
| Rooftops / repetitive structures | Long-tailed heights (57 % of pixels < 1 m) cause systematic underestimation of tall buildings; GBA under-estimates high-rises in Asia/S. America | Indian dense urban cores will be underestimated by any model trained on EU/NA morphology | [E4][E12] |
| Vegetation | Foundation-encoder + LiDAR supervision gives canopy MAE 2.8 m; MHE canopy predictions are blurred; SRTM/Copernicus are biased +3–5 m in forest | Forested terrain will carry both model and reference error | [E13][E4][E19][E21] |
| Shadows | Shadow-length methods are documented as fragile (Phase 2, StyHighNet review); Intermap patents shadow/lean geometry for mono heights | Shadows are a usable but non-general cue | [E33] |
| Terrain relief | SRTM error grows with slope (5.61 ± 15.68 m in high relief); CartoDSM shows sinks/spikes in hill shadows | Hilly scenes stress both the reference and the prediction; no MHE benchmark has a hilly category | [E21][E23][E4] |
| Occlusion | Single-view cannot see under canopy or behind facades; nDSM ground truth under trees is canopy top | "DSM" from one image is a first-surface model by construction | [E28 (nDSM definition)] |
| Texture | GAN RGB→DEM models learn terrain texture, failing in lowlands with subtle relief | Low-texture plains give little elevation signal | [E18] |
| Cross-region morphology | GBH São Paulo/Guangzhou 9–12 m vs LA 3.4 m; GBA S. America 8.9 m vs Oceania 1.5 m | Geography-of-training matters more than "seen/unseen" | [E4][E12] |

---

## 14. Failure-Mode Matrices

### A. ML Failure Modes
| Problem | Known cause | What existing systems do | What remains unsolved | Evidence |
| --- | --- | --- | --- | --- |
| Scale drift / wrong metric scale zero-shot | Training-domain scale priors; no perspective cues | Fine-tune with metric or RPC pseudo-labels (Sat3R); LiDAR supervision (GBA, Tolan) | Zero-shot metric on nadir | [E16][E15] |
| Tall-object underestimation | Long-tailed height distribution | Head-tail cut classification-regression (HTC-DC); F1-HE metric | Persists in GBA Asia/S. America | [E4][E12] |
| Blurred building edges / canopy | Encoder resolution, regression smoothing | ViT + local branch; sharper after fine-tune | Fine detail at 3 m GSD | [E4][E16] |
| Cross-region drop | Morphology/appearance shift | UDA (RS3DAda); global training sets (GBA 168 regions) | Africa untrained; Asia 5.9 m | [E28][E12] |
| Calibration overfits anchors | Few, clustered anchors | RANSAC; dense scale maps | Undocumented for RS GCP counts | [E9] |
| Tile seams | Receptive-field truncation | Overlap + blending (standard) | Consistency of per-tile scale | [E4] (patches) |

### B. Remote-Sensing Failure Modes
| Problem | Known cause | What existing systems do | What remains unsolved | Evidence |
| --- | --- | --- | --- | --- |
| Canopy vs ground confusion | First-surface sensing | Predict canopy height separately (Tolan); land-cover branch (SynRS3D) | Joint terrain/object separation from one image | [E13][E28] |
| Off-nadir parallax / facade visibility | Oblique acquisition | Geocentric pose flow; RPC back-projection | Requires RPC/metadata; PNG path has none | [E11][E16] |
| Shadow misread as low albedo | Illumination | Semantic branch | Not solved specifically | [E4] qualitative |
| GSD mismatch to training | Sensor-specific models | Multi-GSD datasets (SynRS3D 0.09–1 m; GBH 3 m) | 1.13 m MX / 0.25 m pan Cartosat-3 untested anywhere | [E24][E28] |
| Steep-terrain relief shading | Illumination ≠ height | None specific | Hilly category absent from benchmarks | [E4][E21] |
| Water / specular | Texture-poor | Water class masks (GAMUS has water) | Height of water surfaces in absolute DSM | [E5] |

### C. Geospatial Failure Modes
| Problem | Known cause | What existing systems do | What remains unsolved | Evidence |
| --- | --- | --- | --- | --- |
| Vertical datum mismatch (EGM96 vs EGM2008 vs ellipsoid vs national geoid) | Products differ; metadata rarely carried | DEM community documents datums; PROJ grids exist | Adoption in ML pipelines; CartoDSM datum ambiguity | [E2][E3][E23] |
| DEM voids / fill artefacts | SRTM voids filled with ASTER/GMTED | Documented "num" layers | Users ignore fill provenance | [E2] |
| Nodata → 0 confusion | −32768 sentinels; water masks | Explicit nodata handling in GDAL | Pipelines that cast to float and lose sentinel | [E2] |
| Resampling 30 m → sub-metre | Nyquist | cubic/bilinear | Cannot create building-scale relief | §9 |
| Registration between image and reference | Temporal change, orthorectification | Patch adjustment (IM2ELEVATION) | Sub-pixel co-registration for validation | [E36] |
| "DSM" without stated datum/CRS | Convention in ML | GeoNRW ships georef | Most MHE outputs are patches | [E29][E4] |

### D. Visualization Failure Modes
| Problem | Known cause | What existing systems do | What remains unsolved | Evidence |
| --- | --- | --- | --- | --- |
| Spikes/holes from noisy heights | Per-pixel noise | Error-bounded RTIN (MARTINI tolerance) smooths | Visual fidelity vs faithful DSM | [E25] |
| Texture stretching on facades | Heightfield cannot represent vertical walls | Accepted limitation of 2.5D terrain | True 3D needs LoD1 extrusion (GBA) | [E25][E12] |
| LOD popping, large rasters | Memory | Quadtree tiles (Cesium), TileLayer (deck.gl) | Bundling in standalone app | [E25] |
| Vertical exaggeration mistaken for data | UI | Exposed parameter (Qgis2threejs) | Convention only | [E25] |
| Height readout ambiguity (DSM vs above-ground) | Product semantics | GIS tools show raw value | User-facing clarity | §7 terminology |

---

## 15. Claim-vs-Reality Audit

| Claim (as commonly phrased) | Actual capability | Evidence |
| --- | --- | --- |
| "GAMUS provides paired data to translate 2D satellite imagery into depth models" (SAC README) | GAMUS is an *aerial orthophoto* (0.33 m), *segmentation* benchmark whose height modality is **nDSM**; HF release is 3 cities, non-georeferenced `.h5`. It provides RGB↔height-above-ground pairs, not depth, not DSM, not satellite. | [E1][E5][E6] |
| "GAMUS will teach the model to handle urban, sparse, hilly and forested landscapes" (SAC README) | GAMUS contains dense US urban cores only. | [E1][E5][E6] |
| "Metric depth model" (Metric3D, UniDepth, Depth Pro, ZoeDepth) | Metric in the *camera frame* for *perspective* images in the training distribution; δ₁ ≈ 0 % zero-shot on nadir aerial views; no relation to elevation above a datum. | [E15] |
| "DSM" in MHE paper titles (IM2HEIGHT, HTC-DC Net, GAMUS, SynRS3D, DFC2023) | **nDSM** — height above local ground; DTM subtracted during dataset creation. | [E4][E5][E28][E34] |
| "Generating Elevation Surface from a Single RGB Image" (Panagiotou 2020) | Relative elevation "within a single image"; no metric accuracy published. | [E17] |
| "DEM estimation from RGB satellite imagery" (Madani 2025) | Normalized [-1, 1] output; RMSE in normalized units; 30 m Landsat. | [E18] |
| "SRTM gives absolute elevation to calibrate to" | SRTM gives EGM96-geoid heights of the radar phase centre, ±3–16 m, biased +3–5 m in forest, at 30 m posting. | [E2][E21] |
| "3D reconstruction from satellite image" (Sat3R, SatDN, Sat-NeRF family) | Multi-view; single-image variants (GBA LoD1) are extrusions of footprints by one height value. | [E16][E12] |
| "Object height estimation from monocular images" patent = RS prior art (Phase 2) | Zoox street-level object-height patent. | [E33] |
| "HTC-DC Net RMSE 2.12 m on DFC19" | On a random 256-px patch split of the same two cities; official Track-1 test set best RMSE is 5.46 m (all) / 10.69 m (buildings). | [E4][E11][E22] |
| "Depth Anything works out of the box" | Zero-shot DA-V2 on DFC2019: MAE 4.59 m, "severe scale drift and blurry boundaries" (multi-view fused); fine-tuning halves error. | [E16] |
| "Interactive 3D flythrough of reconstructed terrain" | Commodity in Cesium/Qgis2threejs/Unity; visual quality is independent of DSM accuracy. | [E25] |
| "Blackshark.ai produces accurate building heights from satellite imagery" | No public method or accuracy figures found. **UNVERIFIED.** | [E35] |

---

## 16. Novelty Audit

| Component | Status | Evidence |
| --- | --- | --- |
| Pretrained monocular depth backbone | ESTABLISHED / COMMON | [E26] |
| Fine-tuning a depth foundation model for nadir height (canopy, DSM) | PREVIOUSLY DEMONSTRATED (Depth Any Canopy 2024, Sat3R 2026, Tolan 2024 with DINOv2) | [E14][E16][E13] |
| Monocular height (nDSM) from a single satellite/aerial image | ESTABLISHED (2018→2025; operational global product) | [E36][E4][E12] |
| Single oblique satellite image → height + orthorectification | PREVIOUSLY DEMONSTRATED (geocentric pose, open code) | [E11] |
| SRTM / global DEM usage | COMMON | [E2][E3] |
| DEM fusion / DEM as training target for RGB→terrain | PREVIOUSLY DEMONSTRATED (relative outputs) | [E17][E18] |
| DEM as *scale anchor for monocular depth* on RS imagery | NOVELTY NOT ESTABLISHED (no paper found either way — absence of evidence, not proof of novelty) | §9 |
| Sparse-point / low-res prior → dense metric depth | PREVIOUSLY DEMONSTRATED (camera frame) | [E27][E9] |
| GCP calibration of monocular RS depth | NOVELTY NOT ESTABLISHED (not found) | §10 |
| GeoTIFF processing, CRS, vertical CRS | ESTABLISHED / COMMON | [E2][E3] |
| Height-map generation | ESTABLISHED | — |
| Terrain mesh generation (RTIN/quadtree) | ESTABLISHED / COMMON | [E25] |
| Texture mapping of ortho onto heightfield | ESTABLISHED / COMMON | [E25] |
| Three.js / Babylon.js terrain viewers | ESTABLISHED (Qgis2threejs, deck.gl) | [E25] |
| Unity / Unreal terrain + first-person | ESTABLISHED (Cesium for Unity/Unreal) | [E25] |
| First-person navigation | COMMON | [E25] |
| Slope visualization | ESTABLISHED (gdaldem; DEMIX slope criteria) | [E32] |
| Terrain-stratified accuracy evaluation | ESTABLISHED for DEMs (DEMIX); NOT APPLIED to monocular DSMs in found work | [E32] |
| Standalone HTML/glTF export of terrain scene | ESTABLISHED (Qgis2threejs) | [E25] |
| End-to-end: single RGB → calibrated absolute DSM → validated → standalone 3D | PARTIALLY DEMONSTRATED as separate pieces; **no single found system** integrates all with validation across four terrain types | §3 |

---

## 17. True Technical Gaps

**GAP-01 — Zero-shot foundation depth is not metrically meaningful on nadir imagery.** PROBLEM: the PS mandates a pretrained backbone; every zero-shot test on aerial/satellite views shows scale failure. EXISTING: DA-V2/DA3/UniDepth/Metric3D/MoGe. CAN DO: plausible relative structure; adaptable by fine-tuning. CANNOT RELIABLY DO: produce metric or even consistently scaled output zero-shot at nadir (δ₁≈0 %; MAE 4.6 m with scale drift). EVIDENCE: [E15][E16][E14]. CONFIDENCE: HIGH.

**GAP-02 — Relative depth → absolute elevation with only a 30 m DEM.** PROBLEM: DEM contains no sub-60 m wavelength information and carries 3–16 m σ (worse in forest/relief) on an EGM96/EGM2008 datum. EXISTING: scale-shift LS; prior-conditioned depth models (indoor). CAN DO: low-frequency scale/offset. CANNOT RELIABLY DO: building-scale calibration; datum-clean absolute DSM; avoid double-counting canopy present in the DEM. EVIDENCE: [E2][E19][E21][E9][E27]. CONFIDENCE: HIGH.

**GAP-03 — nDSM + terrain → certified absolute DSM from one image.** PROBLEM: all MHE outputs are nDSM; no system adds terrain back and validates. EXISTING: dataset builders do the inverse. CANNOT DO: end-to-end datum-referenced DSM from monocular input (Sat3R is multi-view). EVIDENCE: [E4][E12][E16]. CONFIDENCE: HIGH.

**GAP-04 — Cross-region/cross-morphology generalization.** PROBLEM: 2–3× RMSE increase on cities with different morphology; Asia 5.9 m, S. America 8.9 m for an operational system. EXISTING: UDA (RS3DAda), large training sets. CANNOT RELIABLY DO: predict Indian urban heights at EU/NA accuracy. EVIDENCE: [E4][E12][E28]. CONFIDENCE: HIGH.

**GAP-05 — Hilly and forested terrain: no LiDAR-grade single-view benchmark exists.** PROBLEM: PS demands stability across four terrain types; benchmarks are urban/suburban. EXISTING: GeoNRW (forest class, DSM), NEON canopy, UseGeo UAV. CANNOT DO: report RMSE/MAE/corr on hilly + forested satellite RGB with absolute-DSM truth. EVIDENCE: [E29][E14][E4][E28]. CONFIDENCE: HIGH.

**GAP-06 — Terrain/object separation from one image.** PROBLEM: a DSM entangles ground, canopy and roofs; first-surface sensing cannot see beneath canopy. EXISTING: land-cover branches; FABDEM-style statistical removal for DEMs. CANNOT DO: recover ground under canopy or building height under trees from a single nadir RGB. EVIDENCE: [E28][E19]. CONFIDENCE: HIGH (physical).

**GAP-07 — Tall-object underestimation (long tail).** EXISTING: HTC-DC head-tail cut, F1-HE metric. CANNOT DO: eliminate high-rise underestimation in Asian cities. EVIDENCE: [E4][E12]. CONFIDENCE: HIGH.

**GAP-08 — Sparse-GCP calibration for monocular RS depth: requirements unknown.** EXISTING: 150-point camera-frame result; RANSAC. CANNOT SAY: how many GCPs, what distribution, what residual on terrain-scale scenes. EVIDENCE: [E9] + absence. CONFIDENCE: MEDIUM (negative finding).

**GAP-09 — Validation protocol against coarse or absent reference.** PROBLEM: LiDAR is absent over India; SRTM-grade reference error exceeds expected prediction error. EXISTING: DEMIX terrain-stratified criteria; Höhle & Höhle robust stats; CartoDSM 8 m LE90. CANNOT DO: credibly certify a sub-metre-GSD DSM using a 30 m reference. EVIDENCE: [E32][E23][E21]. CONFIDENCE: HIGH.

**GAP-10 — Spatial leakage in reported accuracies.** PROBLEM: random patch splits inflate performance up to 28 %; same dataset yields 2.1 m vs 5.5–9.3 m depending on split. EXISTING: held-out-city protocols exist but are not universal. CANNOT DO: compare published numbers across papers. EVIDENCE: [E22][E4][E11]. CONFIDENCE: HIGH.

**GAP-11 — Vertical datum and metadata discipline in ML pipelines.** PROBLEM: SRTM EGM96, Copernicus EGM2008, CartoDSM "WGS-84", LiDAR national geoids; ML outputs are patches without vertical CRS. EXISTING: PROJ/EPSG machinery. CANNOT DO: nothing technical — the gap is that no found MHE/DSM ML work carries a vertical CRS through to output. EVIDENCE: [E2][E3][E23]. CONFIDENCE: HIGH.

**GAP-12 — GSD/sensor mismatch to ISRO imagery.** PROBLEM: models are GSD-specific; Cartosat-3 MX is 1.13 m, pan 0.25 m; no found model trained at 1.13 m 4-band Indian imagery. EVIDENCE: [E24][E4]. CONFIDENCE: MEDIUM (sensor at evaluation unknown [E1]).

**GAP-13 — Off-nadir handling on the non-georeferenced path.** PROBLEM: PNG/JPG carry no RPC; geocentric-pose and Sat3R need geometry to convert oblique depth to height. EVIDENCE: [E11][E16]. CONFIDENCE: MEDIUM.

**GAP-14 — Relative-output validation semantics.** PROBLEM: an rDSM cannot be scored with RMSE/MAE in metres; only scale-invariant or rank metrics apply; the PS's metric list assumes absolute output. EXISTING: affine-invariant depth metrics (δ, AbsRel after alignment). EVIDENCE: [E9][E15]. CONFIDENCE: MEDIUM.

**GAP-15 — Uncertainty is not produced by any found single-view height system.** EXISTING: per-pixel uncertainty in some depth-completion work (Aerial depth completion, Phase 2 mention); none in MHE outputs found. EVIDENCE: absence across [E4][E12][E13][E16]. CONFIDENCE: MEDIUM.

**GAP-16 — Standalone packaging of GPU inference + 3D viewer.** EXISTING: Qgis2threejs exports static HTML; Unity builds; but no found system bundles a depth model with a viewer in one installable artefact. EVIDENCE: [E25] + absence. CONFIDENCE: MEDIUM (engineering, not research).

**GAP-17 — Faithful 2.5D visualization of a noisy DSM.** PROBLEM: error-bounded simplification improves looks while hiding geometry error; walls cannot be represented by heightfields. EXISTING: MARTINI tolerance, LoD1 extrusion. EVIDENCE: [E25][E12]. CONFIDENCE: MEDIUM.

---

## 18. Gap Categories

| Category | Gaps |
| --- | --- |
| Technical (ML) | GAP-01, GAP-07, GAP-15 |
| Data | GAP-05, GAP-12 |
| Scale / Calibration | GAP-02, GAP-03, GAP-08 |
| Remote-Sensing Domain | GAP-01, GAP-04, GAP-06, GAP-13 |
| Geospatial | GAP-03, GAP-11 |
| Validation | GAP-09, GAP-10, GAP-14 |
| Generalization | GAP-04, GAP-05, GAP-12 |
| Visualization | GAP-17 |
| Deployment | GAP-16 |
| Usability | GAP-14 (communicating relative vs absolute), GAP-17 (exaggeration/readout semantics) |

---

## 19. SIH26175 Coverage Matrix

| SIH Requirement | Existing Coverage | Strongest Known Approach | Evidence | Remaining Gap |
| --- | --- | --- | --- | --- |
| Accept PNG/JPG and GeoTIFF | FULLY ADDRESSED | GDAL/rasterio | [E25] | none |
| Pretrained monocular depth backbone → relative depth on RS imagery | PARTIALLY ADDRESSED | DA-V2/DA3 zero-shot give structure with scale drift; fine-tuned variants work | [E16][E14] | GAP-01 |
| rDSM for non-georeferenced imagery | PARTIALLY ADDRESSED | Any relative depth model; GAN RGB→relative DEM | [E17][E18][E26] | GAP-13, GAP-14 |
| Absolute metric DSM for GeoTIFF via SRTM | NOT CLEARLY ADDRESSED | No system found; closest: scale-shift LS (camera frame) + DEM as low-frequency anchor | [E9][E2] | GAP-02, GAP-03 |
| Absolute DSM via limited GCPs | NOT CLEARLY ADDRESSED | Photogrammetric analogues only | [E23] | GAP-08 |
| Metric height above ground (single image) | FULLY ADDRESSED in-distribution; PARTIALLY cross-region | HTC-DC Net / GlobalBuildingAtlas / Tolan canopy | [E4][E12][E13] | GAP-04, GAP-07 |
| Standard geospatial output (GeoTIFF with CRS) | FULLY ADDRESSED (tooling) / WEAKLY ADDRESSED (practice) | GDAL + EPSG vertical CRS | [E2][E3] | GAP-11 |
| RMSE/MAE/correlation vs LiDAR | FULLY ADDRESSED (methodology) / WEAKLY ADDRESSED (Indian reference) | Held-out-city protocol; DEMIX stratification; Höhle robust stats | [E4][E32] | GAP-09, GAP-10 |
| Stability across urban/sparse/hilly/forested | WEAKLY ADDRESSED | GBA per-continent; canopy models for forest; none for hilly | [E12][E13] | GAP-05 |
| Texture projection onto 3D terrain mesh | FULLY ADDRESSED | Cesium, deck.gl, Qgis2threejs, Unity | [E25] | none |
| First-person navigation | FULLY ADDRESSED | Unity/Unreal, Cesium for Unity | [E25] | none |
| Arbitrary aerial perspectives | FULLY ADDRESSED | Orbit/fly controls | [E25] | none |
| Structural height analysis | PARTIALLY ADDRESSED | GIS raster query; needs local ground for DSM | [E25] | GAP-06 |
| Slope analysis | FULLY ADDRESSED | gdaldem, shaders | [E32] | noise amplification |
| Upload → visualize → validate against reference (in one UI) | WEAKLY ADDRESSED | No found single system does all three | §3 | GAP-16 |
| Standalone deployment | PARTIALLY ADDRESSED | Qgis2threejs HTML export; Unity builds | [E25] | GAP-16 |

---

## 20. What Must NOT Be Claimed as Innovation

Only claims contradicted by the prior art found:
1. "First to estimate height/DSM from a single satellite image" — 2018→2025 lineage; operational global product. [E36][E4][E12]
2. "First to apply a depth foundation model (Depth Anything/DINOv2) to remote sensing height" — Depth Any Canopy 2024, Tolan 2024, Sat3R 2026, SynRS3D (DINOv2+DPT). [E14][E13][E16][E28]
3. "Novel scale-and-shift alignment from sparse points" — standard since MiDaS/Wofk; RANSAC standard. [E9]
4. "Novel fusion of coarse depth priors with relative depth" — Prior Depth Anything, PromptDA, Marigold-DC (indoor/driving). [E27]
5. "Novel single-view geocentric/orthorectified height with parallax handling" — Christie et al., open code. [E11]
6. "Novel metric that focuses on objects rather than ground pixels" — F1-HE (SynRS3D); building-wise RMSE (HTC-DC). [E28][E4]
7. "Novel terrain-stratified DEM evaluation" — DEMIX. [E32]
8. "Novel heightmap → LOD terrain mesh / texture draping / web 3D terrain export" — Cesium, MARTINI/deck.gl, Qgis2threejs. [E25]
9. "Novel first-person terrain navigation in Unity/Three.js" — Cesium for Unity/Unreal and game-engine controllers. [E25]
10. "Novel removal of forest/building bias from global DEMs" — FABDEM. [E19]
11. "Novel synthetic data for RS height" — SynRS3D. [E28]
12. "Novel building-height product from optical satellite imagery" — GlobalBuildingAtlas, Open Buildings 2.5D, Precision3D. [E12][E31][E30]

---

## 21. Potential Contribution Areas

(Areas only — not designs.)

| Area | Why difficult | What existing research already achieves | What remains open |
| --- | --- | --- | --- |
| Metric calibration from coarse DEM | DEM lacks high-frequency content and is biased in forest; scale varies spatially | Global scale-shift; dense scale maps in camera frame; low-res-prior fusion indoors | Any evidenced protocol for DEM-anchored monocular RS depth, with residual error quantified |
| Remote-sensing adaptation of foundation depth | Zero-shot scale failure; RS height data scarce outside EU/NA | Fine-tuning with LiDAR/RPC pseudo-labels works (canopy, DSM) | Adaptation without dense LiDAR; adaptation at 1–3 m GSD for South Asian morphology |
| Terrain / object separation | First-surface physics; canopy occlusion | Land-cover branches; FABDEM statistical removal for DEMs | Single-image joint DTM + nDSM with quantified error |
| Sparse-reference (GCP) calibration | No RS evidence at all | 150-point indoor result; photogrammetric GCP practice | Count/distribution/robustness study for monocular RS |
| Uncertainty estimation | Not produced by any found MHE system | Depth-completion uncertainty exists in robotics | Per-pixel confidence tied to terrain class and calibration residual |
| Cross-terrain robustness | Benchmarks lack hilly/forest categories | Per-continent RMSE (GBA); canopy models | A hilly/forested single-view benchmark with absolute DSM truth |
| Geospatial correctness | Vertical datum ignored by ML work | PROJ/EPSG machinery; DEM community documents datums | Carrying vertical CRS through an ML DSM pipeline and reporting datum-consistent RMSE |
| Validation | Random splits inflate; coarse reference measures itself | Held-out-city; DEMIX; robust statistics; F1-HE | Protocol for reporting RMSE/MAE/corr with reference-error bounds when only SRTM/CartoDEM-grade truth exists |
| Computational efficiency | Full satellite tiles vs ViT-L inference | DA-V2-S 24.8 M params; tiled inference | Measured end-to-end latency for inference + mesh + render (Phase 2 open, unchanged) |
| Visualization tied to accuracy | Looks can mask error | Error-bounded meshes; measure tools | Displaying uncertainty/residuals inside the flythrough |

---

## 22. ISRO-Style Evaluator Questions

(Questions only; not answered with our future architecture.)

1. Your backbone was trained on egocentric perspective imagery. Show the zero-shot output on a Cartosat-3 scene before any calibration — what is its metric scale, and why should we believe the *shape* is right if the *scale* is off by 4–5 m? [E15][E16]
2. SRTM is a 30 m EGM96-geoid product with a +3–5 m canopy bias. How does a 30 m surface calibrate a 1 m DSM, and what stops you from double-counting canopy that SRTM already contains? [E2][E19][E21]
3. Which vertical datum is your output DSM in — EGM96, EGM2008, WGS84 ellipsoid, or the LiDAR's national geoid — and where is that written in the GeoTIFF? [E2][E3]
4. Your RMSE is against which reference, at what resolution, co-registered how, acquired when? If the reference is SRTM/CartoDEM (8 m LE90), is your RMSE measuring your model or the reference? [E23][E32]
5. Were your train/test splits spatially disjoint? DFC19 gives 2.1 m on random patch splits and 5.5–9.3 m on the official test set — which regime are your numbers in? [E4][E11][E22]
6. How many GCPs, in what spatial distribution, and how did you separate calibration points from checkpoints? What is the residual after removing them? [E9][E32]
7. What happens on a hilly, forested scene with no buildings — what is your ground truth there, and what is the model's error on canopy vs bare slopes? [E13][E21]
8. Are you delivering a DSM or an nDSM? If a DSM, where does the terrain component come from and what is *its* error? [E4][E12]
9. Your urban RMSE — is it dominated by 57 % near-zero ground pixels? Report building-only and building-instance RMSE and F1-HE. [E4][E28]
10. GlobalBuildingAtlas trained on 168 LiDAR regions and still gets 5.9 m RMSE in Asia. What in your training data covers Indian building morphology? [E12]
11. Your correlation is 0.9 — what is the bias? Correlation is invariant to scale and offset, which is exactly what calibration must fix. [E11]
12. The PNG path has no RPC or view geometry. If the image is off-nadir, how do you know depth maps to height and not to parallax? [E11][E16]
13. What is your runtime on a full Cartosat-3 strip at 1.13 m, and was the whole pipeline (inference + mesh + render) timed, not just the model? (Phase 2 open question, unchanged)
14. Your flythrough looks smooth — what mesh error tolerance did you use, and how much of the DSM detail did it remove? [E25]
15. What licence are your model weights under, and is a CC-BY-NC backbone compatible with an ISRO deployment? [E26][E4]

---

## 23. Red-Team Analysis

| Assumption | Why it may fail | Evidence | Implication |
| --- | --- | --- | --- |
| Georeferencing alone gives elevation | A GeoTIFF supplies horizontal CRS, geotransform and GSD — no vertical information; vertical reference must come from an external source with its own datum | [E2][E3] | Absolute DSM is impossible without an external elevation source; datum reconciliation is mandatory |
| Metric depth = absolute elevation | Metric depth is camera-frame distance; even when correct it needs sensor altitude/RPC and a datum to become elevation; and zero-shot metric depth is wrong at nadir anyway | [E15][E16] | "Metric" from a model is at best height-above-ground scale, never datum elevation |
| A coarse DEM can recover building-scale height | 30 m posting has no sub-60 m signal; DEM σ 3–16 m exceeds sparse-settlement building heights; DEM already includes canopy/roof bias | [E2][E19][E21] | DEM can constrain scene-level scale/offset only |
| One image recovers hidden surfaces | First-surface physics; canopy occlusion; facades invisible at nadir | [E28] | Outputs are first-surface models; heightfields cannot represent walls |
| Monocular depth distinguishes trees from terrain | Canopy is texture; models need explicit supervision (Tolan) or land-cover branches; MHE canopy predictions are blurred | [E13][E4][E28] | Forested terrain error will be structurally higher |
| Correlation hides vertical bias | Correlation is scale- and shift-invariant; Christie et al. show RMSE low where R² poor and vice-versa | [E11] | Report bias and RMSE alongside correlation |
| Visual quality hides poor DSM | Error-bounded simplification and texture draping look good regardless of geometry error | [E25] | Visualization score is independent of accuracy score — the PS's 50/50 split encodes this |
| Random geographic splits exaggerate performance | Up to 28 % inflation; 2.1 m vs 5.5–9.3 m on the same cities | [E22][E4][E11] | Only spatially disjoint evaluation is credible |
| GAMUS covers the four terrain types | Three dense US cities in the release | [E5][E6] | Organiser's dataset cannot evidence hilly/forested stability |
| "Pretrained backbone" implies zero-shot suffices | All positive RS results involve fine-tuning; all zero-shot results show scale failure | [E14][E15][E16] | The PS's requirement is satisfiable only with adaptation or external calibration |
| SRTM is ellipsoidal (Phase 2) | It is EGM96 geoid | [E2] | A pipeline built on the Phase 2 statement would introduce a systematic offset equal to the geoid undulation |

---

## 24. Technical Risk Prioritization

Problems, not solutions. Criteria: impact on correctness, impact on SIH evaluation (50/50), difficulty, data dependency, research uncertainty.

| Priority | Problem | Rationale |
| --- | --- | --- |
| **CRITICAL** | GAP-01 zero-shot scale failure on nadir imagery | Breaks the elevation half at the first stage; strongest evidence base |
| **CRITICAL** | GAP-02/03 relative → absolute datum-referenced DSM with 30 m DEM | The PS's defining requirement for GeoTIFF mode; no prior art closes it |
| **CRITICAL** | GAP-09 validation without LiDAR over India | Without credible reference, 50 % of the score is unevidenced |
| **HIGH** | GAP-05 hilly/forested benchmark absence | Explicit PS criterion; no dataset supports it |
| **HIGH** | GAP-04 cross-morphology generalization | Measured 2–3× degradation for best systems |
| **HIGH** | GAP-10 spatial leakage | Determines whether any reported number is credible |
| **HIGH** | GAP-11 vertical datum discipline | Silent tens-of-metres bias; cheap to get right, catastrophic to get wrong |
| **MEDIUM** | GAP-06 terrain/object separation | Affects structural-height analysis and forest accuracy |
| **MEDIUM** | GAP-07 tall-object underestimation | Affects urban building RMSE |
| **MEDIUM** | GAP-08 GCP requirements unknown | PS permits GCPs; no evidence on how many |
| **MEDIUM** | GAP-12 GSD/sensor mismatch | Depends on unknown evaluation sensor |
| **MEDIUM** | GAP-13 off-nadir PNG path | Depends on evaluation imagery geometry |
| **MEDIUM** | GAP-16 standalone packaging | Engineering risk to the visualization half's "deployment" criterion |
| **LOW** | GAP-14 rDSM scoring semantics | Communication/protocol issue |
| **LOW** | GAP-15 uncertainty | Differentiator, not a blocker |
| **LOW** | GAP-17 faithful visualization | Established tooling mitigates |

---

## 25. Master Gap Map

| Gap ID | Gap | Existing State | Evidence | Why It Matters | Confidence | Category |
| --- | --- | --- | --- | --- | --- | --- |
| GAP-01 | Zero-shot foundation depth unreliable at nadir | δ₁≈0 % zero-shot; fine-tuning recovers | [E15][E16][E14] | First pipeline stage | HIGH | Technical / RS domain |
| GAP-02 | 30 m DEM cannot calibrate building-scale DSM | DEM = low-frequency, biased, 3–16 m σ | [E2][E19][E21] | Core GeoTIFF-mode requirement | HIGH | Scale/Calibration |
| GAP-03 | nDSM + terrain → certified absolute DSM undocumented | Inverse operation only | [E4][E12][E16] | Deliverable definition | HIGH | Scale/Calibration, Geospatial |
| GAP-04 | Cross-morphology generalization | 2–3× degradation; Asia 5.9 m | [E4][E12] | Indian evaluation imagery | HIGH | Generalization |
| GAP-05 | No hilly/forested single-view benchmark | GeoNRW/NEON partial | [E29][E14] | Explicit PS criterion | HIGH | Data |
| GAP-06 | Terrain/object separation | Land-cover branches only | [E28][E19] | Height analysis, forest | HIGH | RS domain |
| GAP-07 | Tall-object underestimation | Head-tail cut helps, persists | [E4][E12] | Urban RMSE | HIGH | Technical |
| GAP-08 | GCP requirements for monocular RS | Not found | [E9] | PS permits GCPs | MEDIUM | Scale/Calibration |
| GAP-09 | Validation with coarse/absent reference | DEMIX/robust stats exist for DEMs | [E32][E23] | 50 % of score | HIGH | Validation |
| GAP-10 | Spatial leakage | Up to 28 % inflation; 2.1 vs 5.5–9.3 m | [E22][E4][E11] | Credibility | HIGH | Validation |
| GAP-11 | Vertical datum discipline | Tooling exists, practice absent | [E2][E3][E23] | Silent bias | HIGH | Geospatial |
| GAP-12 | GSD/sensor mismatch | Models GSD-specific | [E24][E4] | Unknown sensor | MEDIUM | Data / Generalization |
| GAP-13 | Off-nadir on PNG path | Needs RPC/geometry | [E11][E16] | rDSM correctness | MEDIUM | RS domain |
| GAP-14 | rDSM scoring semantics | Affine-invariant metrics exist | [E9][E15] | Protocol | MEDIUM | Validation / Usability |
| GAP-15 | No uncertainty output | Absent in MHE | — | Honest reporting | MEDIUM | Technical |
| GAP-16 | Standalone GPU+viewer packaging | Pieces exist separately | [E25] | Deployment criterion | MEDIUM | Deployment |
| GAP-17 | Faithful 2.5D visualization | Error-bounded meshes hide error | [E25] | Visual vs accuracy | MEDIUM | Visualization |

---

## 26. Constraints Inherited From Prior Art

Evidence-backed constraints any future solution must respect (no solution proposed):

1. **Relative depth is not metric elevation.** Zero-shot outputs carry per-image scale/shift; on nadir imagery even "metric" models fail. [E9][E15][E16]
2. **Georeferencing supplies horizontal position and GSD only; vertical reference must come from outside and has its own datum.** [E2][E3]
3. **SRTM is EGM96-geoid, Copernicus is EGM2008, AW3D30 is EGM96, CartoDSM is stated "WGS-84"; mixing them without transformation biases results by the geoid separation.** [E2][E3][E23]
4. **Coarse DEMs contain no building-scale detail and are themselves biased upward in forest (+3–5 m) and built-up areas (+1.6 m).** [E19][E21]
5. **DSM ≠ nDSM ≠ DTM; nearly all monocular RS height literature and the organiser's dataset deliver nDSM.** [E4][E5][E28]
6. **Vegetation is a first-surface confound: single-view models see canopy top; reference DEMs partially penetrate; canopy height needs explicit supervision.** [E13][E21]
7. **Domain shift is measured, not hypothetical: 2–3× RMSE on morphology-shifted cities; Asia 5.9 m for an operational system.** [E4][E12]
8. **Spatial leakage invalidates random splits; only spatially disjoint evaluation is credible.** [E22][E11]
9. **MAE/RMSE are dominated by ground pixels; object-focused metrics are required to see failure on buildings/trees.** [E28][E4]
10. **Correlation is blind to scale and offset — the very quantities calibration must fix — so it cannot certify absolute accuracy alone.** [E11]
11. **Camera-intrinsics-based metric recovery has no basis on orthophotos.** Geometry + [E15]
12. **Model licences constrain deployment: DA-V2 Base/Large are CC-BY-NC-4.0; HTC-DC Net has no licence.** [E26][E4]
13. **GSD is a model hyper-parameter in practice; accuracy scales with GSD (1.3 m at 0.09 m GSD → 4.5 m at 3 m).** [E4]
14. **Heightfield meshes cannot represent vertical walls; visual smoothing removes geometric information.** [E25]
15. **ISRO's own stereo product achieves 8 m LE90; any single-view claim must be framed against that bar.** [E23]

---

## 27. Evidence Ledger

| ID | Claim | Evidence | Source | Source type | Confidence |
| --- | --- | --- | --- | --- | --- |
| E1 | SAC README (both commits) recommends GAMUS, allows "any open-source dataset containing remote-sensing depth data", allows SRTM 30 m; contains **no** statement on evaluation imagery source; repository has no licence and one file | Raw README fetched at commits feb4dc63 (2026-07-29) and ac078d5e (2026-08-29) | github.com/IMG-PROCESS-SAC/SIH2026 | Official organiser repository | HIGH |
| E2 | SRTM V3 elevations are "meters as referenced to the WGS84/EGM96 geoid"; 1″ posting; voids filled from ASTER GDEM2/GMTED2010/NED; −32768 void sentinel in v1/2.1 | SRTM Collection User Guide V3 (Oct 2015), text extracted | lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf | Government (NASA/USGS) | HIGH |
| E3 | Copernicus GLO-30 vertical datum EGM2008 (EPSG:3855); ALOS AW3D30 EGM96 | GEE catalog entry; Copernicus product handbook; regional accuracy papers | developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30; tandfonline 10.1080/10095020.2023.2296010 | Official documentation / peer-reviewed | HIGH |
| E4 | HTC-DC Net: datasets DFC19 (1.3 m, 2,783 tiles → 44,258 random-split patches), GBH (PlanetScope 3 m, 19 + 3 cities, nDSM), Vaihingen (0.09 m, random split); Tables I–IV RMSE (DFC19 B7 2.118; GBH seen B5 4.453; LA 3.377 / São Paulo 9.301 / Guangzhou RMSE-B 11.840; Vaihingen B7 1.303); 57 % pixels < 1 m; repo has no licence | Paper PDF text extracted; GitHub API | arXiv 2309.16486; github.com/zhu-xlab/HTC-DC-Net | Primary paper / official repo | HIGH |
| E5 | GAMUS: 0.33 m orthophotos, 11,507 tiles 1024², five cities, nDSM = DSM − DTM from LiDAR, data from DC/Philadelphia open data; designed as multi-modal *segmentation* benchmark | Paper PDF text | arXiv 2305.14914 (Xiong, Chen, Wang, Mou, Zhu) | Primary paper | HIGH |
| E6 | GAMUS HF release: 8,724 tiles (train 5,004 / val 859 / test 2,861), `.h5` files, prefixes PHL/NYC/DC only, CC-BY-4.0 | HF API file listing parsed | huggingface.co/datasets/earthflow/GAMUS | Official dataset repository | HIGH |
| E7 | THE Benchmark authors are Xiong, Huang, Hu, Zhu (2021) | arXiv API metadata | arXiv 2112.14985 | Primary | HIGH |
| E8 | arXiv 2301.04581 = Mao et al. 2023 "Elevation Estimation-Driven Building 3D Reconstruction from Single-View RS Imagery"; 2507.02148 = underwater metric depth; 2409.04086 = automotive class-aware metric | arXiv API | export.arxiv.org | Primary metadata | HIGH |
| E9 | Wofk, Ranftl, Müller, Koltun 2023: global LS scale-shift on inverse depth; "just 150 sparse metric depth points"; dense scale alignment gives up to 30 % iRMSE reduction; global alignment "will not adequately resolve metric scale in all" regions | Paper PDF text | arXiv 2303.12134 | Primary | HIGH |
| E10 | DFC2019 Track 1: single unrectified image → semantic labels + above-ground heights (m); mIoU-3 with 1 m threshold; MIT repo; winner Kunwar (nest) 0.5571 | pubgeo/dfc2019 README; GRSS results page (via search) | github.com/pubgeo/dfc2019; grss-ieee.org | Official contest repo / organiser | HIGH (metric, task) / MEDIUM (winner score, via search summary) |
| E11 | Christie et al. 2021: US3D JAX/OMA/ATL + ARG; official DFC19 test set RMSE — Kunwar 9.26 all / 19.65 bldg; Christie 2020 8.23/16.87; Ours 5.46/10.69; Mahmud-split: 3.89 vs Mahmud 5.02, Mou&Zhu 5.40, Srivastava 5.85; LiDAR vs MVS supervision comparable; code pubgeo/monocular-geocentric-pose | Paper PDF text | arXiv 2105.08229 | Primary | HIGH |
| E12 | GlobalBuildingAtlas: HTC-DC Net on PlanetScope 3 m (RGB+NIR); 187,239 training pairs from 168 LiDAR regions (NA 39, EU 109, Oceania 17); RMSE global 5.5 m; Oceania 1.5 / Europe 4.1 / NA 5.3 / Asia 5.9 / SA 8.9; ODbL; under-estimates high-rises; no Africa | Journal article | essd.copernicus.org/articles/17/6647/2025; arXiv 2506.04106 | Peer-reviewed | HIGH |
| E13 | Tolan et al. 2024: DINOv2 + dense decoder on Maxar RGB, trained on US aerial LiDAR; MAE 2.8 m, ME 0.6 m; global 1 m map | RSE 300:113888; AWS/GEE catalog | arXiv 2304.07213; registry.opendata.aws/dataforgood-fb-forests | Peer-reviewed / official data | HIGH |
| E14 | Depth Any Canopy: DA-V2 fine-tuned; zero-shot DA-S MAE 0.4116 → 0.1410 fine-tuned (EarthView, normalized); "necessary to adapt the model" | arXiv HTML | arXiv 2408.04523 | Primary (workshop) | HIGH |
| E15 | AerialMetric 2026: 52 K real + 16 K synthetic UAV pairs; nadir to −45°; zero-shot AbsRel: ZoeDepth 93–97 %, Depth Pro 92–98 %, UniDepth v1 80–89 %, v2 17–31 %, MoGe-2 26–71 %, Metric3D v2 63–81 %; δ₁ ≈ 0 % for most; "catastrophic failures at strictly nadir"; fine-tuned MoGe-2 AbsRel 10.3 % | arXiv HTML | arXiv 2606.29716 | Primary (preprint) | HIGH for reported numbers; MEDIUM as not yet peer-reviewed |
| E16 | Sat3R 2026: DA-V2 fine-tuned with RPC pseudo-depth, SiLog, max depth 150 m; **multi-view fusion** then RPC back-projection; DFC2019 six scenes mean MAE: zero-shot DA2 4.59, DA3 4.37, Sat3R 2.82, SatDN 1.44; zero-shot shows "severe scale drift and blurry boundaries" | Paper PDF text | arXiv 2605.07264 | Primary (preprint) | HIGH for numbers; MEDIUM as preprint |
| E17 | Panagiotou et al. 2020: cGAN Sentinel-2 RGB → AW3D30; "estimations of height have been thus far relative within a single image"; MIT | Repo README; journal listing | github.com/Panagiotou/ImageToDEM; Remote Sens. 12(12):2002 | Official repo / peer-reviewed | HIGH (relative-only statement) |
| E18 | Madani et al. 2025: pix2pix Landsat-5/7 30 m → SRTM; outputs scaled [-1,1]; RMSE 0.4671 normalized, SSIM 0.2065; weak in lowlands | arXiv HTML | arXiv 2511.21985 | Primary (preprint) | HIGH |
| E19 | FABDEM: Copernicus GLO-30 MAE forest 5.15 → 2.88 m, built-up 1.61 → 1.12 m after forest/building removal; validated vs LiDAR (12 countries) + ICESat-2 | Journal article | Hawker et al. 2022, ERL 10.1088/1748-9326/ac4d4f | Peer-reviewed | HIGH |
| E20 | Uuemaa et al. 2020: AW3D30/SRTM/NASADEM least slope-sensitive; forest effect tied to slope | Journal (via search) | Remote Sens. 12(21):3482 | Peer-reviewed | MEDIUM (summary-level) |
| E21 | Carabajal & Harding 2005/2006: SRTM positively biased in vegetation; ICESat−SRTM 0.60 ± 3.46 m (low relief, sparse trees), 3.53 ± 8.04 m (flat, dense vegetation), 5.61 ± 15.68 m (high relief) | GRL 2005; PE&RS 2006 (via search) | agupubs 10.1029/2005GL023957 | Peer-reviewed | MEDIUM-HIGH |
| E22 | Kattenborn et al. 2022: random CV overestimates CNN performance by up to 28 % vs spatial CV; cannot be fixed by sample size or regularization | ISPRS Open J. Photogramm. RS 5 | sciencedirect S2667393222000072 | Peer-reviewed | HIGH |
| E23 | CartoDEM/CartoDSM (NRSC): Cartosat-1 stereo 2.5 m; DEM 1/12″ (~2.5 m) and 30 m free on Bhuvan; vertical 8 m LE90, horizontal 15 m CE90; datum "WGS-84 (Horizontal & Vertical)"; nodata −32768; sinks/spikes in hill shadows; ~1 % gaps | NRSC presentation PDF text; Bhuvan docs | bhoonidhi.nrsc.gov.in …/9_UIM2024_CartoDSM.pdf | Government (ISRO/NRSC) | HIGH |
| E24 | Cartosat-3: 0.25 m pan, 1.13 m 4-band MX, 16 km swath; Cartosat-2S: 0.65 m pan, 2 m MX | eoPortal / Wikipedia (via search) | eoportal.org/satellite-missions/cartosat-3 | Technical reference | MEDIUM-HIGH |
| E25 | Qgis2threejs: DEM + imagery → Three.js, standalone HTML, glTF, GPL-3.0, active; deck.gl TerrainLayer: heightmap + texture, MARTINI error tolerance default 4 m; Cesium for Unity/Unreal Apache-2.0 with 3D Tiles; Unity RAW 16-bit heightmap import; Babylon `CreateGroundFromHeightMap` | Repos / docs | github.com/minorua/Qgis2threejs; deck.gl docs; cesium.com; docs.unity3d.com; doc.babylonjs.com | Official documentation | HIGH |
| E26 | DA-V2: Small Apache-2.0; Base/Large/Giant CC-BY-NC-4.0; Giant "coming soon"; params 24.8 M/97.5 M/335.3 M/1.3 B; metric heads fine-tuned on Hypersim (indoor) / Virtual KITTI 2 (outdoor); no generalization-failure statement in repo | Repo README + metric_depth README | github.com/DepthAnything/Depth-Anything-V2 | Official repo | HIGH |
| E27 | Marigold-DC (ICCV 2025): zero-shot depth completion from sparse points via guided diffusion; Prior Depth Anything (ICLR 2026): fuses sparse/low-res metric priors with relative foundation depth across 7 datasets; PromptDA, Omni-DC, DepthLab related | arXiv / repos (via search) | arXiv 2412.13389; arXiv 2505.10565 | Primary | MEDIUM-HIGH |
| E28 | SynRS3D: 69,667 images, GSD 0.09–1 m, six city styles, **nDSM** height maps, DINOv2 + DPT baseline, RS3DAda UDA; 11 real height datasets (Houston, JAX, OMA, GeoNRW-U/R, Potsdam, ATL, ARG, Nagoya, Tokyo, Vaihingen); T.D.2 RS3DAda MAE 4.87 vs train-on-real 5.38; authors show MAE/RMSE/δ can rate all-zero or doubled predictions as competitive → F1-HE | Paper PDF text | arXiv 2406.18151 (NeurIPS 2024 D&B) | Peer-reviewed | HIGH |
| E29 | GeoNRW: 7,783 triplets 1000², 1 m; orthophoto (0.1 → 1 m), **first-return LiDAR DEM** (DSM), 10 classes incl. forest; DL-DE-BY-2.0; "mostly urban" | IEEE DataPort / torchgeo docs (via search) | ieee-dataport.org/open-access/geonrw | Official dataset page | MEDIUM-HIGH |
| E30 | Maxar Precision3D: 50 cm, 3 m SE90 / LE90-CE90 absolute, no GCPs, multi-view stereo | Vendor / Esri pages (via search) | esri.com partner page; apollomapping | Vendor documentation | MEDIUM (vendor-reported) |
| E31 | Open Buildings 2.5D Temporal: Sentinel-2 time series, 4 m effective, height MAE 1.5 m; ground truth Global North only; possible large errors on tall buildings in Global South | Google Research site (via search) | sites.research.google/gr/open-buildings/temporal | Official project page | MEDIUM-HIGH |
| E32 | DEMIX wine contest: 15 criteria on elevation/slope/roughness vs 1–5 m LiDAR reference; CopDEM & FABDEM best 1″ DEMs, then ALOS, then NASADEM/SRTM, then ASTER; Höhle & Höhle 2009 recommend median, NMAD (1.4826·MAD), LE95 for non-Gaussian DEM errors | arXiv 2302.08425; ISPRS J. 64(4) (via search) | geomorphometry.org; vbn.aau.dk | Peer-reviewed | HIGH |
| E33 | US 11361196 B2 "Object height estimation from monocular images" — assignee **Zoox Inc**, street-level; US 12056888 B2 "Methods and apparatuses for calculating building heights from mono imagery" — assignee **Intermap Technologies**, DTM orthorectification + roof-edge geometry | Google Patents metadata | patents.google.com | Patent office record | HIGH |
| E34 | DFC2023 Track 2: optical (GaoFen) + SAR → nDSM; 1,773 images, 12 cities, 5 continents; 125,153 buildings | IEEE DataPort / JSTARS outcome paper (via search) | ieee-dataport.org; ieeexplore 10535214 | Official contest | MEDIUM-HIGH |
| E35 | Blackshark.ai SYNTH3D: ML-derived semantic 3D globe with buildings "with accurate heights"; no public method or accuracy | Vendor pages (via search) | blackshark.ai | Vendor marketing | LOW (UNVERIFIED) |
| E36 | IM2ELEVATION (Liu et al. 2020, Remote Sens. 12(17):2719; Dublin; LiDAR 2015 vs imagery 2017 registration issue); Amirkolaee & Arefi 2019 ISPRS J. 149:50–66; IM2HEIGHT 2018 (Mou & Zhu, arXiv 1802.10249); IMG2nDSM 2021; Mahmud et al. CVPR 2020 (cited with numbers in E11) | Journal listings / E11 tables | mdpi.com; sciencedirect S0924271619300139 | Peer-reviewed | MEDIUM-HIGH (bibliographic) |
| E37 | "Depth Anything 3" exists and was benchmarked zero-shot on aerial/satellite (E15, E16); its own paper not read this phase | E15, E16 | — | Secondary | MEDIUM (existence) / UNVERIFIED (details) |
| E38 | UseGeo (ISPRS Scientific Initiative, FBK): UAV imagery + LiDAR for monocular depth / MVS / NeRF benchmarking; 2024 evaluation paper in ISPRS Open J. | Project site (via search) | usegeo.fbk.eu | Official project | MEDIUM |
| E39 | Phase 2 SRTM regional accuracy figures (Himalaya RMSE 23–47 m; Brazil slope/forest dependence) | Not re-verified this phase | Phase 2 §5 | Inherited | MEDIUM (inherited, plausible, consistent with E21) |

---

## PHASE 3 COMPLETE

### 20 Most Important Prior-Art Findings
1. Single-image → height-above-ground (nDSM) is a mature research field ("monocular height estimation") with an operational global 3 m product (GlobalBuildingAtlas, RMSE 5.5 m; Asia 5.9 m). [E4][E12]
2. No found system produces a **datum-referenced absolute DSM from one image**; all single-view systems deliver nDSM/canopy height; absolute DSMs come from stereo/MVS (Precision3D, CartoDSM) or multi-view depth fusion (Sat3R). [E12][E30][E23][E16]
3. Zero-shot foundation depth models fail metrically at nadir: δ₁ ≈ 0 % on UAV nadir views; "severe scale drift" on DFC2019 satellite scenes (MAE 4.59 m). [E15][E16]
4. Fine-tuning a foundation encoder with LiDAR/RPC supervision works: canopy MAE 2.8 m (Tolan), canopy MAE 3× better (Depth Any Canopy), DFC2019 MAE 4.59 → 2.82 m (Sat3R). [E13][E14][E16]
5. SRTM is **EGM96-geoid**, not ellipsoidal (correcting Phase 2); Copernicus is EGM2008; AW3D30 EGM96; CartoDSM "WGS-84". [E2][E3][E23]
6. Global 30 m DEMs are biased upward by buildings (+1.6 m) and forest (+3–5 m); FABDEM quantifies and partially removes this. [E19][E21]
7. The SAC README does **not** state that evaluation uses ISRO imagery (correcting Phase 1/2); it says only "single-view optical satellite imagery (PNG, JPG, or TIFF)". [E1]
8. GAMUS as released is 8,724 non-georeferenced `.h5` tiles from three US cities (PHL/NYC/DC), 0.33 m aerial, nDSM — a segmentation benchmark; it cannot support SRTM-anchored calibration or hilly/forested evaluation. [E5][E6]
9. The same DFC19 data yields RMSE 2.12 m on random patch splits vs 5.46–9.26 m on the official spatial test set — spatial leakage dominates headline numbers. [E4][E11][E22]
10. Cross-morphology degradation is measured: São Paulo 9.3 m / Guangzhou 11.8 m vs Los Angeles 3.4 m (HTC-DC); S. America 8.9 m vs Oceania 1.5 m (GBA). [E4][E12]
11. 57 % of height pixels are < 1 m; MAE/RMSE/δ can rate all-zero predictions as competitive; object-focused metrics (F1-HE, building-wise RMSE) exist. [E4][E28]
12. Single **oblique** satellite image → height + orthorectifying flow is solved and open-sourced (geocentric pose), RMSE 5.46 m all-pixel on the official DFC19 test set. [E11]
13. Sparse-anchor and low-res-prior fusion with relative foundation depth is mature in camera frame (Wofk 150 points; Prior Depth Anything; Marigold-DC) but untested on nadir RS or with DEM priors. [E9][E27]
14. No paper found calibrates monocular RS depth with GCPs. §10
15. No public dataset pairs RGB with absolute DSM across urban/sparse/hilly/forested; GeoNRW (first-return DSM, forest class) and NEON canopy are the closest. [E29][E14]
16. SynRS3D provides 69,667 synthetic nDSM images and a UDA method, and lists 11 real height datasets — a wider landscape than Phase 2 recorded. [E28]
17. ISRO's own Cartosat-1 stereo pipeline achieves 8 m LE90 (CartoDSM, 2.5 m); Cartosat-3 RGB is 1.13 m MX (0.25 m pan). [E23][E24]
18. The visualization half is commodity: Qgis2threejs (standalone HTML/glTF), deck.gl TerrainLayer (MARTINI), Cesium for Unity/Unreal (Apache-2.0), Unity/Babylon heightmaps. [E25]
19. DEM validation methodology already has terrain-stratified (DEMIX) and robust-statistics (Höhle & Höhle) standards never applied to monocular DSMs. [E32]
20. Licensing is a real constraint: DA-V2 Base/Large are CC-BY-NC-4.0; HTC-DC Net has no licence; the relevant mono-height patent is Intermap's, not Zoox's. [E26][E4][E33]

### 15 Most Important Technical Gaps
GAP-01 zero-shot nadir scale failure · GAP-02 30 m DEM cannot calibrate building-scale DSM · GAP-03 nDSM + terrain → certified absolute DSM undocumented · GAP-04 cross-morphology generalization · GAP-05 no hilly/forested single-view benchmark · GAP-06 terrain/object separation · GAP-07 tall-object underestimation · GAP-08 GCP requirements unknown · GAP-09 validation without LiDAR over India · GAP-10 spatial leakage · GAP-11 vertical datum discipline · GAP-12 GSD/sensor mismatch · GAP-13 off-nadir PNG path · GAP-14 rDSM scoring semantics · GAP-16 standalone GPU + viewer packaging.

### 10 Things We Must NOT Claim as Innovation
1. Single-image height/DSM estimation from satellite imagery. 2. Applying Depth Anything/DINOv2 to remote-sensing height. 3. Scale-and-shift alignment from sparse anchors (with RANSAC). 4. Fusing coarse/sparse depth priors with relative depth. 5. Single-view oblique height with parallax correction. 6. Object-focused height metrics. 7. Terrain-stratified DEM evaluation. 8. Heightmap → LOD mesh → textured web/game-engine terrain. 9. First-person terrain navigation. 10. Removing forest/building bias from global DEMs.

### 10 Questions an ISRO Evaluator Could Attack Us With
1. Show the uncalibrated zero-shot depth on a Cartosat-3 scene — what is its scale? 2. How does a 30 m EGM96 DEM with +3–5 m canopy bias calibrate a 1 m DSM without double-counting? 3. Which vertical datum is the output in, and where is it tagged? 4. Which reference, at what resolution, co-registered how, acquired when? 5. Were train/test splits spatially disjoint? 6. How many GCPs, where, and how were calibration and check points separated? 7. What is the error on a forested hillside with no buildings, and what truth did you use? 8. DSM or nDSM — where does the terrain come from? 9. Is your RMSE dominated by ground pixels — give building-only and F1-HE. 10. GlobalBuildingAtlas gets 5.9 m in Asia with 168 LiDAR regions — what covers Indian morphology in your training data?

### 10 Evidence-Backed Constraints for Future Solution Design
1. Relative depth ≠ metric elevation; zero-shot metric fails at nadir. [E15][E16] 2. Georeferencing gives no vertical reference. [E2][E3] 3. SRTM = EGM96, Copernicus = EGM2008, CartoDSM = "WGS-84" — transform or bias. [E2][E3][E23] 4. 30 m DEMs carry no building-scale detail and are forest/building-biased. [E19][E21] 5. DSM ≠ nDSM ≠ DTM; the field and the organiser's dataset deliver nDSM. [E4][E5] 6. Vegetation is a first-surface confound needing explicit supervision. [E13][E21] 7. Domain shift is measured: 2–3× RMSE. [E4][E12] 8. Only spatially disjoint evaluation is credible. [E22] 9. Report object-focused metrics and bias, not only MAE/RMSE/correlation. [E28][E11] 10. Model licences (CC-BY-NC, none) constrain deployment. [E26][E4]

### Most Critical Unresolved Technical Challenge
**Converting a domain-shifted, scale-ambiguous single-view depth map into a datum-referenced absolute DSM using only a 30 m, geoid-referenced, canopy-biased DEM (or a handful of GCPs) — and proving its accuracy on hilly and forested Indian terrain for which no LiDAR-grade reference exists.** Every component has partial prior art; the combination has none, and the evidence says the coarse DEM can supply at most a scene-level scale/offset while the model must supply all building-scale relief unsupervised by any absolute reference.

READY FOR PHASE 4
