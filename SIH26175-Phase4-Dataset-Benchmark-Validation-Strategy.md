# Phase 4 — Dataset, Ground Truth, Benchmark, Calibration & Validation Strategy

SIH26175 DepthWizard · 2026-09-20

Scope note: Phase 4 answers *what data, reference, experimental design and validation procedure would be needed to scientifically prove an SIH26175 system works*. It does not design the solution, pick a model, stack, engine, split or metric set. Evidence tags: `[E#]` = items verified in the Phase 3 ledger (`SIH26175-Phase3-Prior-Art-and-Gap-Analysis.md` §27); `[F#]` = items verified in this phase (listed immediately below). Anything not tagged is either standard geodetic/statistical definition or explicitly marked UNVERIFIED / UNKNOWN.

**Inputs inspected:** Phase 1 (`SIH26175 DepthWizard — Problem Decomposition (Phase 1).md`, 354 lines), Phase 2 (`SIH26175-Phase2-Deep-Technical-Research.md`, 316 lines), Phase 3 (`SIH26175-Phase3-Prior-Art-and-Gap-Analysis.md`, 760 lines). The Phase 3 corrections table (SRTM = EGM96 geoid, not ellipsoid; no ISRO-evaluation-imagery statement in the SAC README; GAMUS HF release = 3 cities, `.h5`, nDSM; DFC19 random-split vs official-test discrepancy) is carried forward as established and is **not** re-litigated here; it is re-used where relevant.

**Phase 4 evidence verified this session**

| ID | Finding | Source / type | Confidence |
| --- | --- | --- | --- |
| F1 | `IMG-PROCESS-SAC/SIH2026` re-checked 2026-09-20: still 2 commits (2026-07-29, 2026-08-29), one file (`README.md`, 4,692 B, md5 unchanged vs. Phase 3 copy), branch `main` only, 0 releases, 0 issues, no licence. | GitHub API + raw fetch; official organiser repo | HIGH |
| F2 | swissSURFACE3D Raster: LiDAR-derived **DSM**, 0.5 m grid, ±10 cm (1σ), CRS LV95, vertical reference **LN02**, 1 km² COG tiles, nationwide incl. Alps, 1/6 of country refreshed yearly. swisstopo OGD since 1 Mar 2021: free incl. commercial use, attribution only; SWISSIMAGE orthophotos also OGD (GSD not verified this phase). | swisstopo product & OGD pages | HIGH |
| F3 | LINZ New Zealand: national 1 m LiDAR DEM and 1 m DSM composites (400+ GeoTIFF tiles), ±0.2 m (95 %) vertical, ±1.0 m (95 %) horizontal, ≥4 pulses/m², NZTM2000 / **NZVD2016**, **CC BY 4.0**; regional DSM layers 2016–2025; also mirrored on OpenTopography. Aerial imagery on LDS is CC BY 4.0 (GSD not verified). | LINZ Data Service layer pages; OpenTopography | HIGH |
| F4 | AHN4 (Netherlands, 2020–22): 0.5 m **DSM and DTM**, EPSG:28992, vertical **NAP (EPSG:5709)**, **CC0**, COG via PDOK; AHN5 changes DTM interpolation (IDW → unweighted mean). | data.overheid.nl, OpenTopography, GEE catalog | HIGH |
| F5 | USGS 3DEP: LiDAR point clouds public domain on AWS as Entwine Point Tiles (lossless, full density); QL2 spec ANPS ≤ 0.7 m, ≥ 2 pts/m²; QL1/QL2 vertical 10 cm RMSEz; the GEE `USGS/3DEP/1m` product is **bare-earth (DTM)** in NAVD88 / NAD83 UTM; NAIP (`USDA/NAIP/DOQQ`) 1 m (0.6 m in newer years), RGB/RGBN, public domain, growing-season (leaf-on). | USGS pages, AWS Open Data registry, GEE catalog | HIGH |
| F6 | Spaceborne lidar as sparse absolute reference: ICESat-2 ATL08 terrain RMSE ≈ 0.73 m vs airborne lidar (Neuenschwander 2020); ≈ 1.0 m plains → ≈ 4.3 m alpine canyons (central-south China); 2.35 m vs LVIS in forest; GEDI L2A ground RMSE 2.2–4.0 m (mid/low latitude), 5–7 m in forest/complex terrain. | Peer-reviewed validations (RSE, MDPI RS, Frontiers) | MEDIUM-HIGH |
| F7 | ASPRS Positional Accuracy Standards Ed. 2 (2023): min **30 checkpoints**; checkpoints "independent … not used in processing or calibrating the product"; at least **2× the accuracy** of the product; a discrepancy > **3× target RMSE** is a blunder to investigate, not silently drop; mean error > 25 % of RMSE must be reported; NVA has pass/fail, **VVA is report-only**; area split by land cover (vegetated / non-vegetated); "a quantitative methodology for … spatial distribution of checkpoints … does not currently exist"; 95 % = 1.96·RMSEz retained only as reporting convention. | ASPRS Ed. 2 main body PDF (text extracted) | HIGH |
| F8 | Nuth & Kääb 2011 (The Cryosphere 5:271): analytic DEM co-registration using the relationship between elevation difference and terrain slope/aspect over stable terrain; three-step bias correction (shift → elevation-dependent → sensor-specific); basis of modern tooling (xdem). | Primary paper | HIGH |
| F9 | Licences: FABDEM **CC BY-NC-SA 4.0** (non-commercial); Copernicus GLO-30 free to any registered user; AW3D30 free incl. commercial with JAXA account; NASADEM / SRTM public domain (LP DAAC; Earthdata login required). | Official dataset pages | HIGH |
| F10 | Indian Space Policy 2023 as implemented on Bhoonidhi: EO data **≥ 5 m GSD free and open**; **< 5 m priced** for non-government entities via NSIL — Cartosat-1, -2, -2S, -3 are in the priced category; open: ResourceSat-2/2A (LISS-III, LISS-IV), EOS-04/06, Landsat-8/9, Sentinel-1/2 mirror, CartoDEM. | ISRO policy PDF; NRSC/Bhoonidhi docs | HIGH |
| F11 | Spatial CV literature: Roberts et al. 2017 (Ecography) — ignoring structure "results in serious underestimation of predictive error", block CV; Ploton et al. 2020 (Nat. Comm. 11:4540) — random validation over-optimistic; Meyer & Pebesma 2022 (Nat. Comm. 13:2208) — area-of-applicability; Kattenborn 2022 [E22] up to 28 % inflation. | Peer-reviewed | HIGH |
| F12 | LiDAR reference practice: ASPRS LAS classes 2 = ground, 5 = high vegetation, 6 = building; DSM = TIN of first returns rasterised; DTM = ground-class TIN; spike-free / pit-free algorithms exist to suppress interpolation pits (Khosravipour et al.; lidR `dsm_pitfree`). | USGS LBS; rapidlasso; lidR docs | HIGH |
| F13 | Geoid–ellipsoid separation over India computed with PROJ official grids (EPSG:4979 → 9707 EGM96 / 9518 EGM2008): Kanyakumari −98.6 m, Bengaluru −86.4 m, Mumbai −68.5 m, Ahmedabad −55.4 m, Kolkata −56.9 m, Delhi −52.6 m, Dehradun −43.8 m, Guwahati −49.0 m, Leh −23.7 m; EGM96 vs EGM2008 differ by ≤ 1.0 m at these points. | Computed (pyproj 3.x, PROJ network grids) | HIGH |
| F14 | Open-Canopy (Fogel et al. 2024): 87,000 km² France, 1.5 m satellite imagery, airborne-LiDAR canopy height truth, CC BY 4.0. | arXiv 2407.09392 | HIGH (existence/scope) |
| F15 | DFC2018 Houston: 5 cm RGB (DiMAC), multispectral LiDAR, 0.5 m DSM, hyperspectral; access by e-mail request and acknowledgement terms; single urban campus. | Univ. of Houston / GRSS pages | MEDIUM-HIGH |
| F16 | Uncertainty evaluation: sparsification curves, AUSE / AURG (Ilg et al. 2018; Poggi et al. CVPR 2020); usability: System Usability Scale (Brooke 1996) — 10 items, 0–100. | Primary papers | HIGH |
| F17 | GAMUS HF release content, HTC-DC Net tables, DFC19 official-test numbers, SRTM/Copernicus/AW3D30 datums, FABDEM forest/building bias, CartoDSM 8 m LE90, Cartosat-3 GSDs — all inherited from Phase 3 ledger and **not** re-verified this phase (they were verified 2026-09-20 from primary sources). | Phase 3 [E1–E39] | HIGH (as of Phase 3) |

---

## 1. Executive Summary

**The data problem is harder than the modelling problem.** Phase 3 established that every component of SIH26175 has prior art; Phase 4 establishes that *proving* a system works requires data that the organiser has not supplied and that no single public dataset provides. Specifically:

1. **The official repository supplies no data** — one README, no imagery, no reference, no licence, unchanged since 2026-08-29 [F1]. The recommended GAMUS is a non-georeferenced nDSM segmentation benchmark from three US cities [E5][E6]. Everything else — training, calibration, validation, test — must be assembled from third-party sources.
2. **Ground-truth type must match the prediction target.** The PS asks for an *absolute DSM*; almost every RGB→height dataset (GAMUS, DFC19, GBH, SynRS3D, DFC2023) provides *nDSM* (height above local ground) [E4][E5][E28][E34]. Only national LiDAR programmes ship true, datum-referenced DSMs alongside open orthophotos: swisstopo (0.5 m DSM, LN02) [F2], LINZ (1 m DSM, NZVD2016) [F3], AHN (0.5 m DSM, NAP) [F4], USGS 3DEP point clouds (public domain; the ready-made 1 m raster is DTM, so a DSM must be rasterised from first returns) [F5][F12], GeoNRW (first-return 1 m) [E29].
3. **Hilly and forested truth exists — outside India.** Switzerland and New Zealand cover alpine relief and forest with open LiDAR DSMs [F2][F3]; Open-Canopy covers 87,000 km² of French forest at 1.5 m [F14]; NEON covers US forest [E14]. No LiDAR-grade Indian reference was found; the Indian references are CartoDEM/CartoDSM (stereo, 8 m LE90) [E23], global 30 m DEMs, and spaceborne lidar footprints (ICESat-2 ATL08 ≈ 0.7–4 m terrain RMSE depending on relief) [F6].
4. **Indian high-resolution imagery is priced.** Under Indian Space Policy 2023 anything finer than 5 m (Cartosat-1/2/2S/3) is priced for non-government entities; LISS-IV (5.8 m) and CartoDEM are open [F10]. Whether the organiser will supply Cartosat scenes is UNKNOWN.
5. **Validation standards already exist and are strict.** ASPRS Ed. 2 requires ≥ 30 independent checkpoints at ≥ 2× product accuracy, blunder investigation above 3× RMSE, separate reporting of vegetated vs non-vegetated accuracy and of mean error [F7]; DEM literature requires robust statistics (median, NMAD, LE95) [E32] and co-registration before differencing [F8]; the spatial-CV literature requires spatially blocked splits [F11][E22].
6. **The vertical datum is a first-order effect in India.** Geoid–ellipsoid separation ranges from −24 m (Leh) to −99 m (Kanyakumari) [F13] — larger than any building — so an ellipsoidal/orthometric mix-up would dwarf every other error in an "absolute DSM". EGM96 vs EGM2008 differ by ≤ 1 m.
7. **A coarse DEM can supply a scene-level scale/offset and a low-frequency terrain trend; it cannot supply building- or tree-scale relief, and it carries its own +1.6 m (built-up) / +3–5 m (forest) bias** [E19][E21]. It is therefore a *calibration* input, not a *validation* reference, for a sub-metre DSM.
8. **The two halves of the score need two validation regimes.** DSM accuracy is objectively measurable if the above is respected; visualization quality is partly objective (projection residuals, frame time, memory, crash-free runs, clean-machine install) and partly instrument-based (SUS) [F16].

**What an ISRO evaluator would need to see (evidence, not design):** a stated prediction target (nDSM vs DSM) matched to a stated reference type and vertical datum; spatially disjoint train/test regions; per-terrain (urban / sparse / hilly / forested) metric tables with robust statistics and object-class breakdowns; calibration anchors kept separate from checkpoints; residual maps; an external-region test; and reproducibility metadata.

---

## 2. SIH26175 Data Requirements

| Stage | Data Required | Ground Truth Required | Metadata Required | Resolution / GSD | Purpose | Main Risk |
| --- | --- | --- | --- | --- | --- | --- |
| A. Training (backbone adaptation) | Nadir RGB orthoimagery paired pixel-wise with height | nDSM (for height-above-ground) **or** DSM + DTM (for absolute); LiDAR-derived preferred | Pixel grid alignment; GSD; acquisition dates of both layers; land-cover mask desirable | 0.1–3 m per published MHE practice [E4][E28] | Close the natural→nadir domain gap (Phase 3 GAP-01) | Only nDSM available at scale; US/EU urban bias; GSD mismatch to evaluation sensor [E4][E12] |
| B. Fine-tuning (target-domain) | Smaller RGB–height sets closer to evaluation GSD / geography / morphology | Same as A | Same as A + sensor identity | Match expected evaluation GSD (e.g. ~1 m MX or ~0.3 m pan-sharpened if Cartosat-3 [E24]) | Reduce cross-morphology loss (GAP-04) | No Indian RGB–LiDAR pairs found; Cartosat imagery priced [F10] |
| C. Calibration (relative → absolute) | Coarse DEM tiles (SRTM/Copernicus/AW3D30/CartoDEM) and/or sparse elevation anchors (GCPs, ICESat-2/GEDI footprints) covering the scene | None (calibration data ≠ truth) but anchor accuracy must be known | DEM vertical datum & posting; anchor CRS + vertical datum + accuracy; scene geotransform | 30 m DEM; ATL08 100 m / 20 m segments; GEDI 25 m footprints | Fix scale/offset (GAP-02) | Datum mix-up (−24 to −99 m in India [F13]); DEM canopy bias [E19][E21]; anchors clustered |
| D. Validation (model selection, tuning) | RGB–DSM pairs from regions disjoint from A/B | LiDAR DSM (absolute) or nDSM, matching the *claimed* output | Same as A + vertical datum + LiDAR date | Same as A | Choose hyper-parameters honestly | Leakage from A/B via adjacent tiles [E22][F11] |
| E. Testing (held-out) | RGB–DSM pairs from regions never touched in A–D | LiDAR DSM; ≥ 30 independent checkpoints per land-cover class if point-based [F7] | Full geospatial metadata; co-registration evidence [F8] | Same as A | Report headline RMSE/MAE/corr | Same city/sensor as training → optimistic |
| F. External generalization | RGB from a different country / sensor / GSD / morphology with reference | LiDAR DSM where possible; else ICESat-2 footprints [F6] or CartoDEM-grade DEM with stated error [E23] | Sensor, date, off-nadir angle, GSD | Different from A | Evidence that results transfer (GAP-04, GAP-12) | Reference coarser than prediction → measures reference error (Phase 3 §12) |
| G. Visualization validation | Predicted DSM + source RGB + reference DSM; large rasters for stress tests | Reference DSM for geometric fidelity; known control features for projection checks | Geotransform for both texture and height layers | Full scene size | Score projection accuracy, fidelity, navigability, stability, deployment | Visual quality masking geometric error (Phase 3 §23) |

Per-item requirements common to all stages: horizontal CRS with EPSG code; affine geotransform; pixel size; nodata value; **vertical CRS** (EGM96 / EGM2008 / NAVD88 / NAP / LN02 / NZVD2016 / ellipsoidal); acquisition date of imagery *and* of reference; sensor and off-nadir angle where available [E2][E3][F2–F5].

---

## 3. Official ISRO/SAC Reference Data Findings

Re-inspected 2026-09-20 [F1]; identical to the Phase 3 inspection.

| Item | Status | Detail |
| --- | --- | --- |
| Available files | VERIFIED | `README.md` only (4,692 bytes). Tree listing (recursive) shows no other blobs. |
| Imagery / sample data | NOT FOUND | None in repository; no releases (0), no LFS pointers. |
| File formats | VERIFIED (statement only) | README names PNG, JPG, TIFF/GeoTIFF as *accepted inputs*; no sample files. |
| Metadata / dimensions / GSD / coverage / CRS | NOT FOUND | No files to carry metadata. |
| Elevation information (DEM/DSM/DTM) | VERIFIED (statement only) | "A lower-resolution DEM source such as SRTM 30 m may be used"; no DEM shipped. |
| Reference outputs / labels | NOT FOUND | None. |
| Scripts | NOT FOUND | None. |
| Documentation | VERIFIED | README restates the PS background, pipeline, milestones, evaluation table (50/50), deliverables; recommends GAMUS (`earthflow/GAMUS`, Hugging Face); permits "any open-source dataset containing remote-sensing depth data, provided it supports both relative depth training and metric scale calibration". |
| Evaluation information | VERIFIED (statement only) | "RMSE, MAE, and correlation against LiDAR/reference data. Must demonstrate performance stability across urban, sparse, hilly, and forested landscapes." No test set, no protocol, no sensor named. |
| Evaluation imagery source | NOT FOUND | No statement (Phase 3 correction stands). |
| Licensing / access | UNKNOWN | Repository has no licence file; README has no licence statement. Public read access. |
| Recommended dataset content (GAMUS on HF) | VERIFIED [E6] | 8,724 tiles (PHL/NYC/DC), `.h5`, RGB + nDSM + 6-class labels, CC-BY-4.0, no georeferencing. |
| Consistency of README claims with GAMUS content | CONTRADICTION (carried from Phase 3) | README: GAMUS "translate 2D satellite imagery into accurate depth models" and teaches "urban, sparse, hilly, and forested" — GAMUS is aerial 0.33 m, urban, nDSM, three cities. |

Nothing further can be inferred from the repository.

---

## 4. Dataset Landscape

Only datasets with meaningful RGB → depth/height/elevation relevance. "DSM" = true surface elevation on a datum; "nDSM" noted where that is what ships.

| Dataset | RGB | Single View | DSM | DTM | LiDAR | Building Height | GSD | Geography | Terrain | CRS | Vertical Reference | License | Access | Main Limitation | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAMUS (HF) | Yes | Yes | No (nDSM) | No | Derived | Via nDSM | 0.33 m | PHL, NYC, DC (USA) | Dense urban | None shipped | n/a | CC-BY-4.0 | HF, no login | Non-georeferenced `.h5`; 3 cities; urban only | [E5][E6] |
| DFC2019 / US3D | Yes (WV-3 pan+VNIR) | Yes (Track 1) | Track 3 DSM; Track 1 nDSM | Derivable | Yes (80 cm NPS) | Via footprints | ~0.3 (native) / 1.3 m (as used) | Jacksonville, Omaha (+ Atlanta, San Fernando via geopose ext.) | Urban/suburban | RPC / UTM | LiDAR datum (US) | IEEE DataPort open; repo MIT | Registration | Two US cities; official test truth partly withheld | [E10][E11] |
| GBH (TUM) | Yes (PlanetScope) | Yes | No (nDSM) | No | Derived | Yes | 3 m | 19 + 3 global cities | Urban | Not stated | n/a | Not fully public | Request | Availability | [E4] |
| GlobalBuildingAtlas | Yes (+NIR) | Yes | No (building nDSM) | No | Yes | Yes | 3 m | Global outputs; training NA/EU/Oceania | Urban | Yes | n/a | ODbL | Open | Buildings only | [E12] |
| SynRS3D | Yes (synthetic) | Yes | No (nDSM) | No | Synthetic | Via land cover | 0.09–1 m | Synthetic (6 styles) | Synthetic urban + vegetation | None | n/a | Open | Open | Sim-to-real gap | [E28] |
| ISPRS Vaihingen / Potsdam | IRRG / RGB(IR) | Yes | Potsdam ships DSM + nDSM; Vaihingen nDSM | No | Yes | Via nDSM | 0.09 / 0.05 m | Germany | Small-town urban | Tiles | German (unverified) | Open (ISPRS) | Registration | Small; urban | [E4][E5] |
| GeoNRW | Yes | Yes | Yes (first-return 1 m) | No | Yes | No | 1 m | North Rhine-Westphalia | Mostly urban; forest class present | Yes | German (unverified) | DL-DE-BY-2.0 | IEEE DataPort | Terrain mostly flat/urban | [E29] |
| DFC2018 Houston | Yes (5 cm) | Yes | Yes (0.5 m) | Derivable | Yes (multispectral) | Via DSM | 0.05 m / 0.5 m | Houston campus | Urban | Yes | US (unverified) | Terms by e-mail | Request | Single small site | [F15] |
| DFC2023 Track 2 | Yes (GaoFen) + SAR | Partial | No (nDSM) | No | Derived | Yes | ~0.5–0.8 m (unverified) | 12 cities, 5 continents | Urban | Not stated | n/a | Contest terms | Registration | Needs SAR channel | [E34] |
| **swisstopo swissSURFACE3D Raster + SWISSIMAGE** | Yes (ortho) | Yes | **Yes, 0.5 m** | swissALTI3D (DTM, separate product) | Yes | Via DSM−DTM | 0.5 m DSM; ortho 0.1–0.25 m (GSD UNVERIFIED) | Switzerland | **Alpine hilly, forest, urban, sparse** | LV95 (EPSG:2056) | **LN02** (levelling datum, not a geoid) | OGD, free incl. commercial, attribution | Open download | LN02 ≠ EGM; date offset between ortho and LiDAR tiles | [F2] |
| **LINZ LiDAR 1 m DSM + LINZ aerial imagery** | Yes | Yes | **Yes, 1 m** | Yes (1 m DEM) | Yes (≥ 4 pls/m²) | Via DSM−DEM | 1 m; imagery GSD UNVERIFIED | New Zealand (regional) | **Hilly, forest, rural, urban** | NZTM2000 | **NZVD2016** | CC BY 4.0 | Open (LDS, OpenTopography) | Regional patchwork dates | [F3] |
| **AHN4 + PDOK luchtfoto** | Yes | Yes | **Yes, 0.5 m** | Yes | Yes | Via DSM−DTM | 0.5 m; imagery GSD UNVERIFIED | Netherlands | Flat urban / rural / some forest; **no hills** | EPSG:28992 | **NAP** | CC0 (AHN); imagery licence UNVERIFIED | Open (PDOK) | No relief | [F4] |
| **USGS 3DEP point clouds + NAIP** | Yes (NAIP 0.6–1 m) | Yes | **Derivable** (rasterise first returns) | Yes (1 m DEM product) | Yes (QL2 ≥ 2 pts/m²) | Via DSM−DTM | 1 m | CONUS | **All four terrain classes available** | NAD83 UTM | **NAVD88** | Public domain | AWS EPT, GEE, TNM | DSM must be built; NAIP leaf-on and dated independently of lidar | [F5][F12] |
| Open-Canopy | Yes (1.5 m satellite) | Yes | No (canopy height) | No | Yes | No | 1.5 m | France, 87,000 km² | **Forest** | Yes | n/a | CC BY 4.0 | Open | Vegetation only | [F14] |
| NEON / EarthView | Yes (0.1 m) | Yes | No (CHM) | NEON also ships DTM/DSM rasters (UNVERIFIED here) | Yes | No | 0.1 m / 1 m | USA | **Forest** | Yes | US | Open | Open | Vegetation focus | [E14] |
| Tolan/Meta canopy map | (uses Maxar; imagery not redistributed) | Yes | No | No | Yes (training) | No | 1 m | Global | Forest | Yes | n/a | Product CC-BY (UNVERIFIED) | AWS/GEE | Prediction, not truth | [E13] |
| UseGeo (ISPRS) | Yes (UAV) | Yes | Yes (LiDAR) | — | Yes | No | cm | Italy | Rural/hilly | Yes | UNVERIFIED | Open (ISPRS) | Request | UAV geometry ≠ satellite | [E38] |
| SRTM v3 / NASADEM | No | — | DSM-like 30 m | No | No | No | 30 m | Global 60°N–56°S | All | EPSG:4326 | **EGM96** | Public domain | Earthdata login | Coarse; canopy bias | [E2][F9] |
| Copernicus GLO-30 | No | — | DSM 30 m | No | No | No | 30 m | Global | All | EPSG:4326 | **EGM2008** | Free, registered users | CDSE / AWS | Coarse; +5 m forest bias | [E3][E19][F9] |
| FABDEM | No | — | DTM-like 30 m | ≈ | No | No | 30 m | Global | All | EPSG:4326 | EGM2008 | **CC BY-NC-SA 4.0** | Bristol / GEE community | Non-commercial | [E19][F9] |
| ALOS AW3D30 v4.1 | No | — | DSM 30 m | No | No | No | 30 m | Global | All | EPSG:4326 | **EGM96** | Free incl. commercial, account | JAXA / GEE | Coarse | [E3][F9] |
| CartoDEM v3 (30 m) / CartoDSM (2.5 m) | No | — | DSM (stereo) | No | No | No | 30 m free / 2.5 m | India | All Indian terrain | EPSG:4326 | "WGS-84" (ellipsoid vs geoid UNCLEAR) | Free (30 m) / priced (2.5 m, UNVERIFIED) | Bhuvan / Bhoonidhi | 8 m LE90; sinks/spikes in hill shadow | [E23][F10] |
| ICESat-2 ATL08 / ATL03 | No | — | Point/segment elevations (terrain + canopy) | ≈ (terrain) | Spaceborne lidar | No | 100 m / 20 m segments along track | Global | All | WGS84 | **Ellipsoidal (WGS84) with geoid fields supplied** | Public domain | NSIDC / Earthdata login | Sparse tracks; 0.7–4 m RMSE by relief | [F6] |
| GEDI L2A | No | — | Footprint ground + RH metrics | ≈ | Spaceborne lidar | Weak (RH for buildings studied) | 25 m footprints, 51.6°N–S | All | WGS84 | Ellipsoidal | Public domain | Earthdata | 2–7 m ground RMSE | [F6] |
| ResourceSat-2/2A LISS-IV | Yes (3-band) | Yes | No | No | No | No | 5.8 m | India | All | Yes | — | Open (≥ 5 m policy) | Bhoonidhi login | Too coarse for building-scale; only open Indian optical near-HR | [F10] |
| Cartosat-2S / Cartosat-3 | Yes (MX) / pan | Yes | No | No | No | No | 0.65 / 2 m; 0.25 / 1.13 m | India | All | Yes | — | **Priced for NGEs** | NSIL | Cost; availability to team UNKNOWN | [E24][F10] |

---

## 5. Dataset Role Analysis

| Dataset | Training | Fine-Tuning | Calibration | Validation | Testing | External Test | Reference Only | Main Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAMUS | Yes (RGB→nDSM) | Yes | **No** (no coordinates) | Yes (nDSM only) | Yes (nDSM only) | No (same 3 cities) | — | Cannot evidence absolute DSM or non-urban terrain [E6] |
| DFC2019 / US3D | Yes | Yes | No | Yes | Yes (official Track-1 test if truth obtainable) | Partial (Atlanta / San Fernando extension) | — | Random patch splits leak [E4][E11] |
| SynRS3D | Yes (pre-training / augmentation) | No | No | No (synthetic) | No | No | — | Sim-to-real gap; never as truth [E28] |
| GlobalBuildingAtlas outputs | No (predictions) | No | Weak prior only | No | No | No | Comparison baseline | It is a model output, not truth [E12] |
| GeoNRW | Yes (RGB→DSM) | Yes | No | Yes | Yes | Partial (different sensor/GSD) | — | Flat; verify vertical datum |
| swisstopo DSM + SWISSIMAGE | Yes (if tiles co-registered & dated) | Yes | Anchors can be *derived* (as if GCPs) but must then be excluded from validation | **Yes — hilly/forest/urban/sparse** | Yes | Yes (non-US morphology) | — | LN02 vertical datum conversion; ortho/LiDAR date offset [F2] |
| LINZ DSM + imagery | Yes | Yes | Same as above | **Yes — hilly/forest** | Yes | Yes | — | Regional coverage patchwork; NZVD2016 conversion [F3] |
| AHN4 + luchtfoto | Yes | Yes | Same | Yes (flat urban/rural) | Yes | Yes | — | No hilly terrain; NAP datum [F4] |
| USGS 3DEP + NAIP | Yes (all terrain) | Yes | Same | **Yes — all four classes** | Yes | Partial (same country as GAMUS/DFC) | — | DSM must be rasterised; leaf-on NAIP vs lidar date [F5] |
| Open-Canopy / NEON | Yes (forest height) | Yes | No | Yes (forest, canopy height only) | Yes | Yes | — | Canopy height ≠ DSM |
| SRTM / Copernicus / AW3D30 / CartoDEM | No | No | **Yes (scene-level scale/offset, low-frequency terrain)** | No (too coarse) | No | No | Yes | Treating as truth measures the DEM's error [E2][E19][E21] |
| FABDEM | No | No | Yes (terrain trend, non-commercial) | No | No | No | Yes | CC BY-NC-SA [F9] |
| ICESat-2 ATL08 / GEDI | No | No | **Yes (sparse absolute anchors, India-capable)** | Partial (independent footprints as checkpoints if not used for calibration) | Partial | Yes (India) | Yes | Sparse; 0.7–4 m error; must convert ellipsoidal → target datum [F6] |
| CartoDSM 2.5 m | No | No | Partial (coarse anchor) | Partial (with 8 m LE90 error budget stated) | Partial | Yes (India) | Yes | Reference error ≥ expected model error [E23] |
| LISS-IV | Possible input domain test | No | No | No | No | Yes (Indian sensor, but 5.8 m) | — | GSD too coarse for building heights [F10] |
| Cartosat-2S/3 | Ideal fine-tune/test input **if obtained** | Yes | No | Needs external reference | Yes | Yes | — | Priced; availability UNKNOWN [F10] |

**Justification of the key classifications.** (i) A dataset can be *training* for a target only if its truth is that target: GAMUS trains nDSM, not DSM. (ii) A source is *calibration* if it supplies absolute elevation information coarser or sparser than the prediction — coarse DEMs and spaceborne-lidar footprints — and *cannot simultaneously be validation* for the same scene (ASPRS independence rule [F7]). (iii) A source is *validation/test* only if its accuracy is at least ~2× better than the accuracy being claimed [F7]; a 30 m DEM with 3–16 m σ cannot validate a metre-level DSM, and CartoDSM at 8 m LE90 can only validate claims coarser than ~4 m. (iv) *External test* requires a different region and, ideally, sensor.

---

## 6. Ground-Truth Taxonomy

**Definitions (standard geodesy/photogrammetry):**
- **DEM** — generic "digital elevation model"; in practice often bare-earth, but products named DEM (SRTM, Copernicus, AW3D30, CartoDEM) are *surface* models [E2][E3][E23].
- **DTM** — bare-earth terrain; from lidar ground-class points (LAS class 2) [F12].
- **DSM** — first-surface elevation incl. buildings/canopy; from lidar first returns [F12].
- **LiDAR DSM / DTM** — the above from airborne laser scanning; QL2 accuracy ~10 cm RMSEz [F5].
- **Photogrammetric DSM** — from stereo/MVS matching (Cartosat-1 CartoDSM 8 m LE90; Precision3D 3 m SE90); sees canopy top; noisy in shadow [E23][E30].
- **Building-height map** — per-building scalar (median nDSM within footprint) [E4][E12].
- **Relative height map (rDSM)** — height up to unknown affine transform.
- **Canopy height model (CHM)** — DSM − DTM over vegetation [E13][F14].
- **nDSM** — DSM − DTM, all objects [E4][E5].
- **Absolute elevation** — height on a stated vertical datum.
- **Orthometric height** — above the geoid / mean sea level (EGM96, EGM2008, NAVD88, NAP, NZVD2016; LN02 is a levelling datum approximating this).
- **Ellipsoidal height** — above WGS84 ellipsoid (GNSS raw, ICESat-2/GEDI native) [F6][F13].

| Prediction Target | Appropriate Ground Truth | Why | Possible Incorrect Reference | Caveat |
| --- | --- | --- | --- | --- |
| RGB → Relative depth / rDSM (Mode A) | Any dense height or DSM raster, *after* least-squares affine alignment of the prediction to it (scale + shift fitted on the test tile) | Relative output is defined only up to affine transform; only structure can be scored | Scoring rDSM in metres without alignment; scoring against a coarse DEM | Report affine-invariant metrics (δ-thresholds, AbsRel after alignment, Spearman); the fitted (s, t) must be reported and not counted as "calibration" |
| RGB → Metric depth (camera frame) | Camera-frame depth (photogrammetric or lidar reprojected via sensor model / RPC) | Depth is viewpoint-dependent | DSM raster (a height, not a depth) | For orthorectified inputs the concept collapses to height; for off-nadir inputs RPC geometry is needed [E11][E16] |
| RGB → Height above ground (nDSM) | LiDAR nDSM (DSM − DTM) on the image grid; GAMUS/DFC19/GBH-type truth | Datum-free, terrain-free quantity | Absolute DSM (contains terrain); coarse DEM; CHM only (misses buildings) | nDSM under canopy is canopy height; nDSM of bridges/overpasses ambiguous |
| RGB → Absolute elevation (surface) | LiDAR DSM in a stated vertical datum, resampled to the image grid, co-registered [F8] | Matches the claimed physical quantity | nDSM (no terrain); DTM (no objects); SRTM/Copernicus at 30 m (too coarse, biased); ellipsoidal heights unconverted | Requires datum transformation of either side; reference date vs image date |
| RGB → DSM (Mode B deliverable) | LiDAR DSM as above **plus** independent checkpoints (≥ 30, ≥ 2× accuracy) for ASPRS-style reporting [F7] | PS names LiDAR as reference | "DSM" that is actually nDSM (most MHE datasets); photogrammetric DSM with 8 m LE90 for sub-metre claims | If only a coarse or photogrammetric reference exists, the reference error must be propagated into the reported uncertainty |
| RGB → Building height (per instance) | Building footprints × nDSM aggregated (median) | Standard in MHE [E4][E12] | Pixel RMSE alone | Footprint source and date matter |
| RGB → Canopy height | ALS CHM [E13][F14] | Standard | DSM (includes ground trend) | Seasonality; leaf-on/off |

---

## 7. LiDAR Ground Truth

**What makes a lidar-derived DSM a defensible reference** (synthesis of USGS LBS, ASPRS Ed. 2, lidR/LAStools practice, DEM literature) [F5][F7][F12][F8]:

| Aspect | Established practice | Why it matters for SIH26175 |
| --- | --- | --- |
| Point density | QL2 ≥ 2 pts/m² (ANPS ≤ 0.7 m); QL1 ≥ 8 pts/m² [F5] | Density bounds the smallest resolvable structure; 2 pts/m² supports ~1 m DSM, not 0.3 m |
| Classification | LAS classes 2 ground, 5 high vegetation, 6 building; automated ground filters plus QA [F12] | DTM quality depends on ground classification; nDSM inherits its errors |
| DSM generation | TIN of **first returns** (highest returns) rasterised; or max-Z per cell; spike-free / pit-free TIN variants to suppress interpolation pits in canopy [F12] | Naive max-per-cell leaves pits and spikes; method must be recorded |
| DTM generation | TIN of ground-class points rasterised; interpolation across building footprints/water | Interpolation under large buildings is invented terrain; affects nDSM there |
| Rasterisation resolution | Cell size ≈ 1/√density or coarser (1 m for QL2; 0.5 m for dense national programmes [F2][F4]) | Reference posting should be ≤ prediction posting or aggregation is required (§16) |
| Vertical accuracy | NVA tested on ≥ 30 independent checkpoints, RMSEz; VVA reported separately [F7]; QL2 10 cm RMSEz [F5] | Reference error budget must be stated so it can be compared with the claimed model error |
| Horizontal accuracy | Programme-specific (e.g. LINZ ±1.0 m 95 % [F3]) | Horizontal misalignment converts into vertical error at edges/slopes (§14) |
| CRS / datum | Horizontal projected CRS; vertical datum explicit (NAVD88, NAP, LN02, NZVD2016) [F2–F5] | Must be transformed to the datum of the coarse DEM / claimed output |
| Spatial extent and alignment | Reference must fully cover the image footprint; pixel-centre vs pixel-corner convention checked; same grid after resampling | Partial coverage → nodata contamination (§13) |
| Temporal consistency | Lidar and imagery dates recorded; change masks for demolished/new buildings [E36] | IM2ELEVATION showed 2-year offsets corrupt MAE |
| Vegetation handling | Leaf-on vs leaf-off; first-return DSM in leaf-off season is lower than in leaf-on imagery | Forested class error is partly reference-induced |
| Water | Lidar returns over water are sparse/noisy; masked or flattened | Water pixels should be masked or reported separately |
| Documentation | Metadata: survey date, sensor, density, classification method, accuracy report | Required for reproducibility (§27) |

Sources shipping ready-made DSMs (swisstopo, LINZ, AHN, GeoNRW, Potsdam) already embody these steps; for USGS 3DEP the DSM must be generated from EPT point clouds and the method recorded [F5][F12].

---

## 8. DEM/SRTM Calibration Data

| Source | Posting | Coverage | CRS | Vertical reference | Surface | Known errors | Access / licence | Calibration suitability | Validation suitability | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SRTM v3 (SRTMGL1) | 1″ (~30 m) | 60°N–56°S | EPSG:4326 | EGM96 geoid | Radar phase centre (between ground and canopy top) | Spec 16 m abs / 10 m rel (90 %); ICESat: 0.6 ± 3.5 m open flat; 3.5 ± 8.0 m dense veg; 5.6 ± 15.7 m high relief; voids filled from ASTER/GMTED | Public domain; Earthdata login | Scene-level offset/scale only | No (for sub-decametre claims) | [E2][E21][F9] |
| NASADEM | 1″ | Same | EPSG:4326 | EGM96 | Reprocessed SRTM | Improved voids/geolocation | Public domain | Same | No | [F9] |
| Copernicus GLO-30 | 1″ | Global | EPSG:4326 | EGM2008 | DSM (TanDEM-X) | Forest MAE 5.15 m, built-up 1.61 m [E19]; DEMIX best 1″ DEM [E32] | Free, registered | Same, slightly better | No | [E3][E19][E32][F9] |
| FABDEM | 1″ | Global | EPSG:4326 | EGM2008 | DTM-like (forest/buildings removed statistically) | Residual forest MAE 2.9 m | **CC BY-NC-SA 4.0** | Terrain trend without canopy bias (non-commercial only) | No | [E19][F9] |
| ALOS AW3D30 v4.1 | 1″ | Global | EPSG:4326 | EGM96 | Optical stereo DSM | Least slope-sensitive with SRTM/NASADEM [E20] | Free incl. commercial, account | Same | No | [E3][E20][F9] |
| ASTER GDEM v3 | 1″ | Global | EPSG:4326 | EGM96 | Optical stereo DSM | Worst of 1″ set [E32] | Public, Earthdata | Weak | No | [E32] |
| CartoDEM v3 | 1″ (30 m free) | India | EPSG:4326 | "WGS-84" (UNCLEAR) | Stereo DSM | 8 m LE90; hill-shadow sinks/spikes; ~1 % gaps | Free (Bhuvan) | Indian terrain trend; datum must be clarified | Only for claims ≥ ~16 m | [E23] |
| CartoDSM 2.5 m | 2.5 m | India | EPSG:4326 | "WGS-84" | Stereo DSM | 8 m LE90 | Availability/price UNVERIFIED | Dense but coarse-accuracy anchor | Only for claims ≥ ~16 m | [E23] |

**What a coarse DEM can realistically provide to a high-resolution RGB→DSM system:**
- A **scene-level absolute offset** (the mean elevation of the scene, to within the DEM's bias — which is itself land-cover-dependent: +1.6 m in built-up areas, +3–5 m in forest [E19][E21]).
- A **low-frequency terrain trend** (wavelengths ≳ 60–100 m: valley floors, ridgelines, general slope) — this *is* the "hilly" signal at coarse scale.
- A sanity range (min/max elevation) and a coarse slope field.
- Over India, the only nationwide absolute elevation surfaces available at all [E23][F9].

**What it cannot provide:**
- Any relief at wavelengths below ~2 pixels (~60 m): individual buildings, trees, roads, small terraces — Nyquist limit. Resampling a 30 m grid to 1 m adds no information.
- A **bare-earth** reference: every open 30 m product is a surface model (FABDEM approximates DTM statistically, with 2–3 m residual in forest) [E19].
- **Building-scale vertical detail**: the DEM's own σ (3–16 m) exceeds the height of most low-rise structures [E2][E21].
- A **vertical datum matching the LiDAR reference** without transformation (EGM96 / EGM2008 vs NAVD88 / NAP / LN02 / NZVD2016 / ellipsoidal) [E2][E3][F2–F5][F13].
- **Validation** of a metre-level DSM — comparing against it measures the DEM's error, not the model's (Phase 3 §12).

Spatial-frequency statement (established): a monocular DSM calibrated with a 30 m DEM has its low frequencies constrained by the DEM (to the DEM's accuracy) and its high frequencies entirely supplied by the model (constrained by nothing absolute). Any accuracy claim at building scale therefore cannot be inherited from the DEM.

---

## 9. GCP Calibration Data

**Evidence available (Phase 3 §10 unchanged: no paper calibrates monocular RS depth with GCPs).** What research does support:

| Topic | What is established | Source |
| --- | --- | --- |
| Checkpoint accuracy | Checkpoints should be ≥ 2× more accurate than the product (Ed. 2 relaxed the historical 3×); checkpoint error should be propagated into product accuracy when comparable | [F7] |
| Count | ≥ 30 checkpoints for a fully compliant *assessment*; fewer must be reported with a special statement; land-cover stratified | [F7] |
| Distribution | "Well distributed"; no quantitative methodology exists as of 2023 | [F7] |
| Independence | Checkpoints "not used in processing or calibrating the product" | [F7] |
| Blunders | > 3× target RMSE → investigate; removal must be reported | [F7] |
| Sparse anchors for monocular depth | 150 sparse metric points sufficed for global scale-shift alignment indoors; dense residual remains | [E9] |
| GCP types realistically available over India | Surveyed GNSS points (not open); ICESat-2 ATL08/ATL03 terrain segments (~0.7 m RMSE flat → ~4 m alpine, ellipsoidal) [F6]; GEDI L2A footprints (2–7 m) [F6]; CartoDEM/CartoDSM samples (8 m LE90) [E23]; OpenStreetMap `building:levels` × floor height as weak height proxy (used in GBH for Guangzhou at 3 m/floor [E4]) | [F6][E23][E4] |

**Calibration formulations and what evidence exists for each** (analysis, not selection):

| Formulation | Parameters | Minimum anchors | Evidence status | Known failure |
| --- | --- | --- | --- | --- |
| One-point (offset only, scale assumed) | 1 | 1 | UNKNOWN / NOT FOUND for RS; only meaningful if scale is otherwise fixed (e.g. from GSD or DEM) | Any anchor error becomes global bias |
| One-point scale (shift fixed, e.g. water = 0 above local ground) | 1 | 1 | Not found | Needs a known-height object |
| Two-point (scale + shift) | 2 | 2 | Algebraic minimum; not evaluated in RS literature found | Zero redundancy; no error estimate |
| Multi-point affine LS | 2 (or 3 with tilt) | ≥ 3; Wofk used 150 | Mature in camera frame [E9] | Residual spatially varying; sensitive to clustered anchors |
| Robust fit (RANSAC / Huber / Theil–Sen) | 2–3 | ≥ 3 + redundancy | Standard practice [E9] | Needs ≥ 50 % inliers |
| Regression on covariates (land cover, slope) | > 3 | Tens–hundreds | FABDEM-style for DEMs [E19]; not for monocular depth | Overfits with few anchors |
| Dense low-res prior fusion (DEM as prior) | Per-pixel | Dense coarse grid | Demonstrated indoors with low-res depth priors [E27]; not with DEMs | Untested on RS |

**Calibration GCPs vs independent validation GCPs.** The ASPRS rule is unambiguous: any point used to fit scale/offset is *control*, and product accuracy must be assessed on *different* points [F7]. For sparse spaceborne-lidar tracks this implies a split *by track or by region*, since adjacent segments along one ICESat-2 track are strongly correlated (spacing 20–100 m). The number of anchors withheld for validation should itself meet the ≥ 30 recommendation per land-cover class, or be reported as a reduced assessment.

Open questions for later phases (documented, not decided): minimum anchor count vs residual on terrain-scale scenes; effect of anchor clustering; robustness to ATL08 outliers in canopy; whether the DEM's low-frequency trend and sparse footprints are complementary or redundant.

---

## 10. Train/Validation/Test Splitting

| Split Strategy | Advantage | Main Risk | Scientific Strength | Suitable Use |
| --- | --- | --- | --- | --- |
| Random image split | Simple; large test set | Adjacent/overlapping images share geography → leakage | Weak (Roberts 2017; Ploton 2020; Kattenborn 2022: up to 28 % inflation) [F11][E22] | Debugging only |
| Random tile split within a scene | Same | Same, worse (tiles abut); DFC19 2.1 m vs 5.5–9.3 m [E4][E11] | Weak | Never for headline numbers |
| Scene split (whole strips/tiles held out) | Removes tile adjacency | Scenes within one city share morphology/sensor | Moderate | Model selection |
| Geographic block split (spatial blocks with buffer) | Controls autocorrelation range explicitly | Choosing block size; fewer effective samples | Strong (block CV) [F11] | Validation |
| City split | Clear, interpretable | Cities in same country share style | Strong; GBH/GBA practice [E4][E12] | Test |
| Region / country split | Tests morphology + sensor + vegetation shift | Few regions available | Strongest for generalization claims | External test |
| Terrain split (train urban → test hilly) | Isolates terrain generalization | Confounded with geography | Diagnostic | Cross-terrain benchmark |
| Sensor / GSD split | Tests sensor transfer (e.g. NAIP → LISS-IV / Cartosat) | Confounded with region | Diagnostic | External test |
| Cross-dataset (train GAMUS → test swisstopo) | Independent truth pipeline; independent datum | Datum/definition mismatch (nDSM vs DSM) must be handled | Strong if targets aligned | External test |
| Temporal split (same area, later imagery) | Tests robustness to acquisition changes | Reference drift (new buildings) | Moderate | Stability check |

**Why nearby tiles inflate results.** Spatial autocorrelation: neighbouring tiles share illumination, sensor, season, building style, vegetation species and even the same objects at tile borders; a model memorising local appearance→height mappings scores well on neighbours and fails elsewhere. The remote-sensing CNN literature quantifies inflation up to 28 % [E22] and ecology shows over-optimism that flips conclusions [F11]. The DFC19 case is the domain-specific demonstration: identical cities, random 256-px patches → RMSE 2.12 m; official spatially separated test tiles → 5.46–9.26 m [E4][E11]. Buffers around test blocks larger than the autocorrelation range (empirically ≥ a few hundred metres in urban imagery; longer for terrain) are the standard mitigation [F11].

---

## 11. Cross-Terrain Benchmark

| Terrain | Characteristics | Main Challenge | Reference Data (found) | Metrics | Special Analysis |
| --- | --- | --- | --- | --- | --- |
| Urban | Dense buildings, tall structures, shadows, long-tailed heights (57 % of pixels < 1 m) [E4] | Tall-object underestimation; facade/parallax if off-nadir; shadow confusion | GAMUS (nDSM), DFC19, GeoNRW, AHN4, swisstopo (Zürich/Geneva), LINZ (Auckland/Christchurch), 3DEP+NAIP | RMSE/MAE all; **building-only** and **building-instance** RMSE [E4]; F1-HE [E28]; height-bin errors | Error vs building height bin; edge sharpness; high-rise recall |
| Sparse (rural / low-density / bare) | Few objects, low texture, subtle relief | Model has few cues; ground-pixel dominance makes metrics look good; scale ambiguity strongest | GeoNRW-Rural, AHN4 rural, LINZ rural, 3DEP+NAIP agricultural, swisstopo plateau | Robust stats (median, NMAD, LE95) [E32]; bias | Low-relief bias check; texture-poor patches |
| Hilly / mountainous | Slopes > 15–30°, relief shading, valleys | Slope-dependent error (SRTM 5.6 ± 15.7 m high relief [E21]); horizontal shift → vertical error; DEM anchors least reliable | **swisstopo (Alps)** [F2], **LINZ (Southern Alps, West Coast)** [F3], 3DEP+NAIP (Appalachia/Rockies), UseGeo | RMSE stratified by slope class; slope RMSE; aspect-binned bias (Nuth–Kääb plot) [F8] | Co-registration residual; error vs slope curve |
| Forested | Canopy occludes ground; first-surface reference; seasonal change; reference DEMs biased +3–5 m | Canopy height vs DSM confusion; leaf-on/off mismatch; ATL08 error 2.35 m in forest [F6] | **Open-Canopy** (canopy height) [F14], NEON [E14], swisstopo & LINZ forest tiles, 3DEP+NAIP forest, Copernicus/FABDEM difference as canopy proxy | Canopy-height MAE; RMSE on forest mask; VVA reported separately (ASPRS) [F7] | Leaf-on vs leaf-off; canopy vs ground separation |

Note: no found dataset provides all four classes in India; all four exist in the US (3DEP+NAIP), Switzerland and New Zealand [F2][F3][F5]. Terrain class assignment should come from an independent layer (land cover + slope from the reference DTM), not from the model.

---

## 12. Metric Analysis

Let eᵢ = ẑᵢ − zᵢ over N valid pixels (or checkpoints).

| Metric | Definition | Interpretation | Strengths | Weaknesses | Outlier sens. | Bias sens. | Scale sens. | Suitability for DSM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMSE | √(Σeᵢ²/N) | Typical error incl. bias; ASPRS primary statistic [F7] | Standard; comparable; 1.96·RMSE ≈ 95 % if Gaussian | Dominated by large errors; hides distribution shape; ground pixels dominate [E28] | High | Yes (includes bias) | Yes | Required by PS; report with ME and NMAD |
| MAE | Σ|eᵢ|/N | Median-like typical error | Robust-ish; interpretable | Still ground-dominated; hides sign | Medium | Partial | Yes | Required by PS |
| ME (bias) | Σeᵢ/N | Systematic offset | Diagnoses datum/scale problems | Zero mean can hide symmetric error | Medium | — | — | Essential companion (ASPRS requires reporting if > 25 % RMSE) [F7] |
| Pearson r / R² | cov(ẑ,z)/(σẑσz) | Linear co-variation | Scale/shift-invariant — good for rDSM | Blind to bias and scale — cannot certify absolute accuracy [E11]; inflated by terrain range | Medium | **No** | **No** | Required by PS but only meaningful with ME/RMSE |
| Spearman ρ | Rank correlation | Ordinal agreement | Robust; right for relative depth | Ignores magnitudes | Low | No | No | rDSM |
| NMAD | 1.4826·median(|eᵢ − median(e)|) | Robust σ | Outlier-resistant; = σ if Gaussian [E32] | Less familiar | Low | No | Yes | DEM standard |
| Median error | median(e) | Robust bias | — | — | Low | — | — | DEM standard |
| LE90 / LE95 | 90th/95th percentile of |e| | Accuracy bound without Gaussian assumption | Used by ISRO (CartoDEM 8 m LE90) [E23] | — | Medium | Yes | Yes | Comparable to ISRO products |
| AbsRel | Σ|eᵢ|/zᵢ /N | Relative error | Depth-community standard | Undefined for z→0 (ground at 0 in nDSM!) | High near zero | Yes | No | Only for absolute elevation with z ≫ 0 |
| δ < 1.25ᵏ | fraction with max(ẑ/z, z/ẑ) < 1.25ᵏ | Accuracy thresholds | Depth standard [E15] | Same zero problem; misleading for nDSM [E28] | Low | Yes | No | Absolute elevation only |
| SILog / scale-invariant RMSE | RMSE of log ratio after removing mean log | Structure-only error | Right for relative outputs | Ignores scale entirely | Medium | No | No | rDSM |
| Building-wise RMSE (RMSE-B) | RMSE of per-building median heights | Object-level accuracy [E4] | Isolates the class of interest | Needs footprints | Medium | Yes | Yes | Urban |
| F1-HE | Precision/recall of pixels > 1 m within tolerance [E28] | Object detection-like height accuracy | Immune to ground dominance | Threshold choice | Low | Partial | Partial | All object classes |
| mIoU-3 | IoU of (class correct ∧ |e| < 1 m) [E10] | Joint semantic-height | Contest-standard | Needs labels | Low | Yes | Yes | Urban benchmarks |
| Slope RMSE / aspect error | RMSE of derived slope; DEMIX criteria [E32] | Terrain-shape fidelity | Directly relevant to "slope analysis" | Amplifies noise | High | No | Yes | Hilly |
| Height-bin RMSE | RMSE per GT height interval | Long-tail diagnosis [E4] | Reveals underestimation of tall objects | Sparse bins | — | — | — | All |

---

## 13. Metric Pitfalls

| Pitfall | Cause | How it distorts results | How researchers handle it | Evidence |
| --- | --- | --- | --- | --- |
| Horizontal misregistration | Orthorectification residuals, different DEMs used for ortho, CRS/geotransform errors, pixel-centre vs corner | Vertical error ≈ Δx·tan(slope) on terrain; arbitrary metres at building edges → inflated RMSE, edge "double walls" | Co-register on stable terrain (Nuth–Kääb) [F8]; report shift; edge-buffer masks | [F8][E36] |
| Resolution mismatch | Prediction 0.3–1 m vs reference 30 m or vice versa | Coarse reference smooths objects → prediction penalised for correct detail; upsampled reference invents smoothness | Aggregate finer to coarser (block mean/max); never compare at finer than the coarser native grid (§16) | [E32] |
| Vertical-datum mismatch | EGM96 vs EGM2008 (≤ 1 m in India) vs ellipsoid (24–99 m) vs national datums | Constant or slowly varying bias masquerading as model error; correlation unaffected, RMSE destroyed | Transform all to one vertical CRS (PROJ grids); report datum | [F13][E2][E3] |
| Nodata contamination | −32768 / −9999 / 0 sentinels cast to float; voids | Catastrophic RMSE or false zeros ("sea level") | Explicit nodata masks on both rasters; valid-pixel count reported | [E2] |
| Vegetation mismatch | Leaf-on imagery vs leaf-off lidar; canopy growth; first-return vs phase-centre | Systematic forest bias attributed to model | Season matching; forest mask; VVA separate | [F7][E21] |
| Outdated reference | Lidar years before/after imagery | New/demolished buildings appear as errors | Change masks; date metadata; exclude changed footprints | [E36] |
| Interpolation artefacts | TIN pits/spikes, void filling, DTM under buildings | Local spikes dominate RMSE | Pit-free algorithms; median filters documented; robust stats | [F12][E32] |
| Spatial leakage | Random splits | Up to 28 % optimistic; 2.1 vs 5.5–9.3 m | Block/city/region splits with buffers | [E22][F11][E4][E11] |
| Scene-selection bias | Reporting best tiles; cloud-free only; flat only | Non-representative accuracy | Pre-registered test regions; all-tile reporting; per-terrain tables | [F11] |
| Height-distribution imbalance | 57 % pixels < 1 m | All-zero prediction scores well on MAE/RMSE/δ | Object-focused metrics (F1-HE, RMSE-B), height-bin errors | [E4][E28] |
| Terrain imbalance | Urban-dominated data | Aggregate hides hilly/forest failure | Stratified reporting; equal-weight terrain aggregate; DEMIX-style ranking | [E32] |
| Correlation as certificate | r invariant to scale/shift | High r with wrong scale | Always pair r with ME and RMSE; report fitted affine (s,t) for relative outputs | [E11] |
| Blunder removal | Silent deletion of large residuals | Optimistic RMSE | ASPRS: investigate > 3×RMSE, report removals | [F7] |
| Checkpoint reuse | Same points for calibration and validation | Zero residual at anchors; false accuracy | Independence rule | [F7] |

---

## 14. Geospatial Alignment

Correct comparison of predicted and reference elevation rasters (standard GDAL/PROJ practice; failure evidence from DEM literature):

1. **CRS**: both rasters must resolve to EPSG codes (horizontal) and an explicit vertical CRS; compound CRSs (e.g. EPSG:2056 + LN02, EPSG:28992 + 5709) should be recorded as such [F2][F4].
2. **Affine geotransform**: origin, pixel size, rotation/skew; predicted output must inherit the input GeoTIFF's transform exactly (no resampling side-effects from model input resizing).
3. **Pixel-centre vs pixel-corner (area vs point)**: a half-pixel convention error is a 0.5-pixel shift — 15 m for SRTM, 0.5 m for a 1 m DSM.
4. **Bounds and dimensions**: compute the intersection extent; crop both to it; check that width/height agree after resampling.
5. **Reprojection**: reproject the *coarser* raster or the reference to the prediction grid only when the reference is at least as fine; otherwise aggregate the prediction (§16). Use area-appropriate resampling: nearest for masks, bilinear/cubic for continuous surfaces, average for aggregation.
6. **Registration**: even with correct metadata, orthorectification residuals leave sub-pixel to multi-pixel offsets; estimate and remove a horizontal shift on stable terrain before differencing (Nuth–Kääb) and *report* the shift [F8].
7. **Nodata**: propagate masks from both rasters; exclude water and image borders if declared.

**Why small horizontal shifts create large vertical errors.** For a surface with slope θ, a horizontal offset Δx produces an apparent vertical error Δz ≈ Δx·tan θ: 1 m at 30° slope → 0.58 m; 1 m at 45° → 1 m; at a vertical building wall the error equals the full building height for every mis-registered edge pixel. Because RMSE squares these edge errors, a 1-pixel shift over a dense city can dominate the whole-tile RMSE while leaving MAE and correlation modest — a signature worth checking (edge-buffered vs unbuffered RMSE). SRTM's slope-dependent error (5.6 ± 15.7 m in high relief) is partly this mechanism [E21][F8].

---

## 15. Vertical Datum

- **Ellipsoidal height (h)**: above WGS84 (or GRS80) ellipsoid; native to GNSS, ICESat-2, GEDI [F6].
- **Geoid undulation (N)**: geoid minus ellipsoid; EGM96 / EGM2008 global models; national geoids (GEOID18 for NAVD88, NLGEO for NAP, CHGeo for Swiss, NZGeoid2016).
- **Orthometric height (H)**: H ≈ h − N; "above mean sea level"; what SRTM (EGM96), Copernicus (EGM2008), AW3D30 (EGM96), NAVD88, NAP, NZVD2016 approximate [E2][E3][F3][F4][F5].
- **Levelling datums** (LN02 Switzerland): historical levelled heights, not strictly orthometric; differ from LHN95/geoid by decimetres [F2].
- **Mean sea level**: local tide-gauge zero; not a global surface.

**Why two valid datasets disagree numerically.** Over India N ranges from −24 m (Leh) to −99 m (Kanyakumari) [F13]; a GeoTIFF whose heights are ellipsoidal will read ~50–100 m *lower* than an EGM96 DEM of the same place, and both are "correct". EGM96 vs EGM2008 differ by ≤ ~1 m at the Indian points computed — small but comparable to target accuracy. CartoDEM/CartoDSM state "WGS-84 (horizontal & vertical)" without saying ellipsoid or geoid [E23] — this must be resolved empirically (e.g., against ICESat-2 with both hypotheses) before use.

**What must be normalised before comparison:** (1) all rasters/anchors to one vertical CRS via documented geoid grids; (2) the geoid *model version* recorded; (3) any levelling-datum offset (LN02→LHN95) applied where sub-metre claims are made; (4) the output GeoTIFF tagged with a vertical CRS (compound CRS or explicit metadata) so "RMSE vs reference" is well-defined [E2][E3][F13].

---

## 16. Resolution Matching

| Practice | Defensible? | Why |
| --- | --- | --- |
| Aggregate the finer raster to the coarser grid by block **mean** (for terrain trend) or block **max** (for first-surface DSM semantics), then compare | Yes — DEMIX aggregates 1–5 m lidar to 1″ before ranking [E32]; HTC-DC resamples lidar to image grid [E4] | Compares equal information content |
| Compare at the finer grid after **upsampling** the coarser reference (bilinear/cubic) | No for accuracy claims at the fine scale | Invents smooth values; penalises correct detail; measures reference smoothness |
| Nearest-neighbour upsampling of a coarse reference | No | Blocky artefacts create step errors at block edges |
| Downsample the prediction with **nearest/decimation** | Weak | Aliasing; drops objects |
| Evaluate at multiple scales (native, 5 m, 30 m) and report each | Yes | Separates high- vs low-frequency accuracy — directly relevant to "what did the DEM contribute" |
| Compare a 1 m prediction to a 30 m DEM at 30 m and report as "accuracy" | Only as **coarse-scale trend** accuracy, explicitly labelled | Says nothing about buildings/trees |
| Resample reference and prediction independently to a third grid | Acceptable if identical method and grid; record it | Consistency |
| Evaluate on checkpoints instead of rasters | Yes (ASPRS) with bilinear sampling of the raster at the checkpoint | Avoids grid issues; needs ≥ 30 points per class [F7] |

Key rule: the reported accuracy scale = the coarser of (prediction posting, reference posting, reference accuracy footprint). A 0.5 m DSM validated against 0.5 m swisstopo can claim 0.5 m-scale accuracy; validated against CartoDSM it can claim only ~decametre-scale trend accuracy [F2][E23].

---

## 17. Preprocessing

Established practices (not a prescription):

| Data | Practice | Rationale / caveat | Evidence |
| --- | --- | --- | --- |
| RGB satellite/aerial | Radiometric scaling to reflectance or 8-bit stretch per sensor; histogram matching across domains used in UDA (SynRS3D RS3DAda) | Domain gap partly radiometric | [E28] |
| RGB | Tiling to 256–1024 px with overlap; padding strategy recorded | Receptive-field truncation at borders | [E4][E5] |
| RGB | Resizing changes effective GSD → height scale; GSD must travel with the tile | MHE accuracy is GSD-dependent | [E4] |
| GeoTIFF | Read CRS, transform, nodata; keep as metadata sidecar through the pipeline; write COG with same transform | Standard GDAL | [F2][F4] |
| DEM (coarse) | Reproject to image CRS with bilinear/cubic; convert vertical datum first; keep void mask | Voids in SRTM filled from other sources | [E2] |
| DSM / nDSM reference | Resample to image grid (or aggregate prediction) with documented method; mask water/borders; date check | §16 | [E4][E32] |
| LiDAR | Classify, TIN first returns (DSM) / ground (DTM), pit-free, rasterise; record density and method | §7 | [F12] |
| Nodata / missing | Explicit masks; never fill with 0 | Zero = valid elevation | [E2] |
| Shadows | No standard fix; semantic masks used as auxiliary task | Documented confound | [E4] |
| Clouds | Cloud masks for satellite scenes; not applicable to aerial | Standard RS QA | — |
| Spatial consistency | Same grid, same extent, same nodata for RGB, reference, land-cover, slope layers | Prerequisite for masking | — |

---

## 18. Ground-Truth Quality

| Imperfection | Effect | How researchers account for it | Evidence |
| --- | --- | --- | --- |
| LiDAR vertical noise (~10 cm QL2) | Floor on achievable RMSE; negligible vs metre-level claims | State reference accuracy; ASPRS: propagate checkpoint error when comparable | [F5][F7] |
| LiDAR horizontal accuracy (±1 m LINZ) | Edge errors in DSM | Edge buffers; co-registration | [F3][F8] |
| Registration between imagery and lidar | Systematic edge errors | Nuth–Kääb shift; IM2ELEVATION "patch adjustment" | [F8][E36] |
| Temporal mismatch (2–8 yr) | New/demolished buildings, tree growth | Change masks; footprint dates; exclude changed areas; report years | [E36][F2] |
| Vegetation change / season | Forest bias | VVA separate; leaf-on/off matching | [F7] |
| Interpolation under buildings / water | Invented terrain in DTM → nDSM error | Water masks; document DTM method | [F12] |
| Missing data / voids | Bias if treated as 0 | Masks; valid-pixel counts | [E2] |
| Reference derived from a model (MVS DSM, GBA outputs) | Reference carries its own systematic error (8 m LE90 CartoDSM) | Report reference error budget; compare LiDAR-supervised vs MVS-supervised results (Christie: ~1 m difference) | [E11][E23] |
| Label-derived truth (floors × 3 m) | Coarse | Flag as proxy | [E4] |
| Reference smoothing (pit-free, median) | Reduces fine detail | Document | [F12] |

Practice: accuracy statements should carry a reference-error term ("RMSE 2.3 m against a reference of 0.1 m RMSEz, co-registered to 0.3 m, acquired 2 years before imagery").

---

## 19. Calibration Experiment Design

Experiments and the scientific question each answers (no system implied):

| Experiment | Condition | Question answered | Data needed | Reporting |
| --- | --- | --- | --- | --- |
| C0 — No calibration | Raw relative output vs truth after *post-hoc* affine fit on the test tile (oracle) | What is the best any scale/offset calibration could achieve? (upper bound on structure quality) | Dense truth | Affine-invariant metrics; fitted (s,t) |
| C1 — Scene-level statistics | Scale from GSD/known object; offset from mean DEM | Does scene-level information alone recover metric scale? | Coarse DEM | RMSE, ME vs C0 oracle |
| C2 — Coarse DEM calibration | Global affine fit to DEM; variants: mean offset only; low-pass trend replacement | How much of the absolute error does a 30 m DEM remove, at which spatial frequencies? | DEM + dense truth | Multi-scale RMSE (native/5 m/30 m); residual vs DEM error |
| C3 — Sparse GCP calibration | N ∈ {1, 2, 3, 5, 10, 30, 100} anchors; distributions {clustered, uniform, edge-only, along one ICESat-2 track} | How many anchors, how distributed, for what residual? Where does the curve saturate? | Anchors (surveyed or ATL08) + independent checkpoints | Residual vs N curves; per-distribution; variance over random draws |
| C4 — Dense reference calibration | Fit to full lidar (oracle) | Upper bound; how much is scale vs structure error? | Dense truth | Decomposition: bias / scale / residual |
| C5 — DEM + GCP combined | C2 + C3 | Are DEM trend and sparse anchors complementary? | Both | Δ over each alone |
| C6 — Terrain dependence | Repeat C2/C3 per terrain class | Does calibration stability hold on hilly/forest? | Stratified truth | Per-terrain residual |
| C7 — Datum sensitivity | Deliberately mis-specified datum | How large is the datum error relative to model error? | DEM, geoid grids | ME shift (expected 24–99 m for ellipsoid mix-up in India [F13]) |
| C8 — Anchor-error sensitivity | Add noise to anchors matching ATL08 (0.7–4 m) / CartoDSM (8 m LE90) | Does anchor quality bound the result? | Anchors | Residual vs anchor σ |

Design rules inherited from standards: anchors and checkpoints disjoint; ≥ 30 checkpoints per land-cover class where possible; repeated random anchor draws for variance; report blunders [F7].

---

## 20. Ablation Study Requirements

Component comparisons that establish causality (factors, not architecture):

| Factor | Ablation | What it isolates |
| --- | --- | --- |
| Base monocular depth | Zero-shot vs domain-fine-tuned backbone; small vs large backbone | Contribution of adaptation vs capacity (Phase 3 GAP-01 evidence [E14][E16]) |
| Scale calibration | None / scene-level / DEM / GCP / dense oracle (§19) | Where absolute error comes from |
| DEM information | No DEM / DEM as offset / DEM as trend / DEM as input channel | Whether the DEM adds information beyond offset |
| GCP information | Counts and distributions | Marginal value of each anchor |
| Terrain prior | With/without slope or DTM prior | Effect on hilly scenes |
| Semantic information | With/without land-cover auxiliary task or masks | Effect on building/tree accuracy [E4] |
| Refinement / post-processing | With/without edge-aware smoothing, pit filling | Whether looks-better = measures-better |
| Input GSD | Same scene at 0.3 / 1 / 3 m | GSD sensitivity (Phase 3 GAP-12) |
| Training geography | Train US-only vs US+EU vs +synthetic | Generalization source |
| Tiling | Tile size / overlap | Border artefacts |

Each ablation must be run on the same spatially disjoint test regions with the same masks and metrics; otherwise differences are not attributable.

---

## 21. Generalization Evaluation

| Axis | Methodology | Credible external validation | Evidence of need |
| --- | --- | --- | --- |
| Cities | Held-out cities (GBH protocol) | Cities with different morphology (e.g. Indian dense low-rise) | LA 3.4 m vs São Paulo 9.3 m [E4] |
| Regions / countries | Train one country, test another (US → CH/NZ/NL) | Independent national LiDAR pipelines and datums | GBA Asia 5.9 vs Oceania 1.5 m [E12] |
| Terrain | Stratified test (§11) | All four classes with LiDAR truth | No benchmark has hilly/forest [E4][E28] |
| Sensor | Aerial-trained → satellite-tested (NAIP → LISS-IV / Cartosat if available) | Same area imaged by two sensors with one reference | Zero-shot sensor shifts [E15][E16] |
| GSD | Resample test imagery to 0.3/1/3 m | Same reference | GSD-dependent RMSE [E4] |
| Image quality | Compression (JPG q), noise, haze | Same reference | PS accepts JPG |
| Acquisition conditions | Sun elevation, season, off-nadir angle | Multi-date imagery over one lidar reference (DFC19 multi-date) | Shadows/parallax [E11] |
| Georef vs non-georef modes | Same scene as GeoTIFF and as PNG | Same reference | Mode A/B consistency |

**What constitutes credible external validation:** a region never used in any training/tuning/calibration decision; a reference produced by an independent organisation with its own accuracy report; a stated vertical datum transformation; results reported for all tiles in the region (no selection); and, for India, at minimum an ICESat-2-based sparse assessment with error budget since no LiDAR reference was found [F6].

---

## 22. Uncertainty & Error Analysis

| Tool | What it shows | Standard usage | Evidence |
| --- | --- | --- | --- |
| Residual map (ẑ − z) | Spatial error pattern; datum offsets (uniform), scale errors (proportional to relief), edge errors (double walls) | DEM comparison standard | [F8] |
| Signed error histogram + QQ plot | Non-Gaussianity; justifies robust stats | Höhle & Höhle | [E32] |
| Error vs slope / aspect plots | Co-registration shift signature (sinusoid in aspect) | Nuth–Kääb | [F8] |
| Error vs GT height bins | Long-tail underestimation | HTC-DC, geopose | [E4][E11] |
| Per-class error (building / tree / ground / road / water) | Class-specific failure | GAMUS labels; RMSE-M/NM | [E4][E5] |
| Per-region / per-tile box plots | Variance across scenes; worst cases | Block CV reporting | [F11] |
| Confidence / uncertainty maps | Where the model knows it is unsure | Ensembles, MC-dropout, heteroscedastic heads | [F16] |
| Sparsification curves, AUSE / AURG | Whether uncertainty ranks errors correctly | Ilg 2018; Poggi 2020 | [F16] |
| Calibration reliability of intervals | Whether 90 % intervals contain 90 % of truth | Standard UQ | — |
| Calibration residual maps | Where DEM/GCP calibration disagrees with the prediction | New to RS monocular (Phase 3 GAP-15) | — |

Scientific communication of elevation uncertainty: report accuracy as (ME, RMSE, NMAD, LE95) per terrain class and per object class; attach a reference-error statement; provide residual and uncertainty rasters as GeoTIFFs with the same grid; state the vertical datum on every figure.

---

## 23. DSM-Specific Error Analysis

| Class / condition | Why aggregate metrics hide it | Useful class-specific analysis | Evidence |
| --- | --- | --- | --- |
| Ground | 57 % of pixels; near-zero error inflates apparent accuracy | Report ground-only ME/NMAD separately; exclude from object metrics | [E4][E28] |
| Buildings | Rare tall structures; RMSE-B vs pixel RMSE diverge | RMSE-M, RMSE-B, F1-HE, height-bin curves, edge-buffered RMSE | [E4][E11] |
| Trees / forests | Blurred predictions; reference seasonality; DEM canopy bias | Canopy-height MAE; VVA; leaf-on/off comparison | [E13][F7][E21] |
| Roads | Expected ≈ ground; test for spurious relief | ME on road mask | [E5] |
| Bare terrain | Low texture; scale ambiguity | Low-relief bias; slope RMSE | [E18] |
| Water | Reference noise; specular; should be flat | Mask; report separately; flatness test | [E2] |
| Steep slopes | Δx·tanθ; DEM anchors worst | Error vs slope class; aspect plot | [E21][F8] |
| Image borders / tile seams | Receptive-field truncation | Border-band RMSE vs interior | [E4] |
| High structures (> 30 m) | Long-tail; underestimation | Top-height-bin recall; instance analysis | [E4][E12] |
| Low-relief regions | Everything ≈ offset; correlation collapses | ME/NMAD; do not report r alone | [E11] |
| Bridges / overpasses | nDSM vs DSM semantics | Manual inspection | Phase 1 §7 |

---

## 24. Visualization Validation

| Dimension | Aspect | Objectively measurable? | How (methodology, not design) |
| --- | --- | --- | --- |
| **Geometric quality** | Elevation fidelity of the rendered mesh vs the DSM raster | Yes | Sample mesh heights at raster centres; RMSE mesh-vs-DSM; MARTINI-type tolerance reported [E25] |
| | Terrain-shape fidelity | Yes | Slope/aspect of mesh vs reference; hillshade difference |
| | Structural height readout | Yes | Pick-tool height vs reference at ≥ 30 checkpoints (ASPRS-style) [F7] |
| | Slope readout | Yes | Slope from mesh vs `gdaldem slope` of reference |
| | Vertical exaggeration disclosure | Yes (binary) | Exaggeration factor visible and defaulting to 1.0 for measurements |
| **Texture / projection** | Texture alignment | Yes | Reproject known ground features (road intersections, building corners) from mesh back to image; pixel residuals |
| | Warping / stretching | Partly | Texel-density map; count of triangles with stretch ratio > threshold |
| | Seams / holes | Yes | Automated detection of gaps/nodata triangles; count per scene |
| | Projection accuracy (PS term) | Yes | Same as alignment; report mean/95 % pixel residual |
| **Navigation** | First-person navigation | Partly | Collision/clipping incidents per minute; camera-terrain penetration tests; task completion time |
| | Aerial navigation | Partly | Task-based tests (reach viewpoint, measure height) |
| | Responsiveness | Yes | Frame time percentiles (e.g. p50/p95), input latency |
| | Camera behaviour | Partly | Scripted flight path repeatability |
| **Software quality** | Stability | Yes | Crash-free sessions over N runs and N input sizes/types |
| | Memory | Yes | Peak RSS/VRAM vs raster size |
| | Rendering performance | Yes | FPS vs mesh size; LOD behaviour |
| | Loading time | Yes | Time from upload to first frame, by input size |
| | Deployment reliability | Yes | Clean-machine installs (§25) |
| **UI intuitiveness** | Perceived usability | Instrument-based | SUS (10 items, 0–100) with ≥ 5–10 naive users; task success rates [F16] |

Independence principle (Phase 1 §9, Phase 3 §23): visualization scores must be reported *alongside* DSM accuracy, never as a proxy for it; a mesh smoothed for looks should have its mesh-vs-DSM RMSE disclosed.

---

## 25. Standalone Deployment Validation

**Plausible readings of "standalone"** (PS: "deployable as a standalone application"; "successful standalone deployment"): (a) packaged desktop application (installer/executable) including inference and viewer; (b) local/offline web application served from the same machine; (c) browser-only bundle consuming pre-computed outputs; (d) game-engine build (Unity/Unreal) with pre-processed terrain; (e) containerised local service. Prior art shows (b)/(c) via Qgis2threejs static export and (d) via engine builds [E25]; none bundles GPU inference with a viewer (Phase 3 GAP-16).

**Acceptance / validation criteria that are objective regardless of mechanism:**

| Criterion | Test |
| --- | --- |
| Clean-machine install | Fresh OS image without developer tooling; documented prerequisites only |
| Offline operation | Network disabled after install; full pipeline runs on local files (DEM tiles pre-packaged or explicitly requested) |
| Hardware envelope | Runs on stated minimum (CPU-only fallback tested; GPU optional); memory ceiling documented |
| Model packaging | Weights bundled or fetched with checksum; licence compatible with distribution (DA-V2 Base/Large CC-BY-NC [E26]) |
| Data packaging | Sample inputs included; no dependency on external tile servers for the demo |
| Input coverage | PNG, JPG, GeoTIFF incl. edge cases (no CRS, rotated transform, nodata, > 10k px) |
| Determinism | Same input → same output hash across runs |
| Failure behaviour | Malformed metadata → explicit error, not silent default (Phase 1 §11) |
| Startup and processing time | Recorded per input size |
| Crash-free duration | N-minute navigation session across scene sizes |
| Reproducible build | Build script + versions produce identical artefact |

---

## 26. Benchmark Structures

| Structure | Scientific purpose | Research question | Data requirement | Advantages | Limitations | Leakage risk | Computational burden | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. Geographic hold-out (block CV within a country) | Honest in-country accuracy | How accurate on unseen areas of a known domain? | One national RGB+DSM source (e.g. 3DEP+NAIP or swisstopo) split into buffered blocks | Large sample; controls autocorrelation | Same sensor/style | Low if buffers ≥ autocorrelation range | Medium | Baseline accuracy |
| B. Cross-terrain | PS's four-class stability | Does accuracy hold across urban/sparse/hilly/forested? | Stratified test tiles per class with LiDAR DSM (CH/NZ/US) | Directly answers PS | Class assignment subjectivity; unequal areas | Medium (classes co-located) | Medium | Per-class tables; equal-weight aggregate |
| C. Cross-region / cross-country | Morphology and pipeline independence | Does it transfer to a new country with an independent reference? | Train A, test B (e.g. US → CH, NZ, NL) | Strongest generalization evidence | Few regions; datum conversions | Very low | Medium | Degradation ratio vs A |
| D. External dataset | Independence of truth pipeline | Does it survive a different nDSM/DSM definition and sensor? | Public benchmark with fixed test set (DFC19 Track 1 official) | Comparable to literature | Target definition differences | Low | Low | Position vs published numbers [E11] |
| E. Calibration-vs-generalization | Separate scale error from structure error | How much accuracy is calibration-limited vs model-limited? | Same tiles under C0…C5 conditions (§19) | Attribution | Needs anchors + dense truth | Low if anchors ≠ checkpoints | High (many conditions) | Error decomposition |
| F. India sparse-reference assessment | Target-domain evidence without LiDAR | What can be said over India at all? | Open Indian imagery (LISS-IV) or supplied Cartosat; ICESat-2 ATL08 checkpoints; CartoDEM trend | Only India-specific evidence available | Reference error 0.7–8 m; sparse; coarse imagery if LISS-IV | Low | Low | Bounded statements only |
| G. Mode-A relative benchmark | rDSM structure quality | How good is structure independent of scale? | Any dense truth; affine alignment | Fair for non-georeferenced path | Cannot claim metres | Low | Low | Affine-invariant metrics |
| H. Temporal / multi-date | Acquisition robustness | Stability across dates/sun angles | Multi-date imagery over one reference (DFC19) | Isolates acquisition effects | Reference drift | Low | Low | Variance across dates |

---

## 27. Reproducibility Requirements

Record for every reported number:

- **Dataset**: name, version/release date, download URL, file list hash, licence; tile IDs used per split.
- **Reference generation**: lidar survey date, density, classification method, DSM/DTM algorithm (first-return TIN / pit-free / max), rasterisation resolution, vertical datum, horizontal CRS, accuracy report.
- **Preprocessing**: radiometric scaling, tiling size/overlap, resizing (effective GSD), nodata handling, masks (water, border, change).
- **Geospatial**: horizontal EPSG, vertical CRS + geoid grid version, geotransform, resampling method, co-registration shift applied.
- **Splits**: exact block/city/region definitions, buffer widths, tile counts per split and per terrain class.
- **Model**: architecture and checkpoint hash, licence, training data list, fine-tuning data, hyper-parameters, random seeds, epochs.
- **Inference**: input size, tiling/blending, precision (FP16/32), hardware (GPU/CPU model, VRAM), software versions (framework, GDAL/PROJ), runtime per scene.
- **Calibration**: method, anchor source and count, anchor IDs (so checkpoints can be verified disjoint), fitted parameters, residuals, blunders removed.
- **Evaluation**: metric definitions and code version, valid-pixel counts, masks, aggregation scale, per-class tables, reference-error statement.
- **Visualization**: mesh tolerance, exaggeration, engine version, machine spec for performance numbers.
- **Environment**: OS, container/lockfile, build hash of the standalone artefact.

---

## 28. Data Access & Licensing

| Dataset | Public | Registration | Research-only | Commercial | Redistribution | Size (order) | API | Stability | Metadata | Practical risk flag | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAMUS (HF) | Yes | No | No (CC-BY-4.0) | Allowed | Allowed w/ attribution | ~26k files (GB-scale, UNVERIFIED) | HF hub | Good | **Poor (no georef)** | Non-georeferenced | [E6] |
| DFC2019 | Yes | IEEE DataPort account | Contest terms | UNVERIFIED | Restricted | Tens of GB | Download | Good | Good | Terms review needed | [E10] |
| SynRS3D | Yes | No | Open | UNVERIFIED | Allowed | Large | GitHub/HF/Zenodo | Good | Good | Synthetic only | [E28] |
| GeoNRW | Yes | IEEE DataPort | DL-DE-BY-2.0 | Allowed | Allowed | Tens of GB | Download | Good | Good | Datum unverified | [E29] |
| DFC2018 Houston | By request | E-mail | Acknowledgement | UNVERIFIED | Restricted | GBs | Manual | Medium | Good | Manual gate | [F15] |
| swisstopo (DSM + ortho) | Yes | No | No | Allowed | Allowed w/ attribution | TBs nationally; per-tile download | STAC/CSV lists | Excellent | Excellent | LN02 datum conversion | [F2] |
| LINZ (DSM/DEM + imagery) | Yes | Free LDS account for exports | No | Allowed (CC BY 4.0) | Allowed | TBs; regional | LDS API / OpenTopography | Excellent | Excellent | Regional patchwork | [F3] |
| AHN4 (+ PDOK imagery) | Yes | No | No (CC0) | Allowed | Allowed | TBs; COG tiles | WCS/COG | Excellent | Excellent | Imagery licence unverified | [F4] |
| USGS 3DEP EPT + NAIP | Yes | No (AWS anonymous; GEE account for NAIP) | No | Allowed | Allowed | PB-scale; subset by AOI | AWS S3 / GEE | Excellent | Good | DSM must be built; NAIP leaf-on | [F5] |
| Open-Canopy | Yes | No | CC BY 4.0 | Allowed | Allowed | Large | Download | Good | Good | Canopy only | [F14] |
| SRTM / NASADEM | Yes | Earthdata login | No | Allowed | Allowed | 25 MB/tile | LP DAAC / OpenTopography / GEE | Excellent | Good | — | [F9] |
| Copernicus GLO-30 | Yes | CDSE registration (or AWS anon) | No | Allowed | Per licence | 1°×1° tiles | CDSE / AWS | Excellent | Good | Licence text to cite | [F9] |
| FABDEM | Yes | No | **Non-commercial (CC BY-NC-SA)** | **No** | SA | Global | Bristol / GEE community | Good | Good | Non-commercial | [F9] |
| AW3D30 | Yes | JAXA account | No | Allowed | Per terms | Tiles | JAXA / GEE | Excellent | Good | — | [F9] |
| CartoDEM 30 m | Yes | Bhuvan login | UNVERIFIED | UNVERIFIED | UNVERIFIED | Tiles | Manual | Medium | Datum ambiguous | Datum | [E23] |
| CartoDSM 2.5 m | UNVERIFIED | Bhoonidhi | UNVERIFIED | Likely priced (< 5 m) | UNVERIFIED | — | — | UNKNOWN | — | Access unknown | [E23][F10] |
| ICESat-2 ATL08 / GEDI | Yes | Earthdata login | No | Allowed | Allowed | GB per region | NSIDC / earthaccess | Excellent | Excellent | Sparse | [F6] |
| LISS-IV | Yes | Bhoonidhi login | Open (≥ 5 m) | Per policy | UNVERIFIED | Scenes | Bhoonidhi API | Medium | Good | 5.8 m GSD | [F10] |
| Cartosat-2S / -3 | No (priced for NGEs) | NSIL | — | Priced | Restricted | — | — | UNKNOWN | — | **Cost / availability** | [F10] |
| Depth backbones | Yes | No | DA-V2 Base/Large **CC-BY-NC-4.0**; Small Apache-2.0 | Restricted for B/L | Per licence | GB | HF | Good | — | Licence | [E26] |
| HTC-DC Net code/weights | Yes | No | **No licence** | Undefined | Undefined | — | GitHub | Medium | — | No licence | [E4] |

---

## 29. Data Availability Risk

| Source | Risk | Basis |
| --- | --- | --- |
| swisstopo swissSURFACE3D + SWISSIMAGE | **LOW** | Open, commercial-OK, excellent metadata, stable federal service; only datum conversion work [F2] |
| LINZ LiDAR + imagery | **LOW** | CC BY 4.0, stable, OpenTopography mirror [F3] |
| AHN4 | **LOW** | CC0, COG, stable [F4] |
| USGS 3DEP + NAIP | **LOW–MEDIUM** | Public domain, stable; DSM generation effort and large volumes [F5] |
| SRTM / NASADEM / Copernicus / AW3D30 | **LOW** | Global, free, multiple mirrors [F9] |
| ICESat-2 / GEDI | **LOW** | Public, stable; sparse by nature [F6] |
| GAMUS | **MEDIUM** | Open and stable, but non-georeferenced and terrain-limited — cannot evidence key PS claims [E6] |
| DFC2019 / DFC2023 / DFC2018 | **MEDIUM** | Registration/terms; official test truth partly withheld; contest terms may restrict use [E10][E34][F15] |
| GeoNRW | **MEDIUM** | Open, but vertical datum and DSM semantics need confirmation [E29] |
| FABDEM | **MEDIUM** | Non-commercial licence conflicts with a deployable product [F9] |
| CartoDEM 30 m | **MEDIUM** | Free but datum ambiguity and 8 m LE90 [E23] |
| CartoDSM 2.5 m / Cartosat-2S / Cartosat-3 imagery | **HIGH** | Priced for non-government entities; team access UNKNOWN; no open Indian HR RGB–LiDAR pairs exist [F10] |
| Organiser-supplied evaluation data | **HIGH (unknown)** | Repository contains nothing; sensor/GSD/format UNKNOWN [F1] |
| Depth backbone licences | **MEDIUM** | CC-BY-NC for larger DA-V2 variants; HTC-DC Net unlicensed [E26][E4] |

---

## 30. Scientific Validation Blueprint

Conceptual chain (methodology, not architecture):

```
INPUT IMAGE
  record: format, CRS?, geotransform?, GSD, sensor, date, off-nadir, nodata
        ↓
PREDICTION
  record: backbone/checkpoint hash, tiling, effective GSD, raw output units (relative / height / elevation)
        ↓
CALIBRATION / ALIGNMENT
  Mode A: post-hoc affine fit to reference on the test tile (oracle, reported as such)
  Mode B: DEM and/or anchors — anchor IDs logged; checkpoints excluded; fitted parameters + residuals logged
        ↓
GEOSPATIAL NORMALIZATION
  horizontal: reproject/resample to a common grid (record method); co-register on stable terrain (record shift)
  vertical:   transform reference, DEM, anchors and prediction to ONE vertical CRS (record geoid grid)
        ↓
REFERENCE MATCHING
  choose reference type = claimed target (nDSM ↔ nDSM; DSM ↔ LiDAR DSM); aggregate to the coarser posting;
  record reference date, accuracy report, generation method
        ↓
VALID-PIXEL MASKING
  nodata (both), water, image borders, changed footprints, clouds; report valid-pixel counts
        ↓
METRIC COMPUTATION
  ME, RMSE, MAE, NMAD, LE95, Pearson r (with ME), Spearman ρ; object metrics (RMSE-B, F1-HE);
  for Mode A: affine-invariant metrics only
        ↓
TERRAIN-SPECIFIC ANALYSIS
  urban / sparse / hilly / forested from independent land-cover + slope layers; per-class and per-height-bin tables;
  equal-weight aggregate alongside area-weighted
        ↓
ERROR MAPS
  residual GeoTIFF, uncertainty GeoTIFF, error-vs-slope/aspect plots, histograms; worst-tile gallery
        ↓
GENERALIZATION TESTING
  held-out blocks → held-out cities → other countries → external fixed benchmark → India sparse assessment;
  degradation ratios reported
```

---

## 31. Master Data Matrix

| Dataset | RGB | DSM | DTM | LiDAR | DEM | GCP | GSD | Geography | Terrain | CRS | Vertical Reference | License | Access | Potential Role | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAMUS | ✓ | ✗ (nDSM) | ✗ | derived | ✗ | ✗ | 0.33 m | 3 US cities | Urban | none | n/a | CC-BY-4.0 | HF | Train/val nDSM | MEDIUM |
| DFC2019/US3D | ✓ | T3 DSM / T1 nDSM | derivable | ✓ | ✗ | RPC | 0.3–1.3 m | JAX, OMA (+ATL, ARG) | Urban | UTM/RPC | US | DataPort/MIT | Reg. | Train; external fixed test | MEDIUM |
| SynRS3D | ✓ syn | ✗ (nDSM) | ✗ | syn | ✗ | ✗ | 0.09–1 m | Synthetic | Mixed | none | n/a | Open | Open | Pre-train | LOW |
| GeoNRW | ✓ | ✓ 1 m first-return | ✗ | ✓ | ✗ | ✗ | 1 m | NRW, DE | Urban + forest | ✓ | DE (unverified) | DL-DE-BY-2.0 | DataPort | Train/val DSM | MEDIUM |
| DFC2018 Houston | ✓ 5 cm | ✓ 0.5 m | derivable | ✓ | ✗ | ✗ | 0.05/0.5 m | Houston | Urban | ✓ | US | Terms | E-mail | Val DSM (small) | MEDIUM |
| swisstopo | ✓ | ✓ 0.5 m | ✓ (swissALTI3D) | ✓ | ✗ | derivable | 0.5 m | Switzerland | **All four** | LV95 | LN02 | OGD | Open | Train/val/test/external | LOW |
| LINZ | ✓ | ✓ 1 m | ✓ 1 m | ✓ | ✗ | derivable | 1 m | New Zealand | **All four** | NZTM2000 | NZVD2016 | CC BY 4.0 | Open | Val/test/external | LOW |
| AHN4 | ✓ (PDOK) | ✓ 0.5 m | ✓ 0.5 m | ✓ | ✗ | derivable | 0.5 m | Netherlands | Urban/sparse/forest, flat | EPSG:28992 | NAP | CC0 | Open | Train/val/external | LOW |
| USGS 3DEP + NAIP | ✓ | build from EPT | ✓ 1 m | ✓ | ✗ | derivable | 1 m | CONUS | **All four** | NAD83 UTM | NAVD88 | Public domain | Open | Train/val/test (US) | LOW–MED |
| Open-Canopy | ✓ 1.5 m | ✗ (CHM) | ✗ | ✓ | ✗ | ✗ | 1.5 m | France | Forest | ✓ | n/a | CC BY 4.0 | Open | Forest val | LOW |
| NEON/EarthView | ✓ 0.1 m | ✗ (CHM) | (unverified) | ✓ | ✗ | ✗ | 0.1/1 m | USA | Forest | ✓ | US | Open | Open | Forest val | LOW |
| SRTM/NASADEM | ✗ | 30 m surface | ✗ | ✗ | ✓ | ✗ | 30 m | Global | All | 4326 | EGM96 | PD | Earthdata | Calibration | LOW |
| Copernicus GLO-30 | ✗ | 30 m DSM | ✗ | ✗ | ✓ | ✗ | 30 m | Global | All | 4326 | EGM2008 | Free reg. | CDSE/AWS | Calibration | LOW |
| FABDEM | ✗ | ✗ | ≈ 30 m | ✗ | ✓ | ✗ | 30 m | Global | All | 4326 | EGM2008 | CC BY-NC-SA | Open | Calibration (non-comm.) | MEDIUM |
| AW3D30 | ✗ | 30 m DSM | ✗ | ✗ | ✓ | ✗ | 30 m | Global | All | 4326 | EGM96 | Free w/ account | JAXA | Calibration | LOW |
| CartoDEM 30 m | ✗ | 30 m stereo | ✗ | ✗ | ✓ | ✗ | 30 m | India | All | 4326 | "WGS-84" | Free | Bhuvan | India calibration | MEDIUM |
| CartoDSM 2.5 m | ✗ | 2.5 m stereo | ✗ | ✗ | ✓ | ✗ | 2.5 m | India | All | 4326 | "WGS-84" | Priced? | NSIL | India coarse reference | HIGH |
| ICESat-2 ATL08 | ✗ | segments | terrain | spaceborne | ✗ | **✓ sparse** | 20–100 m | Global | All | WGS84 | Ellipsoidal (+geoid fields) | PD | Earthdata | Anchors / checkpoints (India) | LOW |
| GEDI L2A | ✗ | footprints | ground | spaceborne | ✗ | ✓ sparse | 25 m | ±51.6° | All | WGS84 | Ellipsoidal | PD | Earthdata | Anchors (coarser) | LOW |
| LISS-IV | ✓ 5.8 m | ✗ | ✗ | ✗ | ✗ | ✗ | 5.8 m | India | All | ✓ | — | Open | Bhoonidhi | Indian sensor input test | MEDIUM |
| Cartosat-2S/3 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 0.65–2 / 0.25–1.13 m | India | All | ✓ | — | Priced | NSIL | Ideal Indian input if obtained | HIGH |

---

## 32. Master Validation Matrix

| SIH Requirement | Evidence Needed | Reference Data | Metric / Measure | Validation Procedure | Failure Condition |
| --- | --- | --- | --- | --- | --- |
| DSM accuracy (Mode B) | Metre-level absolute accuracy on spatially disjoint regions | LiDAR DSM (CH/NZ/NL/US), one vertical CRS | ME, RMSE, MAE, NMAD, LE95 | Blueprint §30; co-registration; masks | RMSE not better than the coarse DEM's own error; ME > 25 % RMSE unexplained |
| rDSM structure (Mode A) | Structural correctness independent of scale | Any dense DSM/nDSM | Affine-invariant metrics; Spearman | Post-hoc affine fit reported | Claimed metres without anchors |
| RMSE | Reported with ME and NMAD | LiDAR | RMSE | ≥ 30 checkpoints/class or dense raster | Reported alone |
| MAE | Alongside RMSE | LiDAR | MAE | Same | Ground-dominated MAE presented as object accuracy |
| Correlation | Paired with bias/scale | LiDAR | Pearson r + ME + fitted scale | Same | r reported as accuracy |
| Urban | Building-level accuracy | GAMUS/DFC19/GeoNRW/AHN/CH/NZ/US urban tiles | RMSE-M, RMSE-B, F1-HE, height bins | Footprint-based aggregation | Tall-building underestimation unreported |
| Sparse | Low-relief bias control | Rural CH/NZ/NL/US tiles | ME, NMAD | Robust stats | Correlation-only reporting |
| Hilly | Slope-stratified accuracy | swisstopo Alps, LINZ, 3DEP mountains | RMSE by slope class; slope RMSE; aspect plot | Co-registration first | Error grows unbounded with slope; uncorrected shift |
| Forested | Canopy vs ground handling | Open-Canopy, NEON, CH/NZ forest tiles | Canopy MAE; VVA | Season noted | Forest bias hidden in aggregate |
| Projection accuracy | Texture–geometry alignment | Known features | Pixel residuals (mean/95 %) | Back-projection of features | Residual > 1–2 texels systematically |
| Visual fidelity | Mesh faithful to DSM | Predicted DSM | Mesh-vs-DSM RMSE; seams/holes count | Sampling | Smoothing hides > stated tolerance |
| Navigation | Usable first-person + aerial | — | Frame-time percentiles; clipping incidents; task times | Scripted + user tasks | Frame drops / penetration |
| UI intuitiveness | Naive users succeed | — | SUS; task success | ≥ 5–10 users | SUS < ~68 (population mean; contextual) |
| Software stability | No crashes across inputs | Input suite | Crash-free runs; memory | Stress tests | Any crash on PS-named formats |
| Standalone deployment | Runs on clean machine offline | — | Install success; offline run | §25 criteria | Requires developer setup or network |

---

## 33. Benchmark Red-Team

| Problem | Risk | How it could distort the conclusion | How literature addresses it |
| --- | --- | --- | --- |
| Spatial leakage | High | 2.1 m vs 5.5–9.3 m on identical data [E4][E11]; up to 28 % inflation [E22] | Block/city/region CV with buffers [F11] |
| Geographic overlap between training sources | Medium | GAMUS (DC/PHL/NYC), DFC19 (JAX), GBA training (many US cities), 3DEP+NAIP — overlapping footprints across "different" datasets create hidden leakage | Footprint intersection checks across all sources; exclusion zones |
| Resolution mismatch | High | Coarse reference penalises correct detail or, upsampled, rewards smoothness | Aggregate to coarser posting; multi-scale reporting [E32] |
| Coarse DEM used as truth | High | Measures DEM error (3–16 m σ) not model error; a model copying the DEM scores "perfectly" | DEM = calibration only; LiDAR = validation [E2][E21] |
| Vertical-datum mismatch | Very high in India | 24–99 m offsets [F13]; or ≤ 1 m EGM96/2008 drift mistaken for model bias | Single vertical CRS via geoid grids; datum on every table [E2][E3] |
| Horizontal registration error | High | Edge errors inflate RMSE; slope-correlated error | Nuth–Kääb co-registration; report shift; edge buffers [F8] |
| Vegetation bias | High | Forest reference (phase centre / leaf-off) vs leaf-on imagery; DEM +3–5 m | VVA separate; season matching; canopy-specific truth [F7][E19][E21] |
| Terrain imbalance | High | Urban-heavy data make aggregate look good; hilly/forest fail unseen | Stratified, equal-weight reporting [E32] |
| Urban bias (US/EU morphology) | High | Indian dense low-rise underestimated (GBA Asia 5.9 m) | Cross-region test; report degradation [E12] |
| Temporal mismatch | Medium | Changed buildings/trees counted as errors | Change masks; date metadata [E36] |
| Reference-data quality | Medium | 8 m LE90 CartoDSM cannot validate metre claims | Reference error budget; 2× rule [F7][E23] |
| Class imbalance (57 % ground) | High | All-zero model competitive on MAE/RMSE/δ [E28] | Object metrics; height-bin errors [E4][E28] |
| Metric manipulation | Medium | Choosing r on high-relief scenes; MAE on flat scenes; excluding "outlier" tiles | Pre-registered metric set; all-tile reporting; blunder rules [F7] |
| Anchor–checkpoint contamination | High | Zero residual at fitted points | Disjoint IDs; track-level split for ICESat-2 [F7] |
| Oracle alignment reported as calibration | High | Post-hoc affine fit on the test tile presented as achieved metric accuracy | Label as oracle upper bound (C0) [E9] |
| Selection of favourable dates/sun angles | Medium | Optimistic shadows/parallax | Multi-date tests [E11] |
| Mode confusion | Medium | Scoring rDSM in metres, or DSM as nDSM | Target–reference matching table (§6) |

---

## 34. Ideal Data Ecosystem

Characteristics only (no selection):
- **High-resolution RGB** at the expected evaluation GSD range (≈ 0.3–1.5 m), nadir or with known off-nadir angle and RPC, dated, with sensor identity.
- **Reliable DSM reference** from airborne LiDAR (≥ 2 pts/m², ~10 cm RMSEz), true first-surface, with accompanying DTM, generation method and accuracy report; posting ≤ imagery GSD.
- **LiDAR point clouds where available**, so DSM/DTM/CHM can be regenerated consistently and dated.
- **Coarse DEMs** (SRTM/Copernicus/AW3D30/CartoDEM) for the same footprints, with vertical datum known.
- **Accurate CRS and geotransform** on every layer; compound CRS with vertical component; consistent pixel-centre convention.
- **Known vertical reference** for every source and geoid grids to convert; datum-resolved Indian references.
- **Diverse terrain** — urban, sparse, hilly, forested — each with LiDAR truth, from more than one country and morphology, including South-Asian-like dense low-rise.
- **Independent test areas** defined before modelling, with buffers, and an external fixed benchmark (DFC19 Track 1).
- **Sparse absolute anchors** (surveyed GCPs or ICESat-2 segments) in test areas, partitioned into control and checkpoints.
- **Temporal alignment** between imagery and reference, or change masks.
- **Stable, redistributable licensing** (public domain / CC BY / OGD) for data and models that end up in a deployable product.
- **Indian imagery** at high resolution with any absolute reference at all — currently the missing piece (Cartosat priced; no open LiDAR) [F10].

---

## 35. Evidence-Backed Constraints for Phase 5

1. **Calibration data must be disjoint from validation data**; anchors used to fit scale/offset cannot be checkpoints; ≥ 30 independent checkpoints per land-cover class is the standard for a full assessment. [F7]
2. **Metric scale must be explicitly demonstrated**, not inferred from a post-hoc fit; oracle alignment (C0) is an upper bound to be labelled as such. [E9][F7]
3. **The prediction target must be named and its reference matched**: nDSM ↔ nDSM truth; absolute DSM ↔ LiDAR DSM in a stated vertical datum. Most available training truth is nDSM. [E4][E5][E28]
4. **DSM cannot be treated as DTM and a coarse DEM cannot be treated as high-resolution truth**; 30 m products are surface-like, biased +1.6 m (built-up) / +3–5 m (forest), σ 3–16 m, and carry no sub-60 m information. [E2][E19][E21]
5. **Vertical datum must be consistent and declared**: India's geoid–ellipsoid separation is −24 to −99 m; EGM96/EGM2008 differ by ≤ 1 m; CartoDEM's datum is ambiguous. [F13][E2][E3][E23]
6. **Geospatial alignment must be controlled**: co-register on stable terrain and report the shift; compare at the coarser posting; mask nodata/water/borders/changed areas. [F8][E32][E36]
7. **Terrain diversity must be tested with LiDAR-grade truth**; the only open sources found with all four classes are Switzerland, New Zealand and the US (3DEP+NAIP), none in India. [F2][F3][F5]
8. **Spatial leakage must be prevented** by blocked/city/region splits with buffers; random patch splits are inadmissible for headline numbers. [E22][F11][E4][E11]
9. **Claims must match the actual ground truth and its error**: reference accuracy must be ≥ 2× the claimed accuracy; CartoDSM (8 m LE90) can only support decametre claims. [F7][E23]
10. **Metrics must be reported as a set** — ME, RMSE, MAE, NMAD, LE95, r with fitted scale — plus object-level metrics (RMSE-B, F1-HE) and height-bin errors, per terrain class; ground dominance and correlation's scale-blindness are documented failure modes. [E4][E28][E11][E32]
11. **Indian evidence will be sparse-reference evidence** unless Cartosat imagery and a dense reference are supplied: ICESat-2 (0.7–4 m) and CartoDEM (8 m LE90) bound what can be claimed over India; sub-5 m ISRO imagery is priced for non-government entities. [F6][E23][F10]
12. **Licences constrain the deployable artefact**: FABDEM (CC BY-NC-SA), DA-V2 Base/Large (CC-BY-NC-4.0), HTC-DC Net (no licence) versus CC0/CC BY/OGD/public-domain data. [F9][E26][E4]
13. **Visualization quality must be validated separately from and alongside DSM accuracy**, with mesh-vs-DSM residuals, projection residuals, performance percentiles, crash-free runs, clean-machine installs and a usability instrument (SUS). [E25][F16]
14. **Reproducibility metadata (dataset versions, splits, datum grids, checkpoints, anchors, seeds, hardware) must be recorded** for every reported number. [F7][E32]
15. **Mode A (rDSM) can only be scored with affine-invariant/rank metrics**; any metre-valued claim requires Mode-B calibration data. [E9][E15]

---

## PHASE 4 COMPLETE

### 15 Most Important Dataset Findings
1. The official SAC repository still contains only a README (re-checked 2026-09-20) — no imagery, reference, protocol or licence. [F1]
2. GAMUS as released is 8,724 non-georeferenced `.h5` tiles of RGB + **nDSM** from three US cities; it cannot support absolute-DSM calibration or non-urban evaluation. [E6]
3. Almost all RGB→height datasets (GAMUS, DFC19 T1, GBH, SynRS3D, DFC2023) provide nDSM, not DSM. [E4][E5][E28][E34]
4. True, datum-referenced LiDAR DSMs with open orthophotos exist from national programmes: swisstopo 0.5 m (LN02) [F2], LINZ 1 m (NZVD2016) [F3], AHN4 0.5 m (NAP) [F4], GeoNRW 1 m [E29]; USGS 3DEP supplies public-domain point clouds from which a DSM must be built (the ready 1 m raster is DTM) [F5].
5. Hilly + forested LiDAR truth is available in Switzerland, New Zealand and the US; none was found for India. [F2][F3][F5]
6. Forest-specific truth: Open-Canopy (87,000 km², 1.5 m, CC BY 4.0) and NEON — canopy height, not DSM. [F14][E14]
7. Indian high-resolution imagery (Cartosat-1/2/2S/3) is priced for non-government entities under Indian Space Policy 2023; LISS-IV (5.8 m) and CartoDEM 30 m are open. [F10]
8. The only Indian absolute references found are CartoDEM/CartoDSM (stereo, 8 m LE90, datum ambiguous) and spaceborne lidar footprints (ICESat-2 ≈ 0.7–4 m, GEDI 2–7 m). [E23][F6]
9. Geoid–ellipsoid separation over India is −24 m (Leh) to −99 m (Kanyakumari); EGM96 vs EGM2008 ≤ 1 m. [F13]
10. Coarse DEM vertical datums differ: SRTM/NASADEM/AW3D30 EGM96; Copernicus/FABDEM EGM2008; national LiDAR NAVD88/NAP/LN02/NZVD2016. [E2][E3][F2–F5]
11. FABDEM is CC BY-NC-SA (non-commercial); Copernicus needs registration; AW3D30 allows commercial use; SRTM/NASADEM public domain. [F9]
12. Depth-backbone licences are mixed (DA-V2 Small Apache-2.0; Base/Large CC-BY-NC-4.0); HTC-DC Net has no licence. [E26][E4]
13. DFC2019 Track 1 is the only fixed external benchmark with published single-view height numbers (5.46 m all / 10.69 m buildings). [E11]
14. LiDAR reference quality standards are explicit: QL2 ≥ 2 pts/m², 10 cm RMSEz; DSM = first-return TIN, pit-free variants. [F5][F12]
15. Hidden geographic overlap across "different" US datasets (GAMUS, DFC19, GBA training, 3DEP+NAIP) is a leakage risk requiring footprint checks. (§33)

### 15 Most Important Benchmark Findings
1. Random patch splits are inadmissible: 2.1 m vs 5.5–9.3 m on identical DFC19 data; up to 28 % inflation in RS CNNs. [E4][E11][E22]
2. Block / city / region splits with buffers are the literature standard. [F11]
3. ASPRS Ed. 2 sets ≥ 30 independent checkpoints, ≥ 2× checkpoint accuracy, 3×RMSE blunder rule, mean-error reporting, and NVA/VVA separation. [F7]
4. No quantitative methodology for checkpoint spatial distribution exists as of 2023. [F7]
5. Robust statistics (median, NMAD, LE95) are required because DEM errors are non-Gaussian. [E32]
6. Co-registration before differencing (Nuth–Kääb) is standard; Δz ≈ Δx·tan θ turns small shifts into large vertical errors. [F8]
7. Comparison must occur at the coarser posting; upsampled coarse references invent smoothness. [E32]
8. Correlation is scale/shift-blind and cannot certify absolute accuracy; report with ME and RMSE. [E11]
9. Ground-pixel dominance (57 % < 1 m) makes MAE/RMSE/δ rate all-zero predictions competitively; object metrics (RMSE-B, F1-HE) are required. [E4][E28]
10. Terrain-stratified evaluation is established for DEMs (DEMIX) and must be applied per class, not aggregated. [E32]
11. A coarse DEM is calibration data, never validation data, for metre-level DSMs. [E2][E19][E21]
12. Calibration experiments must separate scale error from structure error via oracle (C0) vs DEM (C2) vs sparse anchors (C3, varying N and distribution) vs dense (C4). (§19)
13. External validation requires an independent region, independent reference pipeline, declared datum transform and all-tile reporting. (§21)
14. Uncertainty must be evaluated (sparsification / AUSE) and communicated as residual + uncertainty GeoTIFFs with datum stated. [F16]
15. Visualization has objective measures (mesh-vs-DSM RMSE, projection residuals, frame-time percentiles, crash-free runs, clean-machine installs) plus SUS for intuitiveness. [E25][F16]

### 10 Most Important Validation Risks
1. Vertical-datum mix-up (24–99 m in India). [F13]
2. Spatial leakage through random or adjacent-tile splits. [E22][E11]
3. Validating against a coarse DEM and reporting the DEM's error as model accuracy. [E2][E21]
4. Scoring nDSM predictions as DSM or vice versa. [E4][E5]
5. Anchor–checkpoint contamination. [F7]
6. Horizontal misregistration inflating RMSE at building edges and slopes. [F8]
7. Forest bias from leaf-on imagery vs leaf-off/phase-centre references and biased DEMs. [F7][E19][E21]
8. Reporting correlation without bias, or MAE dominated by ground pixels. [E11][E28]
9. Temporal mismatch between imagery and reference. [E36]
10. Oracle affine alignment presented as achieved calibration. [E9]

### 10 Most Important Data Risks
1. No organiser data; evaluation sensor/GSD unknown. [F1]
2. Indian HR imagery priced; no open Indian RGB–LiDAR pairs. [F10]
3. GAMUS non-georeferenced and terrain-limited. [E6]
4. CartoDEM datum ambiguity and 8 m LE90. [E23]
5. Hidden geographic overlap across US datasets. (§33)
6. Licence conflicts (FABDEM NC; DA-V2 B/L NC; HTC-DC unlicensed). [F9][E26][E4]
7. DSM generation burden and date offsets for 3DEP+NAIP. [F5]
8. Regional patchwork and datum conversions for LINZ/swisstopo. [F2][F3]
9. Contest-term restrictions on DFC datasets; withheld official test truth. [E10][F15]
10. Large download volumes (TB-scale national LiDAR) against hackathon timelines. (§28)

### 10 Most Important Experimental Questions
1. How much absolute error does a 30 m DEM remove, and at which spatial frequencies (native / 5 m / 30 m)?
2. How does residual error fall with anchor count N and distribution, and where does it saturate?
3. What is the oracle (dense-fit) upper bound, i.e. how much error is structure vs scale?
4. Does calibration stability hold on hilly and forested tiles?
5. What is the cross-country degradation ratio (US → CH/NZ/NL) with independent references?
6. How large is the GSD sensitivity (0.3 / 1 / 3 m) on the same scene?
7. Does an Indian sensor input (LISS-IV, or Cartosat if supplied) change behaviour, assessed against ICESat-2 checkpoints?
8. How much does leaf-on/leaf-off or reference date change forest error?
9. Does model uncertainty rank errors correctly (AUSE) and does it correlate with terrain class?
10. How much geometric information does mesh simplification remove at the tolerance needed for real-time navigation?

### Evidence-Backed Constraints for Phase 5
The fifteen constraints in Section 35 (calibration/validation independence; demonstrated metric scale; target–reference matching; DSM ≠ DTM ≠ coarse DEM; consistent declared vertical datum; controlled alignment and resolution; LiDAR-grade multi-terrain testing; no spatial leakage; claims bounded by reference accuracy; metric sets with object-level and per-terrain reporting; sparse-reference limits over India; licence compatibility; separate visualization validation; reproducibility metadata; affine-invariant scoring for Mode A).

### What Phase 5 Must Resolve Before Solution Architecture
1. Which **prediction target** the georeferenced path will actually claim (nDSM + terrain vs direct absolute DSM), because it dictates which datasets are training-eligible and which references are valid.
2. Which **vertical datum** the output will be declared in, and the geoid-grid workflow that every reference and anchor will be transformed to.
3. Whether **Indian imagery** (Cartosat) and any dense Indian reference will be available from the organiser; if not, the India evidence plan reduces to ICESat-2/CartoDEM-bounded statements.
4. Which **multi-terrain LiDAR source(s)** (CH / NZ / US / NL) will be adopted for training vs held-out testing, with footprint-overlap checks against GAMUS/DFC19/GBA.
5. The **calibration-data policy**: which coarse DEM(s) and which sparse-anchor sources are permitted at inference, and how anchors are partitioned from checkpoints.
6. The **licence policy** for data and model weights that will ship in the standalone artefact.
7. The **pre-registered metric set, split definitions and terrain-class rules**, fixed before any model is trained.

READY FOR PHASE 5 — SOLUTION IDEATION
