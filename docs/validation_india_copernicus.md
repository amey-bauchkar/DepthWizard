# Indian validation — Sikkim vs NASA ICESat-2 (measured)

Run at 2026-09-27T01:25:55 · `python scripts/validate_india.py` · imagery: Maxar Open Data Program (WorldView, 0.5 m, CC BY-NC 4.0) · DEM: Copernicus GLO-30 · checkpoints: NASA ICESat-2 ATL08-style 20 m segments (2018–2025) via SlideRule, ellipsoidal → EGM2008 with the bundled geoid grids.

No airborne LiDAR is publicly available for these Indian sites, so accuracy is measured against **independent sparse laser checkpoints**: *ground* = ICESat-2 terrain height, *canopy* = ICESat-2 height of the top surface above ground (trees and buildings), *top of surface* = ground + canopy. Neither the model nor the calibration saw these points or any Indian data.

## Summary (RMSE in metres; lower is better)

| site | checkpoints | DSM vs top of surface: Copernicus → zero-shot → fine-tuned | height above ground (nDSM vs canopy/structure): zero-shot → fine-tuned | terrain vs ground: Copernicus → fine-tuned |
|---|---|---|---|---|
| namchi | 21 | 12.83 → 10.42 → **9.75** | 12.23 → **8.75** | 7.72 → 8.99 |
| chungthang | 115 | 12.55 → 10.37 → **10.34** | 9.27 → **8.01** | 8.56 → 9.23 |
| teesta_east | 160 | 10.49 → 9.08 → **8.97** | 11.66 → **9.50** | 5.05 → 4.07 |
| teesta_west | 137 | 10.02 → 7.82 → **7.78** | 10.43 → **8.11** | 4.27 → 3.93 |
| chungthang_west | 273 | 15.66 → 16.32 → **16.60** | 14.90 → **10.00** | 19.68 → 15.29 |
| north_sikkim_alpine | 408 | 9.32 → 8.08 → **8.08** | 9.11 → **9.21** | 1.68 → 1.67 |

### Pooled over all sites (RMSE = sqrt of checkpoint-weighted mean squared error; metres)

| comparison | Copernicus DEM alone | zero-shot model | fine-tuned model |
|---|---|---|---|
| terrain vs ICESat-2 ground | 10.51 (ME +4.47) | 10.25 (ME +3.24) | **8.54 (ME +1.09)** |
| DSM vs ICESat-2 top of surface | 11.81 (ME -5.71) | 11.03 (ME -2.54) | **11.10 (ME -2.08)** |
| height above ground vs ICESat-2 canopy/structure | – | 11.38 (ME -8.14) | **9.20 (ME -4.94)** |

Checkpoints: 1114 ICESat-2 segments over 6 sites.

## Namchi, South Sikkim (hill town) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Dense hill town on steep slopes; district HQ of Namchi (South Sikkim)._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 21 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 21 | +2.66 | 7.72 | 6.03 | 5.32 |
| DepthWizard terrain vs ICESat-2 ground | 21 | +1.02 | 7.76 | 5.64 | 4.59 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 21 | -11.09 | 12.83 | 11.14 | 5.68 |
| DepthWizard DSM vs ICESat-2 top of surface | 21 | -7.52 | 10.42 | 8.71 | 7.48 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 21 | -10.79 | 12.23 | 10.79 | 6.43 |

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 7.79 (n=20); input_dem_vs_ground: RMSE 7.74 (n=20); ndsm_vs_canopy_height: RMSE 12.51 (n=20)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 21 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 21 | +2.66 | 7.72 | 6.03 | 5.32 |
| DepthWizard terrain vs ICESat-2 ground | 21 | -4.30 | 8.99 | 5.42 | 2.30 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 21 | -11.09 | 12.83 | 11.14 | 5.68 |
| DepthWizard DSM vs ICESat-2 top of surface | 21 | -6.01 | 9.75 | 8.24 | 7.61 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 21 | -2.89 | 8.75 | 7.09 | 8.69 |

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 7.36 (n=20); input_dem_vs_ground: RMSE 7.74 (n=20); ndsm_vs_canopy_height: RMSE 8.22 (n=20)

## Chungthang, North Sikkim (valley town, forest) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CE8D0400; Steep forested valley and town at the Lachen-Lachung confluence; Teesta-III dam area hit by the Oct-2023 GLOF._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 115 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 115 | +1.66 | 8.56 | 5.18 | 2.43 |
| DepthWizard terrain vs ICESat-2 ground | 115 | -0.60 | 9.11 | 5.16 | 2.96 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 115 | -8.08 | 12.55 | 10.01 | 4.58 |
| DepthWizard DSM vs ICESat-2 top of surface | 115 | -3.80 | 10.37 | 7.76 | 5.24 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 115 | -5.63 | 9.27 | 7.12 | 6.16 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 7.49 (n=10); input_dem_vs_ground: RMSE 6.86 (n=10); ndsm_vs_canopy_height: RMSE 4.47 (n=10)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 9.61 (n=89); input_dem_vs_ground: RMSE 8.85 (n=89); ndsm_vs_canopy_height: RMSE 10.34 (n=89)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 115 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 115 | +1.66 | 8.56 | 5.18 | 2.43 |
| DepthWizard terrain vs ICESat-2 ground | 115 | -2.22 | 9.23 | 5.21 | 2.46 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 115 | -8.08 | 12.55 | 10.01 | 4.58 |
| DepthWizard DSM vs ICESat-2 top of surface | 115 | -3.02 | 10.34 | 7.38 | 5.45 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 115 | -2.71 | 8.01 | 5.76 | 6.03 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 7.38 (n=10); input_dem_vs_ground: RMSE 6.86 (n=10); ndsm_vs_canopy_height: RMSE 5.13 (n=10)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 9.67 (n=89); input_dem_vs_ground: RMSE 8.85 (n=89); ndsm_vs_canopy_height: RMSE 8.67 (n=89)

## Teesta valley east, South Sikkim (hill villages) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Terraced slopes and scattered villages east of Namchi._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 160 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 160 | +3.66 | 5.05 | 3.70 | 2.41 |
| DepthWizard terrain vs ICESat-2 ground | 160 | +2.47 | 4.17 | 3.11 | 2.26 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 160 | -7.61 | 10.49 | 8.02 | 6.97 |
| DepthWizard DSM vs ICESat-2 top of surface | 160 | -4.24 | 9.08 | 6.59 | 6.27 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 160 | -8.95 | 11.66 | 9.19 | 7.78 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 2.97 (n=13); input_dem_vs_ground: RMSE 2.81 (n=13); ndsm_vs_canopy_height: RMSE 1.45 (n=13)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 4.64 (n=119); input_dem_vs_ground: RMSE 5.68 (n=119); ndsm_vs_canopy_height: RMSE 13.46 (n=119)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 160 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 160 | +3.66 | 5.05 | 3.70 | 2.41 |
| DepthWizard terrain vs ICESat-2 ground | 160 | -0.86 | 4.07 | 2.54 | 1.89 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 160 | -7.61 | 10.49 | 8.02 | 6.97 |
| DepthWizard DSM vs ICESat-2 top of surface | 160 | -4.13 | 8.97 | 6.31 | 6.27 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 160 | -4.52 | 9.50 | 6.78 | 5.68 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 1.62 (n=13); input_dem_vs_ground: RMSE 2.81 (n=13); ndsm_vs_canopy_height: RMSE 1.94 (n=13)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 4.63 (n=119); input_dem_vs_ground: RMSE 5.68 (n=119); ndsm_vs_canopy_height: RMSE 10.80 (n=119)

## Teesta valley west, South Sikkim (rural slopes) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Rural terraced hillsides and forest patches west of Namchi._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 137 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 137 | +2.73 | 4.27 | 3.39 | 2.69 |
| DepthWizard terrain vs ICESat-2 ground | 137 | +1.03 | 4.19 | 3.37 | 3.36 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 137 | -7.93 | 10.02 | 8.28 | 5.00 |
| DepthWizard DSM vs ICESat-2 top of surface | 137 | -4.05 | 7.82 | 5.85 | 4.86 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 137 | -7.98 | 10.43 | 8.73 | 6.77 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 4.18 (n=8); input_dem_vs_ground: RMSE 2.33 (n=8); ndsm_vs_canopy_height: RMSE 3.55 (n=8)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 3.93 (n=110); input_dem_vs_ground: RMSE 4.35 (n=110); ndsm_vs_canopy_height: RMSE 11.54 (n=110)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 137 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 137 | +2.73 | 4.27 | 3.39 | 2.69 |
| DepthWizard terrain vs ICESat-2 ground | 137 | -1.94 | 3.93 | 3.06 | 3.54 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 137 | -7.93 | 10.02 | 8.28 | 5.00 |
| DepthWizard DSM vs ICESat-2 top of surface | 137 | -3.66 | 7.78 | 5.69 | 4.91 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 137 | -3.31 | 8.11 | 5.92 | 5.54 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 2.89 (n=8); input_dem_vs_ground: RMSE 2.33 (n=8); ndsm_vs_canopy_height: RMSE 4.53 (n=8)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 3.83 (n=110); input_dem_vs_ground: RMSE 4.35 (n=110); ndsm_vs_canopy_height: RMSE 8.62 (n=110)

## Chungthang west, North Sikkim (forested slopes) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CE8D0400; Steep forested mountainside above the Lachen valley._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 273 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 273 | +13.29 | 19.68 | 13.42 | 11.41 |
| DepthWizard terrain vs ICESat-2 ground | 273 | +12.30 | 18.91 | 12.69 | 10.51 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 273 | -0.75 | 15.66 | 11.17 | 11.44 |
| DepthWizard DSM vs ICESat-2 top of surface | 273 | +2.68 | 16.32 | 11.33 | 12.47 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 273 | -12.40 | 14.90 | 12.51 | 8.68 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 12.63 (n=28); input_dem_vs_ground: RMSE 12.87 (n=28); ndsm_vs_canopy_height: RMSE 1.36 (n=28)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 19.05 (n=231); input_dem_vs_ground: RMSE 19.87 (n=231); ndsm_vs_canopy_height: RMSE 16.17 (n=231)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 273 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 273 | +13.29 | 19.68 | 13.42 | 11.41 |
| DepthWizard terrain vs ICESat-2 ground | 273 | +6.93 | 15.29 | 8.99 | 6.28 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 273 | -0.75 | 15.66 | 11.17 | 11.44 |
| DepthWizard DSM vs ICESat-2 top of surface | 273 | +3.88 | 16.60 | 11.38 | 11.63 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 273 | -4.53 | 10.00 | 7.82 | 8.78 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 10.10 (n=28); input_dem_vs_ground: RMSE 12.87 (n=28); ndsm_vs_canopy_height: RMSE 6.30 (n=28)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 15.36 (n=231); input_dem_vs_ground: RMSE 19.87 (n=231); ndsm_vs_canopy_height: RMSE 10.49 (n=231)

## North Sikkim alpine (barren / glacial) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CF621C00; High-altitude barren and glacial terrain in the South Lhonak region (near-nadir image, off-nadir 2 deg)._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 408 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 408 | +0.35 | 1.68 | 1.18 | 1.18 |
| DepthWizard terrain vs ICESat-2 ground | 408 | -0.58 | 2.90 | 1.81 | 1.72 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 408 | -6.60 | 9.32 | 6.82 | 4.97 |
| DepthWizard DSM vs ICESat-2 top of surface | 408 | -4.26 | 8.08 | 5.68 | 4.40 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 408 | -5.58 | 9.11 | 6.65 | 5.77 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 3.48 (n=122); input_dem_vs_ground: RMSE 1.32 (n=122); ndsm_vs_canopy_height: RMSE 4.21 (n=122)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 2.74 (n=201); input_dem_vs_ground: RMSE 1.93 (n=201); ndsm_vs_canopy_height: RMSE 12.41 (n=201)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 408 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM (Copernicus) vs ICESat-2 ground — baseline | 408 | +0.35 | 1.68 | 1.18 | 1.18 |
| DepthWizard terrain vs ICESat-2 ground | 408 | +0.17 | 1.67 | 1.16 | 1.13 |
| input DEM (Copernicus) vs ICESat-2 top of surface — baseline | 408 | -6.60 | 9.32 | 6.82 | 4.97 |
| DepthWizard DSM vs ICESat-2 top of surface | 408 | -4.26 | 8.08 | 5.69 | 4.46 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 408 | -6.66 | 9.21 | 6.80 | 5.58 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 1.32 (n=122); input_dem_vs_ground: RMSE 1.32 (n=122); ndsm_vs_canopy_height: RMSE 1.14 (n=122)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 1.92 (n=201); input_dem_vs_ground: RMSE 1.93 (n=201); ndsm_vs_canopy_height: RMSE 12.91 (n=201)

## Caveats

* Sparse checkpoints (tens to a few hundred per scene) along a few ground tracks; metrics carry sampling uncertainty.
* ICESat-2 dates (2018–2025) differ from the image dates (2022); new construction or clearing shows up as error.
* Each 20 m segment is compared with a 10 m-radius disk of the raster; on steep Himalayan slopes this adds error of its own.
* The fine-tuned model was trained only on Swiss data: these numbers are its first measurement on Indian imagery and terrain.
