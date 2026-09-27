# Indian validation — Sikkim vs NASA ICESat-2 (measured)

Run at 2026-09-27T07:15:09 · `python scripts/validate_india.py` · imagery: Maxar Open Data Program (WorldView, 0.5 m, CC BY-NC 4.0) · DEM: CartoDEM v3 R1 (ISRO/NRSC, ellipsoidal heights auto-detected and converted) where installed in assets/dem/cartodem, else Copernicus GLO-30 · checkpoints: NASA ICESat-2 ATL08-style 20 m segments (2018–2025) via SlideRule, ellipsoidal → EGM2008 with the bundled geoid grids.

No airborne LiDAR is publicly available for these Indian sites, so accuracy is measured against **independent sparse laser checkpoints**: *ground* = ICESat-2 terrain height, *canopy* = ICESat-2 height of the top surface above ground (trees and buildings), *top of surface* = ground + canopy. Neither the model nor the calibration saw these points or any Indian data.

## Summary (RMSE in metres; lower is better)

| site | checkpoints | DSM vs top of surface: input DEM → zero-shot → fine-tuned | height above ground (nDSM vs canopy/structure): zero-shot → fine-tuned | terrain vs ground: input DEM → fine-tuned |
|---|---|---|---|---|
| namchi | 21 | 14.47 → 12.23 → **11.44** | 13.06 → **8.75** | 7.99 → 8.85 |
| chungthang | 115 | 13.07 → 11.37 → **10.92** | 9.18 → **8.01** | 7.56 → 8.11 |
| teesta_east | 160 | 12.38 → 10.36 → **10.15** | 12.02 → **9.50** | 3.28 → 5.34 |
| teesta_west | 137 | 10.89 → 8.52 → **8.49** | 10.45 → **8.11** | 3.83 → 4.50 |
| chungthang_west | 273 | 16.19 → 15.20 → **14.93** | 15.08 → **10.00** | 12.84 → 11.27 |
| north_sikkim_alpine | 408 | 7.34 → 7.73 → **7.73** | 9.24 → **9.21** | 5.91 → 5.72 |

### Pooled over all sites (RMSE = sqrt of checkpoint-weighted mean squared error; metres)

| comparison | input DEM alone | zero-shot model | fine-tuned model |
|---|---|---|---|
| terrain vs ICESat-2 ground | 7.98 (ME +3.89) | 7.77 (ME +2.78) | **7.61 (ME +0.51)** |
| DSM vs ICESat-2 top of surface | 11.92 (ME -6.29) | 10.91 (ME -3.37) | **10.72 (ME -2.87)** |
| height above ground vs ICESat-2 canopy/structure | – | 11.54 (ME -8.39) | **9.20 (ME -4.94)** |

Checkpoints: 1114 ICESat-2 segments over 6 sites.

## Namchi, South Sikkim (hill town) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Dense hill town on steep slopes; district HQ of Namchi (South Sikkim)._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 21 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 21 | +1.80 | 7.99 | 5.74 | 5.59 |
| DepthWizard terrain vs ICESat-2 ground | 21 | +0.55 | 8.13 | 5.71 | 4.78 |
| input DEM vs ICESat-2 top of surface — baseline | 21 | -11.95 | 14.47 | 12.72 | 6.90 |
| DepthWizard DSM vs ICESat-2 top of surface | 21 | -8.75 | 12.23 | 10.44 | 7.71 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 21 | -11.62 | 13.06 | 11.62 | 5.66 |

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 8.32 (n=20); input_dem_vs_ground: RMSE 8.18 (n=20); ndsm_vs_canopy_height: RMSE 13.36 (n=20)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 21 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 21 | +1.80 | 7.99 | 5.74 | 5.59 |
| DepthWizard terrain vs ICESat-2 ground | 21 | -5.14 | 8.85 | 5.93 | 3.77 |
| input DEM vs ICESat-2 top of surface — baseline | 21 | -11.95 | 14.47 | 12.72 | 6.90 |
| DepthWizard DSM vs ICESat-2 top of surface | 21 | -7.05 | 11.44 | 9.61 | 9.00 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 21 | -2.89 | 8.75 | 7.09 | 8.69 |

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 8.29 (n=20); input_dem_vs_ground: RMSE 8.18 (n=20); ndsm_vs_canopy_height: RMSE 8.22 (n=20)

## Chungthang, North Sikkim (valley town, forest) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CE8D0400; Steep forested valley and town at the Lachen-Lachung confluence; Teesta-III dam area hit by the Oct-2023 GLOF._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 115 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 115 | +1.18 | 7.56 | 4.87 | 3.40 |
| DepthWizard terrain vs ICESat-2 ground | 115 | -1.07 | 7.50 | 5.57 | 4.95 |
| input DEM vs ICESat-2 top of surface — baseline | 115 | -8.55 | 13.07 | 11.49 | 6.76 |
| DepthWizard DSM vs ICESat-2 top of surface | 115 | -4.53 | 11.37 | 9.56 | 6.13 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 115 | -5.87 | 9.18 | 7.15 | 6.21 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 9.46 (n=10); input_dem_vs_ground: RMSE 8.84 (n=10); ndsm_vs_canopy_height: RMSE 4.60 (n=10)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 7.10 (n=89); input_dem_vs_ground: RMSE 7.12 (n=89); ndsm_vs_canopy_height: RMSE 10.26 (n=89)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 115 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 115 | +1.18 | 7.56 | 4.87 | 3.40 |
| DepthWizard terrain vs ICESat-2 ground | 115 | -2.70 | 8.11 | 6.37 | 4.88 |
| input DEM vs ICESat-2 top of surface — baseline | 115 | -8.55 | 13.07 | 11.49 | 6.76 |
| DepthWizard DSM vs ICESat-2 top of surface | 115 | -3.59 | 10.92 | 8.69 | 6.71 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 115 | -2.71 | 8.01 | 5.76 | 6.03 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 9.54 (n=10); input_dem_vs_ground: RMSE 8.84 (n=10); ndsm_vs_canopy_height: RMSE 5.13 (n=10)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 7.77 (n=89); input_dem_vs_ground: RMSE 7.12 (n=89); ndsm_vs_canopy_height: RMSE 8.67 (n=89)

## Teesta valley east, South Sikkim (hill villages) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Terraced slopes and scattered villages east of Namchi._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 160 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 160 | +1.36 | 3.28 | 2.28 | 2.30 |
| DepthWizard terrain vs ICESat-2 ground | 160 | +0.35 | 3.28 | 2.33 | 2.24 |
| input DEM vs ICESat-2 top of surface — baseline | 160 | -9.91 | 12.38 | 10.02 | 7.70 |
| DepthWizard DSM vs ICESat-2 top of surface | 160 | -7.01 | 10.36 | 7.91 | 6.83 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 160 | -9.38 | 12.02 | 9.52 | 8.25 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 2.16 (n=13); input_dem_vs_ground: RMSE 1.74 (n=13); ndsm_vs_canopy_height: RMSE 1.35 (n=13)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 3.60 (n=119); input_dem_vs_ground: RMSE 3.67 (n=119); ndsm_vs_canopy_height: RMSE 13.87 (n=119)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 160 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 160 | +1.36 | 3.28 | 2.28 | 2.30 |
| DepthWizard terrain vs ICESat-2 ground | 160 | -3.18 | 5.34 | 3.56 | 2.86 |
| input DEM vs ICESat-2 top of surface — baseline | 160 | -9.91 | 12.38 | 10.02 | 7.70 |
| DepthWizard DSM vs ICESat-2 top of surface | 160 | -6.81 | 10.15 | 7.56 | 6.09 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 160 | -4.52 | 9.50 | 6.78 | 5.68 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 1.22 (n=13); input_dem_vs_ground: RMSE 1.74 (n=13); ndsm_vs_canopy_height: RMSE 1.94 (n=13)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 6.05 (n=119); input_dem_vs_ground: RMSE 3.67 (n=119); ndsm_vs_canopy_height: RMSE 10.80 (n=119)

## Teesta valley west, South Sikkim (rural slopes) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 1040010073381800; Rural terraced hillsides and forest patches west of Namchi._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 137 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 137 | +1.66 | 3.83 | 3.08 | 3.35 |
| DepthWizard terrain vs ICESat-2 ground | 137 | -0.00 | 4.21 | 3.33 | 4.11 |
| input DEM vs ICESat-2 top of surface — baseline | 137 | -8.99 | 10.89 | 9.19 | 5.68 |
| DepthWizard DSM vs ICESat-2 top of surface | 137 | -5.25 | 8.52 | 6.58 | 4.96 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 137 | -8.12 | 10.45 | 8.82 | 6.49 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 5.71 (n=8); input_dem_vs_ground: RMSE 3.03 (n=8); ndsm_vs_canopy_height: RMSE 3.42 (n=8)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 3.79 (n=110); input_dem_vs_ground: RMSE 3.87 (n=110); ndsm_vs_canopy_height: RMSE 11.57 (n=110)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 137 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 137 | +1.66 | 3.83 | 3.08 | 3.35 |
| DepthWizard terrain vs ICESat-2 ground | 137 | -3.01 | 4.50 | 3.59 | 3.72 |
| input DEM vs ICESat-2 top of surface — baseline | 137 | -8.99 | 10.89 | 9.19 | 5.68 |
| DepthWizard DSM vs ICESat-2 top of surface | 137 | -4.88 | 8.49 | 6.41 | 4.83 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 137 | -3.31 | 8.11 | 5.92 | 5.54 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 4.62 (n=8); input_dem_vs_ground: RMSE 3.03 (n=8); ndsm_vs_canopy_height: RMSE 4.53 (n=8)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 4.24 (n=110); input_dem_vs_ground: RMSE 3.87 (n=110); ndsm_vs_canopy_height: RMSE 8.62 (n=110)

## Chungthang west, North Sikkim (forested slopes) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CE8D0400; Steep forested mountainside above the Lachen valley._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 273 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 273 | +5.89 | 12.84 | 6.62 | 3.88 |
| DepthWizard terrain vs ICESat-2 ground | 273 | +5.00 | 12.44 | 6.15 | 3.28 |
| input DEM vs ICESat-2 top of surface — baseline | 273 | -8.14 | 16.19 | 12.69 | 9.47 |
| DepthWizard DSM vs ICESat-2 top of surface | 273 | -5.18 | 15.20 | 11.15 | 9.61 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 273 | -12.59 | 15.08 | 12.72 | 8.30 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 3.97 (n=28); input_dem_vs_ground: RMSE 3.94 (n=28); ndsm_vs_canopy_height: RMSE 1.39 (n=28)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 12.52 (n=231); input_dem_vs_ground: RMSE 12.93 (n=231); ndsm_vs_canopy_height: RMSE 16.37 (n=231)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 273 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 273 | +5.89 | 12.84 | 6.62 | 3.88 |
| DepthWizard terrain vs ICESat-2 ground | 273 | -0.45 | 11.27 | 5.93 | 3.76 |
| input DEM vs ICESat-2 top of surface — baseline | 273 | -8.14 | 16.19 | 12.69 | 9.47 |
| DepthWizard DSM vs ICESat-2 top of surface | 273 | -3.96 | 14.93 | 10.60 | 9.96 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 273 | -4.53 | 10.00 | 7.82 | 8.78 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 2.70 (n=28); input_dem_vs_ground: RMSE 3.94 (n=28); ndsm_vs_canopy_height: RMSE 6.30 (n=28)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 11.60 (n=231); input_dem_vs_ground: RMSE 12.93 (n=231); ndsm_vs_canopy_height: RMSE 10.49 (n=231)

## North Sikkim alpine (barren / glacial) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B

_Maxar Open Data Program (CC BY-NC 4.0), acquisition 10300100CF621C00; High-altitude barren and glacial terrain in the South Lhonak region (near-nadir image, off-nadir 2 deg)._


**zero shot** (tile model `da-v2-small-baseline`, tier T, quality LIMITED; 408 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 408 | +5.16 | 5.91 | 5.23 | 2.28 |
| DepthWizard terrain vs ICESat-2 ground | 408 | +4.37 | 5.63 | 4.90 | 2.73 |
| input DEM vs ICESat-2 top of surface — baseline | 408 | -1.79 | 7.34 | 5.70 | 5.93 |
| DepthWizard DSM vs ICESat-2 top of surface | 408 | +0.49 | 7.73 | 6.30 | 6.24 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 408 | -5.81 | 9.24 | 6.79 | 6.42 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 5.59 (n=122); input_dem_vs_ground: RMSE 6.00 (n=122); ndsm_vs_canopy_height: RMSE 4.28 (n=122)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 5.86 (n=201); input_dem_vs_ground: RMSE 6.02 (n=201); ndsm_vs_canopy_height: RMSE 12.58 (n=201)

**fine tuned** (tile model `da-v2-small-ndsm`, tier T, quality LIMITED; 408 checkpoints in the scene)

| comparison (metres) | n | ME | RMSE | MAE | NMAD |
|---|---|---|---|---|---|
| input DEM vs ICESat-2 ground — baseline | 408 | +5.16 | 5.91 | 5.23 | 2.28 |
| DepthWizard terrain vs ICESat-2 ground | 408 | +4.97 | 5.72 | 5.08 | 2.15 |
| input DEM vs ICESat-2 top of surface — baseline | 408 | -1.79 | 7.34 | 5.70 | 5.93 |
| DepthWizard DSM vs ICESat-2 top of surface | 408 | +0.50 | 7.73 | 6.30 | 6.38 |
| DepthWizard nDSM vs ICESat-2 canopy/structure height | 408 | -6.66 | 9.21 | 6.80 | 5.58 |

_open_canopy_lt2m_: terrain_vs_ground: RMSE 5.81 (n=122); input_dem_vs_ground: RMSE 6.00 (n=122); ndsm_vs_canopy_height: RMSE 1.14 (n=122)

_vegetated_or_built_ge5m_: terrain_vs_ground: RMSE 5.80 (n=201); input_dem_vs_ground: RMSE 6.02 (n=201); ndsm_vs_canopy_height: RMSE 12.91 (n=201)

## Caveats

* Sparse checkpoints (tens to a few hundred per scene) along a few ground tracks; metrics carry sampling uncertainty.
* ICESat-2 dates (2018–2025) differ from the image dates (2022); new construction or clearing shows up as error.
* Each 20 m segment is compared with a 10 m-radius disk of the raster; on steep Himalayan slopes this adds error of its own.
* The fine-tuned model was trained only on Swiss data: these numbers are its first measurement on Indian imagery and terrain.
