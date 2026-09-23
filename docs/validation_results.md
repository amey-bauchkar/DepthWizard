# Demo-tile validation results (measured)

Run at 2026-09-21T12:20:53 · model da-v2-small-baseline@1.0.0 on cpu · 3.12.14 · torch 2.14.0+cpu

All numbers below are MEASURED by `scripts/validate_demo.py` on the bundled swisstopo tiles (1 km × 1 km, 2 m GSD) against swissSURFACE3D (DSM) and swissALTI3D (DTM) 0.5 m LiDAR references, area-averaged onto the 2 m job grid, LN02 → EGM2008 converted with the bundled grids. Border 4 px masked; anchor pixels excluded within 15 m. Units: metres.

Bands are descriptive (A <2 m, B 2–5 m, C 5–10 m, D >10 m RMSE) — they are not a claim of meeting the problem statement.


## Zürich (urban) · GeoTIFF 2 m · Mode B

_swisstopo SWISSIMAGE 10 cm (resampled 2 m) tile 2682-1247, 2019; references swissSURFACE3D / swissALTI3D 0.5 m (LN02)_

| variant | tier | quality | object scale | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DSM reference | 242064 | -1.09 | 9.00 | 7.43 | 10.07 | 13.48 | 0.400 |
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DTM reference | 242064 | +7.15 | 7.97 | 7.17 | 3.51 | 11.38 | 0.546 |
| no_anchors | T | WARNING | none | dsm vs DSM | 242064 | -1.77 | 9.19 | 7.56 | 9.94 | 14.09 | 0.377 |
| no_anchors | T | WARNING | none | terrain vs DTM | 242064 | +6.47 | 7.20 | 6.54 | 3.00 | 10.24 | 0.522 |
| simulated_anchors | A | WARNING | none | dsm vs DSM | 237906 | -8.33 | 12.26 | 9.17 | 9.90 | 20.54 | 0.379 |
| simulated_anchors | A | WARNING | none | terrain vs DTM | 237906 | -0.11 | 3.17 | 2.47 | 3.00 | 5.06 | 0.524 |

**no_anchors** — triggers: mean ground support 0.50 < 0.5; no object scale available: DSM = terrain layer only; object heights remain relative. Timings: {'preprocessing_ms': 135.7, 'inference_ms': 754.8, 'model_forward_ms': 718.7, 'calibration_ms': 542.9, 'total_ms': 1434.1}

**simulated_anchors** — triggers: mean ground support 0.50 < 0.5; anchors: n=8 offset -6.57 m, hold-out NMAD 5.29 m; anchor hold-out NMAD > 3 m; no object scale available: DSM = terrain layer only; object heights remain relative; anchor scale rejected: object layer does not track anchor heights (r=-0.07, scale=177.01, inliers 7/7, implied p99 height 39.5 m). Timings: {'preprocessing_ms': 53.9, 'inference_ms': 655.7, 'model_forward_ms': 639.8, 'calibration_ms': 1143.7, 'total_ms': 1853.7}

## Emmental (rural, hilly, forest) · GeoTIFF 2 m · Mode B

_swisstopo SWISSIMAGE tile 2621-1202, 2021; references swissSURFACE3D / swissALTI3D 0.5 m (LN02)_

| variant | tier | quality | object scale | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DSM reference | 242064 | -0.43 | 7.45 | 4.33 | 2.24 | 13.04 | 0.975 |
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DTM reference | 242064 | +6.08 | 11.49 | 6.65 | 2.72 | 23.98 | 0.968 |
| no_anchors | T | LIMITED | dem_residual_fit (unvalidated) | dsm vs DSM | 242064 | -2.10 | 8.83 | 5.65 | 4.05 | 15.41 | 0.967 |
| no_anchors | T | LIMITED | dem_residual_fit (unvalidated) | terrain vs DTM | 242064 | +4.11 | 10.77 | 6.67 | 4.38 | 21.68 | 0.970 |
| simulated_anchors | A | LIMITED | dem_residual_fit (unvalidated) | dsm vs DSM | 237840 | -1.11 | 8.57 | 5.41 | 4.04 | 14.87 | 0.968 |
| simulated_anchors | A | LIMITED | dem_residual_fit (unvalidated) | terrain vs DTM | 237840 | +5.07 | 11.13 | 6.82 | 4.36 | 22.66 | 0.970 |

**no_anchors** — triggers: object scale from DEM residual fit (n=809, NMAD=1.81 m) — unvalidated. Timings: {'preprocessing_ms': 52.5, 'inference_ms': 676.7, 'model_forward_ms': 648.2, 'calibration_ms': 680.9, 'total_ms': 1410.5}

DSM error by reference object class: ground_lt1m: ME -1.54, RMSE 7.88 (n=220175); objects_ge1m: ME -7.71, RMSE 15.41 (n=21889)

**simulated_anchors** — triggers: anchors: n=8 offset +0.99 m, hold-out NMAD 0.89 m; object scale from DEM residual fit (n=809, NMAD=1.81 m) — unvalidated. Timings: {'preprocessing_ms': 44.9, 'inference_ms': 680.7, 'model_forward_ms': 654.3, 'calibration_ms': 960.4, 'total_ms': 1686.4}

DSM error by reference object class: ground_lt1m: ME -0.58, RMSE 7.73 (n=216437); objects_ge1m: ME -6.44, RMSE 14.54 (n=21403)
