# Demo-tile validation results (measured)

Run at 2026-09-27T00:22:00 · model da-v2-small-baseline@1.0.0 on cpu · Python 3.14.0 · torch 2.14.0+cpu · method tlcsm-1.0 (tiled inference + DEM-preserving detail fusion)

All numbers are MEASURED by `scripts/validate_demo.py` (reproducible, deterministic): every Mode B demo tile is run through the real pipeline and compared with swisstopo airborne-LiDAR rasters — swissSURFACE3D (DSM) and swissALTI3D (DTM), 0.5 m — area-averaged onto the job grid, LN02 → EGM2008 converted with the bundled geoid grids. Border 4 px (16 px at 0.5 m) masked. Units: metres. *Baseline* = the raw Copernicus GLO-30 DEM bilinearly resampled to the same grid (what you get without DepthWizard).

## Summary — DSM and terrain RMSE against LiDAR (lower is better)

| tile | layer vs reference | Copernicus GLO-30 alone | DepthWizard zero-shot model, tier T | DepthWizard fine-tuned model, tier T (no anchors) | DepthWizard fine-tuned model + simulated anchors (tier A where the anchor fit was accepted) |
|---|---|---|---|---|---|
| Zürich (urban) · Ultra-Clear HD 0.5 m · Mode B | DSM vs LiDAR DSM | 9.25 | **8.91** (-3.7 %) | **5.53** (-40.2 %) | **5.57** (-39.8 %) |
| Zürich (urban) · Ultra-Clear HD 0.5 m · Mode B | terrain vs LiDAR DTM | 8.00 | **6.11** (-23.6 %) | **3.17** (-60.4 %) | **4.04** (-49.5 %) |
| Zürich (urban) · GeoTIFF 2 m · Mode B | DSM vs LiDAR DSM | 9.00 | **8.89** (-1.2 %) | **6.34** (-29.5 %) | **6.32** (-29.8 %) |
| Zürich (urban) · GeoTIFF 2 m · Mode B | terrain vs LiDAR DTM | 7.97 | **6.23** (-21.9 %) | **3.25** (-59.3 %) | **3.24** (-59.4 %) |
| Emmental (rural, hilly, forest) · GeoTIFF 2 m · Mode B | DSM vs LiDAR DSM | 7.45 | **7.18** (-3.7 %) | **6.41** (-13.9 %) | **6.35** (-14.7 %) |
| Emmental (rural, hilly, forest) · GeoTIFF 2 m · Mode B | terrain vs LiDAR DTM | 11.49 | **10.35** (-9.9 %) | **5.30** (-53.9 %) | **5.26** (-54.2 %) |

How to read this: the DEM already carries every structure larger than one posting (30 m); DepthWizard keeps it and adds the finer detail from Depth Anything V2 run on ~0.5 m tiles, so at 30 m its DSM is unbiased against the DEM (per 30 m cell it can still differ by a few metres, because the high-pass is Gaussian) and the changes come from sub-30 m structure. *Zero-shot* = the untrained model, whose per-tile scale is fitted against the DEM. *Fine-tuned* = the DepthWizard nDSM model (trained on swisstopo LiDAR from other regions; the tiles below and 5 km around them were excluded from training, validation and model selection), which predicts height above ground in metres directly; terrain = DSM − predicted heights. Tier A anchors fit a gain/offset on the DSM detail (accepted only if leave-one-out error improves).

Caveats (stated, not hidden): only two 1 km² locations (Swiss urban, and rural/forested hilly); no Indian LiDAR reference was available; the tier-A anchors are *simulated* by sampling the same LiDAR tiles (pixels within 15 m of an anchor are excluded from every metric); the fine-tuned model was trained on Swiss data only, so these tiles measure generalisation to unseen Swiss places, not to Indian imagery. Bands (A < 2 m, B 2–5 m, C 5–10 m, D > 10 m RMSE) are descriptive only.


## Zürich (urban) · Ultra-Clear HD 0.5 m · Mode B

_swisstopo SWISSIMAGE 10 cm (0.5 m GSD high-res, 2000×2000 px); references swissSURFACE3D / swissALTI3D 0.5 m (LN02)_

| variant | tier | quality | model detail | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DSM reference | 3968064 | -1.11 | 9.25 | 7.75 | 10.61 | 13.69 | 0.394 |
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DTM reference | 3968064 | +7.17 | 8.00 | 7.19 | 3.50 | 11.40 | 0.547 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | dsm vs DSM | 3968064 | -1.11 | 8.91 | 7.43 | 10.20 | 13.22 | 0.466 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | terrain vs DTM | 3968064 | +5.34 | 6.11 | 5.46 | 2.81 | 8.92 | 0.554 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | dsm vs DSM | 3968064 | -1.11 | 5.53 | 3.78 | 3.89 | 8.65 | 0.840 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | terrain vs DTM | 3968064 | -0.64 | 3.17 | 2.36 | 2.66 | 5.19 | 0.473 |
| simulated_anchors | A | LIMITED | fine-tuned metric nDSM model + anchor gain | dsm vs DSM | 3901842 | -1.11 | 5.57 | 3.80 | 3.86 | 8.58 | 0.843 |
| simulated_anchors | A | LIMITED | fine-tuned metric nDSM model + anchor gain | terrain vs DTM | 3901842 | -2.15 | 4.04 | 3.06 | 2.98 | 6.63 | 0.410 |

**zeroshot_no_anchors** — triggers: mean ground support 0.49 < 0.5; model detail scaled from the DEM band (24/25 tiles, median gain 3.06 m/unit) — unvalidated without anchors or reference. Timings (ms): {'preprocessing_ms': 577.1, 'model_forward_ms': 732.2, 'tiled_inference_ms': 19984.5, 'inference_ms': 20812.2, 'calibration_ms': 16682.7, 'total_ms': 38691.0}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -0.05, RMSE 7.69 (n=1537663); objects_ge1m: ME -1.77, RMSE 9.60 (n=2430401); tall_ge10m: ME -9.56, RMSE 10.65 (n=10526)

DSM error by reference slope class: flat_lt5: RMSE 7.95 (n=1737505); moderate_5_15: RMSE 9.87 (n=285480); steep_15_30: RMSE 10.13 (n=381101); very_steep_gt30: RMSE 9.40 (n=1563978)

**no_anchors** — triggers: mean ground support 0.42 < 0.5; object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene) — no anchors to confirm them here. Timings (ms): {'preprocessing_ms': 624.1, 'model_forward_ms': 680.0, 'tiled_inference_ms': 16482.6, 'inference_ms': 17216.1, 'calibration_ms': 9852.3, 'total_ms': 27693.1}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -0.12, RMSE 3.88 (n=1669575); objects_ge1m: ME -1.83, RMSE 6.47 (n=2298489); tall_ge10m: ME -1.84, RMSE 6.51 (n=1504605)

DSM error by reference slope class: flat_lt5: RMSE 4.15 (n=1737505); moderate_5_15: RMSE 5.85 (n=285480); steep_15_30: RMSE 5.92 (n=381101); very_steep_gt30: RMSE 6.61 (n=1563978)

**simulated_anchors** — triggers: mean ground support 0.42 < 0.5; model detail calibrated by anchors (gain 1.19, offset +0.00 m, leave-one-out RMSE 4.45 m vs 4.56 m tier T). Timings (ms): {'preprocessing_ms': 767.7, 'model_forward_ms': 772.3, 'tiled_inference_ms': 18848.6, 'inference_ms': 19726.9, 'calibration_ms': 17724.7, 'total_ms': 38220.3}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -1.16, RMSE 4.12 (n=1594637); objects_ge1m: ME -1.08, RMSE 6.39 (n=2307205); tall_ge10m: ME -0.40, RMSE 6.27 (n=1598705)

DSM error by reference slope class: flat_lt5: RMSE 4.16 (n=1705976); moderate_5_15: RMSE 5.73 (n=281109); steep_15_30: RMSE 5.68 (n=376060); very_steep_gt30: RMSE 6.76 (n=1538697)

## Zürich (urban) · GeoTIFF 2 m · Mode B

_swisstopo SWISSIMAGE 10 cm (resampled 2 m) tile 2682-1247, 2019; references swissSURFACE3D / swissALTI3D 0.5 m (LN02)_

| variant | tier | quality | model detail | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DSM reference | 242064 | -1.09 | 9.00 | 7.43 | 10.07 | 13.48 | 0.400 |
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DTM reference | 242064 | +7.15 | 7.97 | 7.17 | 3.51 | 11.38 | 0.546 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | dsm vs DSM | 242064 | -1.09 | 8.89 | 7.34 | 9.95 | 13.33 | 0.423 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | terrain vs DTM | 242064 | +5.49 | 6.23 | 5.59 | 2.78 | 9.05 | 0.559 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | dsm vs DSM | 242064 | -1.10 | 6.34 | 4.71 | 5.47 | 10.17 | 0.772 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | terrain vs DTM | 242064 | +0.22 | 3.25 | 2.50 | 2.97 | 5.18 | 0.454 |
| simulated_anchors | T | LIMITED | fine-tuned metric nDSM model | dsm vs DSM | 237906 | -1.10 | 6.32 | 4.69 | 5.46 | 10.14 | 0.773 |
| simulated_anchors | T | LIMITED | fine-tuned metric nDSM model | terrain vs DTM | 237906 | +0.21 | 3.24 | 2.50 | 2.96 | 5.17 | 0.458 |

**zeroshot_no_anchors** — triggers: mean ground support 0.50 < 0.5; model detail scaled from the DEM band (18/25 tiles, median gain 3.18 m/unit) — unvalidated without anchors or reference. Timings (ms): {'preprocessing_ms': 76.0, 'model_forward_ms': 829.8, 'tiled_inference_ms': 19199.5, 'inference_ms': 20076.9, 'calibration_ms': 2859.6, 'total_ms': 23013.3}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -1.06, RMSE 8.15 (n=99619); objects_ge1m: ME -1.11, RMSE 9.38 (n=142445); tall_ge10m: ME -8.89, RMSE 11.80 (n=189)

DSM error by reference slope class: flat_lt5: RMSE 7.64 (n=61826); moderate_5_15: RMSE 10.12 (n=20829); steep_15_30: RMSE 10.48 (n=30596); very_steep_gt30: RMSE 8.83 (n=128813)

**no_anchors** — triggers: mean ground support 0.34 < 0.5; object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene) — no anchors to confirm them here. Timings (ms): {'preprocessing_ms': 73.4, 'model_forward_ms': 784.6, 'tiled_inference_ms': 22761.3, 'inference_ms': 23577.6, 'calibration_ms': 1712.8, 'total_ms': 25364.6}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME +0.66, RMSE 4.37 (n=82852); objects_ge1m: ME -2.01, RMSE 7.16 (n=159212); tall_ge10m: ME -2.86, RMSE 7.63 (n=82596)

DSM error by reference slope class: flat_lt5: RMSE 4.67 (n=61826); moderate_5_15: RMSE 6.40 (n=20829); steep_15_30: RMSE 6.72 (n=30596); very_steep_gt30: RMSE 6.91 (n=128813)

**simulated_anchors** — triggers: mean ground support 0.34 < 0.5; object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene); anchors supplied but not used: rejected: leave-one-out RMSE does not improve on tier T. Timings (ms): {'preprocessing_ms': 75.5, 'model_forward_ms': 748.1, 'tiled_inference_ms': 21731.3, 'inference_ms': 22512.0, 'calibration_ms': 2042.9, 'total_ms': 24630.9}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME +0.64, RMSE 4.35 (n=81342); objects_ge1m: ME -2.00, RMSE 7.13 (n=156564); tall_ge10m: ME -2.85, RMSE 7.61 (n=81095)

DSM error by reference slope class: flat_lt5: RMSE 4.65 (n=60707); moderate_5_15: RMSE 6.37 (n=20539); steep_15_30: RMSE 6.70 (n=30150); very_steep_gt30: RMSE 6.89 (n=126510)

## Emmental (rural, hilly, forest) · GeoTIFF 2 m · Mode B

_swisstopo SWISSIMAGE tile 2621-1202, 2021; references swissSURFACE3D / swissALTI3D 0.5 m (LN02)_

| variant | tier | quality | model detail | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DSM reference | 242064 | -0.43 | 7.45 | 4.33 | 2.24 | 13.04 | 0.975 |
| raw Copernicus GLO-30 (no model, baseline) | — | — | — | DTM reference | 242064 | +6.08 | 11.49 | 6.65 | 2.72 | 23.98 | 0.968 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | dsm vs DSM | 242064 | -0.43 | 7.18 | 4.16 | 2.16 | 12.51 | 0.977 |
| zeroshot_no_anchors | T | LIMITED | tiled_dem_band_fusion (unvalidated) | terrain vs DTM | 242064 | +3.54 | 10.35 | 6.44 | 4.06 | 20.91 | 0.972 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | dsm vs DSM | 242064 | -0.43 | 6.41 | 3.69 | 1.98 | 10.90 | 0.982 |
| no_anchors | T | LIMITED | fine-tuned metric nDSM model | terrain vs DTM | 242064 | +1.29 | 5.30 | 3.03 | 1.76 | 9.12 | 0.991 |
| simulated_anchors | T | LIMITED | fine-tuned metric nDSM model | dsm vs DSM | 237840 | -0.43 | 6.35 | 3.65 | 1.97 | 10.76 | 0.982 |
| simulated_anchors | T | LIMITED | fine-tuned metric nDSM model | terrain vs DTM | 237840 | +1.28 | 5.26 | 3.00 | 1.74 | 9.06 | 0.991 |

**zeroshot_no_anchors** — triggers: model detail scaled from the DEM band (11/25 tiles, median gain 0.00 m/unit) — unvalidated without anchors or reference. Timings (ms): {'preprocessing_ms': 82.1, 'model_forward_ms': 732.9, 'tiled_inference_ms': 19069.0, 'inference_ms': 19839.7, 'calibration_ms': 1887.2, 'total_ms': 21810.0}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME +0.71, RMSE 6.52 (n=107387); objects_ge1m: ME -1.33, RMSE 7.66 (n=134677); tall_ge10m: ME -7.96, RMSE 11.56 (n=9424)

DSM error by reference slope class: flat_lt5: RMSE 1.91 (n=18755); moderate_5_15: RMSE 3.08 (n=52631); steep_15_30: RMSE 4.33 (n=71215); very_steep_gt30: RMSE 10.30 (n=99463)

**no_anchors** — triggers: object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene) — no anchors to confirm them here. Timings (ms): {'preprocessing_ms': 79.3, 'model_forward_ms': 739.6, 'tiled_inference_ms': 29049.3, 'inference_ms': 29828.4, 'calibration_ms': 916.6, 'total_ms': 30825.1}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -0.33, RMSE 2.51 (n=152902); objects_ge1m: ME -0.59, RMSE 10.04 (n=89162); tall_ge10m: ME -0.95, RMSE 10.51 (n=60942)

DSM error by reference slope class: flat_lt5: RMSE 1.77 (n=18755); moderate_5_15: RMSE 2.70 (n=52631); steep_15_30: RMSE 3.71 (n=71215); very_steep_gt30: RMSE 9.26 (n=99463)

**simulated_anchors** — triggers: object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene); anchors supplied but not used: rejected: leave-one-out RMSE does not improve on tier T. Timings (ms): {'preprocessing_ms': 49.6, 'model_forward_ms': 816.8, 'tiled_inference_ms': 30034.5, 'inference_ms': 30882.4, 'calibration_ms': 2160.5, 'total_ms': 33092.9}

DSM error by object class (from the predicted nDSM): ground_lt1m: ME -0.34, RMSE 2.52 (n=150731); objects_ge1m: ME -0.59, RMSE 9.96 (n=87109); tall_ge10m: ME -0.92, RMSE 10.43 (n=59483)

DSM error by reference slope class: flat_lt5: RMSE 1.75 (n=18467); moderate_5_15: RMSE 2.69 (n=52035); steep_15_30: RMSE 3.67 (n=69969); very_steep_gt30: RMSE 9.19 (n=97369)
