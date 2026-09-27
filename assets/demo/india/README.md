# Indian demo and validation package — Sikkim (Teesta basin)

Regenerate with `python scripts/fetch_india_demo.py`. No account or login is needed. ICESat-2 needs `pip install -r requirements/india-data.txt`.

| File | Content | Source and licence |
|---|---|---|
| `<site>_rgb_0.5m.tif` | RGB, 1.2 km × 1.2 km, 0.5 m, EPSG:32645 (UTM 45N). Read from the ARD "visual" COGs (0.31 m) with area averaging | **Maxar Open Data Program**, event *India-Floods-Oct-2023* (South Lhonak glacial-lake outburst flood, Sikkim, 4 Oct 2023). Imagery © Maxar Technologies. **CC BY-NC 4.0**: non-commercial use only, attribution required |
| `<site>_icesat2.csv` | Independent checkpoints: ground elevation (ellipsoidal WGS84) and canopy/structure height above ground, 20 m segments, 2018–2025 | **NASA ICESat-2** ATL03 photons processed with the ATL08 PhoREAL algorithm by the public **SlideRule** service (slideruleearth.io). NASA data are free of restrictions |
| `../../dem/Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif` | 30 m DSM-like DEM (EGM2008) covering all sites | **Copernicus DEM GLO-30** © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA |

## Sites

| Site | Terrain class | Image acquisition | Use |
|---|---|---|---|
| namchi | dense hill town | 1040010073381800, 2022-03-14 | test |
| chungthang | valley town, dam, forest | 10300100CE8D0400, 2022-03-07 | test |
| teesta_east | hill villages, terraces | 1040010073381800 | test |
| teesta_west | rural slopes, forest patches | 1040010073381800 | test |
| chungthang_west | steep forest | 10300100CE8D0400 | test |
| north_sikkim_alpine | barren, glacial | 10300100CF621C00 (off-nadir 2°) | test |

None of these sites, and no Indian data at all, was used to train or tune the model. The one method change considered after seeing the Namchi/Chungthang results was a scene-level estimate of how much object height the DEM contains. It was rejected: it did not improve track-wise cross-validation. The method is therefore unchanged, and all six sites remain test data.

## Recommended official Indian sources (need an account or a request)

- **CartoDEM v3 R1** (Bhuvan / NRSC): the national 30 m DEM. Upload it as a user DEM (vertical CRS per its README).
- **Cartosat-2/3 imagery** (Bhoonidhi / NRSC): request sample scenes through the SIH nodal contacts.
