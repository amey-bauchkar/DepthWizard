# Nepal flood-validation scene

| File | Content | Source and licence |
|---|---|---|
| `sunkoshi_rgb_0.5m.tif` | RGB, 1.2 km × 1.2 km, 0.5 m, EPSG:32645, centred 27.434 N, 85.836 E (Sunkoshi / Roshi Khola confluence) | **Maxar Open Data Program**, event *Nepal-Floods-Sept-2024*, acquisition 10300100FC189500 (29 May 2024, before the flood; 25° off-nadir). Imagery © Maxar Technologies. **CC BY-NC 4.0**: non-commercial use only, attribution "Maxar Technologies, Maxar Open Data Program" |

Used to validate flood screening against the observed extent of the 27–28 Sep 2024 flood (`docs/hazard_validation.md`). Elevation: Copernicus GLO-30 tile N27E085 (`assets/dem/`), downloaded by `scripts/helipad_study.py`'s `ensure_dem` or manually from `https://copernicus-dem-30m.s3.amazonaws.com/`.
