# CartoDEM (ISRO / NRSC): drop-in folder

Put CartoDEM v3 R1 GeoTIFF tiles here, with any file name. DepthWizard then uses CartoDEM instead of Copernicus GLO-30 wherever it covers the scene (`calib.dem_priority` in `configs/default.yaml`).

**Download**
1. Go to Bhuvan (https://bhuvan-app3.nrsc.gov.in/data/download/), then **Open Data Archive → Satellite / Sensor: CartoDEM → Version-3 R1**. A free login is required.
2. Select the tiles covering your area, download and unzip them. Keep the `.tif` files.
3. Copy the `.tif` files into this folder.

**Vertical datum (checked automatically)**

The product documentation should state the vertical datum, but a wrong assumption would cause a 40–90 m error in India. So with `cartodem_vertical_crs: auto`, DepthWizard compares CartoDEM with Copernicus (EGM2008) over the scene:
- median difference ≈ 0 → geoid heights (EGM96);
- difference ≈ the local geoid undulation → ellipsoidal heights;
- anything else → CartoDEM is **refused** for that scene and Copernicus is used. The reason appears in `calib_report.json` under `dem.selection`.

To declare the datum yourself, set `cartodem_vertical_crs` to `EGM96`, `EGM2008` or `ellipsoidal`.

Licence: follow the NRSC / Bhuvan terms of use shown at download. Check them before redistributing tiles with the software.
