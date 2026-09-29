# Mapped waterways for river-flood screening

Rivers, streams, canals and drains from **OpenStreetMap** (ODbL 1.0, (c) OpenStreetMap contributors), one GeoJSON per demo area; regenerate with `python scripts/fetch_waterways.py`. They are burned into the terrain for the height-above-nearest-drainage computation so that channels follow the mapped rivers instead of the coarse DEM.
