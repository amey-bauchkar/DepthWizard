"""Sentinel-2 L2A (Copernicus, via Microsoft Planetary Computer, free, no key) read onto any grid.

Used by the flood-scar mapping (scripts/glof_observed_extent.py) and the landslide scar check
(core.disaster.landslide). Online only: every function raises on network failure and callers must degrade.
"""
from __future__ import annotations

import numpy as np
import rasterio
import requests
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

PC = "https://planetarycomputer.microsoft.com/api"
CLOUD_SCL = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud medium / high, cirrus


def token() -> str:
    return requests.get(f"{PC}/sas/v1/token/sentinel-2-l2a", timeout=60).json()["token"]


def search(bbox: list[float], date_range: str, max_cloud: float = 20.0) -> list[dict]:
    """Items over bbox (W, S, E, N) in 'YYYY-MM-DD/YYYY-MM-DD', least cloudy first."""
    r = requests.post(f"{PC}/stac/v1/search", json={"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": date_range, "limit": 100,
                                                    "query": {"eo:cloud_cover": {"lt": max_cloud}}}, timeout=120).json()
    return sorted(r["features"], key=lambda f: f["properties"]["eo:cloud_cover"])


def read_band(item: dict, band: str, tok: str, crs, transform, shape: tuple[int, int], resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(item["assets"][band]["href"] + "?" + tok) as ds, WarpedVRT(ds, crs=crs, transform=transform, width=shape[1], height=shape[0], resampling=resampling) as v:
        return v.read(1).astype(np.float64)


def indices(item: dict, tok: str, crs, transform, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(NDVI, MNDWI, valid) on the grid; valid excludes no-data, cloud and shadow (scene classification)."""
    g, r, n, s = (read_band(item, b, tok, crs, transform, shape) for b in ("B03", "B04", "B08", "B11"))
    scl = read_band(item, "SCL", tok, crs, transform, shape, Resampling.nearest)
    ok = (g > 0) & ~np.isin(scl, CLOUD_SCL)
    ndvi = (n - r) / np.maximum(n + r, 1)
    mndwi = (g - s) / np.maximum(g + s, 1)
    return ndvi, mndwi, ok
