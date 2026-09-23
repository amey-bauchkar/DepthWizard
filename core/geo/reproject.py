"""Phase 8 C-4: metric raster operations must never run on latitude/longitude degrees.

`ensure_metric_grid` inspects a raster's CRS; if geographic, it reprojects to the local UTM zone
(WGS84) and records the provenance of that step. Sprint 1 only exercises this in tests; Mode B
(Sprint 2) will call it before any GSD-dependent processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Geod
from rasterio.warp import Resampling, calculate_default_transform, reproject

from core.geo.grid import Grid


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def local_gsd_metres(crs: CRS, transform, width: int, height: int) -> tuple[float, float]:
    """Ground pixel size in metres at the raster centre (geodesic for geographic CRS)."""
    if not crs.is_geographic:
        return (abs(transform.a) if transform.b == 0 else (transform.a**2 + transform.d**2) ** 0.5,
                abs(transform.e) if transform.d == 0 else (transform.b**2 + transform.e**2) ** 0.5)
    g = Geod(ellps="WGS84")
    lon, lat = transform * (width / 2 + 0.5, height / 2 + 0.5)
    _, _, dx = g.inv(lon, lat, lon + abs(transform.a), lat)
    _, _, dy = g.inv(lon, lat, lon, lat + abs(transform.e))
    return (dx, dy)


@dataclass
class ReprojectionRecord:
    performed: bool
    src_crs: str
    dst_crs: str
    src_gsd_m: tuple[float, float]
    dst_gsd_m: tuple[float, float] | None
    anisotropy_before: float
    resampling: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def ensure_metric_grid(src_path: str, dst_path: str, *, resampling: Resampling = Resampling.bilinear) -> tuple[Grid, ReprojectionRecord]:
    """If `src_path` is in a geographic CRS, write a UTM-reprojected copy to `dst_path` and return its Grid.
    Otherwise return the source Grid untouched (dst_path unused)."""
    with rasterio.open(src_path) as src:
        crs = CRS.from_user_input(src.crs.to_wkt()) if src.crs else None
        if crs is None:
            raise ValueError("raster has no CRS; cannot establish metric grid")
        gsd = local_gsd_metres(crs, src.transform, src.width, src.height)
        aniso = max(gsd) / min(gsd)
        if not crs.is_geographic:
            grid = Grid(src.width, src.height, src.transform, f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else crs.to_wkt(), str(src.dtypes[0]), src.nodata, "metres", True, None, "R")
            return grid, ReprojectionRecord(False, grid.crs or "", grid.crs or "", gsd, gsd, aniso, "none")
        lon, lat = src.transform * (src.width / 2 + 0.5, src.height / 2 + 0.5)
        dst_epsg = utm_epsg_for(lon, lat)
        dst_crs = CRS.from_epsg(dst_epsg)
        dst_transform, w, h = calculate_default_transform(src.crs, dst_crs.to_wkt(), src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update(crs=dst_crs.to_wkt(), transform=dst_transform, width=w, height=h)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b), destination=rasterio.band(dst, b), src_transform=src.transform, src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs.to_wkt(), resampling=resampling, src_nodata=src.nodata, dst_nodata=src.nodata)
        grid = Grid(w, h, dst_transform, f"EPSG:{dst_epsg}", str(src.dtypes[0]), src.nodata, "metres", True, None, "R")
        rec = ReprojectionRecord(True, f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else "geographic", f"EPSG:{dst_epsg}", gsd, grid.pixel_size, aniso, resampling.name)
        return grid, rec
