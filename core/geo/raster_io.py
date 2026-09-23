"""GeoTIFF read/write helpers built on rasterio, always paired with a `Grid`.

Rules (Phase 6 §9.3): every raster in a job shares the job Grid; writes carry CRS, transform, nodata and the
provenance tags (UNITS, METRIC, CALIBRATION_TIER, VERTICAL_CRS, ...); reads never guess a CRS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

from core.geo.grid import Grid


@dataclass
class GeoImage:
    rgb: np.ndarray  # HxWx3 uint8
    valid_mask: np.ndarray | None
    grid: Grid
    crs_wkt: str | None
    band_count: int
    dtype: str
    vertical_crs_tag: str | None


def _to_uint8(band: np.ndarray, nodata: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-stretch a numeric band to uint8; returns (uint8, valid_mask)."""
    valid = np.isfinite(band)
    if nodata is not None:
        valid &= band != nodata
    if band.dtype == np.uint8:
        return band, valid
    vals = band[valid]
    if vals.size == 0:
        return np.zeros(band.shape, np.uint8), valid
    lo, hi = np.percentile(vals, (1, 99))
    if hi <= lo:
        hi = lo + 1
    out = np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8), valid


def read_georeferenced_rgb(path: str | Path, *, band_map: tuple[int, int, int] = (1, 2, 3), max_dim: int = 4096) -> GeoImage:
    """Read a GeoTIFF as RGB + Grid. Raises ValueError if the file has no CRS or no usable transform."""
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.transform is None or ds.transform.is_identity:
            raise ValueError("raster has no CRS/geotransform")
        if max(ds.width, ds.height) > max_dim:
            raise ValueError(f"{ds.width}x{ds.height} exceeds max_image_dim={max_dim}")
        if ds.count < 3:
            bands = [ds.read(1)] * 3
        else:
            bands = [ds.read(b) for b in band_map]
        nod = ds.nodata
        u8, masks = zip(*[_to_uint8(b, nod) for b in bands])
        rgb = np.dstack(u8)
        valid = masks[0] & masks[1] & masks[2]
        if ds.count >= 4 and ds.colorinterp and any(str(ci).lower().endswith("alpha") for ci in ds.colorinterp):
            valid &= ds.read(ds.count) > 0
        crs = CRS.from_wkt(ds.crs.to_wkt())
        epsg = crs.to_epsg()
        grid = Grid(ds.width, ds.height, ds.transform, f"EPSG:{epsg}" if epsg else crs.to_wkt(), "uint8", None, "metres" if not crs.is_geographic else "relative", False, None, "R")
        tags = ds.tags()
        return GeoImage(rgb, None if valid.all() else valid, grid, crs.to_wkt(), ds.count, str(ds.dtypes[0]), tags.get("VERTICAL_CRS"))


def read_raster_on_grid(path: str | Path, grid: Grid, *, resampling: Resampling = Resampling.bilinear, band: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Reproject/resample band `band` of `path` onto `grid`. Returns (float32 array, valid mask)."""
    with rasterio.open(path) as src:
        dst = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    valid = np.isfinite(dst)
    return dst, valid


def write_raster(path: str | Path, array: np.ndarray, grid: Grid, tags: dict[str, Any] | None = None, *, dtype: str = "float32") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = grid.nodata
    arr = np.asarray(array)
    if dtype == "float32":
        arr = arr.astype(np.float32)
        if nodata is not None:
            arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=dtype, nodata=nodata if dtype == "float32" else 0, compress="lzw", tiled=False)
    if grid.transform is not None:
        profile["transform"] = grid.transform
    if grid.crs is not None:
        profile["crs"] = grid.crs
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
        base = {"UNITS": grid.units, "METRIC": str(grid.metric).lower(), "CALIBRATION_TIER": grid.tier, "VERTICAL_CRS": grid.vertical_reference or "none"}
        if tags:
            base.update({k: str(v) for k, v in tags.items()})
        ds.update_tags(**base)
    return path


def grid_bounds_wgs84(grid: Grid) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    b = grid.bounds
    assert b is not None and grid.crs is not None
    t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    xs, ys = t.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    return (min(xs), min(ys), max(xs), max(ys))


def identity_affine() -> Affine:
    return Affine.identity()
