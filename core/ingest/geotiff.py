"""Mode B ingestion: GeoTIFF with resolvable CRS and geotransform.

Applies Phase 8 C-4: geographic-CRS rasters are reprojected to the local UTM zone (recorded) before any
metric operation. Produces an `InputMeta`-like dict plus a Grid in a projected CRS.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS

from backend.errors import ImageTooLargeError, InvalidFileError
from core.geo.grid import Grid
from core.geo.raster_io import GeoImage, grid_bounds_wgs84, read_georeferenced_rgb
from core.geo.reproject import ensure_metric_grid


@dataclass
class GeoInputMeta:
    mode: str
    mode_label: str
    filename: str
    format: str
    width: int
    height: int
    channels: int
    dtype: str
    has_alpha: bool
    has_georeferencing: bool
    crs: str | None
    crs_is_geographic: bool
    reprojected: bool
    reprojection: dict[str, Any] | None
    transform: list[float]
    bounds: list[float]
    bounds_wgs84: list[float]
    gsd_m: float | None
    gsd_xy_m: list[float]
    vertical_crs_tag: str | None
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeoIngestResult:
    rgb: np.ndarray
    valid_mask: np.ndarray | None
    grid: Grid
    meta: GeoInputMeta
    source_path: Path  # path of the raster actually used (reprojected copy when C-4 applied)


def is_georeferenced_tiff(path: str | Path) -> bool:
    try:
        with rasterio.open(path) as ds:
            return ds.crs is not None and ds.transform is not None and not ds.transform.is_identity
    except Exception:  # noqa: BLE001
        return False


def ingest_geotiff(path: str | Path, work_dir: str | Path, *, max_dim: int = 4096) -> GeoIngestResult:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise InvalidFileError(f"file missing or empty: {p}")
    try:
        with rasterio.open(p) as ds:
            crs = CRS.from_wkt(ds.crs.to_wkt()) if ds.crs else None
            if crs is None or ds.transform is None or ds.transform.is_identity:
                raise InvalidFileError("GeoTIFF has no CRS/geotransform")
            if max(ds.width, ds.height) > max_dim:
                raise ImageTooLargeError(f"{ds.width}x{ds.height} exceeds max_image_dim={max_dim}")
            fmt = ds.driver
    except (InvalidFileError, ImageTooLargeError):
        raise
    except Exception as e:  # noqa: BLE001
        raise InvalidFileError(f"unreadable GeoTIFF: {e}") from e
    src_for_read = p
    reproj_rec = None
    if crs.is_geographic:
        out = Path(work_dir) / "input_utm.tif"
        _grid, rec = ensure_metric_grid(str(p), str(out))
        reproj_rec = rec.to_dict()
        src_for_read = out
    img: GeoImage = read_georeferenced_rgb(src_for_read, max_dim=max_dim)
    g = img.grid
    gsd = g.pixel_size
    meta = GeoInputMeta(
        mode="B",
        mode_label="MODE_B / GEOREFERENCED / METRIC_HORIZONTAL_SCALE",
        filename=p.name,
        format=fmt,
        width=g.width,
        height=g.height,
        channels=img.band_count,
        dtype=img.dtype,
        has_alpha=img.valid_mask is not None,
        has_georeferencing=True,
        crs=g.crs,
        crs_is_geographic=bool(crs.is_geographic),
        reprojected=reproj_rec is not None,
        reprojection=reproj_rec,
        transform=list(g.transform.to_gdal()) if g.transform else [],
        bounds=list(g.bounds) if g.bounds else [],
        bounds_wgs84=list(grid_bounds_wgs84(g)),
        gsd_m=float((gsd[0] * gsd[1]) ** 0.5) if gsd else None,
        gsd_xy_m=[float(gsd[0]), float(gsd[1])] if gsd else [],
        vertical_crs_tag=img.vertical_crs_tag,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        size_bytes=int(p.stat().st_size),
    )
    return GeoIngestResult(img.rgb, img.valid_mask, g, meta, src_for_read)
