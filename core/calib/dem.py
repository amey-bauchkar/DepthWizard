"""DEM discovery and loading for Mode B calibration.

Bundled DEM tiles live under assets/dem with an index (or are discovered by filename convention for Copernicus
GLO-30 COGs: Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif -> 1x1 degree cell). Copernicus GLO-30 heights are
EGM2008 (EPSG:3855) [Phase 3 E3]; SRTM/AW3D30 are EGM96. The output vertical reference defaults to EGM2008;
any other DEM datum is converted through the Phase 8 C-1 guarded transformer.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from core.geo.grid import Grid
from core.geo.raster_io import grid_bounds_wgs84, read_raster_on_grid
from core.geo.vertical import transform_heights

COP_RE = re.compile(r"Copernicus_DSM_COG_(\d+)_([NS])(\d+)_00_([EW])(\d+)_00_DEM\.tif$")


@dataclass
class DemSource:
    name: str
    product: str
    files: list[str]
    vertical_crs: str  # "EGM2008" | "EGM96" | "ellipsoidal"
    posting_m: float
    accuracy_note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _cop_tile_bounds(fn: str) -> tuple[float, float, float, float] | None:
    m = COP_RE.search(fn)
    if not m:
        return None
    lat = int(m.group(3)) * (1 if m.group(2) == "N" else -1)
    lon = int(m.group(5)) * (1 if m.group(4) == "E" else -1)
    return (lon, lat, lon + 1, lat + 1)


def discover_dem(bounds_wgs84: tuple[float, float, float, float], dem_dir: Path, *, user_dem: Path | None = None, user_dem_vcrs: str = "EGM2008") -> DemSource | None:
    """Return the DEM source covering the AOI: a user-supplied DEM wins; else bundled Copernicus tiles (index.json or filename)."""
    if user_dem is not None and Path(user_dem).exists():
        posting = 0.0
        try:
            with rasterio.open(user_dem) as ds:
                px = abs(ds.transform.a)
                posting = float(px * 111320.0) if ds.crs and ds.crs.is_geographic else float(px)  # approx. metres for geographic CRS
        except Exception:  # noqa: BLE001 - reported later when the DEM is actually read
            posting = 0.0
        return DemSource("user", "user-supplied DEM", [str(user_dem)], user_dem_vcrs, posting, "unknown (user-supplied)")
    dem_dir = Path(dem_dir)
    if not dem_dir.exists():
        return None
    idx = dem_dir / "index.json"
    entries: list[tuple[str, tuple[float, float, float, float], dict]] = []
    if idx.exists():
        for e in json.loads(idx.read_text(encoding="utf-8")).get("tiles", []):
            entries.append((str(dem_dir / e["file"]), tuple(e["bounds_wgs84"]), e))
    else:
        for f in dem_dir.glob("*.tif"):
            b = _cop_tile_bounds(f.name)
            if b:
                entries.append((str(f), b, {"product": "Copernicus GLO-30", "vertical_crs": "EGM2008", "posting_m": 30.0}))
    w, s, e_, n = bounds_wgs84
    hits = [(f, meta) for f, b, meta in entries if not (b[2] <= w or b[0] >= e_ or b[3] <= s or b[1] >= n)]
    if not hits:
        return None
    meta = hits[0][1]
    return DemSource("bundled", meta.get("product", "Copernicus GLO-30"), [f for f, _ in hits], meta.get("vertical_crs", "EGM2008"), float(meta.get("posting_m", 30.0)), "Copernicus GLO-30: DSM-like; forest MAE ~5 m, built-up ~1.6 m before de-biasing (Phase 3 E19); DEMIX best 1-arcsec global DEM")


def load_dem_on_grid(src: DemSource, grid: Grid, out_vcrs: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mosaic the DEM tiles onto the job grid (bilinear), convert vertical datum to `out_vcrs` via the C-1 guard.
    Returns (dem float32 on grid with NaN voids, valid mask, provenance)."""
    dem = np.full((grid.height, grid.width), np.nan, np.float32)
    for f in src.files:
        arr, valid = read_raster_on_grid(f, grid, resampling=Resampling.bilinear)
        fill = np.isnan(dem) & valid
        dem[fill] = arr[fill]
    valid = np.isfinite(dem)
    prov = {"source": src.to_dict(), "src_vertical_crs": src.vertical_crs, "dst_vertical_crs": out_vcrs, "datum_transformed": False, "void_fraction": float(1 - valid.mean())}
    if src.vertical_crs != out_vcrs:
        # transform a sparse set of nodes then interpolate (undulation is very smooth); guard raises if unsafe
        from pyproj import Transformer

        step = max(1, min(grid.width, grid.height) // 16)
        rows = np.arange(0, grid.height, step)
        cols = np.arange(0, grid.width, step)
        cc, rr = np.meshgrid(cols, rows)
        xs, ys = grid.transform * (cc + 0.5, rr + 0.5)  # type: ignore[operator]
        t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(xs, ys)
        z0 = np.zeros_like(lon)
        z1 = transform_heights(lon, lat, z0, src.vertical_crs, out_vcrs)
        corr = (z1 - z0).astype(np.float32)  # height correction field at nodes
        from scipy.ndimage import zoom

        corr_full = zoom(corr, (grid.height / corr.shape[0], grid.width / corr.shape[1]), order=1)[: grid.height, : grid.width]
        dem = dem + corr_full
        prov["datum_transformed"] = True
        prov["datum_correction_m"] = {"min": float(corr.min()), "max": float(corr.max())}
    return dem, valid, prov


def grid_bounds(grid: Grid) -> tuple[float, float, float, float]:
    return grid_bounds_wgs84(grid)
