"""Mode A rDSM: turn relative inverse-depth into a stable, documented relative-height raster in [0, 1].

This normalisation creates NO physical scale. Outputs carry metric=false, tier="R", units="relative".
Orientation assumption (config `rdsm.orientation`): DA-V2 emits inverse-depth-like values (larger = nearer the
camera); for nadir imagery nearer = higher, so relative height is monotonically increasing in the raw value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from core.geo.grid import Grid


@dataclass
class RdsmStats:
    method: str
    percentiles: list[float] | None
    raw_min: float
    raw_max: float
    clip_low: float
    clip_high: float
    nodata: float
    valid_fraction: float
    orientation: str
    metric: bool = False
    calibration_tier: str = "R"
    units: str = "relative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_rdsm(relative_depth: np.ndarray, valid_mask: np.ndarray | None = None, *, method: str = "percentile", percentiles: tuple[float, float] = (1.0, 99.0), nodata: float = -9999.0, orientation: str = "higher_value_means_higher_surface") -> tuple[np.ndarray, Grid, RdsmStats]:
    z = np.asarray(relative_depth, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("relative_depth must be 2-D")
    valid = np.isfinite(z)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, bool)
    if valid.sum() < 4:
        raise ValueError("not enough valid pixels to normalise")
    vals = z[valid]
    if method == "percentile":
        lo, hi = np.percentile(vals, percentiles)
    elif method == "minmax":
        lo, hi = float(vals.min()), float(vals.max())
    else:
        raise ValueError(f"unknown normalisation method {method}")
    if hi <= lo:
        hi = lo + 1e-6
    rel = np.clip((z - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if orientation == "higher_value_means_lower_surface":
        rel = 1.0 - rel
    rel[~valid] = nodata
    grid = Grid.pixel_space(z.shape[1], z.shape[0], nodata=nodata)
    stats = RdsmStats(method, list(percentiles) if method == "percentile" else None, float(vals.min()), float(vals.max()), float(lo), float(hi), float(nodata), float(valid.mean()), orientation)
    return rel, grid, stats


def write_raster(path: str | Path, array: np.ndarray, grid: Grid, tags: dict[str, Any] | None = None) -> Path:
    """Write a float32 GeoTIFF (no CRS for Mode A) with nodata and provenance tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype="float32", nodata=grid.nodata, compress="lzw", tiled=False)
    if grid.transform is not None:
        profile["transform"] = grid.transform
    if grid.crs is not None:
        profile["crs"] = grid.crs
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array.astype(np.float32), 1)
        base = {"UNITS": grid.units, "METRIC": str(grid.metric).lower(), "CALIBRATION_TIER": grid.tier, "VERTICAL_CRS": grid.vertical_reference or "none"}
        if tags:
            base.update({k: str(v) for k, v in tags.items()})
        ds.update_tags(**base)
    return path


_RAMP = np.array([  # simple terrain-like ramp: low=dark blue -> green -> yellow -> brown -> white
    [30, 60, 120], [40, 120, 90], [110, 170, 60], [220, 200, 80], [170, 110, 60], [245, 245, 245]
], dtype=np.float32)


def colorize(rel: np.ndarray, nodata: float) -> np.ndarray:
    v = np.clip(np.where(rel == nodata, 0.0, rel), 0, 1)
    idx = v * (len(_RAMP) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, len(_RAMP) - 1)
    f = (idx - i0)[..., None]
    rgb = _RAMP[i0] * (1 - f) + _RAMP[i1] * f
    rgb[rel == nodata] = 0
    return rgb.astype(np.uint8)


def write_preview(path: str | Path, rel: np.ndarray, nodata: float, *, mode: str = "ramp", max_dim: int = 2048) -> Path:
    path = Path(path)
    arr = colorize(rel, nodata) if mode == "ramp" else (np.clip(np.where(rel == nodata, 0, rel), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    im.save(path)
    return path
