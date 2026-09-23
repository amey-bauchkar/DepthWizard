"""Heightfield preparation for the 3D viewer (Sprint 1: dense regular grid, no RTIN yet).

The viewer receives (a) a float32 height grid and (b) the RGB texture; both describe the SAME image footprint
so texture coordinates map linearly (u = col/(W-1), v = row/(H-1)). If the raster is larger than
`max_mesh_dim`, heights are reduced by AREA-AVERAGE block downsampling (not pixel-skipping) and the factor is
recorded. Phase 8 L0-11 showed uniform decimation destroys building edges; this is a stop-gap until RTIN
(Sprint 6) — the factor and residual are exposed so nothing is hidden.

Viewer smoothing (viewer_smooth_sigma > 0): a NaN-aware Gaussian is applied to the heightfield *after*
downsampling, for display purposes only. The underlying GeoTIFF rasters are NEVER modified.
This eliminates blocky staircase artifacts from 30-m DEM upsampling without changing any measured value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


@dataclass
class HeightfieldMeta:
    width: int  # vertices per row
    height: int  # rows
    source_width: int
    source_height: int
    downsample_factor: int
    method: str
    nodata_value: str  # 'NaN' sentinel in the float32 binary (JSON cannot carry NaN)
    valid_fraction: float
    min: float
    max: float
    units: str
    metric: bool
    calibration_tier: str
    vertical_reference: str | None
    texture_width: int
    texture_height: int
    residual_vs_source: dict[str, float]
    dtype: str = "float32"
    byte_order: str = "little"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _block_sum_padded(a: np.ndarray, f: int) -> np.ndarray:
    """Sum over f x f blocks, padding the bottom/right edge with zeros so the FULL footprint is covered
    (ceil(h/f) x ceil(w/f) blocks). Callers divide by the padded valid-count to get a nodata-aware mean."""
    h, w = a.shape
    hh, ww = -(-h // f), -(-w // f)
    pad = np.zeros((hh * f, ww * f), dtype=np.float64)
    pad[:h, :w] = a
    return pad.reshape(hh, f, ww, f).sum(axis=(1, 3))


def _nansmooth(z: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smooth: fills NaNs with the weighted average of valid neighbours.
    Preserves NaN where no valid data exists within the kernel. Uses the normalised-convolution
    approach: smooth the data with zeros at NaN, smooth the mask, divide."""
    if sigma <= 0:
        return z
    mask = np.isfinite(z).astype(np.float64)
    z_filled = np.where(mask, z.astype(np.float64), 0.0)
    blurred = gaussian_filter(z_filled, sigma=sigma, mode="nearest")
    norm = gaussian_filter(mask, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore"):
        result = np.where(norm > 1e-6, blurred / norm, np.nan)
    return result.astype(np.float32)


def build_heightfield(
    rel: np.ndarray,
    nodata: float,
    *,
    max_mesh_dim: int = 768,
    units: str = "relative",
    metric: bool = False,
    tier: str = "R",
    vertical_reference: str | None = None,
    texture_size: tuple[int, int] = (0, 0),
    viewer_smooth_sigma: float = 0.0,
    guide_rgb: np.ndarray | None = None,
    edge_sharpen: bool = True,
) -> tuple[np.ndarray, np.ndarray, HeightfieldMeta]:
    """Return (heights float32 HxW with NaN for nodata, valid bool mask, meta).

    viewer_smooth_sigma: if > 0, apply a NaN-aware Gaussian of this sigma (in viewer-grid pixels)
    to the heightfield for display only. The underlying GeoTIFF is never modified.
    Recommended: 1.5 for Tier-T (terrain-only, 30m DEM upsampled), 0 for Tier-A/H (neural structure added).
    guide_rgb: if provided and edge_sharpen is True, snaps smoothed ViT elevation ramps to high-contrast
    building edges from the aerial photograph to form crisp roof plateaus and steep drops.
    """
    z = np.where(rel == nodata, np.nan, rel).astype(np.float32)
    h, w = z.shape
    f = 1
    while max(-(-h // f), -(-w // f)) > max_mesh_dim:
        f += 1
    if f > 1:
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(z).astype(np.float64)
            zs = _block_sum_padded(np.nan_to_num(z).astype(np.float64), f)
            vs = _block_sum_padded(valid, f)  # number of valid source pixels per block (edge blocks may be partial)
            zh = np.where(vs > 0, zs / np.maximum(vs, 1e-9), np.nan).astype(np.float32)
        # residual of the downsampled surface vs source (upsample by repetition, compare where valid)
        up = np.repeat(np.repeat(zh, f, axis=0), f, axis=1)[:h, :w]
        d = (up - z)[np.isfinite(up) & np.isfinite(z)]
        resid = {"rmse": float(np.sqrt(np.mean(d**2))) if d.size else 0.0, "max_abs": float(np.abs(d).max()) if d.size else 0.0}
        method = f"area_average_block_{f}"
    else:
        zh = z
        resid = {"rmse": 0.0, "max_abs": 0.0}
        method = "full_resolution"
    # Viewer-only smoothing: remove DEM staircase artifacts for display; never touches GeoTIFF data
    if viewer_smooth_sigma > 0:
        zh = _nansmooth(zh, viewer_smooth_sigma)
        method = f"{method}+viewer_gauss_{viewer_smooth_sigma:.1f}px"
    # Edge-snapping: snap ViT smoothed gradients to RGB building footprints
    if edge_sharpen and guide_rgb is not None:
        from core.dsm.guided import snap_heightfield_edges
        zh = snap_heightfield_edges(zh, guide_rgb, radius=5, eps=5e-4, blend=0.85)
        method = f"{method}+rgb_guided_snap"
    valid_mask = np.isfinite(zh)
    finite = zh[valid_mask]
    meta = HeightfieldMeta(zh.shape[1], zh.shape[0], w, h, f, method, "NaN", float(valid_mask.mean()), float(finite.min()) if finite.size else 0.0, float(finite.max()) if finite.size else 0.0, units, metric, tier, vertical_reference, texture_size[0], texture_size[1], resid)
    return zh, valid_mask, meta


def write_heightfield(path_bin: str | Path, heights: np.ndarray) -> Path:
    """float32 little-endian row-major, NaN = nodata."""
    p = Path(path_bin)
    heights.astype("<f4").tofile(p)
    return p


def write_texture(path: str | Path, rgb: np.ndarray, *, max_dim: int = 2048) -> tuple[Path, tuple[int, int]]:
    im = Image.fromarray(rgb, "RGB")
    if max(im.size) > max_dim:
        # keep aspect ratio EXACTLY proportional to the heightfield footprint
        scale = max_dim / max(im.size)
        im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))), Image.Resampling.LANCZOS)
    p = Path(path)
    im.save(p, quality=92)
    return p, im.size
