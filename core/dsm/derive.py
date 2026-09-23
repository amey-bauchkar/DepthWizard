"""Derived layers from an elevation raster: slope/aspect (Horn, metres), flags bitmask, previews."""
from __future__ import annotations

import numpy as np
from PIL import Image

from core.validate.coregister import slope_aspect_deg

FLAG_BORDER = 1
FLAG_TERRAIN_RAW_DEM = 2
FLAG_DEM_VOID = 4
FLAG_NODATA = 8
FLAG_NO_OBJECT_SCALE = 16
FLAG_LOW_SUPPORT = 32
FLAG_BITS = {"BORDER": FLAG_BORDER, "TERRAIN_RAW_DEM": FLAG_TERRAIN_RAW_DEM, "DEM_VOID": FLAG_DEM_VOID, "NODATA": FLAG_NODATA, "NO_OBJECT_SCALE": FLAG_NO_OBJECT_SCALE, "LOW_SUPPORT": FLAG_LOW_SUPPORT}


def slope_layers(dsm: np.ndarray, gsd_x: float, gsd_y: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.where(np.isfinite(dsm), dsm, np.nanmedian(dsm)).astype(np.float64)
    s, a = slope_aspect_deg(z, gsd_x, gsd_y)
    s = s.astype(np.float32)
    a = a.astype(np.float32)
    s[~np.isfinite(dsm)] = np.nan
    a[~np.isfinite(dsm)] = np.nan
    return s, a


def build_flags(shape: tuple[int, int], *, valid: np.ndarray, border_px: int = 2, raw_fallback: np.ndarray | None = None, dem_void: np.ndarray | None = None, no_object_scale: bool = False, low_support: np.ndarray | None = None) -> np.ndarray:
    f = np.zeros(shape, np.uint16)
    f[:border_px, :] |= FLAG_BORDER
    f[-border_px:, :] |= FLAG_BORDER
    f[:, :border_px] |= FLAG_BORDER
    f[:, -border_px:] |= FLAG_BORDER
    f[~valid] |= FLAG_NODATA
    if raw_fallback is not None:
        f[raw_fallback] |= FLAG_TERRAIN_RAW_DEM
    if dem_void is not None:
        f[dem_void] |= FLAG_DEM_VOID
    if no_object_scale:
        f |= FLAG_NO_OBJECT_SCALE
    if low_support is not None:
        f[low_support] |= FLAG_LOW_SUPPORT
    return f


_RAMP = np.array([[30, 60, 120], [40, 120, 90], [110, 170, 60], [220, 200, 80], [170, 110, 60], [245, 245, 245]], dtype=np.float32)
_SLOPE_RAMP = np.array([[240, 248, 255], [173, 216, 230], [255, 236, 130], [255, 140, 0], [180, 0, 0]], dtype=np.float32)


def _apply_ramp(v01: np.ndarray, ramp: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    idx = np.clip(np.nan_to_num(v01), 0, 1) * (len(ramp) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, len(ramp) - 1)
    fr = (idx - i0)[..., None]
    rgb = ramp[i0] * (1 - fr) + ramp[i1] * fr
    rgb[invalid] = 0
    return rgb.astype(np.uint8)


def elevation_preview(path, z: np.ndarray, *, lo: float | None = None, hi: float | None = None, max_dim: int = 2048) -> dict:
    v = np.isfinite(z)
    if lo is None or hi is None:
        lo, hi = (float(np.percentile(z[v], 1)), float(np.percentile(z[v], 99))) if v.any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    rgb = _apply_ramp((z - lo) / (hi - lo), _RAMP, ~v)
    im = Image.fromarray(rgb)
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim))
    im.save(path)
    return {"lo": lo, "hi": hi, "ramp": "terrain (dark-blue low -> white high)"}


def slope_preview(path, slope_deg: np.ndarray, *, max_deg: float = 45.0, max_dim: int = 2048) -> dict:
    v = np.isfinite(slope_deg)
    rgb = _apply_ramp(slope_deg / max_deg, _SLOPE_RAMP, ~v)
    im = Image.fromarray(rgb)
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim))
    im.save(path)
    return {"lo": 0.0, "hi": max_deg, "ramp": "slope 0 (white) -> 45+ deg (dark red)"}


def hillshade(z: np.ndarray, gsd_x: float, gsd_y: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    from core.validate.coregister import horn_gradients

    zz = np.where(np.isfinite(z), z, np.nanmedian(z)).astype(np.float64)
    p, q = horn_gradients(zz, 1.0)
    p /= gsd_x
    q /= gsd_y
    slope = np.arctan(np.hypot(p, q))
    aspect = np.arctan2(-p, q)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1)
    hs[~np.isfinite(z)] = np.nan
    return hs.astype(np.float32)


def hillshade_preview(path, z: np.ndarray, gsd_x: float, gsd_y: float, *, max_dim: int = 2048) -> None:
    hs = hillshade(z, gsd_x, gsd_y)
    g = (np.nan_to_num(hs) * 255).astype(np.uint8)
    im = Image.fromarray(np.dstack([g, g, g]))
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim))
    im.save(path)


def flags_preview(path, flags: np.ndarray, *, max_dim: int = 2048) -> None:
    h, w = flags.shape
    rgb = np.zeros((h, w, 4), np.uint8)
    rgb[(flags & FLAG_TERRAIN_RAW_DEM) > 0] = (255, 0, 255, 140)
    rgb[(flags & FLAG_LOW_SUPPORT) > 0] = (255, 160, 0, 110)
    rgb[(flags & FLAG_DEM_VOID) > 0] = (255, 0, 0, 160)
    rgb[(flags & FLAG_BORDER) > 0] = (255, 255, 255, 90)
    rgb[(flags & FLAG_NODATA) > 0] = (0, 0, 0, 200)
    im = Image.fromarray(rgb, "RGBA")
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim))
    im.save(path)
