"""Terrain layer (ground-weighted DEM reconstruction) and the relative object layer used as its ground mask.

Terrain layer (Phase 6 §7, validated synthetically in Phase 8 L0-04): the coarse DEM is sampled preferentially
where the image model sees bare ground, then reconstructed by normalized convolution so canopy/roof-contaminated
DEM cells are down-weighted; where support is absent the raw DEM is used and flagged.

The metric object layer itself (sub-DEM-posting detail) is produced by core.calib.fusion (tiled inference +
DEM-preserving detail fusion). The earlier global "DEM residual" object-scale fit was removed: measured against
LiDAR it degraded the DEM (docs/validation_results.md, 2026-09-26).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


@dataclass
class TerrainResult:
    terrain: np.ndarray  # float32 on job grid, NaN where DEM void
    ground_mask: np.ndarray  # bool on job grid
    support: np.ndarray  # coarse-cell ground support (0..1) upsampled to grid
    raw_fallback: np.ndarray  # bool where NC support < w_min (raw DEM used)
    params: dict[str, Any]
    stats: dict[str, Any]


@dataclass
class ObjectLayerResult:
    object_rel: np.ndarray  # relative object layer >= 0 (unitless)
    ground_mask: np.ndarray
    params: dict[str, Any]


# ----------------------------------------------------------------------------------------------
def object_layer_from_relative(rel: np.ndarray, valid: np.ndarray, *, gsd_m: float, ground_window_m: float = 60.0, ground_quantile: float = 0.5) -> ObjectLayerResult:
    """Morphological ground filter: object = rel - grey_opening(rel, window). Ground = object below a small threshold.
    Window (metres) should exceed typical building/tree footprints so they are removed by the opening."""
    r = np.where(valid, rel, np.nan).astype(np.float32)
    filled = np.where(np.isfinite(r), r, np.nanmedian(r)).astype(np.float32)
    w = max(3, int(round(ground_window_m / max(gsd_m, 1e-6))) | 1)
    ground_surface = ndimage.grey_opening(filled, size=(w, w))
    obj = np.clip(filled - ground_surface, 0, None)
    obj[~valid] = np.nan
    thr = float(np.nanquantile(obj[valid], ground_quantile)) if valid.any() else 0.0
    thr = max(thr, 1e-3)
    ground = np.isfinite(obj) & (obj <= thr)
    obj_clean = np.where(ground, 0.0, obj).astype(np.float32)
    obj_clean[~valid] = np.nan
    return ObjectLayerResult(obj_clean, ground, {"ground_window_m": ground_window_m, "window_px": w, "ground_threshold_rel": thr, "ground_fraction": float(ground[valid].mean()) if valid.any() else 0.0, "method": "grey_opening_ground_filter (heuristic on zero-shot relative structure)"})


def _block_reduce(a: np.ndarray, f: int, fn=np.nanmean) -> np.ndarray:
    h, w = a.shape
    hh, ww = -(-h // f), -(-w // f)
    pad = np.full((hh * f, ww * f), np.nan, np.float64)
    pad[:h, :w] = a
    with np.errstate(all="ignore"):
        return fn(pad.reshape(hh, f, ww, f), axis=(1, 3))


def _nc_plane(dem_c: np.ndarray, w: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Order-1 normalized convolution: per cell, weighted least-squares plane through neighbouring DEM cells
    (weights = Gaussian(sigma) x ground support). Returns (fitted value at the cell centre, weight sum).
    A plane fit preserves slopes where ground support is one-sided; a plain weighted mean would tilt them."""
    r = max(1, int(np.ceil(3 * sigma)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float64)
    G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    C = lambda a, k: ndimage.correlate(a, k, mode="constant", cval=0.0)  # noqa: E731  (correlate: C-2 orientation rule; zero weight outside the domain)
    z = np.nan_to_num(dem_c)
    S0, Sx, Sy = C(w, G), C(w, G * xx), C(w, G * yy)
    Sxx, Syy, Sxy = C(w, G * xx * xx), C(w, G * yy * yy), C(w, G * xx * yy)
    Sz, Sxz, Syz = C(w * z, G), C(w * z, G * xx), C(w * z, G * yy)
    A = np.stack([np.stack([S0, Sx, Sy], -1), np.stack([Sx, Sxx, Sxy], -1), np.stack([Sy, Sxy, Syy], -1)], -2)
    b = np.stack([Sz, Sxz, Syz], -1)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-9
    fit = np.where(S0 > 1e-9, Sz / np.maximum(S0, 1e-9), z)  # order-0 fallback
    if ok.any():
        sol = np.linalg.solve(A[ok], b[ok][..., None])[..., 0, 0]
        fit[ok] = sol
    return fit, S0


def terrain_layer(dem: np.ndarray, dem_valid: np.ndarray, ground: np.ndarray, *, gsd_m: float, dem_posting_m: float = 30.0, sigma_cells: float = 1.5, w_min: float = 0.1, max_raise_m: float = 0.5) -> TerrainResult:
    """Ground-masked normalized convolution of the DEM at DEM posting, applied as a *correction field* to the
    fine-resampled DEM so that no DEM relief is lost; where support is absent the raw DEM is used and flagged.
    The DEM (DSM-like) is treated as an upper bound: the correction may lower the surface, not raise it (> max_raise_m)."""
    f = max(1, int(round(dem_posting_m / max(gsd_m, 1e-6))))
    dem_c = _block_reduce(np.where(dem_valid, dem, np.nan), f)
    g = ground.astype(np.float64)
    g[~dem_valid] = 0.0
    w = np.nan_to_num(_block_reduce(g, f))  # ground support fraction per cell
    cell_valid = np.isfinite(dem_c)
    w = np.where(cell_valid, w, 0.0)
    fit, den = _nc_plane(dem_c, w, sigma_cells)
    corr_c = fit - np.nan_to_num(dem_c)
    corr_c = np.minimum(corr_c, max_raise_m)
    raw_fb_c = (den < w_min) & cell_valid
    corr_c[raw_fb_c | ~cell_valid] = 0.0
    zh, zw = dem.shape[0] / corr_c.shape[0], dem.shape[1] / corr_c.shape[1]
    corr = ndimage.zoom(corr_c, (zh, zw), order=1)[: dem.shape[0], : dem.shape[1]]
    terrain = (dem + corr).astype(np.float32)
    terrain[~dem_valid] = np.nan
    support = ndimage.zoom(w, (zh, zw), order=1)[: dem.shape[0], : dem.shape[1]].astype(np.float32)
    raw_fb = ndimage.zoom(raw_fb_c.astype(np.float32), (zh, zw), order=0)[: dem.shape[0], : dem.shape[1]] > 0.5
    stats_ = {"block_factor": f, "cells": [int(corr_c.shape[1]), int(corr_c.shape[0])], "support_mean": float(w[cell_valid].mean()) if cell_valid.any() else 0.0, "support_min": float(w[cell_valid].min()) if cell_valid.any() else 0.0, "raw_fallback_fraction": float(raw_fb_c[cell_valid].mean()) if cell_valid.any() else 0.0, "dem_void_fraction": float(1 - dem_valid.mean()), "correction_mean_m": float(corr_c[cell_valid].mean()) if cell_valid.any() else 0.0, "correction_min_m": float(corr_c[cell_valid].min()) if cell_valid.any() else 0.0, "method": "order1_normalized_convolution_correction (plane fit, DEM upper-bound clamp)"}
    return TerrainResult(terrain, ground, support, raw_fb, {"sigma_cells": sigma_cells, "w_min": w_min, "dem_posting_m": dem_posting_m, "max_raise_m": max_raise_m}, stats_)
