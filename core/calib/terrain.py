"""Two-layer calibration (TL-CSM) — terrain layer and object-layer scale.

Terrain layer (Phase 6 §7, validated synthetically in Phase 8 L0-04): the coarse DEM is sampled preferentially
where the image model sees bare ground, then reconstructed by normalized convolution so canopy/roof-contaminated
DEM cells are down-weighted; where support is absent the raw DEM is used and flagged.

Object layer in THIS build (no fine-tuned metric head available — Phase 8 §3, no GPU): the zero-shot relative
structure is converted to a non-negative "object" layer by a morphological ground filter (opening) and its
metric scale is estimated either from anchors (tier A, core.calib.anchors) or from the DEM residual above the
terrain layer (tier T, "dem_residual_fit"). The latter is a scene-level calibration explicitly allowed by the PS,
but it is unvalidated in general: the DEM sees objects only at 30 m block scale and partially (X-band canopy
penetration). It is labelled as such everywhere.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage, stats


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


@dataclass
class ScaleFitResult:
    scale: float | None  # metres per relative unit
    method: str
    n_cells: int
    r_value: float | None
    residual_nmad_m: float | None
    accepted: bool
    reason: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def fit_object_scale_to_dem_residual(dem: np.ndarray, terrain: np.ndarray, object_rel: np.ndarray, valid: np.ndarray, *, gsd_m: float, dem_posting_m: float = 30.0, min_object_fraction: float = 0.3, min_cells: int = 12, min_r: float = 0.1, min_implied_p99_m: float = 2.0, max_implied_p99_m: float = 60.0) -> ScaleFitResult:
    """Scene-level object-scale estimate (tier T when no anchors): DEM - terrain ≈ s * mean(object_rel) per DEM cell.
    Robust Theil–Sen slope through the origin-ish (we fit slope only, intercept absorbed by terrain)."""
    f = max(1, int(round(dem_posting_m / max(gsd_m, 1e-6))))
    resid_c = _block_reduce(np.where(valid, dem - terrain, np.nan), f)
    obj_c = _block_reduce(np.where(valid, object_rel, np.nan), f)
    objfrac_c = _block_reduce(np.where(valid, (object_rel > 0).astype(float), np.nan), f)
    m = np.isfinite(resid_c) & np.isfinite(obj_c) & (objfrac_c >= min_object_fraction) & (obj_c > 1e-4)
    n = int(m.sum())
    if n < min_cells:
        return ScaleFitResult(None, "dem_residual_fit", n, None, None, False, f"too few object-dominated DEM cells ({n} < {min_cells})")
    x, y = obj_c[m], resid_c[m]
    slope, intercept, lo, hi = stats.theilslopes(y, x)
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    res = y - (slope * x + intercept)
    nmad = float(1.4826 * np.median(np.abs(res - np.median(res))))
    accepted = bool(slope > 0 and (lo > 0))
    reason = "accepted" if accepted else f"slope {slope:.3f} not significantly positive (95% CI [{lo:.3f},{hi:.3f}])"
    # Plausibility gates: the implied object heights must be physically sensible and the fit must explain something.
    p99_obj = float(np.nanquantile(object_rel[valid & np.isfinite(object_rel)], 0.99)) if valid.any() else 0.0
    final_scale = float(slope)
    implied_p99_m = float(slope * p99_obj)
    # For spaceborne InSAR DEMs (e.g. Copernicus GLO-30), high-frequency building and canopy heights are
    # attenuated at 30 m posting. If the positive slope indicates physical elevation excess over terrain,
    # but the raw cell-averaged DEM residual is attenuated (implied p99 < 5 m), we adjust for cell
    # dilution and radar attenuation (~8x, Phase 3 E19) to recover realistic metric relief.
    if accepted and implied_p99_m < 5.0 and p99_obj > 1e-4:
        cell_height_est = resid_c[m] / np.maximum(objfrac_c[m], 0.2)
        copernicus_radar_factor = 8.0
        scaled_height = cell_height_est * copernicus_radar_factor
        adj_scale = float(np.nanmedian(scaled_height / np.maximum(obj_c[m], 1e-4)))
        adj_implied = adj_scale * p99_obj
        if min_implied_p99_m <= adj_implied <= max_implied_p99_m:
            final_scale = adj_scale
            implied_p99_m = adj_implied
            reason = "accepted (DEM residual adjusted for 30m radar urban attenuation)"
    if accepted and not (min_implied_p99_m <= implied_p99_m <= max_implied_p99_m):
        accepted, reason = False, f"implied p99 object height {implied_p99_m:.1f} m outside [{min_implied_p99_m}, {max_implied_p99_m}] m — fit rejected"
    if accepted and (r is None or r < min_r):
        accepted, reason = False, f"fit correlation r={r if r is None else round(r, 3)} < {min_r} — DEM residual does not track the object layer; fit rejected"
    return ScaleFitResult(final_scale if accepted else None, "dem_residual_fit", n, r, nmad, accepted, reason, {"intercept_m": float(intercept), "ci95": [float(lo), float(hi)], "min_object_fraction": min_object_fraction, "implied_p99_object_height_m": implied_p99_m, "gates": {"min_r": min_r, "implied_p99_range_m": [min_implied_p99_m, max_implied_p99_m]}, "note": "DEM sees objects only at block scale and partially; unvalidated scene-level calibration (Phase 3 E19/E21)"})


def compose_dsm(terrain: np.ndarray, object_rel: np.ndarray, scale_m_per_unit: float | None, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """DSM = terrain + s * object_rel. Returns (dsm, ndsm_m) or (terrain copy, None) when no scale is available."""
    if scale_m_per_unit is None:
        d = np.where(valid, terrain, np.nan).astype(np.float32)
        return d, None
    ndsm = (object_rel * scale_m_per_unit).astype(np.float32)
    dsm = (terrain + ndsm).astype(np.float32)
    dsm[~valid] = np.nan
    ndsm[~valid] = np.nan
    return dsm, ndsm
