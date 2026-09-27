"""Per-building statistics on the exact footprint pixel set (docs/math_audit.md §C, final specification §L).

Notation for one building B with pixel set P (its label in building_labels.tif), valid subset V = {i in P : nDSM_i and
T_i finite}, cell area A_px = |det J| (J = 2x2 linear part of the pixel->CRS affine):

    h_i        = nDSM_i = DSM_i - T_i                  height above the terrain layer, pixelwise (PREDICTED)
    A          = |P| * A_px                             footprint area, holes excluded (DERIVED)
    V          = A_px * sum_{i in V} h_i * |P| / |V|    integrated above-ground volume, invalid pixels imputed with the
                                                        valid mean (DERIVED)
    H_block    = V / A = mean_{i in V} h_i              LoD-1 block height: the block has exactly volume V (DERIVED)
    H_roof     = median_{i in V} h_i                    typical roof height, 50 % breakdown point (PREDICTED)
    Z_ground   = median_{i in V} T_i                    (REFERENCE-derived: DEM-based terrain layer)
    Z_roof     = median_{i in V} (T_i + h_i)            pixelwise first, then summarised (sloped sites!)
    ground plane: least squares of T_i on centred CRS coordinates -> ground_slope_deg = atan |grad|

The estimator choices are measured in docs/math_audit_results.md (synthetic roofs + Zürich LiDAR).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.screening_params import (
    FLOOR_HEIGHT_RANGE_M,
    MIN_FLOORS_HEIGHT_M,
    MIN_PIXELS_FOR_QUANTILES,
    MIN_VALID_FRACTION,
    PLANE_FIT_MAX_COND,
    PLANE_FIT_MIN_PIXELS,
)

STATS_VERSION = "building-stats-2"


def _r(v: float | None, d: int = 2) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), d)


def ground_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares plane z = a (x - x0) + b (y - y0) + c on coordinates centred at their mean (x0, y0).

    Centring matters: projected coordinates are ~1e6 m, and an uncentred [x, y, 1] design matrix has a condition number
    ~1e12 in float64, i.e. catastrophic cancellation. Returns (a, b, c) or None when fewer than 3 points or the centred
    design is (nearly) rank deficient (collinear pixel centres)."""
    if zs.size < PLANE_FIT_MIN_PIXELS:
        return None
    x0, y0 = float(xs.mean()), float(ys.mean())
    M = np.column_stack([xs - x0, ys - y0, np.ones_like(xs)])
    s = np.linalg.svd(M, compute_uv=False)
    if s[-1] <= 0 or s[0] / s[-1] > PLANE_FIT_MAX_COND:
        return None
    (a, b, c), *_ = np.linalg.lstsq(M, zs, rcond=None)
    return float(a), float(b), float(c)


def floors_range(h_roof: float) -> list[int]:
    lo, hi = FLOOR_HEIGHT_RANGE_M
    if not math.isfinite(h_roof) or h_roof < MIN_FLOORS_HEIGHT_M:
        return [0, 0]
    return [max(1, math.floor(h_roof / hi)), max(1, math.ceil(h_roof / lo))]


def footprint_stats(ndsm: np.ndarray, terrain: np.ndarray, fp: np.ndarray, *, transform, row0: int = 0, col0: int = 0, touches_edge: bool = False) -> dict[str, Any] | None:
    """Statistics of one footprint. ndsm/terrain/fp are same-shape windows; (row0, col0) is the window origin in the
    full raster; transform is the full raster's pixel->CRS affine. Returns None if no valid pixel exists."""
    t = transform
    a_px = abs(t.a * t.e - t.b * t.d)
    n_total = int(fp.sum())
    valid = fp & np.isfinite(ndsm) & np.isfinite(terrain)
    n_valid = int(valid.sum())
    if n_total == 0 or n_valid == 0:
        return None
    h = ndsm[valid].astype(np.float64)
    T = terrain[valid].astype(np.float64)
    area = n_total * a_px
    h_block = float(h.mean())
    volume = area * h_block  # = A_px * sum(h) * N / N_valid
    h_roof = float(np.median(h))
    p10, p90 = (float(v) for v in np.percentile(h, [10, 90]))
    rows, cols = np.nonzero(valid)
    cx, cy = cols + col0 + 0.5, rows + row0 + 0.5  # pixel centres
    xs, ys = t.a * cx + t.b * cy + t.c, t.d * cx + t.e * cy + t.f
    plane = ground_plane(xs, ys, T)
    flags = []
    if n_valid < MIN_PIXELS_FOR_QUANTILES:
        flags.append("SMALL_SAMPLE")
    if n_valid / n_total < MIN_VALID_FRACTION:
        flags.append("LOW_VALID_FRACTION")
    if touches_edge:
        flags.append("TRUNCATED_BY_RASTER_EDGE")  # area and volume are lower bounds
    if plane is None:
        flags.append("GROUND_TILT_UNDEFINED")
    return {
        "height_m": _r(h_block),  # LoD-1 extrusion height (block volume == volume_m3)
        "height_block_m": _r(h_block),
        "height_median_m": _r(h_roof),
        "height_p10_m": _r(p10),
        "height_p90_m": _r(p90),
        "height_nmad_m": _r(1.4826 * float(np.median(np.abs(h - h_roof)))),
        "base_elev_m": _r(float(np.median(T))),
        "ground_p10_m": _r(float(np.percentile(T, 10))),
        "ground_min_m": _r(float(T.min())),
        "ground_max_m": _r(float(T.max())),
        "ground_slope_deg": _r(math.degrees(math.atan(math.hypot(plane[0], plane[1]))), 2) if plane else None,
        "roof_elev_m": _r(float(np.median(T + h))),
        "area_m2": _r(area, 2),  # N x cell area: exact at 2 dp for 0.5 m cells (0.25 m^2)
        "volume_m3": _r(volume, 1),
        "n_pixels": n_valid,
        "n_footprint_pixels": n_total,
        "valid_fraction": _r(n_valid / n_total, 3),
        "floors_range": floors_range(h_roof),
        "quality_flags": flags,
    }
