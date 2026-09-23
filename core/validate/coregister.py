"""Horizontal co-registration of two elevation rasters (Phase 8 C-2).

Phase 8 findings baked in:
  * Horn gradients must be computed with `scipy.ndimage.correlate` (kernel applied as written).
    `convolve` flips the kernel, inverts the gradient sign and makes the iterative estimator DIVERGE
    (results/l0_07c vs l0_07d).
  * A single gradient step is attenuated by regression dilution when the reference is noisy;
    the estimator therefore iterates to convergence with reference smoothing.
  * A deterministic grid search + parabolic sub-pixel refinement is provided as the robust alternative.

Sign convention: a positive result (dx, dy) means `pred` is displaced by +dx columns and +dy rows
relative to `ref` (image axes: columns east, rows south). To align, shift `pred` by (-dx, -dy).
Synthetic regression fixture: true (2, -1) px -> Phase 8 recovered (2.000, -1.012) iterative, (1.979, -1.020) grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

HORN_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8.0  # d/d(col)
HORN_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float) / 8.0  # d/d(row)


def horn_gradients(z: np.ndarray, gsd: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, q) = (dz/dx, dz/dy) per metre using Horn's kernels applied as correlation."""
    p = ndimage.correlate(z, HORN_KX, mode="nearest") / gsd
    q = ndimage.correlate(z, HORN_KY, mode="nearest") / gsd
    return p, q


def slope_aspect_deg(z: np.ndarray, gsd_x: float, gsd_y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north) — Horn method. Edge pixels use
    'nearest' padding and should be flagged (Phase 8 L0-06: up to ~3 deg edge error)."""
    gsd_y = gsd_x if gsd_y is None else gsd_y
    p = ndimage.correlate(z, HORN_KX, mode="nearest") / gsd_x
    q = ndimage.correlate(z, HORN_KY, mode="nearest") / gsd_y
    slope = np.degrees(np.arctan(np.hypot(p, q)))
    aspect = (np.degrees(np.arctan2(p, -q)) + 360.0) % 360.0
    return slope, aspect


@dataclass
class ShiftResult:
    dx: float
    dy: float
    method: str
    iterations: int
    converged: bool
    rmse_before: float
    rmse_after: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = (a - b)[mask]
    return float(np.sqrt(np.mean(d**2))) if d.size else float("nan")


def _shift_image(img: np.ndarray, dx: float, dy: float, order: int = 1) -> np.ndarray:
    """Translate image content by (+dx cols, +dy rows)."""
    return ndimage.shift(img, (dy, dx), order=order, mode="nearest")


def _valid_mask(pred: np.ndarray, ref: np.ndarray, border: int) -> np.ndarray:
    m = np.isfinite(pred) & np.isfinite(ref)
    if border > 0:
        m[:border] = m[-border:] = False
        m[:, :border] = m[:, -border:] = False
    return m


def estimate_shift_gradient(pred: np.ndarray, ref: np.ndarray, *, smooth_sigma: float = 2.0, max_iter: int = 20, tol_px: float = 1e-3, min_slope: float = 0.02, border: int = 8) -> ShiftResult:
    """Iterative linearised-gradient (Nuth–Kääb-type) shift estimate. Returns displacement of pred w.r.t. ref."""
    mask0 = _valid_mask(pred, ref, border)
    r_s = ndimage.gaussian_filter(np.nan_to_num(ref), smooth_sigma) if smooth_sigma > 0 else np.nan_to_num(ref)
    p, q = horn_gradients(r_s, 1.0)
    m = mask0 & (np.hypot(p, q) > min_slope)
    if m.sum() < 50:
        return ShiftResult(0.0, 0.0, "gradient_iterative", 0, False, _rmse(pred, ref, mask0), _rmse(pred, ref, mask0))
    A = np.column_stack([-p[m], -q[m], np.ones(int(m.sum()))])
    acc = np.array([0.0, 0.0])
    cur = pred
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        dh = np.nan_to_num(cur - ref)
        coef, *_ = np.linalg.lstsq(A, dh[m], rcond=None)
        step = coef[:2]
        acc += step
        cur = _shift_image(np.nan_to_num(pred), -acc[0], -acc[1], order=3)
        if np.all(np.abs(step) < tol_px):
            converged = True
            break
    return ShiftResult(float(acc[0]), float(acc[1]), "gradient_iterative", it, converged, _rmse(pred, ref, mask0), _rmse(cur, ref, mask0))


def estimate_shift_grid(pred: np.ndarray, ref: np.ndarray, *, max_shift: int = 4, border: int = 8) -> ShiftResult:
    """Deterministic integer grid search minimising RMSE, then parabolic sub-pixel refinement."""
    mask0 = _valid_mask(pred, ref, border)
    P = np.nan_to_num(pred)
    cost: dict[tuple[int, int], float] = {}
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            cost[(dx, dy)] = _rmse(_shift_image(P, -dx, -dy, order=1), ref, mask0)
    (bx, by) = min(cost, key=cost.get)

    def parab(fm: float, f0: float, fp: float) -> float:
        d = fm - 2 * f0 + fp
        return 0.0 if d == 0 else 0.5 * (fm - fp) / d

    sx = parab(cost[(bx - 1, by)], cost[(bx, by)], cost[(bx + 1, by)]) if abs(bx) < max_shift else 0.0
    sy = parab(cost[(bx, by - 1)], cost[(bx, by)], cost[(bx, by + 1)]) if abs(by) < max_shift else 0.0
    dx, dy = bx + sx, by + sy
    after = _rmse(_shift_image(P, -dx, -dy, order=3), ref, mask0)
    return ShiftResult(float(dx), float(dy), "grid_search_subpixel", 1, True, cost[(0, 0)], after)
