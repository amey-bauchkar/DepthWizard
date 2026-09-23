"""Robust sparse-anchor estimation (Phase 8 C-3).

Evidence (results/l0_05_anchors.json): with 40 % gross outliers the Huber estimator failed (10.99 m vs 3.0 m
truth) while the median recovered the offset within 0.05 m; RANSAC recovered a 1.2x scale within 1 % for N >= 5.
Therefore: median offset, RANSAC scale, N >= 5 minimum, 3-sigma (NMAD) blunder flagging, hold-out reporting.

Not wired into the Sprint 1 pipeline (Mode A has no anchors); provided with tests so Sprint 2/3 build on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_ANCHORS = 5


@dataclass
class OffsetFit:
    offset: float
    n_used: int
    nmad: float
    blunder_idx: list[int] = field(default_factory=list)
    method: str = "median"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ScaleFit:
    scale: float
    n_inliers: int
    n_total: int
    residual_threshold: float
    method: str = "ransac_through_origin"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def robust_offset(residuals: np.ndarray, *, blunder_k: float = 3.0, min_n: int = MIN_ANCHORS) -> OffsetFit:
    """Median offset with NMAD-based blunder flags. residuals = z_anchor - z_predicted (ground anchors)."""
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    if r.size < min_n:
        raise ValueError(f"need at least {min_n} anchors, got {r.size}")
    med = float(np.median(r))
    nmad = float(1.4826 * np.median(np.abs(r - med)))
    thr = blunder_k * max(nmad, 1e-6)
    blunders = [int(i) for i in np.where(np.abs(r - med) > thr)[0]]
    return OffsetFit(offset=med, n_used=int(r.size), nmad=nmad, blunder_idx=blunders)


def ransac_scale(x: np.ndarray, y: np.ndarray, *, residual_threshold: float = 1.5, min_n: int = MIN_ANCHORS, n_iter: int = 200, seed: int = 0) -> ScaleFit:
    """Fit y = a*x (through origin) robustly. x = predicted object height, y = anchor height above terrain."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    x, y = x[m], y[m]
    n = x.size
    if n < min_n:
        raise ValueError(f"need at least {min_n} object anchors, got {n}")
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    k = max(2, n // 2)
    for _ in range(n_iter):
        idx = rng.choice(n, k, replace=False)
        a = float(np.sum(x[idx] * y[idx]) / np.sum(x[idx] ** 2))
        inl = np.abs(y - a * x) <= residual_threshold
        if inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl.sum() < 2:
        best_inl = np.ones(n, bool)
    a = float(np.sum(x[best_inl] * y[best_inl]) / np.sum(x[best_inl] ** 2))
    return ScaleFit(scale=a, n_inliers=int(best_inl.sum()), n_total=int(n), residual_threshold=residual_threshold)


def split_holdout(n: int, holdout_fraction: float = 0.3, *, seed: int = 0, min_n_for_split: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (fit_idx, holdout_idx). Below min_n_for_split everything is used for fitting (holdout empty) — reported as such."""
    idx = np.arange(n)
    if n < min_n_for_split:
        return idx, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = int(round(holdout_fraction * n))
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])
