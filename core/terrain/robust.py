"""Robust location estimators for per-building height statistics.

All functions take a 1-D array of finite samples (metres) and return a float (metres), or NaN for an empty
sample. They are pure and dimension-preserving: est(c * h) = c * est(h) for c > 0 and est(h + k) = est(h) + k
(scale and translation equivariance), which tests/unit/test_building_math.py checks.
"""
from __future__ import annotations

import numpy as np

NMAD_K = 1.4826  # consistency constant: NMAD = 1.4826 * MAD estimates sigma for Gaussian data (1 / Phi^-1(0.75))


def median(h: np.ndarray) -> float:
    return float(np.median(h)) if h.size else float("nan")


def mean(h: np.ndarray) -> float:
    return float(np.mean(h)) if h.size else float("nan")


def percentile(h: np.ndarray, q: float) -> float:
    return float(np.percentile(h, q)) if h.size else float("nan")


def trimmed_mean(h: np.ndarray, lo_q: float = 10.0, hi_q: float = 90.0) -> float:
    """Mean of the samples between the lo_q and hi_q percentiles (inclusive)."""
    if not h.size:
        return float("nan")
    lo, hi = np.percentile(h, [lo_q, hi_q])
    s = h[(h >= lo) & (h <= hi)]
    return float(s.mean()) if s.size else float(np.median(h))


def nmad(h: np.ndarray) -> float:
    """Normalised median absolute deviation (robust sigma)."""
    return float(NMAD_K * np.median(np.abs(h - np.median(h)))) if h.size else float("nan")


def huber(h: np.ndarray, c: float = 1.345, tol: float = 1e-6, max_iter: int = 50) -> float:
    """Huber M-estimate of location with scale fixed at NMAD, by iteratively re-weighted least squares.

    c = 1.345 gives 95 % asymptotic efficiency relative to the mean under Gaussian errors (Huber 1981).
    With NMAD == 0 (more than half the samples identical) the median is already the exact answer and is returned.
    """
    if not h.size:
        return float("nan")
    mu = float(np.median(h))
    s = nmad(h)
    if s <= 0:
        return mu
    for _ in range(max_iter):
        r = (h - mu) / s
        w = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
        new = float(np.sum(w * h) / np.sum(w))
        if abs(new - mu) < tol * max(1.0, abs(mu)):
            return new
        mu = new
    return mu
