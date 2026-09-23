"""Elevation accuracy metrics (Phase 4 §12). Always report the SET: ME, RMSE, MAE, NMAD, LE90, LE95, r, rho.
Phase 8 L0-08: Pearson r = 1.000 with a 60 m bias -> correlation must never be reported alone."""
from __future__ import annotations

import numpy as np
from scipy import stats


def metric_set(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    p = np.asarray(pred, float).ravel()
    r = np.asarray(ref, float).ravel()
    m = np.isfinite(p) & np.isfinite(r)
    if mask is not None:
        m &= np.asarray(mask, bool).ravel()
    p, r = p[m], r[m]
    n = int(p.size)
    if n < 3:
        return {"n": n}
    d = p - r
    med = float(np.median(d))
    out = {
        "n": n,
        "ME": float(d.mean()),
        "RMSE": float(np.sqrt(np.mean(d**2))),
        "MAE": float(np.abs(d).mean()),
        "median": med,
        "NMAD": float(1.4826 * np.median(np.abs(d - med))),
        "LE90": float(np.percentile(np.abs(d), 90)),
        "LE95": float(np.percentile(np.abs(d), 95)),
    }
    if p.std() > 0 and r.std() > 0:
        out["pearson_r"] = float(stats.pearsonr(p, r)[0])
        out["spearman_rho"] = float(stats.spearmanr(p, r)[0])
    return out


def affine_fit(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Least-squares ref ~ s*pred + t. For Mode A (relative) outputs this is the ORACLE alignment and
    must be labelled as such — it is an upper bound, never 'calibration'."""
    p = np.asarray(pred, float).ravel()
    r = np.asarray(ref, float).ravel()
    m = np.isfinite(p) & np.isfinite(r)
    if mask is not None:
        m &= np.asarray(mask, bool).ravel()
    p, r = p[m], r[m]
    A = np.column_stack([p, np.ones_like(p)])
    (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)
    aligned = s * p + t
    return {"scale": float(s), "shift": float(t), "metrics_after_alignment": metric_set(aligned, r), "label": "oracle_affine_alignment"}
