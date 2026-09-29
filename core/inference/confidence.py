"""Per-pixel confidence of the model's heights above ground (nDSM): "where can I trust this?".

Method (test-time augmentation, the same 4 exact-inverse orientations as the v2 training notebook):
    for f in {identity, h-flip, v-flip, rot180}:  h_f = f^-1( model( f(image) ) )     (same tiling as the job)
    spread(x) = std_f h_f(x)
The spread is turned into an error interval with a calibration table measured against LiDAR on VALIDATION tiles
(scripts/calibrate_confidence.py -> confidence_calibration.json): for each spread bin, the 80 % quantile of
|model nDSM - LiDAR nDSM|. So interval80(x) = Q80[bin(spread(x))]: "80 % of such pixels are within +/- this".
It costs 4 extra model passes, so it runs on demand, never inside the main job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION = Path(__file__).with_name("confidence_calibration.json")
FLIPS = [
    (lambda a: a, lambda a: a),
    (lambda a: a[:, ::-1], lambda a: a[:, ::-1]),
    (lambda a: a[::-1, :], lambda a: a[::-1, :]),
    (lambda a: a[::-1, ::-1], lambda a: a[::-1, ::-1]),
]


def tta_heights(rgb: np.ndarray, predictor: Any, gsd_m: float | None, fusion_cfg: Any, progress_cb=None) -> tuple[np.ndarray, np.ndarray]:
    """(mean, spread) of the stitched metric heights over the 4 orientations, on the job grid (Welford: 2 maps in RAM)."""
    from core.calib.fusion import plan_upsample, stitch_tiles, tiled_relative

    h, w = rgb.shape[:2]
    desired = (gsd_m / fusion_cfg.inference_gsd_m) if gsd_m else 1.0
    desired = float(min(max(desired, fusion_cfg.min_upsample), fusion_cfg.max_upsample))
    up = plan_upsample(h, w, desired, tile=fusion_cfg.tile_px, overlap=fusion_cfg.overlap, max_tiles=fusion_cfg.max_tiles)
    mean = m2 = None
    for k, (fwd, inv) in enumerate(FLIPS):
        tp = tiled_relative(lambda x: predictor.predict(x).relative_depth, np.ascontiguousarray(fwd(rgb)), upsample=up, tile=fusion_cfg.tile_px,
                            overlap=fusion_cfg.overlap, inference_gsd_m=(gsd_m / up) if gsd_m else None)
        p = np.ascontiguousarray(inv(np.clip(stitch_tiles(tp), 0.0, None))).astype(np.float64)
        del tp
        if mean is None:
            mean, m2 = p.copy(), np.zeros_like(p)
        else:
            d = p - mean
            mean += d / (k + 1)
            m2 += d * (p - mean)
        if progress_cb:
            progress_cb(k + 1, len(FLIPS))
    return mean.astype(np.float32), np.sqrt(m2 / len(FLIPS)).astype(np.float32)


def load_calibration() -> dict[str, Any] | None:
    return json.loads(CALIBRATION.read_text(encoding="utf-8")) if CALIBRATION.exists() else None


def interval_from_spread(spread: np.ndarray, cal: dict[str, Any], q: str = "80%") -> np.ndarray:
    edges = np.asarray(cal["spread_bin_edges_m"], float)
    qv = np.asarray([np.nan if v is None else v for v in cal["abs_error_quantiles_m"][q]], float)
    b = np.clip(np.searchsorted(edges, spread, side="right") - 1, 0, len(qv) - 1)
    out = qv[b].astype(np.float32)
    out[~np.isfinite(spread)] = np.nan
    return out
