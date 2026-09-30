"""Speed-ups of the preview writers must not change a single output byte."""
import numpy as np

from core.dsm.derive import _RAMP, _SLOPE_RAMP, _apply_ramp


def _reference_ramp(v01, ramp, invalid):  # the implementation before the per-channel rewrite
    idx = np.clip(np.nan_to_num(v01), 0, 1) * (len(ramp) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, len(ramp) - 1)
    fr = (idx - i0)[..., None]
    rgb = ramp[i0] * (1 - fr) + ramp[i1] * fr
    rgb[invalid] = 0
    return rgb.astype(np.uint8)


def test_apply_ramp_is_byte_identical_to_the_reference():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        z = rng.normal(500.0, 300.0, (257, 311)).astype(dtype)
        z[rng.random(z.shape) < 0.05] = np.nan
        for ramp in (_RAMP, _SLOPE_RAMP):
            v01 = (z - 100.0) / 800.0
            inv = ~np.isfinite(z)
            assert np.array_equal(_apply_ramp(v01, ramp, inv), _reference_ramp(v01, ramp, inv))
