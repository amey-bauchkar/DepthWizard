import numpy as np
from scipy.ndimage import gaussian_filter
from core.dsm.guided import fast_guided_filter, snap_heightfield_edges


def test_guided_filter_sharpens_blurred_step_edge():
    # Crisp step edge in guidance (square in center)
    I = np.zeros((80, 80), dtype=np.float32)
    I[20:60, 20:60] = 1.0

    # Heavily blurred elevation (like ViT-S patch blur)
    p = gaussian_filter(I, sigma=6.0)

    # Guided filtering with I
    q = fast_guided_filter(I, p, radius=5, eps=1e-4)

    # At the center (roof), elevation should remain close to 1.0
    assert abs(q[40, 40] - 1.0) < 0.05

    # At edge transition (e.g. row 20, col 20), gradient should be significantly steeper
    grad_p = np.abs(np.gradient(p, axis=0))
    grad_q = np.abs(np.gradient(q, axis=0))
    assert grad_q.max() > grad_p.max() * 1.3


def test_guided_filter_preserves_nan_mask():
    I = np.random.rand(50, 50).astype(np.float32)
    p = np.ones((50, 50), dtype=np.float32)
    p[10:20, 10:20] = np.nan

    q = fast_guided_filter(I, p, radius=3, eps=1e-3)
    assert np.isnan(q[15, 15])
    assert np.isfinite(q[0, 0])


def test_snap_heightfield_edges():
    rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    rgb[25:75, 25:75] = 240  # white building roof on gray road

    heights = np.zeros((100, 100), dtype=np.float32)
    heights[25:75, 25:75] = 20.0
    blurred_heights = gaussian_filter(heights, sigma=5.0)

    snapped = snap_heightfield_edges(blurred_heights, rgb, radius=4, eps=1e-3)
    assert snapped.shape == heights.shape
    assert np.all(np.isfinite(snapped))
    # Roof center should remain close to 20m
    assert abs(snapped[50, 50] - 20.0) < 3.0
