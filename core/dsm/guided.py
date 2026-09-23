"""Fast Guided Filter for remote sensing heightfield edge snapping (He et al. 2013).

Transfers high-contrast building boundaries from the input RGB aerial/satellite
photograph into the smooth ViT-predicted elevation field, snapping blurred ramps
into crisp building roof plateaus and steep vertical boundaries.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    *,
    radius: int = 5,
    eps: float = 1e-3,
    subsample: int = 1,
) -> np.ndarray:
    """Apply an edge-preserving Guided Filter to `src` using `guide`.

    Parameters
    ----------
    guide : np.ndarray
        Guidance image (2D float32 normalized in [0, 1] or 3D RGB float32).
    src : np.ndarray
        Source elevation / relative depth image (2D float32, may contain NaNs).
    radius : int
        Filter radius in pixels (default: 5).
    eps : float
        Regularization parameter controlling edge preservation vs smoothing.
        Smaller eps (e.g. 1e-4) preserves sharper contrast edges.
    subsample : int
        Subsampling factor for acceleration (default: 1 = full resolution).

    Returns
    -------
    np.ndarray
        Refined, edge-snapped 2D float32 surface.
    """
    if guide.ndim == 3:
        # Convert RGB to perceived luminance (Rec. 709)
        I = (0.2126 * guide[..., 0] + 0.7152 * guide[..., 1] + 0.0722 * guide[..., 2]).astype(np.float32)
    else:
        I = guide.astype(np.float32)

    # Normalize guidance to [0, 1]
    imin, imax = np.nanmin(I), np.nanmax(I)
    if imax > imin:
        I = (I - imin) / (imax - imin)
    else:
        I = np.zeros_like(I)

    p = src.astype(np.float32)
    valid = np.isfinite(p)
    if not valid.any():
        return p

    # Temporarily fill NaNs with median for continuous box filtering
    p_filled = np.where(valid, p, np.nanmedian(p)).astype(np.float32)

    # Box filter size
    r = max(1, radius)
    size = 2 * r + 1

    # Guided filter linear coefficients: a = cov(I, p) / (var(I) + eps), b = mean(p) - a * mean(I)
    mean_I = uniform_filter(I, size=size, mode="reflect")
    mean_p = uniform_filter(p_filled, size=size, mode="reflect")

    corr_I = uniform_filter(I * I, size=size, mode="reflect")
    corr_Ip = uniform_filter(I * p_filled, size=size, mode="reflect")

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / np.maximum(var_I + eps, 1e-9)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=size, mode="reflect")
    mean_b = uniform_filter(b, size=size, mode="reflect")

    q = mean_a * I + mean_b

    # Preserve nodata / NaNs
    q[~valid] = np.nan
    return q.astype(np.float32)


def snap_heightfield_edges(
    heights: np.ndarray,
    rgb: np.ndarray,
    *,
    radius: int = 5,
    eps: float = 5e-4,
    blend: float = 0.85,
) -> np.ndarray:
    """Snaps smoothed heightfield ramps to RGB building edges while preserving true scale.

    Parameters
    ----------
    heights : np.ndarray
        2D float32 heights.
    rgb : np.ndarray
        2D or 3D RGB uint8/float32 array matching heights resolution.
    radius : int
        Spatial edge radius.
    eps : float
        Regularization parameter.
    blend : float
        Blend factor between guided output and original heights [0, 1].
        0.85 preserves 85% of guided crispness while anchoring to calibrated scale.
    """
    if rgb.shape[:2] != heights.shape[:2]:
        from PIL import Image
        im = Image.fromarray(rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8))
        im_resized = im.resize((heights.shape[1], heights.shape[0]), Image.Resampling.BILINEAR)
        guide = np.array(im_resized, dtype=np.float32) / 255.0
    else:
        guide = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb

    q = fast_guided_filter(guide, heights, radius=radius, eps=eps)
    # Blend with original to preserve exact global height datum
    snapped = (1.0 - blend) * heights + blend * q
    return snapped.astype(np.float32)
