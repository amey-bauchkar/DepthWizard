"""Building footprint mask from RGB + nDSM (rule-based spectral/height fusion; no trained segmentation model).

Evidence combined per pixel:
  1. Height: nDSM >= min_height -> elevated object candidate.
  2. Vegetation: Excess-Green / VDVI indices on plain RGB -> trees, grass and parks are removed.
  3. Spectral score: not green, not shadow, not saturated, locally uniform -> roof-like.
  4. Edge strength (Sobel): sharp boundaries typical of roofs.
Final mask = weighted fusion >= threshold, plus clearly elevated non-vegetation, then morphological cleanup.

This is a heuristic used only to draw LoD-1 blocks in the viewer; it is not an accuracy-assessed building
detector. A trained segmentation model would replace compute_spectral_building_score in a later build.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_closing, binary_opening

log = logging.getLogger(__name__)


# ─── Vegetation Index (works on plain RGB, no NIR needed) ─────────────────────

def compute_vegetation_mask(rgb: np.ndarray, *, threshold: float = 0.05) -> np.ndarray:
    """Compute an Excess Green vegetation mask from RGB.

    Uses the Excess Green Index (ExG = 2G - R - B) normalized to [0, 1].
    Pixels with ExG above threshold are classified as vegetation.
    This reliably separates trees/grass from buildings/roads in aerial imagery.

    Parameters
    ----------
    rgb : np.ndarray
        HxWx3 uint8 aerial image.
    threshold : float
        ExG threshold (default 0.05 works well for European & Indian urban areas).

    Returns
    -------
    np.ndarray
        Boolean mask, True = vegetation pixel.
    """
    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0

    # Excess Green Index (ExG): standard remote-sensing vegetation index for RGB
    total = r + g + b + 1e-8
    exg = (2.0 * g - r - b) / total

    # Visible-band Difference Vegetation Index (VDVI) as additional check
    vdvi = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-8)

    # Combined: pixel is vegetation if either index exceeds threshold
    veg_mask = (exg > threshold) | (vdvi > threshold * 1.5)

    # Morphological cleanup: fill small gaps in tree canopies
    veg_mask = binary_closing(veg_mask, structure=np.ones((5, 5), dtype=bool))
    veg_mask = binary_opening(veg_mask, structure=np.ones((3, 3), dtype=bool))

    return veg_mask


# ─── Edge-based building boundary detection ──────────────────────────────────

def compute_building_edges(rgb: np.ndarray) -> np.ndarray:
    """Detect strong rectilinear edges typical of building boundaries.

    Uses Sobel gradient magnitude on the grayscale image.
    Buildings have sharp, straight edges while vegetation has soft, organic boundaries.

    Returns a float32 edge strength map normalized to [0, 1].
    """
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)

    # Sobel edge detection
    from scipy.ndimage import sobel
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    edges = np.hypot(sx, sy)

    # Normalize to [0, 1]
    emax = edges.max()
    if emax > 0:
        edges = edges / emax

    return edges


# ─── Spectral building probability (color-based) ────────────────────────────

def compute_spectral_building_score(rgb: np.ndarray) -> np.ndarray:
    """Estimate building probability from spectral properties.

    Buildings in aerial imagery tend to have:
    - Low vegetation index (not green)
    - Higher red/blue relative to green (rooftops: grey, brown, red, white)
    - Higher saturation uniformity than vegetation
    - Distinct color from shadows (not too dark)

    Returns float32 probability map [0, 1], higher = more likely building.
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0

    # Not vegetation (inverse of green dominance)
    total = r + g + b + 1e-8
    not_green = 1.0 - np.clip((2.0 * g - r - b) / total, 0, 1)

    # Not shadow (reasonably bright)
    not_shadow = np.clip(brightness / 128.0, 0, 1)

    # Not pure white sky (not overexposed)
    not_sky = 1.0 - np.clip((brightness - 230) / 25.0, 0, 1)

    # Color uniformity in local 5x5 patches (buildings are uniform, vegetation is noisy)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(brightness, size=7)
    local_sq = uniform_filter(brightness**2, size=7)
    local_var = np.maximum(local_sq - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    uniformity = 1.0 - np.clip(local_std / 40.0, 0, 1)

    # Combined spectral building score
    score = not_green * not_shadow * not_sky * uniformity
    return np.clip(score, 0, 1).astype(np.float32)


# ─── Combined building mask ──────────────────────────────────────────────────

def extract_building_mask(
    rgb: np.ndarray,
    ndsm: np.ndarray,
    *,
    min_height_m: float = 1.5,
    veg_threshold: float = 0.05,
    building_score_threshold: float = 0.3,
) -> np.ndarray:
    """Building mask from height + vegetation + spectral + edge evidence (see module docstring).

    Parameters
    ----------
    rgb : np.ndarray
        HxWx3 uint8 aerial image (same spatial extent as nDSM).
    ndsm : np.ndarray
        HxW float32 normalized DSM (height above ground, metres).
    min_height_m : float
        Minimum height to consider as elevated structure (default 1.5m).
    veg_threshold : float
        Vegetation index threshold (default 0.05).
    building_score_threshold : float
        Minimum combined building probability to accept.

    Returns
    -------
    np.ndarray
        Boolean mask, True = building pixel. Same shape as ndsm.
    """
    H, W = ndsm.shape

    # Ensure RGB matches nDSM dimensions
    if rgb.shape[:2] != (H, W):
        from PIL import Image as PILImage
        rgb = np.array(PILImage.fromarray(rgb).resize((W, H), PILImage.Resampling.BILINEAR))

    log.info("Building segmentation: computing vegetation mask...")
    veg_mask = compute_vegetation_mask(rgb, threshold=veg_threshold)
    veg_count = int(veg_mask.sum())
    log.info("Vegetation pixels: %d (%.1f%%)", veg_count, 100.0 * veg_count / (H * W))

    # Height evidence: anything above ground is a candidate
    height_mask = (ndsm >= min_height_m) & np.isfinite(ndsm) & (ndsm < 200.0)

    # Spectral building score
    log.info("Building segmentation: computing spectral building score...")
    spectral_score = compute_spectral_building_score(rgb)

    # Edge evidence
    edge_map = compute_building_edges(rgb)

    # ── Fusion: combine all evidence ──────────────────────────────────────
    # Weight each source:
    #   - Height evidence is strongest (if nDSM says it's elevated, it probably is)
    #   - Vegetation removal is a strong negative signal
    #   - Spectral score helps distinguish buildings from roads at ground level
    #   - Edge strength confirms structural boundaries

    # Elevated + not vegetation = strong building candidate
    elevated_nonveg = height_mask & ~veg_mask

    # Spectral building evidence at elevated locations
    spectral_elevated = spectral_score * height_mask.astype(np.float32)

    # Combined probability
    combined = (
        0.55 * elevated_nonveg.astype(np.float32) +
        0.25 * spectral_elevated +
        0.20 * edge_map * elevated_nonveg.astype(np.float32)
    )

    # Apply threshold
    building_mask = combined >= building_score_threshold

    # Also include any pixel that is clearly elevated and not vegetation,
    # even if other scores are low (catch-all for grey/uniform rooftops)
    building_mask |= (elevated_nonveg & (ndsm >= min_height_m * 1.5))

    # Morphological cleanup
    # Close small gaps within buildings (e.g., courtyards, balconies)
    building_mask = binary_closing(building_mask, structure=np.ones((3, 3), dtype=bool))
    # Remove tiny noise blobs
    building_mask = binary_opening(building_mask, structure=np.ones((2, 2), dtype=bool))

    building_count = int(building_mask.sum())
    log.info(
        "Building mask complete: %d building pixels (%.1f%%), vegetation removed: %d pixels",
        building_count, 100.0 * building_count / (H * W), veg_count,
    )

    return building_mask
