"""Tiled inference + DEM-preserving multi-scale detail fusion (TL-CSM v1.0, the Mode B object layer).

Why (measured on the bundled swisstopo tiles, see docs/validation_results.md):
  * Depth Anything V2 run ONCE on a whole 1 km tile (resized to 518 px) carries almost no building-scale
    information at nadir: correlation with LiDAR sub-30 m detail was 0.05-0.07 in Zurich.
  * The same model run on overlapping 518 px tiles at ~0.5 m inference GSD reaches 0.50-0.54.
  * The DEM (Copernicus GLO-30) already carries the >30 m surface correctly; replacing it (terrain layer +
    scaled object layer) lost accuracy against LiDAR. So the DEM is kept and only the band it cannot resolve
    (< one DEM posting) is taken from the model.

Method:
  1. Tiled inference: the image is resampled to the inference GSD and cut into overlapping tiles; each tile is
     predicted independently (relative depth is affine-ambiguous per tile).
  2. Per tile, the affine scale `a` (metres per relative unit) is estimated by least squares between the
     band-pass of the model and the band-pass of the base surface (the DEM) in the band the DEM DOES resolve
     (one to four DEM postings). An affine map has the same scale at every spatial frequency, so the same `a`
     is applied to the model's high-pass detail (< one DEM posting). Negative fits are clamped to 0 (no detail),
     and regression dilution shrinks `a` automatically when the model and the DEM disagree.
  3. Detail tiles are feather-blended and re-high-passed on the job grid, so every DEM-cell mean is preserved:
     at the DEM's own resolution the fused surface equals the DEM (the fusion cannot degrade the DEM there).

The same routine refines Mode A: the base is then the whole-image prediction (unitless) instead of a DEM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class TiledPrediction:
    tiles: list[tuple[int, int, np.ndarray]]  # (row0, col0, relative depth tile) on the inference grid
    inference_shape: tuple[int, int]  # (Hb, Wb)
    job_shape: tuple[int, int]  # (H, W)
    upsample: float  # inference grid px per job-grid px
    tile: int
    overlap: float
    elapsed_ms: float
    inference_gsd_m: float | None = None
    quantity: str = "relative_inverse_depth"  # or "metric_ndsm_metres" (fine-tuned model: tiles already in metres)
    model_info: dict = field(default_factory=dict)  # name / version / measured object-height error from the model card

    def summary(self) -> dict[str, Any]:
        return {
            "n_tiles": len(self.tiles),
            "tile_px": self.tile,
            "overlap": self.overlap,
            "upsample": round(self.upsample, 4),
            "inference_shape": list(self.inference_shape),
            "inference_gsd_m": self.inference_gsd_m,
            "quantity": self.quantity,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class FusionResult:
    detail: np.ndarray  # job grid, base units (metres for Mode B), zero mean at the base's resolution
    gains: list[float]  # per-tile scale (base units per relative unit)
    params: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _tile_origins(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    out = list(range(0, n - tile + 1, step))
    if out[-1] + tile < n:
        out.append(n - tile)
    return out


def _resize(a: np.ndarray, width: int, height: int, how: Image.Resampling) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((width, height), how), dtype=np.float64)


def stitch_tiles(tp: TiledPrediction) -> np.ndarray:
    """Feather-blend tiles whose values are directly comparable (metric model output) and resample to the job grid."""
    hb, wb = tp.inference_shape
    H, W = tp.job_shape
    w1 = np.hanning(tp.tile + 2)[1:-1]
    win = np.outer(w1, w1) + 1e-3
    acc = np.zeros((hb, wb))
    wsum = np.zeros((hb, wb))
    for r0, c0, t in tp.tiles:
        th, tw = t.shape
        acc[r0 : r0 + th, c0 : c0 + tw] += win[:th, :tw] * t
        wsum[r0 : r0 + th, c0 : c0 + tw] += win[:th, :tw]
    out = acc / np.maximum(wsum, 1e-9)
    if (hb, wb) != (H, W):
        out = _resize(out, W, H, Image.Resampling.BOX)
    return out.astype(np.float32)


def highpass(a: np.ndarray, scale_px: float) -> np.ndarray:
    """Structure below scale_px (same filter the fusion uses to preserve DEM cell means)."""
    return (a - _gauss(a.astype(np.float64), max(0.5, scale_px / 2.0))).astype(np.float32)


def plan_upsample(h: int, w: int, desired: float, *, tile: int, overlap: float, max_tiles: int) -> float:
    """Largest upsample factor <= desired whose tile count stays within max_tiles."""
    step = max(1, int(tile * (1 - overlap)))
    up = desired
    for _ in range(64):
        hb, wb = max(tile, int(round(h * up))), max(tile, int(round(w * up)))
        n = len(_tile_origins(hb, tile, step)) * len(_tile_origins(wb, tile, step))
        if n <= max_tiles:
            return up
        up *= 0.9
    return up


def tiled_relative(predict: Callable[[np.ndarray], np.ndarray], rgb: np.ndarray, *, upsample: float = 1.0, tile: int = 518, overlap: float = 0.25, inference_gsd_m: float | None = None) -> TiledPrediction:
    """Run `predict` (HxWx3 uint8 -> HxW relative depth) on overlapping tiles of `rgb` resampled by `upsample`."""
    t0 = time.perf_counter()
    h, w = rgb.shape[:2]
    hb, wb = max(tile, int(round(h * upsample))), max(tile, int(round(w * upsample)))
    big = rgb if (hb, wb) == (h, w) else np.asarray(Image.fromarray(rgb).resize((wb, hb), Image.Resampling.BICUBIC))
    step = max(1, int(tile * (1 - overlap)))
    tiles = []
    for r0 in _tile_origins(hb, tile, step):
        for c0 in _tile_origins(wb, tile, step):
            tiles.append((r0, c0, np.asarray(predict(big[r0 : r0 + tile, c0 : c0 + tile]), dtype=np.float64)))
    return TiledPrediction(tiles, (hb, wb), (h, w), hb / h, tile, overlap, (time.perf_counter() - t0) * 1000.0, inference_gsd_m)


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass (reflect borders). Large sigmas are computed on a block-averaged grid and interpolated
    back (block size <= sigma/4 adds < 1 % to the kernel variance) — ~100x faster for sigma ~ 100 px."""
    if sigma <= 8.0:
        return ndimage.gaussian_filter(a, sigma, mode="reflect")
    f = max(2, int(sigma // 4))
    h, w = a.shape
    hh, ww = -(-h // f), -(-w // f)
    pad = np.pad(a, ((0, hh * f - h), (0, ww * f - w)), mode="reflect")
    small = pad.reshape(hh, f, ww, f).mean(axis=(1, 3))
    small = ndimage.gaussian_filter(small, sigma / f, mode="reflect")
    yy = (np.arange(h) + 0.5) / f - 0.5
    xx = (np.arange(w) + 0.5) / f - 0.5
    return ndimage.map_coordinates(small, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")


def fuse_detail(base: np.ndarray, tp: TiledPrediction, *, detail_scale_px: float, band_high_px: float, max_gain: float | None = None, max_detail_band_ratio: float = 3.0) -> FusionResult:
    """Model detail below `detail_scale_px` (job-grid px), scaled per tile against `base` in the band
    [detail_scale_px, band_high_px]. Returns detail on the job grid in base units; add it to `base`.

    Spectral plausibility guard: a tile whose model detail is far stronger than its model band (std ratio above
    `max_detail_band_ratio`) is noise-dominated at fine scales; its gain is shrunk so that the injected detail std
    is at most `max_detail_band_ratio` times the band std the DEM confirmed. Natural surfaces and DA-V2 outputs have
    red spectra, so the guard is inactive on them; it only bites on noise-like predictions."""
    H, W = tp.job_shape
    hb, wb = tp.inference_shape
    up = tp.upsample
    fill = float(np.nanmean(base)) if np.isfinite(base).any() else 0.0
    base_f = np.where(np.isfinite(base), base, fill).astype(np.float64)
    base_b = base_f if (hb, wb) == (H, W) else _resize(base_f, wb, hb, Image.Resampling.BILINEAR)
    s1 = max(0.5, detail_scale_px * up / 2.0)  # gaussian sigma ~ half the scale (inference-grid px)
    s2 = max(s1 * 1.5, band_high_px * up / 2.0)
    w1 = np.hanning(tp.tile + 2)[1:-1]
    win = np.outer(w1, w1) + 1e-3
    acc = np.zeros((hb, wb))
    wsum = np.zeros((hb, wb))
    gains: list[float] = []
    corrs: list[float] = []
    ratios: list[float] = []
    G = _gauss
    for r0, c0, rel in tp.tiles:
        th, tw = rel.shape
        low1 = G(rel, s1)
        det = rel - low1
        rb = low1 - G(rel, s2)
        b = base_b[r0 : r0 + th, c0 : c0 + tw]
        b1 = G(b, s1)
        db = b1 - G(b, s2)
        e = int(min(s2, max(0, (min(th, tw) - 64) // 2)))  # drop the filter edge zone from the fit
        rbi = rb[e : th - e, e : tw - e] if e else rb
        dbi = db[e : th - e, e : tw - e] if e else db
        den = float(np.sum(rbi * rbi))
        a = float(np.sum(rbi * dbi) / den) if den > 1e-12 else 0.0
        c = float(np.corrcoef(rbi.ravel(), dbi.ravel())[0, 1]) if rbi.std() > 0 and dbi.std() > 0 else 0.0
        a = max(0.0, a)
        det_i = det[e : th - e, e : tw - e] if e else det
        ratio = float(det_i.std() / max(rbi.std(), 1e-12))
        ratios.append(ratio)
        if ratio > max_detail_band_ratio:
            a *= max_detail_band_ratio / ratio
        if max_gain is not None:
            a = min(a, max_gain)
        gains.append(a)
        corrs.append(c)
        acc[r0 : r0 + th, c0 : c0 + tw] += win[:th, :tw] * a * det
        wsum[r0 : r0 + th, c0 : c0 + tw] += win[:th, :tw]
    det_b = acc / np.maximum(wsum, 1e-9)
    detail = det_b if (hb, wb) == (H, W) else _resize(det_b, W, H, Image.Resampling.BOX)
    # re-impose mean preservation at the base's resolution on the job grid
    detail = detail - G(detail, max(0.5, detail_scale_px / 2.0))
    g = np.asarray(gains)
    stats = {
        "n_tiles": len(gains),
        "gain_median": float(np.median(g)) if g.size else 0.0,
        "gain_min": float(g.min()) if g.size else 0.0,
        "gain_max": float(g.max()) if g.size else 0.0,
        "tiles_with_detail": int((g > 0).sum()),
        "band_corr_median": float(np.median(corrs)) if corrs else 0.0,
        "detail_band_ratio_median": float(np.median(ratios)) if ratios else 0.0,
        "detail_band_ratio_max": float(np.max(ratios)) if ratios else 0.0,
        "tiles_ratio_capped": int(sum(r > max_detail_band_ratio for r in ratios)),
        "detail_std": float(np.std(detail)),
        "detail_p01_p99": [float(np.percentile(detail, 1)), float(np.percentile(detail, 99))] if detail.size else [0.0, 0.0],
    }
    params = {"detail_scale_px": detail_scale_px, "band_high_px": band_high_px, "sigma_detail_inf_px": s1, "sigma_band_inf_px": s2, "max_gain": max_gain, "max_detail_band_ratio": max_detail_band_ratio}
    return FusionResult(detail.astype(np.float32), gains, params, stats)


def _irls(y: np.ndarray, A: np.ndarray, n_iter: int = 20, min_n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Tukey-biweight IRLS; returns (coefficients, final weights)."""
    w = np.ones(len(y))
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = r / (4.685 * s)
        w_new = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if (w_new > 0).sum() < min_n:
            break
        w = w_new
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    return coef, w


def fit_anchor_gain(z: np.ndarray, base: np.ndarray, detail: np.ndarray, *, min_n: int = 5, k_range: tuple[float, float] = (0.0, 4.0), offset_z: float = 2.0) -> dict[str, Any]:
    """Tier A refinement of the fused surface at anchor points: z ~ base + c + k * detail (robust IRLS).

    `detail` is the Tier-T detail (k = 1, c = 0 reproduces Tier T). A single point anchor cannot tell a datum
    bias from sub-DEM-cell mixing (a street pixel inside a 30 m cell that also holds roofs), so the offset c is
    kept only when it is statistically significant (|c| > offset_z * SE); otherwise c = 0 and only the detail gain
    is fitted. The refinement is accepted only if its leave-one-out anchor RMSE beats Tier T's."""
    z, base, detail = (np.asarray(v, float) for v in (z, base, detail))
    m = np.isfinite(z) & np.isfinite(base) & np.isfinite(detail)
    z, base, detail = z[m], base[m], detail[m]
    n = int(z.size)
    if n < min_n:
        return {"accepted": False, "reason": f"need >= {min_n} anchors with valid surface values (have {n})", "n": n, "offset_m": 0.0, "detail_gain": 1.0}
    y = z - base
    lo, hi = k_range

    def fit(yy: np.ndarray, xx: np.ndarray) -> tuple[float, float, np.ndarray]:
        coef, w = _irls(yy, np.column_stack([np.ones(len(yy)), xx]))
        c, k = float(coef[0]), float(coef[1])
        r = yy - (c + k * xx)
        inl = w > 0
        se_c = 1.4826 * float(np.median(np.abs(r[inl] - np.median(r[inl])))) / np.sqrt(max(int(inl.sum()), 1))
        if abs(c) <= offset_z * se_c:  # offset not significant -> gain only, through the origin
            coef, w = _irls(yy, xx[:, None])
            c, k = 0.0, float(coef[0])
        k = float(min(max(k, lo), hi))
        if c != 0.0:
            c = float(np.median(yy - k * xx))
        return c, k, w

    c, k, w = fit(y, detail)
    # leave-one-out comparison against Tier T (c = 0, k = 1)
    loo = []
    for i in range(n):
        keep = np.arange(n) != i
        ci, ki, _ = fit(y[keep], detail[keep])
        loo.append(y[i] - (ci + ki * detail[i]))
    loo = np.asarray(loo)
    base_r = y - detail
    rmse = lambda v: float(np.sqrt(np.mean(v**2)))  # noqa: E731
    nmad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))  # noqa: E731
    accepted = rmse(loo) <= rmse(base_r)
    return {
        "accepted": bool(accepted),
        "reason": "accepted (leave-one-out RMSE improves on tier T)" if accepted else "rejected: leave-one-out RMSE does not improve on tier T",
        "offset_m": c if accepted else 0.0,
        "detail_gain": k if accepted else 1.0,
        "fitted_offset_m": c,
        "fitted_detail_gain": k,
        "offset_significant": c != 0.0,
        "n": n,
        "inliers": int((w > 0).sum()),
        "tier_t_rmse_m": rmse(base_r),
        "loo_rmse_m": rmse(loo),
        "loo_nmad_m": nmad(loo),
        "gain_range": [lo, hi],
        "method": "IRLS Tukey biweight z = DEM + c + k*detail; c kept only if |c| > %.1f SE; accepted on leave-one-out RMSE" % offset_z,
    }


def default_detail_scales(dem_posting_m: float, job_gsd_m: float) -> tuple[float, float]:
    """(detail_scale_px, band_high_px) on the job grid: below one DEM posting comes from the model; the gain is
    fitted between one and four postings."""
    d = dem_posting_m / max(job_gsd_m, 1e-6)
    return max(1.0, d), max(3.0, 4.0 * d)


def mode_a_scales(h: int, w: int, model_input: int = 518) -> tuple[float, float]:
    """Mode A: the whole-image prediction resolves ~2 of its own pixels; detail below that comes from tiles."""
    px = 2.0 * min(h, w) / float(model_input)
    return max(1.0, px), max(3.0, 4.0 * px)


def needs_tiling(h: int, w: int, model_input: int = 518) -> bool:
    return min(h, w) > 1.3 * model_input or max(h, w) > 2.0 * model_input


__all__ = ["TiledPrediction", "FusionResult", "tiled_relative", "fuse_detail", "fit_anchor_gain", "plan_upsample", "default_detail_scales", "mode_a_scales", "needs_tiling", "stitch_tiles", "highpass"]
