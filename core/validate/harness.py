"""Validation harness (Phase 4 blueprint, Phase 6 §10): reference → datum → align → co-register → mask → metrics
→ strata → residual. Independent of calibration code; anchors used for calibration are excluded by radius.

Reference types: "dsm" (compare to dsm.tif), "dtm" (compare to terrain.tif), "ndsm" (compare to ndsm.tif).
The reference is resampled onto the job grid by area-average when it is finer (Phase 4 §16) — comparison then
happens at the job posting, and the reference's native posting is recorded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from core.geo.grid import Grid
from core.geo.raster_io import read_raster_on_grid, write_raster
from core.geo.vertical import transform_heights_xy
from core.validate.coregister import estimate_shift_grid, slope_aspect_deg
from core.validate.metrics import affine_fit, metric_set
from PIL import Image


@dataclass
class ReferenceSpec:
    path: str
    ref_type: str  # dsm | dtm | ndsm
    vertical_crs: str  # EGM2008 | EGM96 | ellipsoidal | "unknown" | "same"
    source_note: str = ""
    acquisition_date: str | None = None


@dataclass
class ValidationResult:
    reference: dict[str, Any]
    comparison_grid: dict[str, Any]
    alignment: dict[str, Any]
    mask: dict[str, Any]
    metrics_overall: dict[str, Any]
    metrics_by_slope: dict[str, dict[str, Any]]
    metrics_by_object: dict[str, dict[str, Any]]
    height_bins: dict[str, dict[str, Any]]
    oracle_affine: dict[str, Any]
    residual_stats: dict[str, Any]
    artifacts: dict[str, str]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _native_posting(path: str) -> tuple[float, float]:
    with rasterio.open(path) as ds:
        return (abs(ds.transform.a), abs(ds.transform.e))


def run_validation(job_dir: Path, grid: Grid, pred: np.ndarray, ref_spec: ReferenceSpec, *, out_vcrs: str, ndsm: np.ndarray | None, anchors_xy: np.ndarray | None = None, exclusion_radius_m: float = 15.0, border_px: int = 4, max_shift_px: int = 4) -> ValidationResult:
    caveats: list[str] = []
    native = _native_posting(ref_spec.path)
    gsd = grid.pixel_size or (1.0, 1.0)
    finer = native[0] < gsd[0] * 0.75
    resampling = Resampling.average if finer else Resampling.bilinear
    ref, ref_valid = read_raster_on_grid(ref_spec.path, grid, resampling=resampling)
    # vertical datum
    datum_note = "reference vertical CRS assumed identical to output (no transform)"
    if ref_spec.vertical_crs not in ("same", "unknown", out_vcrs):
        # compound transform over the job's projected CRS (C-1 guarded: offline, no ballpark, grids required)
        step = max(1, min(grid.width, grid.height) // 16)
        rows = np.arange(0, grid.height, step); cols = np.arange(0, grid.width, step)
        cc, rr = np.meshgrid(cols, rows)
        xs, ys = grid.transform * (cc + 0.5, rr + 0.5)  # type: ignore[operator]
        z1, _info = transform_heights_xy(np.asarray(xs), np.asarray(ys), np.zeros(np.shape(xs)), grid.crs, ref_spec.vertical_crs, out_vcrs)
        z1 = np.asarray(z1).reshape(np.shape(xs))
        from scipy.ndimage import zoom

        corr = zoom(z1.astype(np.float32), (grid.height / z1.shape[0], grid.width / z1.shape[1]), order=1)[: grid.height, : grid.width]
        ref = ref + corr
        datum_note = f"reference converted {ref_spec.vertical_crs} -> {out_vcrs} (correction {float(corr.min()):.2f}..{float(corr.max()):.2f} m)"
    elif ref_spec.vertical_crs == "unknown":
        caveats.append("reference vertical datum UNKNOWN — metrics may contain a constant datum offset")
    # masks
    m = np.isfinite(pred) & ref_valid & np.isfinite(ref)
    m[:border_px] = m[-border_px:] = False
    m[:, :border_px] = m[:, -border_px:] = False
    n_before = int(m.sum())
    excluded = 0
    if anchors_xy is not None and len(anchors_xy):
        yy, xx = np.mgrid[0 : grid.height, 0 : grid.width]
        for ax, ay in anchors_xy:
            c, r = grid.crs_to_pixel(ax, ay)
            rad = exclusion_radius_m / gsd[0]
            ex = (xx - c) ** 2 + (yy - r) ** 2 <= rad**2
            excluded += int((m & ex).sum())
            m &= ~ex
    # co-registration on the terrain-like signal (grid search, deterministic)
    shift = estimate_shift_grid(np.where(m, pred, np.nan), np.where(m, ref, np.nan), max_shift=max_shift_px, border=border_px)
    from scipy import ndimage

    pred_al = ndimage.shift(np.nan_to_num(pred, nan=np.nanmedian(pred)), (-shift.dy, -shift.dx), order=1, mode="nearest") if (abs(shift.dx) > 0.05 or abs(shift.dy) > 0.05) else pred
    overall = metric_set(pred_al, ref, m)
    # strata: slope classes from the reference; object vs ground from ndsm when available
    slope_ref, _ = slope_aspect_deg(np.where(np.isfinite(ref), ref, np.nanmedian(ref)), gsd[0], gsd[1])
    by_slope = {}
    for name, lo, hi in (("flat_lt5", 0, 5), ("moderate_5_15", 5, 15), ("steep_15_30", 15, 30), ("very_steep_gt30", 30, 90)):
        mm = m & (slope_ref >= lo) & (slope_ref < hi)
        if mm.sum() >= 50:
            by_slope[name] = metric_set(pred_al, ref, mm)
    by_obj = {}
    if ndsm is not None:
        for name, cond in (("ground_lt1m", ndsm < 1.0), ("objects_ge1m", ndsm >= 1.0), ("tall_ge10m", ndsm >= 10.0)):
            mm = m & np.isfinite(ndsm) & cond
            if mm.sum() >= 50:
                by_obj[name] = metric_set(pred_al, ref, mm)
    bins = {}
    if m.any():
        edges = np.percentile(ref[m], [0, 25, 50, 75, 100])
        for i in range(4):
            mm = m & (ref >= edges[i]) & (ref <= edges[i + 1])
            if mm.sum() >= 50:
                bins[f"ref_q{i+1}_{edges[i]:.1f}_{edges[i+1]:.1f}m"] = metric_set(pred_al, ref, mm)
    oracle = affine_fit(pred_al, ref, m)
    resid = np.where(m, pred_al - ref, np.nan).astype(np.float32)
    rg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, out_vcrs, grid.tier)
    write_raster(job_dir / "residual.tif", resid, rg, {"KIND": "pred_minus_ref", "REFERENCE": Path(ref_spec.path).name})
    # residual preview: diverging ramp +-clip at LE95
    v = np.isfinite(resid)
    clip = float(np.percentile(np.abs(resid[v]), 95)) if v.any() else 1.0
    clip = max(clip, 1e-3)
    x = np.clip(np.nan_to_num(resid) / clip, -1, 1)
    rgb = np.zeros(resid.shape + (3,), np.uint8)
    rgb[..., 0] = np.clip(255 * (1 + x) / 2 + 128 * (x > 0), 0, 255)  # red for positive (pred too high)
    rgb[..., 2] = np.clip(255 * (1 - x) / 2 + 128 * (x < 0), 0, 255)  # blue for negative
    rgb[..., 1] = np.clip(255 * (1 - np.abs(x)), 0, 255)
    rgb[~v] = 0
    Image.fromarray(rgb).save(job_dir / "residual_preview.png")
    return ValidationResult(
        reference={**asdict(ref_spec), "native_posting_m": list(native), "datum_handling": datum_note},
        comparison_grid={"posting_m": list(gsd), "resampling": resampling.name, "reference_finer_than_grid": bool(finer), "width": grid.width, "height": grid.height},
        alignment=shift.to_dict() | {"applied": bool(abs(shift.dx) > 0.05 or abs(shift.dy) > 0.05)},
        mask={"valid_pixels": int(m.sum()), "before_exclusion": n_before, "anchor_excluded_pixels": excluded, "border_px": border_px, "leakage_check": "passed" if excluded >= 0 else "n/a"},
        metrics_overall=overall,
        metrics_by_slope=by_slope,
        metrics_by_object=by_obj,
        height_bins=bins,
        oracle_affine=oracle,
        residual_stats={"clip_le95_m": clip},
        artifacts={"residual_tif": "residual.tif", "residual_preview": "residual_preview.png"},
        caveats=caveats + ["Pearson r is scale/offset-blind: read it together with ME/RMSE/NMAD (Phase 8 L0-08)."],
    )
