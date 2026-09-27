"""Terrain accessibility screening (Disaster Management)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from PIL import Image


def run_accessibility_screening(
    job_dir: Path,
    max_slope_deg: float,
    result: dict[str, Any]
) -> dict[str, Any]:
    """Calculate terrain accessibility based on slope."""
    slope_path = job_dir / "slope.tif"
    if not slope_path.exists():
        raise FileNotFoundError("Disaster analysis requires a valid slope raster (slope.tif).")

    with rasterio.open(slope_path) as ds:
        slope = ds.read(1)
        nodata = ds.nodata
        grid = Grid(
            width=ds.width, height=ds.height,
            transform=ds.transform, crs=ds.crs.to_string() if ds.crs else None,
            dtype="float32", nodata=-9999.0, units="deg", metric=True,
            vertical_reference=None,
            tier=result.get("calibration_tier")
        )
        gsd_m = grid.pixel_size[0] if grid.pixel_size else 1.0

    valid = np.isfinite(slope)
    if nodata is not None:
        valid &= (slope != nodata)

    # Accessibility mask
    accessible = valid & (slope <= max_slope_deg)
    
    accessible_pixels = int(accessible.sum())
    accessible_area_m2 = accessible_pixels * (gsd_m * gsd_m)

    # Export Raster
    out_mask_path = job_dir / "accessibility.tif"
    out_arr = np.where(valid, accessible.astype(np.float32), -9999.0).astype(np.float32)
    write_raster(out_mask_path, out_arr, grid, tags={
        "KIND": "accessibility_mask",
        "SCENARIO_MAX_SLOPE_DEG": str(max_slope_deg),
        "WARNING": "Terrain accessibility screening only. Not an emergency route prediction."
    })

    # Preview Image
    out_preview_path = job_dir / "accessibility_preview.png"
    _generate_accessibility_preview(out_preview_path, slope, valid, max_slope_deg)

    summary = {
        "scenario": "accessibility_screening",
        "elevationSource": "slope.tif",
        "maxSlopeDeg": max_slope_deg,
        "accessibleAreaM2": accessible_area_m2,
        "rasterResult": "accessibility.tif",
        "previewResult": "accessibility_preview.png",
        "warnings": [
            "Terrain accessibility screening only.",
            "Not an emergency route prediction.",
            "Ignores land cover and obstacles."
        ]
    }
    
    summary_path = job_dir / "disaster_accessibility.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _generate_accessibility_preview(path: Path, slope: np.ndarray, valid: np.ndarray, max_slope: float) -> None:
    h, w = slope.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Accessible: semi-transparent green
    rgba[valid & (slope <= max_slope)] = (0, 255, 0, 100)
    # Inaccessible: semi-transparent red
    rgba[valid & (slope > max_slope)] = (255, 0, 0, 150)
    
    Image.fromarray(rgba, "RGBA").save(path)
