"""Terrain accessibility screening (Disaster Management) — NOT emergency routing. docs/math_audit.md §11.

    theta(x) = atan |grad T(x)|        slope of the TERRAIN layer T (Horn, exact affine gradient, core.dsm.derive)
    accessible(x) = theta(x) <= theta_max  and  x not inside a building footprint  and  T(x) valid

Before the audit the slope came from slope.tif, i.e. the DSM: flat roofs counted as accessible ground and building
walls as cliffs. Buildings are obstacles, not terrain, so they are excluded explicitly and reported separately.
Only slope is used: DepthWizard has no land-cover, road or obstacle data, and none is invented here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from core.dsm.derive import pixel_area_m2, slope_layers
from core.geo.grid import Grid
from core.geo.raster_io import write_raster


def run_accessibility_screening(job_dir: Path, max_slope_deg: float, result: dict[str, Any]) -> dict[str, Any]:
    if not math.isfinite(max_slope_deg) or not 0.0 <= max_slope_deg <= 90.0:
        raise ValueError("max slope must be between 0 and 90 degrees")
    terrain_path, slope_path = job_dir / "terrain.tif", job_dir / "slope.tif"
    warnings = ["Terrain accessibility screening only.", "Not an emergency route prediction.", "Ignores land cover, roads and obstacles other than detected buildings."]
    if terrain_path.exists():
        with rasterio.open(terrain_path) as ds:
            T = ds.read(1).astype(np.float64)
            if ds.nodata is not None:
                T[T == ds.nodata] = np.nan
            transform, crs = ds.transform, ds.crs
        slope, _ = slope_layers(T, transform)
        source = "terrain.tif (Horn slope of the terrain layer)"
    elif slope_path.exists():  # legacy / tier-H jobs without a terrain layer
        with rasterio.open(slope_path) as ds:
            slope = ds.read(1).astype(np.float64)
            if ds.nodata is not None:
                slope[slope == ds.nodata] = np.nan
            transform, crs = ds.transform, ds.crs
        source = "slope.tif (surface slope — includes buildings; no terrain layer in this job)"
        warnings.append("No terrain layer: slope is of the surface model, so roofs and walls are included.")
    else:
        raise FileNotFoundError("Accessibility screening requires terrain.tif or slope.tif.")

    a_px = pixel_area_m2(transform)
    valid = np.isfinite(slope)
    building = np.zeros_like(valid)
    footprint_method = None
    if (job_dir / "building_labels.tif").exists() or (job_dir / "buildings.json").exists():
        from core.disaster.flood import _footprint_sets  # one footprint definition for every hazard product

        b_data = json.loads((job_dir / "buildings.json").read_text(encoding="utf-8")) if (job_dir / "buildings.json").exists() else {}
        lab, footprint_method = _footprint_sets(job_dir, b_data, valid.shape)
        if lab is not None:
            building = lab > 0
    ground = valid & ~building
    accessible = ground & (slope <= max_slope_deg)
    steep = ground & (slope > max_slope_deg)

    grid = Grid(width=slope.shape[1], height=slope.shape[0], transform=transform, crs=crs.to_string() if crs else None, dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference=None, tier=result.get("calibration_tier") or "T")
    out = np.full(slope.shape, -9999.0, np.float32)
    out[ground] = accessible[ground].astype(np.float32)
    write_raster(job_dir / "accessibility.tif", out, grid, tags={"KIND": "accessibility_mask", "VALUES": "1 = slope <= threshold, 0 = steeper, nodata = invalid or building", "SLOPE_SOURCE": source, "SCENARIO_MAX_SLOPE_DEG": repr(float(max_slope_deg)), "WARNING": "Terrain accessibility screening only. Not an emergency route prediction."})
    _generate_accessibility_preview(job_dir / "accessibility_preview.png", accessible, steep, building & valid)

    n_ground = int(ground.sum())
    summary = {
        "scenario": "accessibility_screening",
        "elevationSource": source,
        "maxSlopeDeg": float(max_slope_deg),
        "pixelAreaM2": a_px,
        "accessibleAreaM2": int(accessible.sum()) * a_px,
        "steepAreaM2": int(steep.sum()) * a_px,
        "buildingAreaM2": int((building & valid).sum()) * a_px,
        "accessiblePctOfGround": round(100.0 * int(accessible.sum()) / n_ground, 1) if n_ground else 0.0,
        "meanTerrainSlopeDeg": round(float(slope[ground].mean()), 2) if n_ground else None,
        "footprintMethod": footprint_method,
        "rasterResult": "accessibility.tif",
        "previewResult": "accessibility_preview.png",
        "quantityCategory": {"maxSlopeDeg": "SCENARIO", "accessibleAreaM2": "SCENARIO", "steepAreaM2": "SCENARIO", "buildingAreaM2": "DERIVED"},
        "warnings": warnings,
    }
    (job_dir / "disaster_accessibility.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _generate_accessibility_preview(path: Path, accessible: np.ndarray, steep: np.ndarray, building: np.ndarray) -> None:
    rgba = np.zeros((*accessible.shape, 4), dtype=np.uint8)
    rgba[accessible] = (0, 200, 0, 100)
    rgba[steep] = (220, 0, 0, 150)
    rgba[building] = (90, 90, 90, 120)
    Image.fromarray(rgba, "RGBA").save(path)
