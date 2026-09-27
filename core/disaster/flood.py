"""Elevation-based flood screening and analysis (Disaster Management).
Static thresholding on terrain raster, NOT a hydraulic simulation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from PIL import Image


@dataclass
class ExposureClass:
    label: str
    min_depth_m: float
    max_depth_m: float


EXPOSURE_RULES = [
    ExposureClass("LOW", 0.0, 0.5),
    ExposureClass("MODERATE", 0.5, 1.5),
    ExposureClass("HIGH", 1.5, 3.0),
    ExposureClass("VERY HIGH", 3.0, 9999.0),
]


def classify_exposure(depth_m: float) -> str:
    if depth_m <= 0:
        return "NONE"
    for rule in EXPOSURE_RULES:
        if rule.min_depth_m < depth_m <= rule.max_depth_m:
            return rule.label
    return "VERY HIGH"


def run_flood_screening(
    job_dir: Path,
    water_level_m: float,
    result: dict[str, Any]
) -> dict[str, Any]:
    """Calculate elevation-based flood screening from terrain.tif."""
    terrain_path = job_dir / "terrain.tif"
    if not terrain_path.exists():
        raise FileNotFoundError("Disaster analysis requires a valid terrain/elevation surface (terrain.tif).")

    with rasterio.open(terrain_path) as ds:
        terrain = ds.read(1)
        nodata = ds.nodata
        grid = Grid(
            width=ds.width, height=ds.height,
            transform=ds.transform, crs=ds.crs.to_string() if ds.crs else None,
            dtype="float32", nodata=-9999.0, units="m", metric=True,
            vertical_reference=result.get("vertical_reference"),
            tier=result.get("calibration_tier")
        )
        gsd_m = grid.pixel_size[0] if grid.pixel_size else 1.0

    valid = np.isfinite(terrain)
    if nodata is not None:
        valid &= (terrain != nodata)

    # Inundation calculation
    inundated = valid & (terrain <= water_level_m)
    
    depth = np.zeros_like(terrain, dtype=np.float32)
    depth[inundated] = water_level_m - terrain[inundated]
    depth[~valid] = np.nan

    affected_pixels = int(inundated.sum())
    affected_area_m2 = affected_pixels * (gsd_m * gsd_m)

    max_depth = float(depth[inundated].max()) if affected_pixels > 0 else 0.0
    mean_depth = float(depth[inundated].mean()) if affected_pixels > 0 else 0.0

    total_valid_pixels = int(valid.sum())
    affected_pct = round(100.0 * affected_pixels / max(1, total_valid_pixels), 1) if total_valid_pixels > 0 else 0.0

    # Building Analysis
    buildings_path = job_dir / "buildings.json"
    affected_buildings = 0
    buildings_result = []

    if buildings_path.exists():
        b_data = json.loads(buildings_path.read_text(encoding="utf-8"))
        for b in b_data.get("buildings", []):
            base_elev = b.get("base_elev_m")
            b_height = b.get("height_m")
            area = b.get("area_m2")
            if base_elev is not None and base_elev < water_level_m:
                b_depth = water_level_m - base_elev
                exp = classify_exposure(b_depth)
                buildings_result.append({
                    "id": b.get("id"),
                    "base_elev_m": round(base_elev, 2),
                    "height_m": round(b_height, 2) if b_height else None,
                    "area_m2": round(area, 2) if area else None,
                    "flood_depth_m": round(b_depth, 2),
                    "exposure": exp,
                    "coords": b.get("coords"),
                    "pixel_bbox": b.get("pixel_bbox"),
                })
                affected_buildings += 1

    # Export Raster
    out_depth_path = job_dir / "flood_depth.tif"
    out_depth_arr = np.where(valid, depth, -9999.0).astype(np.float32)
    write_raster(out_depth_path, out_depth_arr, grid, tags={
        "KIND": "flood_depth",
        "SCENARIO_WATER_LEVEL_M": str(water_level_m),
        "WARNING": "Elevation-based screening only. Not a hydraulic simulation."
    })

    # Preview Image for Viewer/UI
    out_preview_path = job_dir / "flood_preview.png"
    _generate_flood_preview(out_preview_path, depth, inundated)

    summary = {
        "scenario": "flood_screening",
        "elevationSource": "terrain.tif",
        "waterLevel_m": water_level_m,
        "affectedAreaM2": affected_area_m2,
        "affectedAreaPct": affected_pct,
        "totalAreaM2": total_valid_pixels * (gsd_m * gsd_m),
        "affectedBuildingsCount": affected_buildings,
        "maxDepth_m": round(max_depth, 2),
        "meanDepth_m": round(mean_depth, 2),
        "rasterResult": "flood_depth.tif",
        "previewResult": "flood_preview.png",
        "qualityMask": "flags.tif",
        "buildings": buildings_result,
        "warnings": [
            "Elevation-based screening only.",
            "No hydraulic connectivity model.",
            "Not a hydrodynamic simulation."
        ]
    }
    
    summary_path = job_dir / "disaster_flood.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def _generate_flood_preview(path: Path, depth: np.ndarray, mask: np.ndarray, max_depth_ramp: float = 5.0) -> None:
    # Colors: shallow = light cyan, deep = dark blue
    ramp = np.array([
        [173, 216, 230, 200], # Shallow
        [0,   191, 255, 200], # Medium-shallow
        [0,   0,   205, 200], # Medium-deep
        [0,   0,   128, 200], # Deep
    ], dtype=np.float32)
    
    h, w = depth.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    if mask.any():
        idx = np.clip(depth[mask] / max_depth_ramp, 0, 1) * (len(ramp) - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, len(ramp) - 1)
        fr = (idx - i0)[..., None]
        color = ramp[i0] * (1 - fr) + ramp[i1] * fr
        rgba[mask] = color.astype(np.uint8)
        
    Image.fromarray(rgba, "RGBA").save(path)
