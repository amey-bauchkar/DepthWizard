"""Elevation-based flood screening (Disaster Management). Static "bathtub" thresholding on the terrain layer — NOT a
hydraulic or hydrodynamic simulation. Mathematics: docs/math_audit.md §5-8; constants: core/screening_params.py.

For water level W (metres, same vertical datum as the terrain layer T) and valid cells Omega (T finite, not NoData):

    I(x)       = 1  if T(x) <= W                         inundation indicator (SCENARIO)
    D(x)       = max(W - T(x), 0)                         screened water depth above the terrain layer (SCENARIO)
    A_wet      = sum_x I(x) * |det J|                     wet area; |det J| = exact cell area of the affine grid
    connected  = wet cells 8-connected (within the wet set) to an open boundary cell: a wet cell on the raster edge or
                 next to a NoData cell (NoData is unknown ground, not a wall). Wet cells without such a path are
                 "isolated": below W, but with no surface path for water in this scene.

Depth is terrain-relative (never DSM-relative: roofs are not ground). Per building B (exact footprint pixel set):

    f_wet      = |{i in B : T_i <= W}| / |B_valid|        wet footprint fraction
    D_mean     = mean over wet footprint cells of (W - T_i)
    D_max      = max over the footprint of D_i
    D_exp      = max(W - Q10(T_B), 0)                     exposure depth on the low side of the footprint
    class      = bin(D_exp)  by EXPOSURE_BINS_M           (policy bins, returned with every result)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

from core.dsm.derive import pixel_area_m2
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.screening_params import EXPOSURE_BINS_M, EXPOSURE_GROUND_QUANTILE, EXPOSURE_LABELS, FLOOD_CONNECTIVITY

PREVIEW_MAX_DEPTH_M = 5.0  # display only: the preview colour ramp saturates here (4 colours, linear in depth)
PREVIEW_COLOURS = ["#add8e6", "#00bfff", "#0000cd", "#000080"]
METHOD = "bathtub-threshold-2 (terrain layer, exact footprints, 8-connectivity diagnostic)"
QUANTITY_CATEGORY = {
    "waterLevel_m": "SCENARIO", "affectedAreaM2": "SCENARIO", "affectedAreaPct": "SCENARIO", "maxDepth_m": "SCENARIO",
    "meanDepth_m": "SCENARIO", "isolatedAreaM2": "SCENARIO", "totalAreaM2": "DERIVED", "terrain": "REFERENCE",
    "flood_depth_m": "SCENARIO", "wet_fraction": "SCENARIO", "base_elev_m": "REFERENCE", "height_m": "PREDICTED",
}


def exposure_rules() -> list[dict[str, Any]]:
    """The single definition of the exposure classes, serialised into every response (the UI must not re-derive it)."""
    lo = 0.0
    rules = []
    for label, hi in zip(EXPOSURE_LABELS, (*EXPOSURE_BINS_M, None)):
        rules.append({"label": label, "gt_m": lo, "le_m": hi})
        lo = hi if hi is not None else lo
    return rules


def classify_exposure(depth_m: float) -> str:
    """NONE if d <= 0 (or not finite); LOW if 0 < d <= b0; MODERATE if b0 < d <= b1; HIGH if b1 < d <= b2; else VERY HIGH."""
    if not math.isfinite(depth_m) or depth_m <= 0:
        return "NONE"
    for label, hi in zip(EXPOSURE_LABELS, EXPOSURE_BINS_M):
        if depth_m <= hi:
            return label
    return EXPOSURE_LABELS[-1]


def connectivity_structure(n: int = FLOOD_CONNECTIVITY) -> np.ndarray:
    if n == 8:
        return np.ones((3, 3), bool)
    if n == 4:
        return ndimage.generate_binary_structure(2, 1)
    raise ValueError("connectivity must be 4 or 8")


def boundary_connected(wet: np.ndarray, valid: np.ndarray, connectivity: int = FLOOD_CONNECTIVITY) -> np.ndarray:
    """Wet cells connected (within the wet set) to an open boundary cell: raster edge or adjacent to NoData."""
    st = connectivity_structure(connectivity)
    open_b = np.zeros_like(wet)
    open_b[0, :] = open_b[-1, :] = True
    open_b[:, 0] = open_b[:, -1] = True
    open_b |= ndimage.binary_dilation(~valid, structure=st)
    lab, n = ndimage.label(wet, structure=st)
    if n == 0:
        return np.zeros_like(wet)
    seeds = np.unique(lab[wet & open_b])
    keep = np.zeros(n + 1, bool)
    keep[seeds[seeds > 0]] = True
    return keep[lab]


def _footprint_sets(job_dir: Path, b_data: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """Exact label raster if the job has one; else rasterise the stored pixel polygons (cell-centre rule)."""
    lp = job_dir / "building_labels.tif"
    if lp.exists():
        with rasterio.open(lp) as ds:
            lab = ds.read(1)
        if lab.shape == shape:
            return lab.astype(np.int32), "exact_label_raster"
    shapes = [({"type": "Polygon", "coordinates": [b["pixel_coords"]]}, int(b["id"])) for b in b_data.get("buildings", []) if len(b.get("pixel_coords") or []) >= 4]
    if shapes:
        from rasterio.features import rasterize

        return rasterize(shapes, out_shape=shape, transform=rasterio.Affine.identity(), fill=0, dtype="int32"), "polygon_cell_centre"
    return None, "scalar_ground_legacy"


def building_exposure(terrain: np.ndarray, wet: np.ndarray, W: float, labels: np.ndarray | None, b_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-building exposure. With labels: pixelwise over the footprint. Without: legacy scalar rule on base_elev_m."""
    slices = ndimage.find_objects(labels) if labels is not None else []
    out = []
    for b in b_data.get("buildings", []):
        bid = int(b.get("id"))
        base = b.get("base_elev_m")
        rec = {"id": bid, "base_elev_m": round(base, 2) if base is not None else None, "height_m": b.get("height_median_m", b.get("height_m")), "area_m2": b.get("area_m2"), "coords": b.get("coords"), "pixel_bbox": b.get("pixel_bbox")}
        sl = slices[bid - 1] if labels is not None and 0 < bid <= len(slices) else None
        if sl is not None:
            fp = labels[sl] == bid
            T = terrain[sl][fp]
            ok = np.isfinite(T)
            if not ok.any():
                continue
            T = T[ok].astype(np.float64)
            w = wet[sl][fp][ok]
            d = np.where(w, W - T, 0.0)
            q = float(np.percentile(T, EXPOSURE_GROUND_QUANTILE))
            d_exp = max(W - q, 0.0) if w.any() else 0.0  # wet mask may be the connected subset
            rec.update({"ground_p10_m": round(q, 2), "flood_depth_m": round(d_exp, 2), "wet_fraction": round(float(w.mean()), 3), "mean_depth_wet_m": round(float(d[w].mean()), 2) if w.any() else 0.0, "max_depth_m": round(float(d.max()), 2), "contact": bool(w.any())})
        elif base is not None:  # legacy: one ground value per building (pre-audit behaviour, flagged)
            d_exp = max(W - float(base), 0.0)
            rec.update({"flood_depth_m": round(d_exp, 2), "wet_fraction": None, "contact": d_exp > 0})
        else:
            continue
        rec["exposure"] = classify_exposure(rec["flood_depth_m"])
        if rec["contact"]:
            out.append(rec)
    return out


def run_flood_screening(job_dir: Path, water_level_m: float, result: dict[str, Any], *, connected_only: bool = False) -> dict[str, Any]:
    """Flood screening for water level W on terrain.tif. connected_only=False: every valid cell with T <= W is wet
    (upper bound). True: only wet cells connected to an open boundary (excludes isolated depressions)."""
    if not math.isfinite(water_level_m):
        raise ValueError("water level must be a finite number of metres")
    terrain_path = job_dir / "terrain.tif"
    if not terrain_path.exists():
        raise FileNotFoundError("Disaster analysis requires a valid terrain/elevation surface (terrain.tif).")

    with rasterio.open(terrain_path) as ds:
        terrain = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        grid = Grid(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs.to_string() if ds.crs else None, dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference=result.get("vertical_reference"), tier=result.get("calibration_tier") or "T")
        a_px = pixel_area_m2(ds.transform)

    valid = np.isfinite(terrain)
    if nodata is not None:
        valid &= terrain != nodata
    W = float(water_level_m)
    vref = result.get("vertical_reference") or "unknown (vertical datum of terrain.tif)"
    below = valid & (terrain <= W)
    connected = boundary_connected(below, valid)
    wet = connected if connected_only else below

    depth = np.where(wet, W - terrain, 0.0)
    depth[~valid] = np.nan
    n_wet, n_valid = int(wet.sum()), int(valid.sum())

    buildings_path = job_dir / "buildings.json"
    buildings_result: list[dict[str, Any]] = []
    footprint_method = None
    if buildings_path.exists():
        b_data = json.loads(buildings_path.read_text(encoding="utf-8"))
        labels, footprint_method = _footprint_sets(job_dir, b_data, terrain.shape)
        buildings_result = building_exposure(terrain, wet, W, labels, b_data)
    affected = [b for b in buildings_result if b["exposure"] != "NONE"]

    write_raster(job_dir / "flood_depth.tif", np.where(valid, depth, -9999.0).astype(np.float32), grid, tags={
        "KIND": "flood_depth", "QUANTITY": "max(W - terrain, 0), metres above the terrain layer",
        "SCENARIO_WATER_LEVEL_M": repr(W), "WATER_LEVEL_VERTICAL_REFERENCE": vref, "CONNECTED_ONLY": str(connected_only), "METHOD": METHOD,
        "WARNING": "Elevation-based screening only. Not a hydraulic simulation."})
    _generate_flood_preview(job_dir / "flood_preview.png", depth, wet, PREVIEW_MAX_DEPTH_M)

    counts = {lab: sum(b["exposure"] == lab for b in buildings_result) for lab in EXPOSURE_LABELS}
    summary = {
        "scenario": "flood_screening",
        "method": METHOD,
        "elevationSource": "terrain.tif",
        "waterLevel_m": W,
        # W is compared with T directly, so it is read in T's vertical datum; there is no conversion of W
        "waterLevelVerticalReference": vref,
        "connectedOnly": connected_only,
        "connectivity": FLOOD_CONNECTIVITY,
        "pixelAreaM2": a_px,
        "affectedAreaM2": n_wet * a_px,
        "affectedAreaPct": round(100.0 * n_wet / n_valid, 1) if n_valid else 0.0,
        "totalAreaM2": n_valid * a_px,
        "isolatedAreaM2": int((below & ~connected).sum()) * a_px,
        "affectedBuildingsCount": len(affected),
        "contactBuildingsCount": len(buildings_result),
        "exposureCounts": counts,
        "exposureRules": exposure_rules(),
        "exposureDepthDefinition": f"W - P{EXPOSURE_GROUND_QUANTILE:g} of footprint ground elevation (depth on the low side of the building)",
        "footprintMethod": footprint_method,
        "maxDepth_m": round(float(depth[wet].max()), 2) if n_wet else 0.0,
        "meanDepth_m": round(float(depth[wet].mean()), 2) if n_wet else 0.0,
        "rasterResult": "flood_depth.tif",
        "previewResult": "flood_preview.png",
        # the legend must be drawn from this: colour i sits at depth stops_m[i] (linear interpolation in between)
        "previewRamp": {"colours": PREVIEW_COLOURS, "stops_m": [round(PREVIEW_MAX_DEPTH_M * i / (len(PREVIEW_COLOURS) - 1), 2) for i in range(len(PREVIEW_COLOURS))], "saturates_above_m": PREVIEW_MAX_DEPTH_M},
        "qualityMask": "flags.tif",
        "buildings": affected,
        "quantityCategory": QUANTITY_CATEGORY,
        "warnings": [
            "Elevation-based screening only.",
            f"Water level is read in {vref}, the terrain layer's vertical datum; convert heights from other datums first.",
            "Connectivity is a diagnostic, not a hydraulic model." if not connected_only else "Only cells with a surface path to the scene boundary are shown.",
            "Not a hydrodynamic simulation.",
        ] + (["Building footprints rasterised from simplified polygons (job predates exact footprints)."] if footprint_method == "polygon_cell_centre" else [])
          + (["Building exposure uses one ground value per building (legacy job)."] if footprint_method == "scalar_ground_legacy" else []),
    }
    (job_dir / "disaster_flood.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _generate_flood_preview(path: Path, depth: np.ndarray, mask: np.ndarray, max_depth_ramp: float = 5.0) -> None:
    # Colors: shallow = light cyan, deep = dark blue (display only; the ramp saturates at max_depth_ramp metres)
    ramp = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] + [200] for c in PREVIEW_COLOURS], dtype=np.float32)
    h, w = depth.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask.any():
        idx = np.clip(depth[mask] / max_depth_ramp, 0, 1) * (len(ramp) - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, len(ramp) - 1)
        fr = (idx - i0)[..., None]
        rgba[mask] = (ramp[i0] * (1 - fr) + ramp[i1] * fr).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
