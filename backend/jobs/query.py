"""Server-side measurement authority (Phase 6 §9): every number shown to a user is sampled from the job's
rasters here — never from the viewer mesh. Every value carries its quantity label, units, tier and vertical reference.

Also hosts the validation entry point (reference upload -> core.validate.harness) so results are persisted per job.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer

from backend.config.settings import Settings
from backend.errors import DepthWizardError, InvalidFileError, JobStateError
from core.dsm.derive import FLAG_BITS
from core.geo.grid import Grid
from core.validate.harness import ReferenceSpec, run_validation

# layer file -> (quantity label, units, is_absolute)
LAYERS_B = {
    "dsm": ("ABSOLUTE ELEVATION (DSM)", "m"),
    "terrain": ("TERRAIN LAYER (DEM-derived, NOT a DTM)", "m"),
    "ndsm": ("METRIC RELATIVE HEIGHT ABOVE TERRAIN (nDSM)", "m"),
    "slope": ("SLOPE", "deg"),
    "aspect": ("ASPECT", "deg"),
    "relative": ("RELATIVE STRUCTURE (non-metric)", "rel"),
}
LAYERS_A = {"rdsm": ("RELATIVE STRUCTURE (non-metric)", "rel")}


class _RasterCache:
    """Per-process cache of opened job rasters (small; jobs are few)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], tuple[np.ndarray, Grid, float | None]] = {}

    def get(self, job_dir: Path, name: str) -> tuple[np.ndarray, Grid, float | None] | None:
        key = (str(job_dir), name)
        if key in self._data:
            return self._data[key]
        p = job_dir / f"{name}.tif"
        if not p.exists():
            return None
        with rasterio.open(p) as ds:
            arr = ds.read(1)
            grid = Grid(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs.to_string() if ds.crs else None, dtype=str(arr.dtype), nodata=ds.nodata, units=ds.tags().get("UNITS", ""), metric=ds.tags().get("METRIC", "false") == "true", vertical_reference=ds.tags().get("VERTICAL_REFERENCE"), tier=ds.tags().get("CALIBRATION_TIER"))
        self._data[key] = (arr, grid, ds.nodata)
        return self._data[key]

    def drop(self, job_dir: Path) -> None:
        for k in [k for k in self._data if k[0] == str(job_dir)]:
            self._data.pop(k, None)


CACHE = _RasterCache()


def _to_pixel(grid: Grid, x: float, y: float, crs: str) -> tuple[float, float]:
    if crs == "pixel":
        return x, y
    if crs == "job":
        c, r = grid.crs_to_pixel(x, y)  # centre-based index -> continuous raster coords (integer = pixel edge)
        return c + 0.5, r + 0.5
    if crs == "wgs84":
        if not grid.crs:
            raise InvalidFileError("job has no CRS; use crs=pixel")
        t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        xx, yy = t.transform(x, y)
        c, r = grid.crs_to_pixel(xx, yy)
        return c + 0.5, r + 0.5
    raise InvalidFileError(f"unknown crs {crs!r}; expected pixel|job|wgs84")


def sample(job_dir: Path, result: dict[str, Any], x: float, y: float, crs: str = "pixel") -> dict[str, Any]:
    mode = result.get("mode", "A")
    layers = LAYERS_B if mode == "B" else LAYERS_A
    first = None
    out: dict[str, Any] = {"mode": mode, "calibration_tier": result.get("calibration_tier"), "quality": result.get("quality"), "vertical_reference": result.get("vertical_reference"), "values": {}}
    for name, (label, units) in layers.items():
        got = CACHE.get(job_dir, name)
        if got is None:
            continue
        arr, grid, nodata = got
        if first is None:
            first = grid
            col, row = _to_pixel(grid, x, y, crs)
            ci, ri = int(math.floor(col)), int(math.floor(row))
            out["pixel"] = {"col": ci, "row": ri}
            if grid.crs:
                px, py = grid.pixel_center_to_crs(ci, ri)
                out["position"] = {"x": px, "y": py, "crs": grid.crs}
                try:
                    t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
                    lon, lat = t.transform(px, py)
                    out["position"]["lon"], out["position"]["lat"] = float(lon), float(lat)
                except Exception:  # noqa: BLE001
                    pass
            if not (0 <= ci < grid.width and 0 <= ri < grid.height):
                out["in_bounds"] = False
                return out
            out["in_bounds"] = True
        v = float(arr[ri, ci])
        valid = np.isfinite(v) and (nodata is None or v != nodata)
        entry: dict[str, Any] = {"value": round(v, 3) if valid else None, "valid": bool(valid), "quantity": label, "units": units}
        if name == "relative" or name == "rdsm":
            entry["metric"] = False
            entry["tier"] = "R"
        elif name in ("dsm", "terrain"):
            entry["metric"] = True
            entry["absolute"] = True
            entry["tier"] = result.get("calibration_tier")
            entry["vertical_reference"] = result.get("vertical_reference")
        elif name == "ndsm":
            entry["metric"] = True
            entry["absolute"] = False
            entry["tier"] = result.get("calibration_tier")
            entry["scale_source"] = result.get("object_scale_source")
        out["values"][name] = entry
    fl = CACHE.get(job_dir, "flags")
    if fl is not None and out.get("in_bounds"):
        f = int(fl[0][ri, ci])
        out["flags"] = [k for k, bit in FLAG_BITS.items() if f & bit]
    return out


def measure(job_dir: Path, result: dict[str, Any], points: list[dict[str, float]], crs: str = "pixel") -> dict[str, Any]:
    """Two-or-more-point measurement: per-point samples + horizontal distance (metres when metric grid) + dz."""
    if len(points) < 2:
        raise InvalidFileError("need at least two points")
    samples = [sample(job_dir, result, float(p["x"]), float(p["y"]), crs) for p in points]
    segs = []
    metric_h = bool(result.get("metric_horizontal")) or result.get("mode") == "B"
    for a, b in zip(samples[:-1], samples[1:]):
        if "position" in a and "position" in b:
            dx, dy = b["position"]["x"] - a["position"]["x"], b["position"]["y"] - a["position"]["y"]
            dist = math.hypot(dx, dy)
            seg: dict[str, Any] = {"horizontal_distance": round(dist, 3), "distance_units": "m" if metric_h else "pixel"}
        else:
            dx, dy = b["pixel"]["col"] - a["pixel"]["col"], b["pixel"]["row"] - a["pixel"]["row"]
            seg = {"horizontal_distance": round(math.hypot(dx, dy), 3), "distance_units": "pixel"}
        for lname in ("dsm", "ndsm", "rdsm", "relative"):
            va, vb = a["values"].get(lname), b["values"].get(lname)
            if va and vb and va["valid"] and vb["valid"]:
                seg[f"dz_{lname}"] = round(vb["value"] - va["value"], 3)
                seg[f"dz_{lname}_units"] = va["units"]
                if seg.get("distance_units") == "m" and va["units"] == "m" and seg["horizontal_distance"] > 0:
                    seg[f"grade_{lname}_deg"] = round(math.degrees(math.atan2(vb["value"] - va["value"], seg["horizontal_distance"])), 2)
        segs.append(seg)
    return {"mode": result.get("mode"), "calibration_tier": result.get("calibration_tier"), "points": samples, "segments": segs, "authority": "server rasters (never the viewer mesh)"}


def validate_job(job_dir: Path, result: dict[str, Any], ref_path: Path, ref_type: str, vertical_crs: str, source_note: str, settings: Settings, acquisition_date: str | None = None) -> dict[str, Any]:
    if result.get("mode") != "B":
        raise JobStateError("validation against a georeferenced reference requires a Mode B (GeoTIFF) job")
    if ref_type not in ("dsm", "dtm", "ndsm"):
        raise InvalidFileError("ref_type must be dsm|dtm|ndsm")
    layer = {"dsm": "dsm", "dtm": "terrain", "ndsm": "ndsm"}[ref_type]
    got = CACHE.get(job_dir, layer)
    if got is None:
        raise JobStateError(f"job has no {layer} layer to validate")
    arr, grid, nodata = got
    pred = arr.astype(np.float64)
    if nodata is not None:
        pred[pred == nodata] = np.nan
    ndsm = None
    nd = CACHE.get(job_dir, "ndsm")
    if nd is not None:
        ndsm = nd[0].astype(np.float64)
        if nd[2] is not None:
            ndsm[ndsm == nd[2]] = np.nan
    anchors_xy = None
    au = job_dir / "anchors_used.json"
    if au.exists():
        pts = json.loads(au.read_text(encoding="utf-8")).get("points", [])
        if pts:
            anchors_xy = np.array([[p["x_grid"], p["y_grid"]] for p in pts], float)
    grid = Grid(width=grid.width, height=grid.height, transform=grid.transform, crs=grid.crs, dtype="float32", nodata=nodata, units="m", metric=True, vertical_reference=result.get("vertical_reference"), tier=result.get("calibration_tier"))
    t0 = time.perf_counter()
    try:
        vr = run_validation(job_dir, grid, pred, ReferenceSpec(str(ref_path), ref_type, vertical_crs, source_note, acquisition_date), out_vcrs=result.get("vertical_reference") or "EGM2008", ndsm=ndsm, anchors_xy=anchors_xy, exclusion_radius_m=settings.validation.anchor_exclusion_radius_m, border_px=settings.validation.border_px, max_shift_px=settings.validation.max_shift_px)
    except rasterio.errors.RasterioIOError as e:
        raise InvalidFileError(f"reference raster unreadable: {e}") from e
    d = vr.to_dict()
    d["compared_layer"] = layer
    d["job"] = {"calibration_tier": result.get("calibration_tier"), "quality": result.get("quality"), "object_scale_source": result.get("object_scale_source")}
    d["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    d["ran_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    d["verdict"] = _verdict(d)
    history_path = job_dir / "validation.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"runs": []}
    history["runs"].append(d)
    history["latest"] = d
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    return d


def _verdict(d: dict[str, Any]) -> dict[str, Any]:
    """Plain-language reading of the metrics; thresholds from Phase 4 §21 acceptance bands (NOT a pass/fail of the PS)."""
    m = d.get("metrics_overall", {})
    rmse, me, nmad = m.get("RMSE"), m.get("ME"), m.get("NMAD")
    if rmse is None:
        return {"band": "NO_DATA", "text": "no overlapping valid pixels"}
    band = "A (<2 m RMSE)" if rmse < 2 else "B (2-5 m)" if rmse < 5 else "C (5-10 m)" if rmse < 10 else "D (>10 m)"
    txt = f"RMSE {rmse:.2f} m, bias {me:+.2f} m, NMAD {nmad:.2f} m over {m.get('n')} pixels vs. the uploaded reference."
    return {"band": band, "text": txt, "note": "Bands are descriptive only; the comparison grid is the job grid (reference area-averaged if finer)."}
