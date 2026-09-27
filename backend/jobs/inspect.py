"""Input check (before a job runs): reads only the header and a small overview of the upload and says, in plain terms,
what DepthWizard will be able to deliver for it — mode, resolution fit for the model, DEM coverage, expected tier and
the typical error behind that tier. Nothing here is a guess dressed as a number: errors come from the model card and
from DepthWizard's own measured validations (backend.jobs.pipeline_b.MEASURED_DEM_RMSE)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from backend.config.settings import Settings
from backend.jobs.pipeline_b import MEASURED_DEM_RMSE
from core.calib.dem import select_dem
from core.calib.fusion import plan_upsample
from core.geo.grid import Grid
from core.geo.raster_io import grid_bounds_wgs84

TRAINING_GSD_M = 0.5


def _check(level: str, title: str, detail: str) -> dict[str, str]:
    return {"level": level, "title": title, "detail": detail}


def _gsd_m(ds) -> float | None:  # noqa: ANN001
    if not ds.crs:
        return None
    px, py = abs(ds.transform.a), abs(ds.transform.e)
    if ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        return float(math.sqrt(px * 111320.0 * math.cos(math.radians(lat)) * py * 110574.0))
    f = ds.crs.linear_units_factor[1] if ds.crs.linear_units_factor else 1.0
    return float(math.sqrt(px * py) * f)


def _inspect(filename: str, data: bytes, settings: Settings, *, metric_card: Any | None, has_user_dem: bool = False, has_anchors: bool = False) -> dict[str, Any]:
    ext = Path(filename).suffix.lower()
    checks: list[dict[str, str]] = []
    out: dict[str, Any] = {"filename": filename, "size_mb": round(len(data) / 1e6, 2), "checks": checks}
    if ext not in (".tif", ".tiff"):
        out.update(mode="A", expected_tier="R", expected_accuracy=None)
        checks.append(_check("warn", "Not a GeoTIFF: relative shape only", "PNG/JPEG carry no location or scale, so the result is relative structure without metres. Upload a GeoTIFF for heights in metres."))
        return out
    with MemoryFile(data) as mf, mf.open() as ds:
        out.update(width=ds.width, height=ds.height, bands=ds.count, dtype=ds.dtypes[0], crs=ds.crs.to_string() if ds.crs else None)
        # --- bands / bit depth
        if ds.count < 3:
            checks.append(_check("bad", f"{ds.count} band(s): the model needs RGB", "Single-band (panchromatic) input is replicated to grey RGB; heights will be much less reliable. Use a 3-band true-colour product."))
        elif ds.count > 4:
            checks.append(_check("warn", f"{ds.count} bands", "Bands 1-3 are used as RGB; make sure they are red, green, blue in that order."))
        else:
            checks.append(_check("ok", "RGB bands", f"{ds.count} bands, {ds.dtypes[0]}"))
        if ds.dtypes[0] != "uint8":
            checks.append(_check("warn", f"{ds.dtypes[0]} pixels", "Non-8-bit imagery is contrast-stretched (2-98 %) to 8-bit; very dark or hazy scenes lose detail."))
        # --- nodata share from a small overview
        try:
            f = max(1, max(ds.width, ds.height) // 512)
            arr = ds.read(1, out_shape=(max(1, ds.height // f), max(1, ds.width // f)), masked=True)
            empty = float(np.ma.getmaskarray(arr).mean()) if ds.nodata is not None else float((np.asarray(arr) == 0).mean())
            out["empty_fraction"] = round(empty, 3)
            if empty > 0.5:
                checks.append(_check("warn", f"{empty:.0%} of the image is empty", "Large nodata/black areas; only the imaged part is processed."))
        except Exception:  # noqa: BLE001 - informational only
            pass
        if not ds.crs:
            out.update(mode="A", expected_tier="R", expected_accuracy=None)
            checks.append(_check("bad", "No georeferencing", "The GeoTIFF has no CRS: no ground resolution, no DEM, no metres. Export it with its map projection."))
            return out
        gsd = _gsd_m(ds)
        out["gsd_m"] = round(gsd, 3) if gsd else None
        ext_km = (ds.width * gsd / 1000, ds.height * gsd / 1000) if gsd else None
        out["extent_km"] = [round(ext_km[0], 2), round(ext_km[1], 2)] if ext_km else None
        grid = Grid(ds.width, ds.height, ds.transform, ds.crs.to_string())
    # --- resolution fit
    if gsd is not None:
        if gsd <= 0.8:
            checks.append(_check("ok", f"Resolution {gsd:.2f} m", f"Matches the model's training resolution ({TRAINING_GSD_M} m); buildings and trees are resolved."))
        elif gsd <= 2.0:
            checks.append(_check("warn", f"Resolution {gsd:.2f} m", f"Coarser than the {TRAINING_GSD_M} m the model was trained on: small buildings merge and heights are less accurate. Accuracy figures below do not strictly apply."))
        else:
            checks.append(_check("bad", f"Resolution {gsd:.2f} m", "Too coarse for individual structures: expect terrain-level output only (the DEM dominates). Use imagery of 1 m or finer."))
        fz = settings.fusion
        desired = min(max(gsd / fz.inference_gsd_m, fz.min_upsample), fz.max_upsample)
        up = plan_upsample(ds.height, ds.width, desired, tile=fz.tile_px, overlap=fz.overlap, max_tiles=fz.max_tiles)
        eff = gsd / up
        out["inference_gsd_m"] = round(eff, 3)
        if eff > fz.inference_gsd_m * 1.3:
            checks.append(_check("warn", f"Large scene: processed at {eff:.2f} m", f"To stay within {fz.max_tiles} tiles the model sees the image at {eff:.2f} m instead of {fz.inference_gsd_m} m; crop to the area of interest for full detail."))
    # --- DEM coverage
    dem_name = None
    if has_user_dem:
        dem_name = "user"
        checks.append(_check("ok", "DEM: your upload", "Absolute elevations will follow your DEM; its accuracy is not measured by DepthWizard."))
    else:
        try:
            src, _sel = select_dem(grid_bounds_wgs84(grid), settings.dem_dir, grid, priority=tuple(settings.calib.dem_priority), cartodem_vcrs=settings.calib.cartodem_vertical_crs)
        except Exception as e:  # noqa: BLE001
            src = None
            checks.append(_check("warn", "DEM lookup failed", f"{type(e).__name__}: {e}"))
        if src is not None:
            dem_name = src.name
            checks.append(_check("ok", f"DEM: {src.product}", f"{src.posting_m:.0f} m posting, vertical datum {src.vertical_crs} (converted to {settings.calib.output_vertical_crs})."))
        else:
            checks.append(_check("warn", "No DEM covers this area", "Heights above ground only (no absolute elevation). Add the CartoDEM / Copernicus tile for this area or upload a DEM."))
    # --- expected tier and accuracy
    tier = "A" if has_anchors and dem_name else "T" if dem_name else "H" if metric_card is not None else "R"
    acc: dict[str, Any] = {}
    if metric_card is not None:
        ft = (((metric_card.extra or {}).get("validation") or {}).get("finetuned")) or {}
        if ft.get("RMSE_objects_ge2.5m"):
            acc["height_above_ground"] = {"objects_m": round(ft["RMSE_objects_ge2.5m"], 1), "ground_m": round(ft.get("RMSE_ground_lt2.5m", float("nan")), 1), "source": f"{metric_card.name}@{metric_card.version}, held-out Swiss LiDAR"}
    d = MEASURED_DEM_RMSE.get(dem_name or "")
    if d:
        acc["terrain_m"] = d["terrain_m"]
        acc["dsm_m"] = d["dsm_m"]
        acc["elevation_source"] = d["source"]
    out.update(mode="B", expected_tier=tier, dem=dem_name, expected_accuracy=acc or None)
    if metric_card is None:
        checks.append(_check("warn", "No fine-tuned height model installed", "Object heights come from zero-shot detail scaled to the DEM: uncalibrated."))
    if has_anchors:
        checks.append(_check("ok", "Ground control points supplied", "Tier A if at least 5 points fit consistently; otherwise the job falls back to tier T."))
    return out


def inspect_upload(filename: str, data: bytes, settings: Settings, *, metric_card: Any | None, has_user_dem: bool = False, has_anchors: bool = False) -> dict[str, Any]:
    out = _inspect(filename, data, settings, metric_card=metric_card, has_user_dem=has_user_dem, has_anchors=has_anchors)
    lv = {c["level"] for c in out["checks"]}
    out["verdict"] = "bad" if "bad" in lv else "warn" if "warn" in lv else "ok"
    return out
