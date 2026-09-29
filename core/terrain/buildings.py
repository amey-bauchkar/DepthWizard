"""Building intelligence: a per-building table from buildings.json, with filters and GIS exports (GeoJSON, CSV).

Each building record carries its footprint, robust height statistics from the nDSM inside the footprint, ground and
roof elevation, footprint area, the integrated above-ground volume and a floor-count *range*. It also carries the
model's measured typical height error (from the model card, held-out LiDAR), never an invented confidence.
The mathematics is specified in core/terrain/building_stats.py and docs/math_audit.md.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import statistics

from affine import Affine
from pyproj import Transformer

from core.terrain.building_stats import floors_range

# What every output number IS (docs/math_audit.md §19). PREDICTED = model output; DERIVED = computed from other
# quantities; REFERENCE = from the DEM-based terrain layer; SCENARIO = depends on a user-chosen parameter.
QUANTITY_CATEGORY = {
    "height_m": "PREDICTED", "height_block_m": "DERIVED", "height_p10_m": "PREDICTED", "height_p90_m": "PREDICTED",
    "height_interval_m": "DERIVED", "ground_elev_m": "REFERENCE", "ground_slope_deg": "DERIVED", "roof_elev_m": "DERIVED",
    "area_m2": "DERIVED", "volume_m3": "DERIVED", "volume_range_m3": "DERIVED", "floors_range": "DERIVED",
}


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_crs(data: dict[str, Any], coords: list[list[float]]) -> list[tuple[float, float]]:
    """Scene coordinates (metres, centred on the grid) -> job CRS via pixel coordinates."""
    g = data.get("grid") or {}
    gsd = float(data.get("gsd_m") or 1.0)
    ex, ey = data.get("extent") or [0.0, 0.0]
    tr = Affine.from_gdal(*g["transform"]) if g.get("transform") else None
    out = []
    for sx, sy in coords:
        col, row = (sx + ex / 2.0) / gsd, (ey / 2.0 - sy) / gsd
        out.append(tuple(tr @ (col, row)) if tr is not None else (col, row))
    return out


def records(data: dict[str, Any], *, min_height_m: float = 0.0, min_area_m2: float = 0.0) -> list[dict[str, Any]]:
    err = data.get("height_error") or {}
    typ = err.get("typical_m")
    crs = (data.get("grid") or {}).get("crs")
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs else None
    out = []
    for b in data.get("buildings", []):
        h = float(b.get("height_median_m", b["height_m"]))  # typical roof height (median of nDSM in the footprint)
        if h < min_height_m or b["area_m2"] < min_area_m2:
            continue
        exact = "volume_m3" in b  # jobs processed before the audit lack the pixelwise quantities
        h_block = float(b.get("height_block_m", h))
        vol = float(b["volume_m3"]) if exact else b["area_m2"] * h
        pts = _to_crs(data, b["coords"])
        cx, cy = sum(p[0] for p in pts[:-1]) / max(1, len(pts) - 1), sum(p[1] for p in pts[:-1]) / max(1, len(pts) - 1)
        lon, lat = to_ll.transform(cx, cy) if to_ll else (None, None)
        scx = sum(p[0] for p in b["coords"][:-1]) / max(1, len(b["coords"]) - 1)
        scy = sum(p[1] for p in b["coords"][:-1]) / max(1, len(b["coords"]) - 1)
        out.append({
            "id": b["id"],
            "height_m": round(h, 1),
            "height_p10_m": b.get("height_p10_m"),
            "height_p90_m": b.get("height_p90_m"),
            "height_interval_m": [round(max(0.0, h - typ), 1), round(h + typ, 1)] if typ else None,
            "height_block_m": round(h_block, 2),
            "ground_elev_m": b["base_elev_m"],
            "ground_slope_deg": b.get("ground_slope_deg"),
            # pixelwise median(T_i + h_i); legacy jobs: base + median h (differs on sloped sites)
            "roof_elev_m": round(float(b["roof_elev_m"]), 2) if exact else round(b["base_elev_m"] + h, 2),
            "area_m2": b["area_m2"],
            "volume_m3": round(vol),
            # +-1 sigma band if the model's per-object error were fully correlated across the roof (conservative bound:
            # V = A * H_block, dV = A * sigma_h). Independent pixel errors would give A_px * sigma * sqrt(N), far smaller.
            "volume_range_m3": [round(b["area_m2"] * max(0.0, h_block - typ)), round(b["area_m2"] * (h_block + typ))] if typ else None,
            "floors_range": floors_range(h),
            "valid_fraction": b.get("valid_fraction"),
            "quality_flags": b.get("quality_flags", []) if exact else ["LEGACY_STATS_REPROCESS_JOB"],
            # False: the model reads this known footprint below the detection height (LOW_PREDICTED_HEIGHT), so the
            # height is not resolved and is at best a lower bound
            "height_resolved": "LOW_PREDICTED_HEIGHT" not in (b.get("quality_flags") or []),
            "lon": round(lon, 6) if lon is not None else None,
            "lat": round(lat, 6) if lat is not None else None,
            "scene_xy": [round(scx, 2), round(scy, 2)],
            "n_pixels": b.get("n_pixels"),
        })
    out.sort(key=lambda r: -r["height_m"])
    return out


def summary(data: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
    hs = sorted(r["height_m"] for r in recs)
    hs_ok = sorted(r["height_m"] for r in recs if r.get("height_resolved", True))
    method = "pixelwise" if any("volume_m3" in b for b in data.get("buildings", [])) else "legacy (area x median height; reprocess the job)"
    classes = {"low_lt10m": sum(h < 10 for h in hs), "mid_10_25m": sum(10 <= h < 25 for h in hs), "high_ge25m": sum(h >= 25 for h in hs)}
    allb = data.get("buildings", [])  # all buildings, not the filtered table: the warning must not hide behind a height filter
    low = sum("LOW_PREDICTED_HEIGHT" in (b.get("quality_flags") or []) for b in allb)
    notes_extra = []
    min_h = data.get("min_height_m", 2.2)
    if allb and low / len(allb) >= 0.25:
        notes_extra.append(f"{low} of {len(allb)} buildings ({100 * low / len(allb):.0f} %) are read below {min_h:g} m by the model although the footprint says a building is there: "
                           "their height is NOT resolved (single-storey houses are about 3 m or more). Small rural buildings are under-read "
                           "(the model was trained on Swiss / US buildings), so treat these heights as lower bounds; the median above excludes them.")
    hsrc = data.get("height_source") or {}
    return {
        "low_height_buildings": low,
        "count_total": data.get("count", len(data.get("buildings", []))),
        "count_filtered": len(recs),
        "height_classes": classes,
        "max_height_m": hs[-1] if hs else None,
        "median_height_m": round(statistics.median(hs), 1) if hs else None,
        "median_height_resolved_m": round(statistics.median(hs_ok), 1) if hs_ok else None,
        "unresolved_filtered": len(hs) - len(hs_ok),
        "min_height_m": min_h,
        "height_source": hsrc or None,
        "total_footprint_m2": round(sum(r["area_m2"] for r in recs)),
        "total_volume_m3": round(sum(r["volume_m3"] for r in recs)),
        "height_error": data.get("height_error"),
        "vertical_reference": data.get("vertical_reference"),
        "segmentation_method": data.get("segmentation_method"),
        "footprints": data.get("footprints"),
        "stats_method": method,
        "quantity_category": QUANTITY_CATEGORY,
        "notes": notes_extra + [
            "height = typical roof height = median of the model's nDSM over the footprint pixels; p10-p90 shows the spread (a wide spread means several roof levels or a pitched roof)."
            + (f" Height layer: {hsrc['layer']}." if hsrc.get("layer") else ""),
            "volume = integral of the nDSM over the footprint (sum h_i x cell area); the 3D block height is volume / area, so the extruded block has exactly this volume.",
            "roof elevation is the median of (terrain + nDSM) per pixel, so it stays correct on sloped ground.",
            "height interval and volume range use the model card's per-object RMSE; they are not per-building calibrated uncertainties.",
            "floors_range assumes 3.0-3.5 m per storey and is an approximation, not a measurement.",
            "footprints come from a rule-based RGB + nDSM mask (not a trained segmentation model) and may merge or split buildings.",
        ],
    }


def to_geojson(data: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
    crs = (data.get("grid") or {}).get("crs")
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs else None
    by_id = {b["id"]: b for b in data.get("buildings", [])}
    feats = []
    for r in recs:
        pts = _to_crs(data, by_id[r["id"]]["coords"])
        ring = [list(map(lambda v: round(v, 7), to_ll.transform(x, y))) for x, y in pts] if to_ll else [list(p) for p in pts]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        props = {k: v for k, v in r.items() if k not in ("scene_xy",)}
        feats.append({"type": "Feature", "id": r["id"], "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "name": "depthwizard_buildings", "features": feats,
            "properties": {"height_error": data.get("height_error"), "vertical_reference": data.get("vertical_reference"), "source": "DepthWizard LoD-1 building extraction"}}


def to_csv(recs: list[dict[str, Any]]) -> str:
    cols = ["id", "lon", "lat", "height_m", "height_p10_m", "height_p90_m", "height_interval_m", "height_block_m", "ground_elev_m", "ground_slope_deg", "roof_elev_m", "area_m2", "volume_m3", "volume_range_m3", "floors_range", "valid_fraction", "quality_flags", "n_pixels"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in recs:
        w.writerow([("|" if c == "quality_flags" else "-").join(map(str, r[c])) if isinstance(r.get(c), list) else r.get(c) for c in cols])
    return buf.getvalue()
