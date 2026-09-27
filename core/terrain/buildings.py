"""Building intelligence: a per-building table from buildings.json, with filters and GIS exports (GeoJSON, CSV).

Each building record carries its footprint, robust height statistics from the nDSM inside the footprint, ground and
roof elevation, footprint area, an approximate volume and a floor-count *range*. It also carries the model's measured
typical height error (from the model card, held-out LiDAR), never an invented confidence.
"""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from affine import Affine
from pyproj import Transformer

FLOOR_M = (3.0, 3.5)  # typical storey heights used for the approximate floor range


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
        h = float(b.get("height_median_m", b["height_m"]))
        if h < min_height_m or b["area_m2"] < min_area_m2:
            continue
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
            "ground_elev_m": b["base_elev_m"],
            "roof_elev_m": round(b["base_elev_m"] + h, 1),
            "area_m2": b["area_m2"],
            "volume_m3": round(b["area_m2"] * h),
            "floors_range": [max(1, math.floor(h / FLOOR_M[1])), max(1, math.ceil(h / FLOOR_M[0]))] if h >= 2.5 else [0, 0],
            "lon": round(lon, 6) if lon is not None else None,
            "lat": round(lat, 6) if lat is not None else None,
            "scene_xy": [round(scx, 2), round(scy, 2)],
            "n_pixels": b.get("n_pixels"),
        })
    out.sort(key=lambda r: -r["height_m"])
    return out


def summary(data: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
    hs = sorted(r["height_m"] for r in recs)
    classes = {"low_lt10m": sum(h < 10 for h in hs), "mid_10_25m": sum(10 <= h < 25 for h in hs), "high_ge25m": sum(h >= 25 for h in hs)}
    return {
        "count_total": data.get("count", len(data.get("buildings", []))),
        "count_filtered": len(recs),
        "height_classes": classes,
        "max_height_m": hs[-1] if hs else None,
        "median_height_m": hs[len(hs) // 2] if hs else None,
        "total_footprint_m2": round(sum(r["area_m2"] for r in recs)),
        "total_volume_m3": round(sum(r["volume_m3"] for r in recs)),
        "height_error": data.get("height_error"),
        "vertical_reference": data.get("vertical_reference"),
        "segmentation_method": data.get("segmentation_method"),
        "notes": [
            "height = median of the model's nDSM inside the footprint; p10-p90 shows the spread across the roof.",
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
    cols = ["id", "lon", "lat", "height_m", "height_p10_m", "height_p90_m", "height_interval_m", "ground_elev_m", "roof_elev_m", "area_m2", "volume_m3", "floors_range", "n_pixels"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in recs:
        w.writerow(["-".join(map(str, r[c])) if isinstance(r.get(c), list) else r.get(c) for c in cols])
    return buf.getvalue()
