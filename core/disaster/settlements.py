"""Settlements and population estimate from the job's buildings (+ OSM place names). Shared by road access and the
damage report.

    cluster      buildings whose footprints are closer than SETTLEMENT_CLUSTER_M (dilation by half the distance)
    floors(b)    mean of the building's floors_range (from its nDSM height), at least 1
    population   sum over buildings of floors(b) x HOUSEHOLD_SIZE (one household per floor: a rough, labelled estimate)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from core.screening_params import HOUSEHOLD_SIZE, SETTLEMENT_CLUSTER_M, SETTLEMENT_MIN_BUILDINGS, SETTLEMENT_NAME_M

POPULATION_NOTE = (f"Population is a rough estimate: {HOUSEHOLD_SIZE:g} persons (Census 2011 mean household) per floor of every detected "
                   "building, including non-residential ones. Use census or survey figures for decisions.")


def floors(b: dict[str, Any]) -> float:
    fr = b.get("floors_range")
    if isinstance(fr, list) and fr:
        return max(1.0, float(np.mean(fr)))
    h = b.get("height_median_m") or b.get("height_m")
    return max(1.0, round(float(h) / 3.0)) if isinstance(h, (int, float)) else 1.0


def population(buildings: list[dict[str, Any]]) -> float:
    return sum(floors(b) for b in buildings) * HOUSEHOLD_SIZE


def load_buildings(job_dir: Path) -> list[dict[str, Any]]:
    p = job_dir / "buildings.json"
    return json.loads(p.read_text(encoding="utf-8")).get("buildings", []) if p.exists() else []


def building_centres(buildings: list[dict[str, Any]]) -> np.ndarray:
    """(n, 2) pixel centres (col, row) from pixel_bbox = [col0, row0, col1, row1]."""
    out = []
    for b in buildings:
        bb = b.get("pixel_bbox") or [0, 0, 0, 0]
        out.append(((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def clusters(job_dir: Path, shape: tuple[int, int], px_m: float, places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Settlement clusters: {id, name, named_from, col, row, buildings (ids), n_buildings, population}.
    places: [{name, col, row}] in pixel coordinates (OSM place nodes)."""
    from core.disaster.flood import _footprint_sets

    bl = load_buildings(job_dir)
    if not bl:
        return []
    lab, _ = _footprint_sets(job_dir, {"buildings": bl}, shape)
    if lab is None:
        return []
    r = max(1, int(round(SETTLEMENT_CLUSTER_M / 2.0 / px_m)))
    # dilate on a coarser grid (factor f) for speed: the cluster distance is >> one pixel
    f = max(1, int(r // 4))
    small = ndimage.maximum_filter(lab > 0, size=f)[::f, ::f]
    rs = max(1, int(round(r / f)))
    yy, xx = np.mgrid[-rs:rs + 1, -rs:rs + 1]
    grown = ndimage.binary_dilation(small, structure=(xx * xx + yy * yy) <= rs * rs)
    cl, n = ndimage.label(grown)
    by_id = {int(b["id"]): b for b in bl}
    cen = building_centres(bl)
    pos = {int(b["id"]): i for i, b in enumerate(bl)}
    members: dict[int, list[int]] = {}
    for b, (c, rr) in zip(bl, cen):
        k = int(cl[min(cl.shape[0] - 1, int(rr // f)), min(cl.shape[1] - 1, int(c // f))])
        if k:
            members.setdefault(k, []).append(int(b["id"]))
    out = []
    for k, ids in sorted(members.items(), key=lambda kv: -len(kv[1])):
        if len(ids) < SETTLEMENT_MIN_BUILDINGS:
            continue
        pts = cen[[pos[i] for i in ids]]
        c0, r0 = pts.mean(axis=0)
        name, src = None, None
        if places:
            d = [math.hypot(p["col"] - c0, p["row"] - r0) * px_m for p in places]
            j = int(np.argmin(d))
            if d[j] <= SETTLEMENT_NAME_M:
                name, src = places[j]["name"], "OpenStreetMap place"
        out.append({"id": len(out) + 1, "name": name or f"Settlement {len(out) + 1}", "namedFrom": src, "col": round(float(c0), 1), "row": round(float(r0), 1),
                    "buildings": ids, "nBuildings": len(ids), "population": round(population([by_id[i] for i in ids]))})
    return out
