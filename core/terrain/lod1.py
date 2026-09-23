"""LoD-1 3D Building Extrusion (Level of Detail 1 Vector Building Blocks).

Extracts building footprints from the nDSM object layer, calculates per-building
median heights, simplifies polygon perimeters (RDP algorithm), and generates
architectural 3D building instances for clean, un-smeared urban visualization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.features
from scipy.ndimage import binary_closing, binary_opening


def rdp_simplify(points: list[list[float]], epsilon: float = 1.0) -> list[list[float]]:
    """Ramer-Douglas-Peucker polygon simplification to eliminate pixel-grid stairsteps."""
    if len(points) <= 3:
        return points

    pts = np.array(points, dtype=np.float64)
    start, end = pts[0], pts[-1]

    # Vector along chord
    line_vec = end - start
    line_len = np.hypot(line_vec[0], line_vec[1])

    if line_len < 1e-9:
        dists = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        # Distance from points to line segment
        dists = np.abs(line_vec[1] * pts[:, 0] - line_vec[0] * pts[:, 1] + end[0] * start[1] - end[1] * start[0]) / line_len

    idx = int(np.argmax(dists))
    max_dist = float(dists[idx])

    if max_dist > epsilon and 0 < idx < len(points) - 1:
        left = rdp_simplify(points[: idx + 1], epsilon)
        right = rdp_simplify(points[idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def extract_lod1_buildings(
    ndsm: np.ndarray,
    terrain: np.ndarray,
    *,
    gsd_m: float = 1.0,
    min_height_m: float = 2.5,
    min_area_m2: float = 25.0,
    max_buildings: int = 500,
    simplify_tol_m: float = 1.2,
) -> dict[str, Any]:
    """Extracts LoD-1 vector building geometries with height and base elevation attributes.

    Parameters
    ----------
    ndsm : np.ndarray
        2D float32 object height above ground in metres.
    terrain : np.ndarray
        2D float32 ground terrain elevation in metres.
    gsd_m : float
        Ground sampling distance in metres per pixel.
    min_height_m : float
        Minimum object height to classify as building structure (default: 2.5m).
    min_area_m2 : float
        Minimum footprint area in square metres (default: 25.0m²).
    max_buildings : int
        Cap on total building instances rendered for performance.
    simplify_tol_m : float
        Polygon simplification tolerance in metres.

    Returns
    -------
    dict[str, Any]
        JSON-serializable catalog of 3D building instances.
    """
    H, W = ndsm.shape
    pixel_area_m2 = gsd_m * gsd_m
    min_pixels = max(4, int(round(min_area_m2 / max(pixel_area_m2, 1e-6))))

    # Building mask: height above ground threshold
    mask = (ndsm >= min_height_m) & np.isfinite(ndsm) & np.isfinite(terrain)
    if not mask.any():
        return {"buildings": [], "count": 0, "min_height_m": min_height_m, "extent": [W, H]}

    # Morphological cleanup: eliminate single-pixel noise and close small internal roof holes
    cleaned = binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    cleaned = binary_closing(cleaned, structure=np.ones((2, 2), dtype=bool))

    # Raster to vector polygon shapes (in pixel coordinates)
    shapes_gen = rasterio.features.shapes(
        cleaned.astype(np.int16),
        mask=cleaned,
        transform=rasterio.Affine.identity(),
    )

    buildings = []
    extent_x = (W - 1) * gsd_m
    extent_y = (H - 1) * gsd_m

    for geom, val in shapes_gen:
        if val != 1:
            continue
        coords = geom["coordinates"][0]  # outer ring
        if len(coords) < 4:
            continue

        # Convert coordinates to pixel arrays
        pts_col = np.array([p[0] for p in coords])
        pts_row = np.array([p[1] for p in coords])

        # Approximate polygon area (Shoelace formula)
        area_px = 0.5 * np.abs(np.dot(pts_col[:-1], pts_row[1:]) - np.dot(pts_col[1:], pts_row[:-1]))
        if area_px < min_pixels:
            continue

        # Sample interior pixels to determine median building height and ground elevation
        c_min, c_max = max(0, int(np.floor(pts_col.min()))), min(W - 1, int(np.ceil(pts_col.max())))
        r_min, r_max = max(0, int(np.floor(pts_row.min()))), min(H - 1, int(np.ceil(pts_row.max())))

        sub_ndsm = ndsm[r_min : r_max + 1, c_min : c_max + 1]
        sub_mask = cleaned[r_min : r_max + 1, c_min : c_max + 1]
        sub_terr = terrain[r_min : r_max + 1, c_min : c_max + 1]

        valid_vals = sub_ndsm[sub_mask & np.isfinite(sub_ndsm)]
        if not valid_vals.size:
            continue

        height_m = float(np.median(valid_vals))
        if height_m < min_height_m:
            continue

        terr_vals = sub_terr[sub_mask & np.isfinite(sub_terr)]
        base_elev_m = float(np.median(terr_vals)) if terr_vals.size else 0.0

        # Simplify coordinates to remove pixel staircase steps
        simplified_pts = rdp_simplify([[float(p[0]), float(p[1])] for p in coords], epsilon=simplify_tol_m / gsd_m)

        # Map to scene coordinates centered at origin (0, 0)
        scene_coords = []
        for p in simplified_pts:
            col, row = p[0], p[1]
            sx = (col * gsd_m) - (extent_x / 2.0)
            sy = (extent_y / 2.0) - (row * gsd_m)
            scene_coords.append([round(sx, 2), round(sy, 2)])

        buildings.append({
            "id": len(buildings) + 1,
            "height_m": round(height_m, 2),
            "base_elev_m": round(base_elev_m, 2),
            "area_m2": round(float(area_px * pixel_area_m2), 1),
            "coords": scene_coords,
            "pixel_bbox": [c_min, r_min, c_max, r_max],
        })

        if len(buildings) >= max_buildings:
            break

    return {
        "buildings": buildings,
        "count": len(buildings),
        "min_height_m": min_height_m,
        "gsd_m": gsd_m,
        "extent": [extent_x, extent_y],
    }


def save_lod1_buildings(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
