"""LoD-1 3D Building Extrusion — Clean Single-Block Buildings.

Each building = ONE polygon + ONE height = ONE vertical block.
No stacking. No tier splitting. No merged blobs.

Fast extraction using connected-component labeling + gradient boundary carving.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.features
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    label as nd_label,
    sobel,
)

log = logging.getLogger(__name__)


def rdp_simplify(points: list[list[float]], epsilon: float = 1.0) -> list[list[float]]:
    """Ramer-Douglas-Peucker polygon simplification."""
    if len(points) <= 3:
        return points
    pts = np.array(points, dtype=np.float64)
    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = np.hypot(line_vec[0], line_vec[1])
    if line_len < 1e-9:
        dists = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        dists = np.abs(
            line_vec[1] * pts[:, 0] - line_vec[0] * pts[:, 1]
            + end[0] * start[1] - end[1] * start[0]
        ) / line_len
    idx = int(np.argmax(dists))
    if float(dists[idx]) > epsilon and 0 < idx < len(points) - 1:
        left = rdp_simplify(points[: idx + 1], epsilon)
        right = rdp_simplify(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _separate_building_instances(
    mask: np.ndarray,
    ndsm: np.ndarray,
    *,
    gsd_m: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Separate touching buildings into distinct instances using watershed.

    Uses distance transform peaks guided by height gradients. Unlike destructive
    carving, watershed partitions touching structures along natural ridges
    without erasing any building footprint area (preserving ~100% of structures).
    """
    try:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from scipy.ndimage import distance_transform_edt, sobel

        # Distance transform gives geometry peaks (building centres)
        dist = distance_transform_edt(mask)
        if dist.max() < 2.0:
            labeled, n = nd_label(mask)
            return labeled, n

        # Height gradient inside building mask acts as watershed barriers between roof levels
        ndsm_clean = np.where(mask & np.isfinite(ndsm), ndsm, 0.0)
        gx = sobel(ndsm_clean, axis=1)
        gy = sobel(ndsm_clean, axis=0)
        grad = np.hypot(gx, gy)
        ws_surface = -dist + np.clip(grad * 1.5, 0, 15)

        # Minimum distance between building centres: ~4.5m
        min_dist_px = max(6, int(round(4.5 / max(gsd_m, 0.1))))
        coords = peak_local_max(dist, min_distance=min_dist_px, threshold_abs=2.5)

        if len(coords) == 0:
            labeled, n = nd_label(mask)
            return labeled, n

        markers = np.zeros_like(mask, dtype=np.int32)
        for idx, (r, c) in enumerate(coords, start=1):
            markers[r, c] = idx

        labels = watershed(ws_surface, markers, mask=mask)
        return labels, int(labels.max())
    except Exception as e:
        log.warning("Watershed instance separation fallback to connected components: %s", e)
        labeled, n = nd_label(mask)
        return labeled, n


def extract_lod1_buildings(
    ndsm: np.ndarray,
    terrain: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    gsd_m: float = 1.0,
    min_height_m: float = 2.2,
    min_area_m2: float = 15.0,
    max_buildings: int = 3000,
    simplify_tol_m: float = 1.0,
) -> dict[str, Any]:
    """Extract LoD-1 buildings — one clean block per building.

    Pipeline:
      1. Build building mask (RGB spectral + nDSM rules, or height threshold)
      2. Marker-controlled watershed to separate touching buildings without area loss
      3. Vectorize all components in a single pass
      4. Sample exact roof height & base elevation per building using fast bounding slices
      5. Simplify polygon + map to centered scene coordinates
    """
    from scipy.ndimage import find_objects

    H, W = ndsm.shape
    pixel_area_m2 = gsd_m * gsd_m
    min_area_px = max(4, int(round(min_area_m2 / max(pixel_area_m2, 1e-6))))
    max_area_m2 = 50000.0

    # ── Step 1: Build unified building mask ───────────────────────────────
    if rgb is not None:
        try:
            from core.terrain.building_segmentation import extract_building_mask
            base_mask = extract_building_mask(
                rgb, ndsm,
                min_height_m=min_height_m,
                veg_threshold=0.05,
                building_score_threshold=0.3,
            )
            segmentation_method = "rgb_spectral_ndsm_rules"
        except Exception as e:
            log.warning("Building mask failed, falling back to height threshold: %s", e)
            base_mask = (ndsm >= min_height_m) & np.isfinite(ndsm) & np.isfinite(terrain) & (ndsm < 200.0)
            segmentation_method = "height_threshold_fallback"
    else:
        base_mask = (ndsm >= min_height_m) & np.isfinite(ndsm) & np.isfinite(terrain) & (ndsm < 200.0)
        segmentation_method = "height_threshold"

    if not base_mask.any():
        return {
            "buildings": [], "count": 0, "min_height_m": min_height_m,
            "gsd_m": gsd_m, "extent": [(W - 1) * gsd_m, (H - 1) * gsd_m],
            "segmentation_method": segmentation_method,
        }

    # Cleanup noise
    clean = binary_opening(base_mask, structure=np.ones((2, 2), dtype=bool))
    clean = binary_closing(clean, structure=np.ones((3, 3), dtype=bool))

    # ── Step 2: Separate touching buildings via watershed ─────────────────
    labeled, n_components = _separate_building_instances(clean, ndsm, gsd_m=gsd_m)
    log.info("Building instances separated via watershed: %d", n_components)

    # Fast spatial slice precomputation for per-building height & terrain sampling
    slices = find_objects(labeled)

    # ── Step 3: Vectorize ALL components at once ──────────────────────────
    shapes_gen = rasterio.features.shapes(
        labeled.astype(np.int32),
        mask=(labeled > 0),
        transform=rasterio.Affine.identity(),
    )

    # Collect best polygon per label
    best_per_label: dict[int, tuple[Any, float]] = {}
    for geom, val in shapes_gen:
        val = int(val)
        if val == 0:
            continue
        coords = geom["coordinates"][0]
        if len(coords) < 4:
            continue
        pts_col = np.array([p[0] for p in coords])
        pts_row = np.array([p[1] for p in coords])
        area_px = 0.5 * np.abs(
            np.dot(pts_col[:-1], pts_row[1:]) - np.dot(pts_col[1:], pts_row[:-1])
        )
        if val not in best_per_label or area_px > best_per_label[val][1]:
            best_per_label[val] = (geom, area_px)

    log.info("Unique building labels with valid polygons: %d", len(best_per_label))

    buildings: list[dict[str, Any]] = []
    extent_x = (W - 1) * gsd_m
    extent_y = (H - 1) * gsd_m

    for val, (geom, area_px) in best_per_label.items():
        area_m2_val = area_px * pixel_area_m2
        if area_m2_val < min_area_m2 or area_m2_val > max_area_m2:
            continue

        coords = geom["coordinates"][0]
        pts_col = np.array([p[0] for p in coords])
        pts_row = np.array([p[1] for p in coords])

        # ── Fast sub-slice height & terrain sampling ─────────────────
        if val - 1 < len(slices) and slices[val - 1] is not None:
            sl = slices[val - 1]
            sub_mask = (labeled[sl] == val)
            sub_ndsm = ndsm[sl]
            valid_heights = sub_ndsm[sub_mask & np.isfinite(sub_ndsm)]
            if valid_heights.size == 0:
                continue
            height_m = float(np.percentile(valid_heights, 85))
            if height_m < min_height_m:
                continue

            sub_terr = terrain[sl]
            valid_terr = sub_terr[sub_mask & np.isfinite(sub_terr)]
            base_elev_m = float(np.median(valid_terr)) if valid_terr.size else 0.0
        else:
            comp_mask = (labeled == val)
            valid_heights = ndsm[comp_mask & np.isfinite(ndsm)]
            if valid_heights.size == 0:
                continue
            height_m = float(np.percentile(valid_heights, 85))
            if height_m < min_height_m:
                continue
            terr_vals = terrain[comp_mask & np.isfinite(terrain)]
            base_elev_m = float(np.median(terr_vals)) if terr_vals.size else 0.0

        # ── Simplify polygon ─────────────────────────────────────────
        simplified = rdp_simplify(
            [[float(p[0]), float(p[1])] for p in coords],
            epsilon=simplify_tol_m / gsd_m,
        )
        if len(simplified) < 4:
            continue

        # Map to scene coordinates centered at origin
        scene_coords = []
        for p in simplified:
            col, row = p[0], p[1]
            sx = (col * gsd_m) - (extent_x / 2.0)
            sy = (extent_y / 2.0) - (row * gsd_m)
            scene_coords.append([round(sx, 2), round(sy, 2)])

        buildings.append({
            "id": len(buildings) + 1,
            "height_m": round(height_m, 2),
            "base_elev_m": round(base_elev_m, 2),
            "area_m2": round(float(area_m2_val), 1),
            # robust within-footprint statistics of the nDSM (for the building table; height_m drives the extrusion)
            "height_p10_m": round(float(np.percentile(valid_heights, 10)), 2),
            "height_median_m": round(float(np.median(valid_heights)), 2),
            "height_p90_m": round(float(np.percentile(valid_heights, 90)), 2),
            "n_pixels": int(valid_heights.size),
            "coords": scene_coords,
            "pixel_bbox": [
                int(np.floor(pts_col.min())),
                int(np.floor(pts_row.min())),
                int(np.ceil(pts_col.max())),
                int(np.ceil(pts_row.max())),
            ],
        })

        if len(buildings) >= max_buildings:
            break

    # ── Step 5: Deduplicate courtyard hole polygons & contained fragments ──
    # Watershed labels are disjoint, but outer rings ignore holes: a block inside a courtyard overlaps the filled
    # outer polygon. Candidate pairs come from a vectorised bbox test; only those are rasterised and compared.
    if len(buildings) > 1:
        try:
            from PIL import Image, ImageDraw

            bb = np.array([b["pixel_bbox"] for b in buildings], dtype=np.int64)
            area = np.maximum(1, (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]))
            to_remove: set[int] = set()

            def raster(idx: int, min_x: int, min_y: int, w: int, h: int) -> np.ndarray:
                im = Image.new("1", (w, h), 0)
                pts = [(((p[0] + extent_x / 2.0) / gsd_m) - min_x, ((extent_y / 2.0 - p[1]) / gsd_m) - min_y) for p in buildings[idx]["coords"]]
                ImageDraw.Draw(im).polygon(pts, fill=1)
                return np.array(im)

            for i in range(len(buildings) - 1):
                if i in to_remove:
                    continue
                j = np.arange(i + 1, len(buildings))
                ox = np.clip(np.minimum(bb[i, 2], bb[j, 2]) - np.maximum(bb[i, 0], bb[j, 0]), 0, None)
                oy = np.clip(np.minimum(bb[i, 3], bb[j, 3]) - np.maximum(bb[i, 1], bb[j, 1]), 0, None)
                cand = j[(ox * oy) / np.minimum(area[i], area[j]) > 0.5]
                for jj in cand.tolist():
                    if jj in to_remove:
                        continue
                    min_x, min_y = int(max(bb[i, 0], bb[jj, 0])), int(max(bb[i, 1], bb[jj, 1]))
                    w = max(1, int(min(bb[i, 2], bb[jj, 2])) - min_x + 1)
                    h = max(1, int(min(bb[i, 3], bb[jj, 3])) - min_y + 1)
                    a1, a2 = raster(i, min_x, min_y, w, h), raster(jj, min_x, min_y, w, h)
                    n1, n2 = int(a1.sum()), int(a2.sum())
                    if min(n1, n2) > 0 and int((a1 & a2).sum()) / min(n1, n2) > 0.4:
                        if n1 <= n2:
                            to_remove.add(i)
                            break
                        to_remove.add(jj)
            if to_remove:
                buildings = [b for idx, b in enumerate(buildings) if idx not in to_remove]
                for idx, b in enumerate(buildings):
                    b["id"] = idx + 1
                log.info("LoD-1: Removed %d courtyard/contained overlap polygons", len(to_remove))
        except Exception as e:  # noqa: BLE001
            log.debug("LoD-1 overlap deduplication skipped: %s", e)

    log.info("LoD-1: %d clean non-overlapping buildings via %s", len(buildings), segmentation_method)

    return {
        "buildings": buildings,
        "count": len(buildings),
        "min_height_m": min_height_m,
        "gsd_m": gsd_m,
        "extent": [extent_x, extent_y],
        "segmentation_method": segmentation_method,
    }


def save_lod1_buildings(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
