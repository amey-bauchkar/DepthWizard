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
from core.terrain.building_stats import STATS_VERSION, footprint_stats
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    label as nd_label,
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
        # watershed only floods basins that contain a marker: a mask component without a peak (e.g. a building
        # within min_distance of the image border, where peak_local_max excludes peaks) would be silently dropped
        # (Zürich demo: 11 buildings >= 15 m^2). Every such component becomes its own instance, so labels partition
        # the mask exactly.
        orphan = mask & (labels == 0)
        if orphan.any():
            extra, k = nd_label(orphan)
            labels = np.where(extra > 0, extra + labels.max(), labels)
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
    return_labels: bool = False,
    transform: Any = None,
    footprints: tuple[np.ndarray, dict[int, list[tuple[float, float]]]] | None = None,
    object_filter: bool = True,
) -> dict[str, Any]:
    """Extract LoD-1 buildings — one clean block per building.

    Pipeline:
      1. Build building mask (RGB spectral + nDSM rules, or height threshold)
      2. Marker-controlled watershed to separate touching buildings without area loss
      3. Vectorize all components in a single pass
      4. Sample exact roof height & base elevation per building using fast bounding slices
      5. Simplify polygon + map to centered scene coordinates

    footprints=(labels, {id: outer ring in pixel coordinates}) (core.terrain.footprints.to_labels) replaces steps 1-2:
    the outlines come from a footprint dataset and only the heights from the nDSM. Without footprints, detected
    candidates are scored by the object filter (core.terrain.building_filter) and trees / rock are dropped.

    With return_labels=True the result also carries "_labels": an int32 raster where pixel value k is building id k
    (0 = none). It is the exact pixel set every per-building statistic is computed on; callers must pop it before
    JSON serialisation.
    """

    H, W = ndsm.shape
    # pixel->CRS affine; without one, a north-up grid of gsd_m cells (pixel area = |det J| either way)
    tr = transform if transform is not None else rasterio.Affine(gsd_m, 0.0, 0.0, 0.0, -gsd_m, 0.0)
    pixel_area_m2 = abs(tr.a * tr.e - tr.b * tr.d)
    max_area_m2 = 50000.0

    filter_report: dict[str, Any] | None = None
    if footprints is not None:
        return _finish(footprints[0], {k: v + [v[0]] for k, v in footprints[1].items()}, ndsm, terrain, tr, pixel_area_m2, gsd_m, H, W,
                       min_height_m=min_height_m, min_area_m2=min_area_m2, max_area_m2=max_area_m2, simplify_tol_m=0.0,
                       max_buildings=10 ** 9, return_labels=return_labels, segmentation_method="reference_footprints", gate_height=False, dedupe=False)

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
            **({"_labels": np.zeros((H, W), np.int32)} if return_labels else {}),
            "buildings": [], "count": 0, "min_height_m": min_height_m,
            "gsd_m": gsd_m, "extent": [(W - 1) * gsd_m, (H - 1) * gsd_m],
            "segmentation_method": segmentation_method,
        }

    # Cleanup noise. Pixels outside the raster are unknown, not background: edge-replicate before the morphology and
    # crop after, otherwise the opening erodes (and the closing cannot restore) every building touching the raster
    # edge by one pixel (measured: a 16 x 10 px edge block lost 10 % of its area).
    pad = 2
    padded = np.pad(base_mask, pad, mode="edge")
    padded = binary_opening(padded, structure=np.ones((2, 2), dtype=bool))
    clean = binary_closing(padded, structure=np.ones((3, 3), dtype=bool))[pad:-pad, pad:-pad]

    # ── Step 2: Separate touching buildings via watershed ─────────────────
    labeled, n_components = _separate_building_instances(clean, ndsm, gsd_m=gsd_m)
    log.info("Building instances separated via watershed: %d", n_components)
    if object_filter and rgb is not None and labeled.max() > 0:
        from core.terrain.building_filter import filter_labels

        labeled, filter_report = filter_labels(labeled, rgb, ndsm, gsd_m)
        if filter_report.get("applied"):
            segmentation_method += f"+object_filter_{filter_report['model']}"
            log.info("Object filter kept %d of %d candidates", filter_report["kept"], filter_report["candidates"])
            filter_report = {k: v for k, v in filter_report.items() if k != "probability"}

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
    out = _finish(labeled, {k: g["coordinates"][0] for k, (g, _a) in best_per_label.items()}, ndsm, terrain, tr, pixel_area_m2, gsd_m, H, W,
                  min_height_m=min_height_m, min_area_m2=min_area_m2, max_area_m2=max_area_m2, simplify_tol_m=simplify_tol_m,
                  max_buildings=max_buildings, return_labels=return_labels, segmentation_method=segmentation_method, gate_height=True, dedupe=True)
    if filter_report:
        out["object_filter"] = filter_report
    return out


def _finish(labeled: np.ndarray, rings: dict[int, list], ndsm: np.ndarray, terrain: np.ndarray, tr: Any, pixel_area_m2: float, gsd_m: float, H: int, W: int, *,
            min_height_m: float, min_area_m2: float, max_area_m2: float, simplify_tol_m: float, max_buildings: int, return_labels: bool,
            segmentation_method: str, gate_height: bool, dedupe: bool) -> dict[str, Any]:
    """Per-building statistics + scene-coordinate polygons for labelled footprints (detected or from a dataset)."""
    from scipy.ndimage import find_objects

    slices = find_objects(labeled)
    buildings: list[dict[str, Any]] = []
    extent_x = (W - 1) * gsd_m
    extent_y = (H - 1) * gsd_m

    for val, coords in rings.items():
        pts_col = np.array([p[0] for p in coords])
        pts_row = np.array([p[1] for p in coords])
        sl = slices[val - 1] if val - 1 < len(slices) else None
        if sl is None:
            continue
        sub_mask = labeled[sl] == val
        # area from the label's pixel set (holes excluded, all parts included): the same set the statistics use.
        # The outer-ring polygon area used before this audit counted courtyard holes as building area.
        area_m2_val = float(sub_mask.sum()) * pixel_area_m2
        if area_m2_val < min_area_m2 or area_m2_val > max_area_m2:
            continue
        sub_ndsm = ndsm[sl]
        det_h = sub_ndsm[sub_mask & np.isfinite(sub_ndsm)]
        low = det_h.size == 0 or float(np.percentile(det_h, 85)) < min_height_m
        # detection gate (segmentation rule): the upper roof level must clear min_height_m. A footprint from a dataset
        # is a known building: it is kept and flagged when the model sees it lower than that.
        if det_h.size == 0 or (gate_height and low):
            continue
        touches = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        st = footprint_stats(sub_ndsm, terrain[sl], sub_mask, transform=tr, row0=sl[0].start, col0=sl[1].start, touches_edge=touches)
        if st is None:
            continue
        if low:
            st["quality_flags"] = list(st.get("quality_flags", [])) + ["LOW_PREDICTED_HEIGHT"]

        # ── Simplify polygon ─────────────────────────────────────────
        ring = [[float(p[0]), float(p[1])] for p in coords]
        simplified = rdp_simplify(ring, epsilon=simplify_tol_m / gsd_m) if simplify_tol_m > 0 else ring
        if len(simplified) < 4:
            continue

        # Map to scene coordinates centered at origin
        scene_coords = []
        pixel_coords = []
        for p in simplified:
            col, row = float(p[0]), float(p[1])
            sx = (col * gsd_m) - (extent_x / 2.0)
            sy = (extent_y / 2.0) - (row * gsd_m)
            scene_coords.append([round(sx, 2), round(sy, 2)])
            pixel_coords.append([round(col, 1), round(row, 1)])

        buildings.append({
            "id": len(buildings) + 1,
            "_label": val,
            # height_m = V/A (LoD-1 extrusion; block volume == volume_m3); height_median_m = typical roof height
            **st,
            "coords": scene_coords,
            "pixel_coords": pixel_coords,
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
    if dedupe and len(buildings) > 1:
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

    # relabel so the raster value equals the final building id (dropped labels -> 0)
    lut = np.zeros(int(labeled.max()) + 1, dtype=np.int32)
    for b in buildings:
        lut[b.pop("_label")] = b["id"]
    extra: dict[str, Any] = {"_labels": lut[labeled]} if return_labels else {}

    return {
        **extra,
        "buildings": buildings,
        "count": len(buildings),
        "min_height_m": min_height_m,
        "gsd_m": gsd_m,
        "extent": [extent_x, extent_y],
        "segmentation_method": segmentation_method,
        "stats_version": STATS_VERSION,
    }


def save_lod1_buildings(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
