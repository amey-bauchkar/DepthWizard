"""Building footprints from open data for the LoD-1 city: exact outlines from a footprint dataset, heights from the
DepthWizard nDSM.

Why: footprints detected from one image confuse tree crowns and rock with buildings. Measured against Microsoft Global ML
Building Footprints (docs/building_detection_validation.md), only 2-3 % of the "building" area detected on the rural
Sikkim demos was on real buildings. Footprint datasets exist for all of India (Microsoft, Google Open Buildings,
OpenStreetMap, Bhuvan), so DepthWizard uses one whenever it covers the scene and falls back to detection otherwise.

Sources, in order: a GeoJSON uploaded with the job > bundled footprints under assets/footprints (index.json, written by
scripts/fetch_building_footprints.py) that cover the scene > none (detection).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize

MIN_COVER = 0.9  # a bundled file is used only if its coverage box holds >= 90 % of the scene


def _polygons(geom: dict[str, Any]) -> list[list[list[float]]]:
    """Outer rings of a Polygon / MultiPolygon."""
    if geom.get("type") == "Polygon":
        return [geom["coordinates"][0]]
    if geom.get("type") == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


def load_geojson(path: str | Path) -> tuple[list[list[list[float]]], str]:
    """Rings (outer boundaries) and their CRS ("EPSG:4326" unless the file declares another) from a GeoJSON file."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    crs = "EPSG:4326"
    name = ((d.get("crs") or {}).get("properties") or {}).get("name")
    if name:
        crs = CRS.from_user_input(name).to_string()
    feats = d.get("features", []) if d.get("type") == "FeatureCollection" else [d]
    rings = [r for f in feats if f.get("geometry") for r in _polygons(f["geometry"])]
    return rings, crs


def discover(bounds_wgs84: tuple[float, float, float, float], fp_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    idx = Path(fp_dir) / "index.json"
    if not idx.exists():
        return None
    w, s, e, n = bounds_wgs84
    area = max(1e-12, (e - w) * (n - s))
    best = None
    for it in json.loads(idx.read_text(encoding="utf-8")).get("files", []):
        bw, bs, be, bn = it["bounds_wgs84"]
        ix, iy = max(0.0, min(e, be) - max(w, bw)), max(0.0, min(n, bn) - max(s, bs))
        cover = ix * iy / area
        if cover >= MIN_COVER and (best is None or cover > best[0]):
            best = (cover, it)
    if best is None:
        return None
    return Path(fp_dir) / best[1]["file"], {**best[1], "coverage": round(best[0], 3)}


def to_labels(rings: list[list[list[float]]], rings_crs: str, transform: Affine, crs: str, shape: tuple[int, int]) -> tuple[np.ndarray, dict[int, list[tuple[float, float]]]]:
    """Rasterise footprints onto the job grid. Returns (labels int32: pixel value = building id, 0 = none;
    {id: outer ring in continuous pixel coordinates (col, row), pixel corners at integers})."""
    H, W = shape
    to = Transformer.from_crs(rings_crs, crs, always_xy=True) if CRS.from_user_input(rings_crs) != CRS.from_user_input(crs) else None
    inv = ~transform
    shapes, polys = [], {}
    k = 0
    for ring in rings:
        xy = [to.transform(x, y) for x, y in ring] if to else [(float(x), float(y)) for x, y in ring]
        px = [inv * p for p in xy]
        cs, rs = [p[0] for p in px], [p[1] for p in px]
        if max(cs) < 0 or min(cs) > W or max(rs) < 0 or min(rs) > H:
            continue
        k += 1
        shapes.append(({"type": "Polygon", "coordinates": [xy]}, k))
        polys[k] = [(float(c), float(r)) for c, r in px]
    if not shapes:
        return np.zeros(shape, np.int32), {}
    labels = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    present = np.unique(labels)
    polys = {i: p for i, p in polys.items() if i in set(present.tolist())}  # footprints smaller than a pixel vanish
    return labels, polys


MAX_SHIFT_M = 10.0
MIN_GAIN = 0.10  # the shift must raise the roof evidence under the footprints by >= 10 % to be applied


def coregister(labels: np.ndarray, rgb: np.ndarray, ndsm: np.ndarray, gsd_m: float, max_shift_m: float = MAX_SHIFT_M) -> dict[str, Any]:
    """Shift (in pixels) that best places the footprint outlines on the roofs of THIS image.

    Footprint datasets come from other imagery with its own orthorectification; on hills their outlines can sit metres
    off the roofs, and the median height over a misplaced footprint samples the ground. Roof evidence = the nDSM
    (clipped to 8 m so tall trees do not dominate) on non-green pixels. The shift maximising the evidence under all
    footprints together is found by FFT cross-correlation, limited to +-max_shift_m, and applied only when it raises the
    evidence by >= 10 %. Returns {"applied", "dcol", "drow", "shift_m", "gain"}."""
    mask = (labels > 0).astype(np.float32)
    if mask.sum() < 50 or rgb is None:
        return {"applied": False, "dcol": 0, "drow": 0, "reason": "too few footprint pixels"}
    if rgb.shape[:2] != labels.shape:
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb).resize((labels.shape[1], labels.shape[0]), Image.BILINEAR))
    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1e-6)
    roof = (np.clip(np.nan_to_num(ndsm), 0.0, 8.0) * (exg < 0.02)).astype(np.float32)
    corr = np.fft.irfft2(np.fft.rfft2(roof) * np.conj(np.fft.rfft2(mask)), s=roof.shape)  # corr[k] = sum_p mask(p) roof(p + k)
    r = max(1, int(round(max_shift_m / gsd_m)))
    ks = np.arange(-r, r + 1)
    win = corr[np.ix_(ks % roof.shape[0], ks % roof.shape[1])]
    i, j = np.unravel_index(int(np.argmax(win)), win.shape)
    drow, dcol = int(ks[i]), int(ks[j])
    zero = float(corr[0, 0])
    gain = (float(win[i, j]) - zero) / max(zero, 1e-6)
    on_edge = abs(drow) == r or abs(dcol) == r  # the maximum sits on the search limit: no real peak, do not trust it
    applied = gain >= MIN_GAIN and (drow, dcol) != (0, 0) and not on_edge
    why = "applied" if applied else ("not applied (best shift on the search limit: no clear alignment peak)" if on_edge and gain >= MIN_GAIN else f"not applied (roof-evidence gain {gain:.0%} < {MIN_GAIN:.0%})")
    return {"applied": bool(applied), "dcol": dcol if applied else 0, "drow": drow if applied else 0, "shift_m": round(float(np.hypot(drow, dcol)) * gsd_m, 2) if applied else 0.0,
            "gain": round(gain, 3), "reason": why}


def shifted(transform: Affine, reg: dict[str, Any]) -> Affine:
    """Transform that rasterises the footprints moved by (dcol, drow) pixels."""
    return transform * Affine.translation(-reg.get("dcol", 0), -reg.get("drow", 0))
