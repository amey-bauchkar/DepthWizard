"""Object-level building / not-building decision for LoD-1 candidates (trees, rocks and terrain bumps are rejected).

The pixel mask in building_segmentation.py is permissive: anything elevated that is not clearly green passes, so shaded
tree crowns, rock outcrops and forest edges became "buildings" (measured on the Sikkim demos: 3-10x more objects than
the Microsoft ML building footprints). This module scores every candidate OBJECT with a small logistic model on
explainable features (colour, texture, roof flatness, shape, size) and keeps only probable buildings.

The weights are learned by scripts/validate_buildings.py against Microsoft Global ML Building Footprints (ODbL) on the
demo scenes (India / Switzerland / Türkiye) and stored in building_filter_model.json with their leave-one-scene-out
scores. At runtime only numpy + scikit-image are needed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

MODEL_PATH = Path(__file__).with_name("building_filter_model.json")
FEATURES = ("exg_med", "green_frac", "sat_med", "bright_med", "bright_cv", "texture", "rg_med", "bg_med",
            "h_med", "h_flat", "h_grad", "log_area", "solidity", "rect_fill", "elong", "dark_frac")


def _pixel_maps(rgb: np.ndarray, ndsm: np.ndarray, gsd_m: float) -> dict[str, np.ndarray]:
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-6
    mx, mn = f.max(axis=2), f.min(axis=2)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    hz = np.where(np.isfinite(ndsm), ndsm, 0.0).astype(np.float32)
    gy, gx = np.gradient(hz, gsd_m)
    return {
        "exg": (2 * g - r - b) / s,
        "sat": (mx - mn) / (mx + 1e-6),
        "bright": gray,
        "texture": np.abs(ndimage.laplace(ndimage.gaussian_filter(gray, 0.7))),
        "rg": (r - g) / (r + g + 1e-6),
        "bg": (b - g) / (b + g + 1e-6),
        "h": hz,
        "hgrad": np.hypot(gx, gy),
    }


def _oriented_fill(rr: np.ndarray, cc: np.ndarray) -> tuple[float, float]:
    """Area / area of the principal-axis-aligned bounding box (1 = perfect rectangle) and elongation."""
    if rr.size < 5:
        return 1.0, 1.0
    y, x = rr - rr.mean(), cc - cc.mean()
    cov = np.cov(np.vstack([x, y]))
    w, v = np.linalg.eigh(cov)
    u = np.vstack([x, y]).T @ v
    ext = (u.max(axis=0) - u.min(axis=0)) + 1.0
    return float(rr.size / (ext[0] * ext[1])), float(max(ext) / max(1.0, min(ext)))


def object_features(labels: np.ndarray, rgb: np.ndarray, ndsm: np.ndarray, gsd_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Features (len(FEATURES) columns) of every labelled object. Returns (ids, X)."""
    from skimage.measure import regionprops

    if rgb.shape[:2] != labels.shape:
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb).resize((labels.shape[1], labels.shape[0]), Image.BILINEAR))
    M = _pixel_maps(rgb, ndsm, gsd_m)
    dark = M["bright"] < 50
    ids, rows = [], []
    for rp in regionprops(labels):
        sl = rp.slice
        m = labels[sl] == rp.label
        v = {k: a[sl][m] for k, a in M.items()}
        h = v["h"]
        hmed = float(np.median(h))
        nmad = float(1.4826 * np.median(np.abs(h - hmed)))
        bmean = float(v["bright"].mean())
        rr, cc = np.nonzero(m)
        fill, elong = _oriented_fill(rr, cc)
        rows.append([
            float(np.median(v["exg"])), float((v["exg"] > 0.02).mean()), float(np.median(v["sat"])), float(np.median(v["bright"])),
            float(v["bright"].std() / (bmean + 1e-6)), float(v["texture"].mean()), float(np.median(v["rg"])), float(np.median(v["bg"])),
            hmed, nmad / max(1.0, hmed), float(np.median(v["hgrad"])), math.log(rp.area * gsd_m * gsd_m),
            float(rp.solidity), fill, math.log(elong), float(dark[sl][m].mean()),
        ])
        ids.append(rp.label)
    return np.asarray(ids, np.int64), np.asarray(rows, np.float64).reshape(-1, len(FEATURES))


def load_model(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def building_probability(X: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = Z @ np.asarray(model["coef"]) + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def filter_labels(labels: np.ndarray, rgb: np.ndarray, ndsm: np.ndarray, gsd_m: float, model: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only objects the model calls buildings. Returns (labels with rejected objects set to 0, report)."""
    model = model if model is not None else load_model()
    if model is None or labels.max() == 0:
        return labels, {"applied": False, "reason": "no model" if model is None else "no candidates"}
    ids, X = object_features(labels, rgb, ndsm, gsd_m)
    p = building_probability(X, model)
    keep = ids[p >= model["threshold"]]
    lut = np.zeros(int(labels.max()) + 1, bool)
    lut[keep] = True
    out = np.where(lut[labels], labels, 0).astype(labels.dtype)
    return out, {"applied": True, "candidates": int(ids.size), "kept": int(keep.size), "threshold": model["threshold"], "model": model.get("version"),
                 "probability": {int(i): round(float(q), 3) for i, q in zip(ids, p)}}
