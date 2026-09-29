"""Flood screening vs an OBSERVED real flood extent (scripts/glof_observed_extent.py):
  chungthang       4 Oct 2023 Teesta GLOF             -> docs/flood_validation_glof.json
  nepal_sunkoshi   27-28 Sep 2024 Sunkoshi flood      -> docs/flood_validation_nepal_sunkoshi.json

Observed flooded = Sentinel-2 flood scar (new sediment / water) OR the river channel that was water before.
Modelled flooded = wet cells of the screening, averaged to the 10 m Sentinel-2 grid (wet if >= 50 % of the cell).
Scores: precision, recall, F1 (= Dice) and IoU over the scene, for every water height of each model, and the best one.
The GLOF's true stage at Chungthang is not known here, so the best stage is REPORTED (with its F1) rather than
assumed; the question answered is: which model can reproduce the observed footprint, and how well.

Prerequisites: python scripts/hazard_study_prepare.py <scene> ; python scripts/glof_observed_extent.py <scene>
Usage: python scripts/validate_flood.py [scene]   (default chungthang)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.disaster.flood import run_flood_screening  # noqa: E402


def to10(a: np.ndarray, src_tr, crs, dst_tr, shape) -> np.ndarray:
    out = np.zeros(shape, np.float32)
    reproject(a.astype(np.float32), out, src_transform=src_tr, src_crs=crs, dst_transform=dst_tr, dst_crs=crs, resampling=Resampling.average)
    return out


def scores(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> dict:
    p, o = pred[valid], obs[valid]
    tp, fp, fn = float((p & o).sum()), float((p & ~o).sum()), float((~p & o).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0, "iou": round(tp / (tp + fp + fn), 3) if tp else 0.0,
            "wet_ha": round(float(p.sum()) * 0.01, 2)}


OUTPUT = {"chungthang": "flood_validation_glof.json"}


def main() -> int:
    scene = sys.argv[1] if len(sys.argv) > 1 else "chungthang"
    d = ROOT / "data" / "hazard_study" / scene
    event = scene.split("__")[0]  # "<scene>__<tag>": the same scene processed with another DEM
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    g = ROOT / "data" / "glof"
    with rasterio.open(g / f"{event}_affected.tif") as ds:
        aff = ds.read(1); tr10, crs, shape = ds.transform, ds.crs, ds.shape
    with rasterio.open(g / f"{event}_river_before.tif") as ds:
        riv = ds.read(1)
    valid = (aff != 255) & (riv != 255)
    obs = ((aff == 1) | (riv == 1)) & valid
    with rasterio.open(d / "terrain.tif") as ds:
        tr0 = ds.transform
        lo, hi = float(np.nanpercentile(ds.read(1, masked=True).filled(np.nan), 1)), float(np.nanpercentile(ds.read(1, masked=True).filled(np.nan), 60))
    out = {"observed_ha": round(float(obs.sum()) * 0.01, 2), "observed": json.loads((g / f"{event}_s2.json").read_text(encoding="utf-8")), "river": [], "level": []}
    for model, levels in (("river", np.arange(0.0, 40.1, 1.0)), ("level", np.arange(np.floor(lo), np.ceil(hi) + 1, 2.0))):
        for h in levels:
            run_flood_screening(d, float(h), res, model=model)
            with rasterio.open(d / "flood_depth.tif") as ds:
                dep = ds.read(1, masked=True).filled(np.nan)
            wet = to10(np.nan_to_num(dep) > 0, tr0, crs, tr10, shape) >= 0.5
            out[model].append({"h": float(h), **scores(wet, obs, valid)})
    for model in ("river", "level"):
        b = max(out[model], key=lambda r: r["f1"])
        out[f"best_{model}"] = b
        print(f"{model:5s} best: h {b['h']:g} m -> F1 {b['f1']} IoU {b['iou']} (precision {b['precision']}, recall {b['recall']}, wet {b['wet_ha']} ha vs observed {out['observed_ha']} ha)")
    rows = [r for r in out["river"] if r["h"] in (3, 5, 10, 15, 20, 25, 30)]
    print("river stage sweep:", [(r["h"], r["f1"]) for r in rows])
    (ROOT / "docs" / OUTPUT.get(scene, f"flood_validation_{scene}.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
