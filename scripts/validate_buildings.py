"""Building-detection study: how many LoD-1 "buildings" are real, measured against an independent reference, and the
object-level filter (core/terrain/building_filter.py) that is trained and evaluated here.

Reference: Microsoft Global ML Building Footprints (ODbL; https://github.com/microsoft/GlobalMLBuildingFootprints), the
footprints inside each demo scene (assets/footprints/<scene>.geojson, clipped to the scene). It is itself an ML product from other imagery
dates, so the metrics are agreement with an independent detector, not with a survey. Area metrics tolerate 2 m of
misalignment (different imagery / building lean).

Protocol: leave-one-scene-out. For every scene the filter is trained on the OTHER scenes only and evaluated on it, so
every reported number is on a scene the model never saw. The shipped model is then trained on all scenes.

Prerequisites: python scripts/building_study_prepare.py  (processes the scenes with the current pipeline)
Usage:         python scripts/validate_buildings.py
Writes core/terrain/building_filter_model.json, docs/building_detection_validation.{md,json}
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.terrain.building_filter import FEATURES, MODEL_PATH, building_probability, object_features  # noqa: E402

STUDY = ROOT / "data" / "building_study"
REF_DIR = ROOT / "assets" / "footprints"  # the bundled Microsoft footprints (scripts/fetch_building_footprints.py)
TOL_M = 2.0
POS, NEG = 0.5, 0.1  # object label: >= 50 % of its area on a reference footprint = building; <= 10 % = not; else ambiguous


def load_scene(name: str) -> dict:
    d = STUDY / name
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    g = res["grid"]
    tr = Affine.from_gdal(*g["transform"])
    gsd = float(res.get("gsd_m") or abs(tr.a))
    with rasterio.open(d / "building_labels.tif") as ds:
        labels = ds.read(1).astype(np.int32)
    with rasterio.open(d / "ndsm.tif") as ds:
        ndsm = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
    rgb = np.asarray(Image.open(d / res["artifacts"]["input_preview"]).convert("RGB"))
    stem = Path(json.loads((STUDY / "index.json").read_text(encoding="utf-8"))[name]["input"]).stem
    to = Transformer.from_crs("EPSG:4326", g["crs"], always_xy=True)
    x0, y0 = tr * (0, 0)
    x1, y1 = tr * (g["width"], g["height"])
    shapes = []
    for f in json.loads((REF_DIR / f"{stem}.geojson").read_text(encoding="utf-8"))["features"]:
        ring = [to.transform(x, y) for x, y in f["geometry"]["coordinates"][0]]
        cx, cy = sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)
        if min(x0, x1) <= cx <= max(x0, x1) and min(y0, y1) <= cy <= max(y0, y1):
            shapes.append(({"type": "Polygon", "coordinates": [ring]}, 1))
    feats = shapes
    ref = rasterize(shapes, out_shape=labels.shape, transform=tr, fill=0, dtype="uint8").astype(bool) if shapes else np.zeros(labels.shape, bool)
    k = max(1, int(round(TOL_M / gsd)))
    ref_buf = ndimage.binary_dilation(ref, iterations=k)
    ids, X = object_features(labels, rgb, ndsm, gsd)
    frac = np.asarray(ndimage.mean(ref_buf.astype(np.float32), labels, ids)) if ids.size else np.zeros(0)
    y = np.where(frac >= POS, 1, np.where(frac <= NEG, 0, -1))
    return {"name": name, "labels": labels, "ref": ref, "ref_buf": ref_buf, "k": k, "ids": ids, "X": X, "y": y, "gsd": gsd, "n_ref": len(feats), "ref_shapes": shapes, "tr": tr}


def scores(S: dict, keep_ids: np.ndarray) -> dict:
    lut = np.zeros(int(S["labels"].max()) + 1, bool)
    lut[keep_ids] = True
    pred = lut[S["labels"]]
    pred_buf = ndimage.binary_dilation(pred, iterations=S["k"]) if pred.any() else pred
    px = S["gsd"] ** 2
    tp_p = float((pred & S["ref_buf"]).sum())
    prec = tp_p / pred.sum() if pred.any() else float("nan")
    rec = float((S["ref"] & pred_buf).sum()) / S["ref"].sum() if S["ref"].any() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else float("nan")
    # reference footprints found: >= 30 % of the footprint covered by the (tolerant) prediction
    found = 0
    for shp, _ in S["ref_shapes"]:
        m = rasterize([(shp, 1)], out_shape=S["labels"].shape, transform=S["tr"], fill=0, dtype="uint8").astype(bool)
        if m.any() and (m & pred_buf).sum() >= 0.3 * m.sum():
            found += 1
    return {"objects": int(keep_ids.size), "reference_footprints": S["n_ref"], "building_area_ha": round(float(pred.sum()) * px / 1e4, 2),
            "reference_area_ha": round(float(S["ref"].sum()) * px / 1e4, 2), "area_precision": round(prec, 3) if np.isfinite(prec) else None,
            "area_recall": round(rec, 3) if np.isfinite(rec) else None, "area_f1": round(f1, 3) if np.isfinite(f1) else None,
            "footprints_found": found, "footprint_recall": round(found / S["n_ref"], 3) if S["n_ref"] else None}


def fit(Xs: list[np.ndarray], ys: list[np.ndarray]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.vstack(Xs); y = np.concatenate(ys)
    m = y >= 0
    sc = StandardScaler().fit(X[m])
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(X[m]), y[m])
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def main() -> int:
    names = list(json.loads((STUDY / "index.json").read_text(encoding="utf-8")))
    scenes = [load_scene(n) for n in names]
    for S in scenes:
        print(f"{S['name']:20s} candidates {S['ids'].size:5d} | labelled building {int((S['y'] == 1).sum()):4d} not {int((S['y'] == 0).sum()):5d} ambiguous {int((S['y'] == -1).sum()):4d} | reference {S['n_ref']}", flush=True)
    out: dict = {"ran_at": _dt.datetime.now().isoformat(timespec="seconds"), "reference": "Microsoft Global ML Building Footprints (ODbL)", "tolerance_m": TOL_M, "features": list(FEATURES), "scenes": {}}
    for i, S in enumerate(scenes):
        model = fit([T["X"] for j, T in enumerate(scenes) if j != i], [T["y"] for j, T in enumerate(scenes) if j != i])
        model["threshold"] = 0.5
        p = building_probability(S["X"], model) if S["ids"].size else np.zeros(0)
        before, after = scores(S, S["ids"]), scores(S, S["ids"][p >= 0.5])
        out["scenes"][S["name"]] = {"before_filter": before, "after_filter_held_out": after}
        print(f"{S['name']:20s} BEFORE objects {before['objects']:5d} P {before['area_precision']} R {before['area_recall']} F1 {before['area_f1']} | AFTER (held out) objects {after['objects']:5d} P {after['area_precision']} R {after['area_recall']} F1 {after['area_f1']} | ref {S['n_ref']}", flush=True)
    final = fit([S["X"] for S in scenes], [S["y"] for S in scenes])
    final.update(threshold=0.5, features=list(FEATURES), version="bf-1.0", trained_on=names, reference="Microsoft Global ML Building Footprints (ODbL)",
                 protocol="logistic regression on standardised object features; labels = overlap with reference footprints (>= 50 % building, <= 10 % not)",
                 leave_one_scene_out=out["scenes"], created=out["ran_at"])
    MODEL_PATH.write_text(json.dumps(final, indent=1), encoding="utf-8")
    out["coefficients"] = dict(zip(FEATURES, [round(c, 3) for c in final["coef"]]))
    (ROOT / "docs" / "building_detection_validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("coefficients (standardised):", out["coefficients"])
    print("wrote", MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
