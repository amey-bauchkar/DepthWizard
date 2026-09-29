"""Calibrate the per-pixel confidence (core/inference/confidence.py) on VALIDATION tiles, check it on TEST tiles.

Validation: the six Swiss Aarau / Fribourg tiles (fine-tuning validation regions; downloaded by scripts/select_dem_trust.py
into data/dem_trust/swiss). Test: DepthWizard's Swiss test tiles (assets/demo + assets/reference), never used here for
fitting. Reference = LiDAR nDSM (swissSURFACE3D - swissALTI3D) averaged to the image grid.
For each fixed spread bin: 50 / 80 / 90 % quantiles of |model nDSM - LiDAR nDSM| -> core/inference/confidence_calibration.json.
    python scripts/calibrate_confidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.warp import reproject  # noqa: E402

SPREAD_EDGES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
ERR_EDGES = np.linspace(0, 150, 15001)
QS = (0.5, 0.8, 0.9)
VAL = ["2645-1249", "2645-1248", "2646-1249", "2646-1248", "2578-1183", "2579-1183"]
TEST = [("swissimage_2019_2682-1247_2m.tif", "urban_2682-1247"), ("swissimage_2021_2621-1202_2m.tif", "rural_2621-1202"), ("swissimage_2019_2682-1247_0.5m.tif", "urban_2682-1247")]


def ref_ndsm(img: Path, dsm: Path, dtm: Path) -> np.ndarray:
    with rasterio.open(img) as t:
        out = []
        for p in (dsm, dtm):
            a = np.full((t.height, t.width), np.nan, np.float64)
            with rasterio.open(p) as r:
                reproject(rasterio.band(r, 1), a, src_transform=r.transform, src_crs=r.crs, dst_transform=t.transform, dst_crs=t.crs, resampling=Resampling.average, src_nodata=r.nodata, dst_nodata=np.nan)
            out.append(a)
    return np.clip(out[0] - out[1], 0.0, None)


def spread_err_hist(img: Path, ref: np.ndarray, predictor, fusion) -> np.ndarray:
    from core.inference.confidence import tta_heights

    with rasterio.open(img) as ds:
        rgb = ds.read((1, 2, 3)).transpose(1, 2, 0)
        gsd = abs(ds.transform.a)
    mean, spread = tta_heights(np.ascontiguousarray(rgb), predictor, gsd, fusion)
    m = np.isfinite(ref) & np.isfinite(spread)
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    b = np.clip(np.searchsorted(SPREAD_EDGES, spread[m], side="right") - 1, 0, len(SPREAD_EDGES) - 1)
    e = np.abs(mean[m] - ref[m])
    H = np.zeros((len(SPREAD_EDGES), len(ERR_EDGES) - 1), np.int64)
    for k in range(H.shape[0]):
        H[k] = np.histogram(e[b == k], bins=ERR_EDGES)[0]
    return H


def main() -> int:
    from backend.config.settings import load_settings
    from core.inference.predictor import build_metric_predictor

    s = load_settings()
    pred = build_metric_predictor(s)
    d = ROOT / "data" / "dem_trust" / "swiss"
    Hv = None
    for k in VAL:
        img = d / f"{k}_rgb_2m.tif"
        if not img.exists():
            raise SystemExit(f"{img} missing: run scripts/select_dem_trust.py first (it downloads the validation tiles)")
        H = spread_err_hist(img, ref_ndsm(img, d / f"{k}_dsm.tif", d / f"{k}_dtm.tif"), pred, s.fusion)
        Hv = H if Hv is None else Hv + H
        print("val", k, "pixels", int(H.sum()), flush=True)
    q = []
    for k in range(Hv.shape[0]):
        c = np.cumsum(Hv[k])
        q.append([float(ERR_EDGES[1:][np.searchsorted(c, qq * c[-1])]) if c[-1] > 2000 else None for qq in QS])
    # bins without enough samples inherit the next populated bin (monotone, conservative for large spreads)
    for j in range(len(QS)):
        last = None
        for k in range(len(q) - 1, -1, -1):
            if q[k][j] is None:
                q[k][j] = last
            else:
                last = q[k][j]
    for j in range(len(QS)):  # sparse top bins: the widest calibrated interval below them (conservative)
        last = None
        for k in range(len(q)):
            if q[k][j] is None:
                q[k][j] = last
            else:
                last = q[k][j]
    cal = {"method": "4-way TTA spread of the fine-tuned nDSM model; |error| quantiles per fixed spread bin vs LiDAR nDSM",
           "calibrated_on": "Swiss validation tiles " + ", ".join(VAL), "model": f"{pred.card.name}@{pred.card.version}",
           "spread_bin_edges_m": SPREAD_EDGES, "abs_error_quantiles_m": {f"{int(qq * 100)}%": [row[j] for row in q] for j, qq in enumerate(QS)}}
    # held-out check: coverage of the intervals on the Swiss test tiles
    ref_dir, demo = ROOT / "assets" / "reference", ROOT / "assets" / "demo"
    cov = {f"{int(qq * 100)}%": [0, 0] for qq in QS}
    for img_name, ref in TEST:
        img = demo / img_name
        H = spread_err_hist(img, ref_ndsm(img, ref_dir / f"swisssurface3d_{ref}_dsm_0.5m.tif", ref_dir / f"swissalti3d_{ref}_dtm_0.5m.tif"), pred, s.fusion)
        for j, qq in enumerate(QS):
            key = f"{int(qq * 100)}%"
            for k in range(H.shape[0]):
                lim = cal["abs_error_quantiles_m"][key][k]
                if lim is None:
                    continue
                c = np.cumsum(H[k])
                cov[key][0] += int(c[np.searchsorted(ERR_EDGES[1:], lim)])
                cov[key][1] += int(c[-1])
        print("test", img_name, flush=True)
    cal["test_coverage_swiss_test_tiles"] = {k: round(a / max(b, 1), 3) for k, (a, b) in cov.items()}
    out = ROOT / "core" / "inference" / "confidence_calibration.json"
    out.write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(json.dumps(cal, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
