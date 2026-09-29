"""Which layer gives the better per-building height: the metric model's own nDSM, or the fused dsm - terrain?

The fused layers are chosen for the DSM and the terrain (scripts/select_dem_trust.py: g = 1, f = 0.75); their
difference is hp + f x lp, i.e. the model nDSM lowered by (1 - f) x its low-pass, most where trees surround a house.
This study scores both against airborne LiDAR per building (median height over the footprint), on the validation /
test tiles of select_dem_trust.py, with OpenStreetMap footprints (ODbL, Overpass API; cached in data/osm_buildings).
Footprints whose LiDAR median is < 2 m (demolished / not built in the LiDAR year) and < 25 m2 are skipped.

Measured (2026-09-30, model v2):                         RMSE (m)  model nDSM | dsm - terrain
    val   Aarau A 2645-1249        n = 831   ref median  9.2 m        2.67      |   3.55
    val   Aarau B 2646-1248        n = 813                7.4 m        2.35      |   2.93
    val   Fribourg 2578-1183       n = 487               13.0 m        4.60      |   5.89
    val   Tucson                   n = 677                2.8 m        1.06      |   1.01
    val   Pittsburgh               n = 1029               6.1 m        1.63      |   2.11
    test  Zurich                   n = 979               17.8 m        3.09      |   4.01
    test  State College            n = 357                7.4 m        3.42      |   4.18
    test  Las Cruces               n = 6                  3.8 m        1.44      |   1.06   (too few OSM buildings)
    (seen in v2 training: Gatlinburg n = 147: 3.21 | 3.84)
-> backend/jobs/pipeline_b.py BUILDING_HEIGHT_SOURCE: building heights from the model nDSM (chosen on validation).

    python scripts/validate_building_heights.py   -> data/osm_buildings/building_height_check.json + table
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.select_dem_trust as T  # noqa: E402  (runs jobs at g = f = 1, so every fusion can be rebuilt exactly)

import numpy as np  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402

CACHE = ROOT / "data" / "osm_buildings"
F_FUSED = 0.75


def tiles() -> list[tuple[str, str, Path, Path, Path, str]]:
    sw = T.OUT / "swiss"
    out = [("val", k, sw / f"{k}_rgb_2m.tif", sw / f"{k}_dsm.tif", sw / f"{k}_dtm.tif", "EPSG:5728") for k in ("2645-1249", "2646-1248", "2578-1183")]
    out += [("val", n, T.OUT / "us" / n / "naip_rgb_0.5m.tif", T.OUT / "us" / n / "lidar_dsm.tif", T.OUT / "us" / n / "lidar_dtm.tif", "EPSG:5703") for n in ("tucson", "pittsburgh")]
    out.append(("test", "zurich", ROOT / "assets/demo/swissimage_2019_2682-1247_0.5m.tif", ROOT / "assets/reference/swisssurface3d_urban_2682-1247_dsm_0.5m.tif",
                ROOT / "assets/reference/swissalti3d_urban_2682-1247_dtm_0.5m.tif", "EPSG:5728"))
    for n in ("state_college", "las_cruces", "gatlinburg"):
        d = ROOT / "data" / "us_study" / n
        out.append(("seen" if n == "gatlinburg" else "test", n, d / "naip_rgb_0.5m.tif", d / "lidar_dsm.tif", d / "lidar_dtm.tif", "EPSG:5703"))
    return out


def osm_labels(name: str, grid) -> np.ndarray:
    """OSM building footprints on the job grid (value = footprint index + 1), fetched once and cached."""
    p = CACHE / f"{name}.json"
    if not p.exists():
        to = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
        x0, y0 = grid.transform * (0, 0)
        x1, y1 = grid.transform * (grid.width, grid.height)
        lo, la = to.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        query = f'[out:json][timeout:120];way["building"]({min(la)},{min(lo)},{max(la)},{max(lo)});out geom;'
        for k in range(4):
            try:
                req = urllib.request.Request("https://overpass-api.de/api/interpreter", data=urllib.parse.urlencode({"data": query}).encode(), headers={"User-Agent": "DepthWizard-validation"})
                p.write_bytes(urllib.request.urlopen(req, timeout=180).read())
                break
            except Exception as e:  # noqa: BLE001
                print("overpass retry:", e, flush=True)
                time.sleep(20 * (k + 1))
    fr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    shapes = []
    for i, w in enumerate(json.loads(p.read_text(encoding="utf-8"))["elements"]):
        g = w.get("geometry")
        if g and len(g) >= 4:
            xs, ys = fr.transform([v["lon"] for v in g], [v["lat"] for v in g])
            shapes.append(({"type": "Polygon", "coordinates": [list(zip(xs, ys))]}, i + 1))
    if not shapes:
        return np.zeros((grid.height, grid.width), np.int32)
    return rasterize(shapes, out_shape=(grid.height, grid.width), transform=grid.transform, fill=0, dtype="int32")


def score(a: np.ndarray, col: int) -> dict:
    e = a[:, col] - a[:, 0]
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "ME": float(e.mean()), "MAE": float(np.abs(e).mean()), "share_below_2m": float((a[:, col] < 2).mean())}


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    CACHE.mkdir(parents=True, exist_ok=True)
    c = TestClient(create_app())
    res = {}
    for split, name, img, dsm, dtm, vcrs in tiles():
        if not img.exists():
            print("missing input, skipped:", name)
            continue
        grid, _dem, hp, lp = T.job_layers(T.run_job(c, img))
        model = np.clip(hp + lp, 0, None)           # the model nDSM (f = 1)
        fused = np.clip(hp + F_FUSED * lp, 0, None)  # dsm - terrain as the app composes it
        ref = np.clip(T.ref_on_grid(dsm, grid, vcrs) - T.ref_on_grid(dtm, grid, vcrs), 0, None)
        lab = osm_labels(name, grid)
        lab[:6] = lab[-6:] = 0
        lab[:, :6] = lab[:, -6:] = 0
        px_area = abs(grid.transform.a * grid.transform.e)
        rows = []
        for b in np.unique(lab[lab > 0]):
            m = (lab == b) & np.isfinite(ref) & np.isfinite(model)
            if m.sum() * px_area < 25:
                continue
            r = float(np.median(ref[m]))
            if r >= 2.0:
                rows.append((r, float(np.median(model[m])), float(np.median(fused[m]))))
        if not rows:
            print(split, name, "no buildings")
            continue
        a = np.array(rows)
        res[name] = {"split": split, "n": len(a), "ref_median_m": float(np.median(a[:, 0])), "model_ndsm": score(a, 1), "dsm_minus_terrain": score(a, 2)}
        x = res[name]
        print(f"{split:5s} {name:14s} n={x['n']:5d} ref median {x['ref_median_m']:5.1f} m | model nDSM RMSE {x['model_ndsm']['RMSE']:5.2f} ME {x['model_ndsm']['ME']:+5.2f}"
              f" | dsm - terrain RMSE {x['dsm_minus_terrain']['RMSE']:5.2f} ME {x['dsm_minus_terrain']['ME']:+5.2f}", flush=True)
    val = [v for v in res.values() if v["split"] == "val"]
    if val:
        print("\nvalidation mean RMSE: model nDSM %.2f m, dsm - terrain %.2f m" % (np.mean([v["model_ndsm"]["RMSE"] for v in val]), np.mean([v["dsm_minus_terrain"]["RMSE"] for v in val])))
    (CACHE / "building_height_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
