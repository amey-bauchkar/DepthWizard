"""DepthWizard vs USGS 3DEP airborne LiDAR on three US terrain types it was never trained or tuned on.

Sites were fixed by terrain class BEFORE any result was seen (the problem statement asks for urban, sparse, hilly and
forested terrain; Switzerland covers urban and rural / forest):
    gatlinburg     forested mountain town, Great Smoky Mountains, Tennessee   (hilly + forest; the Sikkim analogue)
    state_college  town between forested ridges, Pennsylvania               (urban / suburban + forest)
    las_cruces     sparse, arid, scattered buildings, New Mexico            (sparse)
Input image: NAIP (USDA, public domain) 0.6 m RGB of the LiDAR year (or the closest), resampled to 0.5 m.
Reference: USGS 3DEP LiDAR DSM and DTM, 2 m, NAVD88 (converted to EGM2008 with NOAA GEOID18 by the app's C-1 guard).
Both via Microsoft Planetary Computer (free, no key). DEM for the app: Copernicus GLO-30.
Scores (RMSE / MAE / r, the problem statement's metrics) come from the app's own validation endpoint, next to the
Copernicus DEM alone on the same pixels.

    python scripts/validate_us.py   -> docs/validation_us.{md,json}
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "us_study"
os.environ["DW_DATA_DIR"] = str(OUT / "_data")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import requests  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import Resampling  # noqa: E402

PC = "https://planetarycomputer.microsoft.com/api"
SITES = {
    "gatlinburg": dict(lon=-83.512, lat=35.714, year=2016, label="Gatlinburg, TN - forested mountain town (hilly + forest)"),
    "state_college": dict(lon=-77.860, lat=40.795, year=2019, label="State College, PA - town between forested ridges (urban + forest)"),
    "las_cruces": dict(lon=-106.735, lat=32.330, year=2018, label="Las Cruces, NM - sparse arid terrain"),
}
SIZE_M, GSD = 1000, 0.5
GEOID18 = ("https://cdn.proj.org/us_noaa_g2018u0.tif", ROOT / "assets" / "proj" / "us_noaa_g2018u0.tif")


def token(collection: str) -> str:
    return requests.get(f"{PC}/sas/v1/token/{collection}", timeout=60).json()["token"]


def search(collection: str, lon: float, lat: float, limit: int = 50) -> list[dict]:
    r = requests.post(f"{PC}/stac/v1/search", json={"collections": [collection], "intersects": {"type": "Point", "coordinates": [lon, lat]}, "limit": limit}, timeout=120)
    return r.json().get("features", [])


def grid(s: dict) -> tuple[int, object, int]:
    epsg = 32600 + int((s["lon"] + 180) // 6) + 1
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(s["lon"], s["lat"])
    return epsg, from_origin(round(x - SIZE_M / 2), round(y + SIZE_M / 2), GSD, GSD), int(SIZE_M / GSD)


def warp(href: str, bands: list[int], epsg: int, tr, n: int, res: Resampling) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(src, crs=f"EPSG:{epsg}", transform=tr, width=n, height=n, resampling=res) as v:
        return v.read(bands).astype(np.float64), src.nodata


def fetch(name: str, s: dict) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    epsg, tr, n = grid(s)
    meta: dict = {"site": name, **s, "epsg": epsg}
    # reference LiDAR DSM / DTM (2 m, NAVD88), written at 2 m on the site grid (the app resamples to its grid)
    tr2, n2 = from_origin(tr.c, tr.f, 2.0, 2.0), int(SIZE_M / 2)
    for coll, kind in (("3dep-lidar-dsm", "dsm"), ("3dep-lidar-dtm", "dtm")):
        tok = token(coll)
        items = search(coll, s["lon"], s["lat"])
        if not items:
            raise RuntimeError(f"no {coll} at {name}")
        it = items[0]
        arr, nod = warp(it["assets"]["data"]["href"] + "?" + tok, [1], epsg, tr2, n2, Resampling.average)
        a = arr[0]
        if nod is not None:
            a[a == nod] = np.nan
        a[(a < -500) | (a > 9000)] = np.nan
        with rasterio.open(d / f"lidar_{kind}.tif", "w", driver="GTiff", width=n2, height=n2, count=1, dtype="float32", crs=f"EPSG:{epsg}", transform=tr2, nodata=-9999.0) as ds:
            ds.write(np.where(np.isfinite(a), a, -9999.0).astype(np.float32), 1)
        meta[f"lidar_{kind}"] = {"item": it["id"], "years": [it["properties"].get("start_datetime", "")[:4], it["properties"].get("end_datetime", "")[:4]], "valid": round(float(np.isfinite(a).mean()), 3)}
    # NAIP RGB of the closest year
    tok = token("naip")
    naip = sorted(search("naip", s["lon"], s["lat"]), key=lambda f: (abs(int(f["properties"]["naip:year"]) - s["year"]), -int(f["properties"]["naip:year"])))
    rgb, got = np.zeros((3, n, n)), np.zeros((n, n), bool)
    year = None
    for it in naip:
        if year is not None and it["properties"]["naip:year"] != year:
            break
        arr, _ = warp(it["assets"]["image"]["href"] + "?" + tok, [1, 2, 3], epsg, tr, n, Resampling.bilinear)
        m = (arr.max(axis=0) > 0) & ~got
        rgb[:, m] = arr[:, m]
        got |= m
        year = it["properties"]["naip:year"]
        if got.mean() > 0.999:
            break
    if got.mean() < 0.95:
        raise RuntimeError(f"NAIP covers only {got.mean():.0%} of {name}")
    with rasterio.open(d / "naip_rgb_0.5m.tif", "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=f"EPSG:{epsg}", transform=tr, compress="deflate") as ds:
        ds.write(np.clip(rgb, 0, 255).astype(np.uint8))
        ds.update_tags(SOURCE=f"USDA NAIP {year} via Microsoft Planetary Computer", LICENSE="public domain")
    meta["naip_year"] = year
    # Copernicus DEM tile for the app
    lat_t, lon_t = int(np.floor(s["lat"])), int(np.floor(s["lon"]))
    cop = f"Copernicus_DSM_COG_10_N{lat_t:02d}_00_W{abs(lon_t):03d}_00_DEM"
    p = ROOT / "assets" / "dem" / f"{cop}.tif"
    if not p.exists():
        urllib.request.urlretrieve(f"https://copernicus-dem-30m.s3.amazonaws.com/{cop}/{cop}.tif", p)
    meta["dem_tile"] = p.name
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    url, gp = GEOID18
    if not gp.exists():
        print("downloading NOAA GEOID18 grid (16.7 MB, public domain) ->", gp)
        urllib.request.urlretrieve(url, gp)
    from fastapi.testclient import TestClient

    from backend.config.settings import load_settings
    from backend.main import create_app
    from scripts.validate_demo import raw_dem_metrics

    c = TestClient(create_app(load_settings()))
    rows = []
    for name, s in SITES.items():
        try:
            meta = fetch(name, s)
        except Exception as e:  # noqa: BLE001
            print("SKIP", name, e, flush=True)
            continue
        d = OUT / name
        f = d / "naip_rgb_0.5m.tif"
        jid = c.post("/api/jobs", files={"file": (f"{name}_naip.tif", f.read_bytes(), "image/tiff")}).json()["job_id"]
        c.post(f"/api/jobs/{jid}/run")
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(1)
        if j["status"] != "READY":
            print("FAILED", name, j.get("error"))
            continue
        res = c.get(f"/api/jobs/{jid}/result").json()
        row = {"site": name, "label": s["label"], "meta": meta, "tier": res["calibration_tier"], "quality": res["quality"], "validation": {}, "dem_alone": {}}
        jd = OUT / "_data" / "jobs" / jid
        for rt in ("dsm", "dtm"):
            ref = d / f"lidar_{rt}.tif"
            v = c.post(f"/api/jobs/{jid}/validate", files={"reference": (ref.name, ref.read_bytes(), "image/tiff")}, data={"ref_type": rt, "vertical_crs": "EPSG:5703", "source_note": f"USGS 3DEP {meta[f'lidar_{rt}']['item']}"})
            if v.status_code != 200:
                row["validation"][rt] = {"error": v.text[:300]}
                continue
            vj = v.json()
            row["validation"][rt] = {"overall": vj["metrics_overall"], "by_slope": vj.get("metrics_by_slope"), "by_object": vj.get("metrics_by_object"), "caveats": vj.get("caveats")}
            row["dem_alone"][rt] = raw_dem_metrics(jd, ref, "EPSG:5703", res["vertical_reference"])
        rows.append(row)
        keep = OUT / f"{name}_job"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(jd, keep)
        print(name, {rt: (row["validation"][rt].get("overall", {}).get("RMSE"), row["dem_alone"].get(rt, {}).get("RMSE")) for rt in ("dsm", "dtm")}, flush=True)
    (ROOT / "docs" / "validation_us.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    L = ["# DepthWizard vs USGS 3DEP airborne LiDAR (three US terrain types, never used for training or tuning)", "",
         "Generated by `python scripts/validate_us.py`. Image: USDA NAIP (0.6 m, the LiDAR year or closest) resampled to 0.5 m. Reference: USGS 3DEP LiDAR DSM / DTM (2 m, NAVD88 → EGM2008 with NOAA GEOID18). DEM given to the app: Copernicus GLO-30. Sites fixed by terrain class before any result was seen. RMSE / MAE in metres, r = Pearson.", "",
         "| Site | Layer vs LiDAR | Copernicus alone RMSE | **DepthWizard RMSE** | DepthWizard MAE | DepthWizard r | Change |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for rt, name in (("dsm", "DSM"), ("dtm", "terrain vs DTM")):
            m, b = r["validation"].get(rt, {}).get("overall"), r["dem_alone"].get(rt)
            if not m or not b:
                L.append(f"| {r['label']} | {name} | – | error | – | – | – |")
                continue
            L.append(f"| {r['label']} (NAIP {r['meta'].get('naip_year')}, LiDAR {r['meta']['lidar_dsm']['years'][1]}) | {name} | {b['RMSE']:.2f} | **{m['RMSE']:.2f}** | {m['MAE']:.2f} | {m['pearson_r']:.3f} | {100 * (m['RMSE'] - b['RMSE']) / b['RMSE']:+.0f} % |")
    L += ["", "Caveats: the image and LiDAR years can differ by up to a few years (new buildings, tree growth, clearing); NAIP is flown leaf-on in summer, so forest canopy is at its fullest; 3DEP DSMs are first-return surfaces at 2 m.", ""]
    (ROOT / "docs" / "validation_us.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(L) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
