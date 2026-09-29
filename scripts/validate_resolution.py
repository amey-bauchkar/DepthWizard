"""Accuracy across input resolutions 0.35 - 10 m (ISRO FAQ: evaluation on Cartosat-2S 0.6 m, but the solution must work
from 0.35 m to 10 m and must not be overfitted to 0.6 m).

Images at each GSD are made from a FINER source, never by upscaling a coarse one:
  * Swiss tiles: the swisstopo SWISSIMAGE 0.1 m COG read directly at the target GSD (area average);
  * US tiles: NAIP (0.6 m) averaged to GSDs >= 0.6 m only.
Each image runs through the full Mode B pipeline (Copernicus DEM, model detail at g = f = 1 so every fusion setting
can be rebuilt exactly, as in scripts/select_dem_trust.py) and is scored against the LiDAR DSM / DTM on the job grid.

Selection rule (validation tiles only: Swiss Aarau / Fribourg, US Tucson / Pittsburgh): for each GSD, the DSM detail
gain g in {0, 0.25, 0.5, 0.75, 1} with the lowest mean validation RMSE (f fixed at the configured value). The result is
a per-GSD gain table for the pipeline; it is then reported unchanged on the TEST tiles (Zurich, Emmental, State College,
Las Cruces).
    python scripts/validate_resolution.py   -> data/resolution/scores.json + printed tables
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.select_dem_trust as T  # noqa: E402  (sets DW_DATA_DIR and g = f = 1 for the runs)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

OUT = ROOT / "data" / "resolution"
GSDS = (0.35, 0.6, 1.0, 2.5, 5.0, 10.0)
G_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
F_FIXED = 0.75
SWISS = {  # key: (split, label)
    "2645-1249": ("val", "Aarau A"), "2646-1248": ("val", "Aarau B"), "2578-1183": ("val", "Fribourg"),
    "2682-1247": ("test", "Zurich urban"), "2621-1202": ("test", "Emmental rural / forest"),
}
US = {"tucson": ("val", None), "pittsburgh": ("val", None), "state_college": ("test", "State College"), "las_cruces": ("test", "Las Cruces")}


def swiss_urls(key: str):
    STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/{}/items?bbox={}&limit=100"
    to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    e, n = map(int, key.split("-"))
    lon, lat = to_wgs.transform(e * 1000 + 500, n * 1000 + 500)
    bb = f"{lon - 0.001:.5f},{lat - 0.001:.5f},{lon + 0.001:.5f},{lat + 0.001:.5f}"

    def get(coll, suffix, year=None):
        feats = json.load(urllib.request.urlopen(STAC.format(coll, bb), timeout=60))["features"]
        c = sorted((int(f["id"].split("_")[1]), a["href"]) for f in feats if f["id"].endswith(key) for a in f["assets"].values() if a["href"].endswith(suffix))
        return min(c, key=lambda x: abs(x[0] - year)) if year else c[-1]
    ys, dsm = get("ch.swisstopo.swisssurface3d-raster", "_0.5_2056_5728.tif")
    _, dtm = get("ch.swisstopo.swissalti3d", "_0.5_2056_5728.tif")
    _, rgb = get("ch.swisstopo.swissimage-dop10", "_0.1_2056.tif", year=ys)
    return rgb, dsm, dtm


def swiss_image(key: str, gsd: float) -> tuple[Path, Path, Path]:
    d = OUT / "swiss"
    d.mkdir(parents=True, exist_ok=True)
    img = d / f"{key}_{gsd:g}m.tif"
    dsm, dtm = d / f"{key}_dsm.tif", d / f"{key}_dtm.tif"
    if not (img.exists() and dsm.exists()):
        u_rgb, u_dsm, u_dtm = swiss_urls(key)
        if not dsm.exists():
            urllib.request.urlretrieve(u_dsm, dsm)
            urllib.request.urlretrieve(u_dtm, dtm)
        n = int(round(1000 / gsd))
        with rasterio.open("/vsicurl/" + u_rgb) as ds:
            a = ds.read((1, 2, 3), out_shape=(3, n, n), resampling=Resampling.average)
            tr = ds.transform * ds.transform.scale(ds.width / n, ds.height / n)
            crs = ds.crs
        with rasterio.open(img, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=crs, transform=tr, compress="deflate") as o:
            o.write(a)
    return img, dsm, dtm


def us_image(name: str, gsd: float) -> tuple[Path, Path, Path] | None:
    if gsd < 0.6:
        return None  # NAIP is 0.6 m: finer would be upscaled, not a real image at that GSD
    src_dir = (ROOT / "data" / "us_study" / name) if (ROOT / "data" / "us_study" / name / "naip_rgb_0.5m.tif").exists() else (T.OUT / "us" / name)
    base = src_dir / "naip_rgb_0.5m.tif"
    out = OUT / "us"
    out.mkdir(parents=True, exist_ok=True)
    img = out / f"{name}_{gsd:g}m.tif"
    if not img.exists():
        with rasterio.open(base) as ds:
            n = int(round(ds.width * ds.transform.a / gsd))
            a = ds.read(out_shape=(3, n, n), resampling=Resampling.average)
            tr = from_origin(ds.transform.c, ds.transform.f, gsd, gsd)
            with rasterio.open(img, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=ds.crs, transform=tr, compress="deflate") as o:
                o.write(a)
    return img, src_dir / "lidar_dsm.tif", src_dir / "lidar_dtm.tif"


def score(client, img, dsm, dtm, vcrs):
    jd = T.run_job(client, img)
    grid, dem, hp, lp = T.job_layers(jd)
    rs, rt = T.ref_on_grid(dsm, grid, vcrs), T.ref_on_grid(dtm, grid, vcrs)
    out = {"dem_only": {"dsm": T.rmse(dem, rs), "terrain": T.rmse(dem, rt)}}
    for g in G_GRID:
        dsm_c, ter_c = T.candidate(dem, hp, lp, g, F_FIXED)
        out[str(g)] = {"dsm": T.rmse(dsm_c, rs), "terrain": T.rmse(ter_c, rt)}
    tiles = json.loads((jd / "tiled_inference.json").read_text()) if (jd / "tiled_inference.json").exists() else {}
    out["inference_gsd_m"] = tiles.get("inference_gsd_m")
    return out


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "scores.json"
    S = json.loads(cache.read_text()) if cache.exists() else {}
    c = TestClient(create_app())
    for gsd in GSDS:
        for key, (split, label) in SWISS.items():
            k = f"ch_{key}@{gsd:g}"
            if k not in S:
                t0 = time.time()
                S[k] = {"split": split, "gsd": gsd, **score(c, *swiss_image(key, gsd), "EPSG:5728")}
                cache.write_text(json.dumps(S, indent=1))
                print(k, split, f"{time.time() - t0:.0f}s", "DEM", {x: round(v, 2) for x, v in S[k]["dem_only"].items()}, "g=1", {x: round(v, 2) for x, v in S[k]["1.0"].items()}, flush=True)
        for name, (split, label) in US.items():
            k = f"us_{name}@{gsd:g}"
            files = us_image(name, gsd)
            if files is None or k in S:
                continue
            S[k] = {"split": split, "gsd": gsd, **score(c, *files, "EPSG:5703")}
            cache.write_text(json.dumps(S, indent=1))
            print(k, split, "DEM", {x: round(v, 2) for x, v in S[k]["dem_only"].items()}, "g=1", {x: round(v, 2) for x, v in S[k]["1.0"].items()}, flush=True)
    table = {}
    for gsd in GSDS:
        val = [v for v in S.values() if v["gsd"] == gsd and v["split"] == "val"]
        best = min(G_GRID, key=lambda g: (round(float(np.mean([r[str(g)]["dsm"] for r in val])), 3), -g))
        table[gsd] = best
    print("\nVALIDATION choice of DSM detail gain per GSD:", table)
    print(f"\n{'TEST':34s} {'gsd':>5s} {'layer':8s} {'DEM':>6s} {'g=1':>6s} {'chosen':>7s}")
    for k, v in S.items():
        if v["split"] != "test":
            continue
        g = table[v["gsd"]]
        for layer in ("dsm", "terrain"):
            print(f"{k:34s} {v['gsd']:5g} {layer:8s} {v['dem_only'][layer]:6.2f} {v['1.0'][layer]:6.2f} {v[str(g)][layer]:7.2f}")
    (OUT / "gain_table.json").write_text(json.dumps({str(k): v for k, v in table.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
