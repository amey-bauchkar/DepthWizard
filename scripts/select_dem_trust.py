"""Choose how much DepthWizard should trust the model against the DEM - on VALIDATION regions only - then test it.

The metric composition is  DSM = DEM + g * hp(nDSM)   and   terrain = DEM - f * lp(nDSM),  with hp / lp the high- and
low-pass of the model's heights at the DEM posting. Until now g = f = 1: the DEM is assumed to contain the FULL smoothed
object height (Copernicus / CartoDEM are surface models), and all model detail is added. Measured: on flat sparse
ground (Las Cruces) and some towns the DEM holds far less object height than that, so terrain came out too low.

Both knobs are computed from each job's own rasters (hp = dsm - dem, lp = ndsm - hp at the current g = f = 1), so every
candidate is exact without re-running the model. Selection: the (g, f) with the lowest mean RMSE over the VALIDATION
tiles - Swiss Aarau / Fribourg (the fine-tuning validation regions) and US Tucson / Pittsburgh (v2 validation regions);
ties go to the current (1, 1). Then reported unchanged on the TEST sets: DepthWizard's Swiss test tiles, US 3DEP test
sites (Gatlinburg, State College, Las Cruces) and the six Sikkim scenes vs ICESat-2.

    python scripts/select_dem_trust.py   -> data/dem_trust/selection.json (+ printed table)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def _model_version() -> str:
    import json as _j
    idx = _j.loads((ROOT / "models" / "INDEX.json").read_text(encoding="utf-8"))
    return max(idx["models"].get("da-v2-small-ndsm", {"1.0.0": 0}), key=lambda v: tuple(map(int, v.split("."))))


MODEL_VERSION = _model_version()
OUT = ROOT / "data" / ("dem_trust" if MODEL_VERSION == "1.0.0" else f"dem_trust_{MODEL_VERSION}")
os.environ["DW_DATA_DIR"] = str(OUT / "_data")
# candidates are rebuilt from each job's outputs, which is exact only for jobs run at g = f = 1
os.environ["DW_FUSION_METRIC_DETAIL_GAIN"] = "1.0"
os.environ["DW_FUSION_METRIC_DEM_OBJECT_FRACTION"] = "1.0"
# test sites that lie in a TRAINING region of this model version (not an unseen-region test for it)
SEEN_IN_TRAINING = {"2.0.0": {"us_test_gatlinburg": "v2 trained on the 'smoky_mountains' region ~7 km away"}}.get(MODEL_VERSION, {})
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

G_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
F_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SWISS_VAL = ["2645-1249", "2645-1248", "2646-1249", "2646-1248", "2578-1183", "2579-1183"]  # Aarau x4, Fribourg x2 (v1 split)
US_VAL = {"tucson": dict(lon=-110.970, lat=32.220, year=2021, label="Tucson, AZ (val)"), "pittsburgh": dict(lon=-79.990, lat=40.440, year=2019, label="Pittsburgh, PA (val)")}


def run_job(client, path: Path) -> Path:
    jid = client.post("/api/jobs", files={"file": (path.name, path.read_bytes(), "image/tiff")}).json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    while (j := client.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
        time.sleep(1)
    if j["status"] != "READY":
        raise RuntimeError(f"{path.name}: {j.get('error')}")
    return client.app.state.jobs._job_dir(jid)  # ask the app: another script's import may have changed DW_DATA_DIR


def job_layers(jd: Path):
    from core.geo.grid import Grid

    with rasterio.open(jd / "dsm.tif") as ds:
        dsm = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        grid = Grid(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs.to_string(), dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference="EGM2008", tier="T")
    rd = lambda n: rasterio.open(jd / n).read(1, masked=True).astype(np.float64).filled(np.nan)  # noqa: E731
    dem, ndsm = rd("dem.tif"), rd("ndsm.tif")
    hp = dsm - dem
    lp = ndsm - hp
    return grid, dem, hp, lp


def candidate(dem, hp, lp, g, f):
    dsm = dem + g * hp
    terrain = np.fmin(dem - f * lp, dsm)
    return dsm, terrain


def ref_on_grid(path: Path, grid, vcrs: str) -> np.ndarray:
    from core.geo.raster_io import read_raster_on_grid
    from core.geo.vertical import transform_heights_xy

    ref, ok = read_raster_on_grid(str(path), grid, resampling=Resampling.average)
    ref = np.where(ok, ref, np.nan).astype(np.float64)
    if vcrs != "EGM2008":
        cc, rr = np.meshgrid(np.arange(0, grid.width, 64), np.arange(0, grid.height, 64))
        xs, ys = grid.transform @ (cc + 0.5, rr + 0.5)
        z, _ = transform_heights_xy(np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel(), np.zeros(cc.size), grid.crs, vcrs, "EGM2008")
        ref = ref + float(np.mean(z))
    return ref


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def dense_scores(jd, ref_dsm, ref_dtm, vcrs):
    grid, dem, hp, lp = job_layers(jd)
    rs, rt = ref_on_grid(ref_dsm, grid, vcrs), ref_on_grid(ref_dtm, grid, vcrs)
    out = {"dem_only": {"dsm": rmse(dem, rs), "terrain": rmse(dem, rt)}}
    for g in G_GRID:
        for f in F_GRID:
            dsm, ter = candidate(dem, hp, lp, g, f)
            out[f"{g},{f}"] = {"dsm": rmse(dsm, rs), "terrain": rmse(ter, rt)}
    return out


def nearest_coverage(lon: float, lat: float, radius: float = 0.1) -> dict:
    """Centre of the nearest 3DEP DSM + DTM tile within ~10 km (same region, same terrain type)."""
    import requests

    PC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    best = None
    for it in requests.post(PC, json={"collections": ["3dep-lidar-dsm"], "bbox": [lon - radius, lat - radius, lon + radius, lat + radius], "limit": 100}, timeout=120).json().get("features", []):
        g = it.get("properties", {}).get("proj:geometry") and it["geometry"]
        w, s_, e, n = it["bbox"]
        if e - w < 0.02 or n - s_ < 0.02:
            continue
        cx, cy = (w + e) / 2, (s_ + n) / 2
        d = (cx - lon) ** 2 + (cy - lat) ** 2
        if best is None or d < best[0]:
            best = (d, cx, cy)
    if best is None:
        raise RuntimeError(f"no 3DEP LiDAR within {radius} deg of {lon}, {lat}")
    return {"lon": round(best[1], 5), "lat": round(best[2], 5)}


def fetch_swiss(key: str) -> tuple[Path, Path, Path]:
    STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/{}/items?bbox={}&limit=100"
    to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def item_url(coll, suffix, year=None):
        e, n = map(int, key.split("-"))
        lon, lat = to_wgs.transform(e * 1000 + 500, n * 1000 + 500)
        bb = f"{lon - 0.001:.5f},{lat - 0.001:.5f},{lon + 0.001:.5f},{lat + 0.001:.5f}"
        feats = json.load(urllib.request.urlopen(STAC.format(coll, bb), timeout=60))["features"]
        c = sorted((int(f["id"].split("_")[1]), a["href"]) for f in feats if f["id"].endswith(key) for a in f["assets"].values() if a["href"].endswith(suffix))
        return min(c, key=lambda x: abs(x[0] - year)) if year else c[-1]

    d = OUT / "swiss"
    d.mkdir(parents=True, exist_ok=True)
    img, dsm, dtm = d / f"{key}_rgb_2m.tif", d / f"{key}_dsm.tif", d / f"{key}_dtm.tif"
    if not img.exists():
        ys, u_dsm = item_url("ch.swisstopo.swisssurface3d-raster", "_0.5_2056_5728.tif")
        _, u_dtm = item_url("ch.swisstopo.swissalti3d", "_0.5_2056_5728.tif")
        _, u_rgb = item_url("ch.swisstopo.swissimage-dop10", "_2_2056.tif", year=ys)
        for u, p in ((u_dsm, dsm), (u_dtm, dtm), (u_rgb, img)):
            urllib.request.urlretrieve(u, p)
    return img, dsm, dtm


def india_scores(client):
    """Six Sikkim scenes run with the CURRENT model vs ICESat-2 at the checkpoints, per candidate."""
    from core.validate.points import load_checkpoints, validate_points

    out = {}
    for site in ("namchi", "chungthang", "teesta_east", "teesta_west", "chungthang_west", "north_sikkim_alpine"):
        jd = run_job(client, ROOT / "assets" / "demo" / "india" / f"{site}_rgb_0.5m.tif")
        grid, dem, hp, lp = job_layers(jd)
        cp = load_checkpoints(ROOT / "assets" / "demo" / "india" / f"{site}_icesat2.csv")
        rows = {}
        for key, gf in [("dem_only", None)] + [(f"{g},{f}", (g, f)) for g in G_GRID for f in F_GRID]:
            dsm, ter = (dem, dem) if gf is None else candidate(dem, hp, lp, *gf)
            pv = validate_points(grid, {"terrain": ter.astype(np.float32), "dsm": dsm.astype(np.float32)}, cp)
            m = pv.metrics
            rows[key] = {"terrain": (m.get("terrain_vs_ground") or {}).get("RMSE"), "dsm": (m.get("dsm_vs_top_of_surface") or {}).get("RMSE"), "n": (m.get("terrain_vs_ground") or {}).get("n")}
        out[site] = rows
    return out


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.config.settings import load_settings
    from backend.main import create_app
    from scripts.validate_us import SITES as US_TEST
    from scripts.validate_us import fetch as fetch_us

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "scores.json"
    S = json.loads(cache.read_text()) if cache.exists() else {}
    c = None

    def client():
        nonlocal c
        c = c or TestClient(create_app(load_settings()))
        return c

    def score(name, split, fn):
        if name not in S:
            S[name] = {"split": split, **fn()}
            cache.write_text(json.dumps(S, indent=1))
            print(name, split, "dem-only", S[name]["dem_only"], "| current", S[name]["1.0,1.0"], flush=True)

    # validation: Swiss Aarau / Fribourg + US Tucson / Pittsburgh
    for k in SWISS_VAL:
        score(f"ch_val_{k}", "val", lambda k=k: dense_scores(run_job(client(), fetch_swiss(k)[0]), *fetch_swiss(k)[1:], "EPSG:5728"))
    import scripts.validate_us as vu
    vu.OUT = OUT / "us"
    for name, s in US_VAL.items():
        def us_val(name=name, s=s):
            s = {**s, **nearest_coverage(s["lon"], s["lat"])}   # region centre may lack 3DEP LiDAR: nearest covered point
            meta = fetch_us(name, s)
            d = vu.OUT / name
            return dense_scores(run_job(client(), d / "naip_rgb_0.5m.tif"), d / "lidar_dsm.tif", d / "lidar_dtm.tif", "EPSG:5703")
        score(f"us_val_{name}", "val", us_val)
    # test: Swiss demo tiles, US 3DEP test sites, (India separately, points)
    for item, img in (("zurich_05", "swissimage_2019_2682-1247_0.5m.tif"), ("zurich_2", "swissimage_2019_2682-1247_2m.tif"), ("emmental_2", "swissimage_2021_2621-1202_2m.tif")):
        ref = "urban_2682-1247" if "2682" in img else "rural_2621-1202"
        score(f"ch_test_{item}", "test", lambda img=img, ref=ref: dense_scores(run_job(client(), ROOT / "assets" / "demo" / img),
                                                                              ROOT / "assets" / "reference" / f"swisssurface3d_{ref}_dsm_0.5m.tif",
                                                                              ROOT / "assets" / "reference" / f"swissalti3d_{ref}_dtm_0.5m.tif", "EPSG:5728"))
    for name in US_TEST:
        d = ROOT / "data" / "us_study" / name
        score(f"us_test_{name}", "test", lambda d=d: dense_scores(run_job(client(), d / "naip_rgb_0.5m.tif"), d / "lidar_dsm.tif", d / "lidar_dtm.tif", "EPSG:5703"))
    if "india" not in S:
        S["india"] = india_scores(client())
        cache.write_text(json.dumps(S, indent=1))

    val = [v for k, v in S.items() if k != "india" and v["split"] == "val"]
    cands = [f"{g},{f}" for g in G_GRID for f in F_GRID]

    def mean_over(rows, key, layer):
        return float(np.mean([r[key][layer] for r in rows]))
    best_g = min(G_GRID, key=lambda g: (round(mean_over(val, f"{g},1.0", "dsm"), 3), -g))
    best_f = min(F_GRID, key=lambda f: (round(mean_over(val, f"{best_g},{f}", "terrain"), 3), -f))
    chosen, current = f"{best_g},{best_f}", "1.0,1.0"
    print(f"\nVALIDATION choice: g (DSM detail) = {best_g}, f (object height removed from the DEM) = {best_f}")
    print("validation mean RMSE  DSM: current %.2f -> chosen %.2f | terrain: current %.2f -> chosen %.2f | DEM alone: DSM %.2f terrain %.2f" % (
        mean_over(val, current, "dsm"), mean_over(val, chosen, "dsm"), mean_over(val, current, "terrain"), mean_over(val, chosen, "terrain"),
        mean_over(val, "dem_only", "dsm"), mean_over(val, "dem_only", "terrain")))
    print(f"\n{'TEST site':34s} {'layer':8s} {'DEM':>7s} {'current':>8s} {'chosen':>8s}")
    worse = {"current": 0, "chosen": 0}
    for k, v in S.items():
        if k == "india" or v["split"] != "test":
            continue
        seen = k in SEEN_IN_TRAINING
        for layer in ("dsm", "terrain"):
            dm, cu, ch = v["dem_only"][layer], v[current][layer], v[chosen][layer]
            if not seen:
                worse["current"] += cu > dm * 1.05
                worse["chosen"] += ch > dm * 1.05
            print(f"{k:34s} {layer:8s} {dm:7.2f} {cu:8.2f} {ch:8.2f}" + ("   (training region: not counted)" if seen else ""))
    for site, rows in S.get("india", {}).items():
        for layer in ("dsm", "terrain"):
            dm, cu, ch = rows["dem_only"][layer], rows[current][layer], rows[chosen][layer]
            if None in (dm, cu, ch):
                continue
            worse["current"] += cu > dm * 1.05
            worse["chosen"] += ch > dm * 1.05
            print(f"{'india_' + site:34s} {layer:8s} {dm:7.2f} {cu:8.2f} {ch:8.2f}")
    print(f"\ntest cases more than 5 % worse than the DEM alone: current {worse['current']}, chosen {worse['chosen']}")
    (OUT / "selection.json").write_text(json.dumps({"g": best_g, "f": best_f, "worse_than_dem": worse}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
