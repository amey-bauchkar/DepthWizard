"""Choose fusion.metric_composition on VALIDATION-region tiles of the fine-tuning split (never the demo test tiles).

Downloads the Aarau / Fribourg validation tiles (2 m RGB + 0.5 m swissSURFACE3D / swissALTI3D) from swisstopo, runs the
pipeline with the zero-shot model and with the fine-tuned model under both compositions, validates against the LiDAR
and writes docs/metric_composition_selection.json. Usage: python scripts/select_metric_composition.py
"""
import json, os, sys, time, zipfile, tempfile, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer

OUT = ROOT / "data" / "valsel"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
rep = json.loads((ROOT / "models" / "da-v2-small-ndsm" / "1.0.0" / "training_report.json").read_text(encoding="utf-8"))
keys = [t.split(" ")[0] for t in rep["tiles"]["val"] if "(aarau)" in t or "(fribourg)" in t][:6]
print("validation tiles:", keys)
to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/{}/items?bbox={}&limit=100"

def item_url(coll, key, suffix, pick="latest", year=None):
    e, n = map(int, key.split("-"))
    lon, lat = to_wgs.transform(e * 1000 + 500, n * 1000 + 500)
    bb = f"{lon-0.001:.5f},{lat-0.001:.5f},{lon+0.001:.5f},{lat+0.001:.5f}"
    feats = json.load(urllib.request.urlopen(STAC.format(coll, bb), timeout=60))["features"]
    cands = []
    for f in feats:
        if not f["id"].endswith(key):
            continue
        y = int(f["id"].split("_")[1])
        for a in f["assets"].values():
            if a["href"].endswith(suffix):
                cands.append((y, a["href"]))
    cands.sort()
    if year is not None:
        return min(cands, key=lambda c: abs(c[0] - year))
    return cands[-1]

tiles = []
for k in keys:
    img = OUT / f"{k}_rgb_2m.tif"; dsm = OUT / f"{k}_dsm.tif"; dtm = OUT / f"{k}_dtm.tif"
    if not img.exists():
        ys, u_dsm = item_url("ch.swisstopo.swisssurface3d-raster", k, "_0.5_2056_5728.tif")
        _, u_dtm = item_url("ch.swisstopo.swissalti3d", k, "_0.5_2056_5728.tif")
        _, u_rgb = item_url("ch.swisstopo.swissimage-dop10", k, "_2_2056.tif", year=ys)
        for u, dst in ((u_dsm, dsm), (u_dtm, dtm), (u_rgb, img)):
            urllib.request.urlretrieve(u, dst)
        print("downloaded", k, flush=True)
    tiles.append((k, img, dsm, dtm))

from fastapi.testclient import TestClient
results = {}
for cfg_name, env in (("zeroshot_fusion", {"DW_MODEL_METRIC_ENABLED": "false"}), ("metric_highpass", {"DW_MODEL_METRIC_ENABLED": "true", "DW_FUSION_METRIC_COMPOSITION": "highpass"}), ("metric_terrain_plus_ndsm", {"DW_MODEL_METRIC_ENABLED": "true", "DW_FUSION_METRIC_COMPOSITION": "terrain_plus_ndsm"})):
    for k in ("DW_MODEL_METRIC_ENABLED", "DW_FUSION_METRIC_COMPOSITION"):
        os.environ.pop(k, None)
    os.environ.update(env); os.environ["DW_DATA_DIR"] = str(OUT / "data")
    from backend.config.settings import load_settings
    from backend.main import create_app
    c = TestClient(create_app(load_settings()))
    for k, img, dsm, dtm in tiles:
        jid = c.post("/api/jobs", files={"file": (img.name, img.read_bytes(), "image/tiff")}).json()["job_id"]
        c.post(f"/api/jobs/{jid}/run")
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(0.5)
        if j["status"] != "READY":
            print(cfg_name, k, "FAILED", j["error"]); continue
        row = {}
        for rt, ref in (("dsm", dsm), ("dtm", dtm)):
            v = c.post(f"/api/jobs/{jid}/validate", files={"reference": (ref.name, ref.read_bytes(), "image/tiff")}, data={"ref_type": rt, "vertical_crs": "EPSG:5728"}).json()
            row[rt] = v["metrics_overall"]["RMSE"]; row[rt + "_dem"] = v["metrics_baseline"]["RMSE"]
        results.setdefault(cfg_name, {})[k] = row
        print(cfg_name, k, {a: round(b, 2) for a, b in row.items()}, flush=True)
print("\nMEAN RMSE over validation tiles (m):")
for cfg_name, rows in results.items():
    m = {rt: np.mean([r[rt] for r in rows.values()]) for rt in ("dsm", "dsm_dem", "dtm", "dtm_dem")}
    print(f"  {cfg_name:26s} DSM {m['dsm']:.2f} (DEM {m['dsm_dem']:.2f})  terrain {m['dtm']:.2f} (DEM {m['dtm_dem']:.2f})")
json.dump(results, open(ROOT / "docs" / "metric_composition_selection.json", "w"), indent=1)
print("wrote docs/metric_composition_selection.json")
