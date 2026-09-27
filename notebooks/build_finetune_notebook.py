"""Generates notebooks/DepthWizard_FineTune_nDSM_v2_Colab.ipynb (keeps the notebook reviewable as plain Python).

v2 (2026-09-27): Swiss + US (USGS 3DEP LiDAR height-above-ground + NAIP) data, building-weighted loss, warm start from
v1, TTA-based calibrated uncertainty intervals. v1 (Swiss only) is kept as DepthWizard_FineTune_nDSM_Colab.ipynb.

Run:  python notebooks/build_finetune_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


md(r"""
# DepthWizard — model v2: metric height above ground (nDSM) from Swiss + US LiDAR, with calibrated uncertainty

**What this does (end to end, ~4–6 h on a free T4; resumable):**
1. Training data, all open and login-free:
   * **Switzerland (swisstopo)**: SWISSIMAGE RGB at 0.5 m + nDSM = swissSURFACE3D − swissALTI3D (0.5 m LiDAR), ~6 tiles in each of 36 regions.
   * **USA**: NAIP aerial RGB (0.3–1 m, resampled to 0.5 m) + USGS 3DEP LiDAR nDSM = DSM − DTM (2 m), via Microsoft Planetary Computer (anonymous access), 20 regions. They include hot/arid cities, forests, the Gulf coast and Puerto Rico (dense tropical low-rise housing, the closest open analogue to Indian towns).
2. Splits **by region** into train / val / test for both countries. The DepthWizard test tiles (Zürich 2682-1247, Emmental 2621-1202) and everything within 5 km of them are **never downloaded**, and no Indian data is used.
3. Fine-tunes Depth Anything V2 Small (Apache-2.0), **warm-starting from v1** if you upload `depth_anything_v2_ndsm_s.pth`. Buildings and trees are weighted ×2 in the loss, and the edge (gradient) term applies only to sharp 0.5 m Swiss targets.
4. **Uncertainty**: each prediction is repeated under 4 rotations/flips. The spread is calibrated on the validation regions into 50 / 80 / 90 % error intervals, and their coverage is checked on the test regions.
5. Evaluates on the held-out test regions (Swiss and US separately) against the zero-shot baseline (with an oracle per-tile affine fit).
6. Exports `depthwizard_ndsm_model_v2.zip` (weights + `training_report.json` with the measured numbers and the uncertainty calibration).

**Before running:** Runtime → Change runtime type → **T4 GPU** (or better). Then Runtime → Run all.
If the session disconnects: reconnect and Runtime → Run all. Checkpoints live in Google Drive (`USE_DRIVE = True`, the default), the dataset is re-downloaded automatically and training resumes where it stopped. Never run a single cell after a reset: each cell checks this and tells you. Once training is complete, a new runtime downloads only the validation + test tiles and goes straight to evaluation and export.

**Memory:** every step fits a free Colab runtime (12.7 GB RAM): downloads read only the 1 km window they need, evaluation streams one tile at a time (running sums + a 1 cm residual histogram), and each heavy cell prints its RAM use.

**Optional warm start:** upload v1 `depth_anything_v2_ndsm_s.pth` (from `models/da-v2-small-ndsm/1.0.0/` in DepthWizard) to `/content/` before running.

**After it finishes:** copy `depthwizard_ndsm_model_v2.zip` into the DepthWizard folder and run
`python scripts/install_finetuned_model.py depthwizard_ndsm_model_v2.zip` (it installs as version 2.0.0; the app uses the newest version).

Data: © swisstopo (Open Government Data, attribution required); NAIP and USGS 3DEP are US public domain (served by Microsoft Planetary Computer). Model: Depth Anything V2 Small, Apache-2.0.
""")

code(r"""
# 1. Runtime check + dependencies
import subprocess, sys, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", torch.cuda.get_device_name(0), "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "pyproj", "huggingface_hub", "requests"], check=True)
""")

code(r"""
# 2. Settings (edit here)
USE_DRIVE = True           # keep checkpoints in Google Drive (~1 GB free needed) so training resumes after a disconnect
CFG = dict(
    gsd=0.5,               # training GSD (m/px) = DepthWizard Mode B inference GSD
    crop=518,              # DA-V2 native tile size (multiple of 14)
    batch=6,
    iters=20000,           # ~3-4 h on a T4 (resumable); validation RMSE was still falling at 8000 in v1
    warmup=300,
    lr_encoder=5e-6,       # DA-V2 metric fine-tuning recipe: small LR for the ViT, 10x for the DPT head
    lr_head=5e-5,
    weight_decay=0.01,
    grad_weight=0.5,       # multi-scale gradient-matching term (sharp roof / canopy edges)
    val_every=1000,
    max_height_m=120.0,    # nDSM clipped to [0, 120] m (tallest Swiss structures/trees are below)
    max_year_gap=3,        # RGB and LiDAR acquisitions at most 3 years apart
    exclusion_km=5.0,      # no tile within 5 km of a DepthWizard validation tile
    tiles_per_region=6,
    use_us=True,           # add USGS 3DEP LiDAR + NAIP regions (Planetary Computer, anonymous)
    us_tiles_per_region=5,
    us_sample_prob=0.35,   # fraction of training crops drawn from US tiles
    building_weight=2.0,   # loss weight for pixels with target height >= 2.5 m (buildings / trees)
    init_from="/content/depth_anything_v2_ndsm_s.pth",  # v1 weights for a warm start (skipped if the file is absent)
    model_version="2.0.0",
    run_name="v2b",        # checkpoints and cached tile lists live under this name: a NEW name = a fresh run
    workers=3,             # parallel downloads; more than 3 can exhaust the 12.7 GB RAM of a free Colab runtime
    seed=0,
)
# DepthWizard's own validation tiles (LV95 km). Never used for training, validation or model selection.
DW_VALIDATION_TILES = [(2682, 1247), (2621, 1202)]

# Regions (lon, lat, split). Split is BY REGION so test scores measure generalisation to unseen places.
REGIONS = {
    # cities / towns
    "geneve": (6.143, 46.204, "train"), "lausanne": (6.633, 46.520, "train"), "bern": (7.444, 46.948, "train"),
    "luzern": (8.310, 47.050, "train"), "winterthur": (8.724, 47.500, "train"), "stgallen": (9.376, 47.424, "train"),
    "biel": (7.247, 47.137, "train"), "neuchatel": (6.931, 46.990, "train"), "chur": (9.531, 46.850, "train"),
    "sion": (7.360, 46.231, "train"), "thun": (7.628, 46.758, "train"), "schaffhausen": (8.635, 47.697, "train"),
    "zug": (8.516, 47.167, "train"), "olten": (7.903, 47.350, "train"), "baden": (8.306, 47.473, "train"),
    "aarau": (8.044, 47.392, "val"), "fribourg": (7.161, 46.803, "val"),
    "basel": (7.590, 47.559, "test"), "lugano": (8.952, 46.004, "test"),
    # farmland / forest / hills
    "seeland": (7.150, 47.010, "train"), "broye": (6.930, 46.800, "train"), "napf": (7.940, 47.005, "train"),
    "jura_franches": (7.000, 47.250, "train"), "toggenburg": (9.200, 47.280, "train"), "appenzell": (9.410, 47.330, "train"),
    "sihlwald": (8.560, 47.260, "train"),
    "entlebuch": (8.063, 46.990, "val"),
    "thurgau": (9.100, 47.550, "test"), "jura_ajoie": (7.080, 47.420, "test"),
    # Alps / valleys
    "grindelwald": (8.034, 46.624, "train"), "zermatt": (7.749, 46.020, "train"), "stmoritz": (9.838, 46.498, "train"),
    "visp": (7.882, 46.293, "train"), "bellinzona": (9.022, 46.193, "train"),
    "engelberg": (8.405, 46.820, "val"),
    "davos": (9.830, 46.800, "test"),
}
import os, random, numpy as np, torch
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
ROOT = "/content/drive/MyDrive/depthwizard_ft" if USE_DRIVE else "/content/depthwizard_ft"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
DATA = "/content/tiles"                          # dataset (~6 GB) on local disk (rebuilt after a disconnect)
RUN = f"{ROOT}/{CFG['run_name']}"                # checkpoints (~0.4 GB) + tile lists of THIS run (Drive when USE_DRIVE)
CKPT = f"{RUN}/ckpt"
os.makedirs(DATA, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
print("run dir:", RUN, "| dataset:", DATA, "| regions:", len(REGIONS))

def ram():
    try:
        import psutil
        v = psutil.virtual_memory(); return f"RAM {v.used / 1e9:.1f} / {v.total / 1e9:.1f} GB"
    except Exception:
        return "RAM ?"

# A finished run (all iterations done) only needs the validation + test tiles for evaluation and export:
# no training tiles are downloaded again after a disconnect.
TRAINING_DONE = os.path.exists(f"{CKPT}/done.json")
if not TRAINING_DONE and os.path.exists(f"{CKPT}/last.pt"):
    _ck = torch.load(f"{CKPT}/last.pt", map_location="cpu")
    TRAINING_DONE = _ck.get("it", -1) + 1 >= CFG["iters"]
    del _ck
NEED_SPLITS = ("val", "test") if TRAINING_DONE else ("train", "val", "test")
if TRAINING_DONE:
    print(f"Training of run '{CFG['run_name']}' is COMPLETE: only validation + test tiles are downloaded, then evaluation and export.")
elif os.path.exists(f"{CKPT}/last.pt"):
    print(f"NOTE: a checkpoint of run '{CFG['run_name']}' exists and training will RESUME from it. For a fresh run, change CFG['run_name'].")
print(ram())
""")

code(r"""
# 3. Depth Anything V2 code (pinned upstream commit, same as DepthWizard's vendored copy) + baseline weights
_missing = [n for n in ("CFG",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import os, subprocess, sys
if not os.path.exists("/content/Depth-Anything-V2"):
    subprocess.run(["git", "clone", "-q", "https://github.com/DepthAnything/Depth-Anything-V2", "/content/Depth-Anything-V2"], check=True)
    subprocess.run(["git", "-C", "/content/Depth-Anything-V2", "checkout", "-q", "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"], check=True)
sys.path.insert(0, "/content/Depth-Anything-V2")
from huggingface_hub import hf_hub_download
BASE_W = hf_hub_download("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth")
import hashlib
print("baseline weights sha256:", hashlib.sha256(open(BASE_W, "rb").read()).hexdigest())  # DepthWizard expects 715fade1...
""")

code(r"""
# 4. Find tiles via the swisstopo STAC API and pair RGB / DSM / DTM acquisitions
_missing = [n for n in ("CFG",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import json, math, re, urllib.request
from pyproj import Transformer

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/{}/items?bbox={}&limit=100"
COLL = {"rgb": "ch.swisstopo.swissimage-dop10", "dsm": "ch.swisstopo.swisssurface3d-raster", "dtm": "ch.swisstopo.swissalti3d"}
to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

def stac_items(coll, bbox):
    url, out = STAC.format(coll, ",".join(f"{v:.5f}" for v in bbox)), []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        out += d["features"]
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
    return out

def asset(item, suffix):
    return next((a["href"] for a in item["assets"].values() if a["href"].endswith(suffix)), None)

def near_validation(e, n):
    return any(math.hypot(e - ve, n - vn) <= CFG["exclusion_km"] for ve, vn in DW_VALIDATION_TILES)

TILES_JSON = f"{RUN}/tiles.json"
if os.path.exists(TILES_JSON):
    TILES = json.load(open(TILES_JSON))
else:
    TILES = []
    for name, (lon, lat, split) in REGIONS.items():
        bbox = (lon - 0.013, lat - 0.009, lon + 0.013, lat + 0.009)   # ~2 x 2 km
        found = {k: stac_items(c, bbox) for k, c in COLL.items()}
        by = {k: {} for k in COLL}
        for k, items in found.items():
            for it in items:
                m = re.search(r"_(\d{4})_(\d{4})-(\d{4})$", it["id"])
                if m:
                    by[k].setdefault((int(m.group(2)), int(m.group(3))), []).append((int(m.group(1)), it))
        keys = sorted(set(by["rgb"]) & set(by["dsm"]) & set(by["dtm"]))
        cx, cy = to_lv95.transform(lon, lat)
        keys.sort(key=lambda k: math.hypot(k[0] + 0.5 - cx / 1000, k[1] + 0.5 - cy / 1000))
        n_ok = 0
        for key in keys:
            if near_validation(*key):
                continue
            ys, dsm_it = max(by["dsm"][key], key=lambda t: t[0])
            yt, dtm_it = max(by["dtm"][key], key=lambda t: t[0])
            yr, rgb_it = min(by["rgb"][key], key=lambda t: abs(t[0] - ys))
            if abs(yr - ys) > CFG["max_year_gap"]:
                continue
            u_rgb, u_dsm, u_dtm = asset(rgb_it, "_0.1_2056.tif"), asset(dsm_it, "_0.5_2056_5728.tif"), asset(dtm_it, "_0.5_2056_5728.tif")
            if u_rgb and u_dsm and u_dtm:
                TILES.append(dict(key=f"{key[0]}-{key[1]}", region=name, split=split, rgb=u_rgb, dsm=u_dsm, dtm=u_dtm, years=[yr, ys, yt]))
                n_ok += 1
            if n_ok >= CFG["tiles_per_region"]:
                break
        print(f"{name:14s} {split:5s} tiles: {n_ok}")
    if len(TILES) >= 150:
        json.dump(TILES, open(TILES_JSON, "w"), indent=1)
    else:
        print(f"only {len(TILES)} Swiss tiles found (swisstopo STAC unreachable?) - list NOT cached, re-run this cell")
from collections import Counter
print("total:", len(TILES), Counter(t["split"] for t in TILES))
assert not any(near_validation(*map(int, t["key"].split("-"))) for t in TILES)
""")

code(r"""
# 5. Download + prepare tiles: RGB read at 0.5 m from the COG overviews, nDSM = DSM - DTM (both LN02, datum cancels)
_missing = [n for n in ("TILES",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import rasterio, numpy as np, time
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", GDAL_CACHEMAX="256")
N = int(1000 / CFG["gsd"])   # 2000 px per 1 km tile

def prepare(t):
    out_rgb, out_h = f"{DATA}/{t['key']}_rgb.npy", f"{DATA}/{t['key']}_ndsm.npy"
    if os.path.exists(out_rgb) and os.path.exists(out_h):
        return t["key"], "cached"
    try:
        with rasterio.open("/vsicurl/" + t["rgb"]) as ds:
            rgb = ds.read((1, 2, 3), out_shape=(3, N, N), resampling=Resampling.average)
            b_rgb = ds.bounds
        hs = []
        for u in (t["dsm"], t["dtm"]):
            with rasterio.open("/vsicurl/" + u) as ds:
                a = ds.read(1, out_shape=(N, N), resampling=Resampling.average, masked=True).astype("float32").filled(np.nan)
                assert max(abs(x - y) for x, y in zip(ds.bounds, b_rgb)) < 1.0, "tile bounds differ"
                hs.append(a)
        ndsm = np.clip(hs[0] - hs[1], 0.0, CFG["max_height_m"])
        ndsm[~(np.isfinite(hs[0]) & np.isfinite(hs[1]))] = np.nan
        if np.isfinite(ndsm).mean() < 0.8 or rgb.max() == 0:
            return t["key"], "skipped (nodata)"
        np.save(out_rgb, np.ascontiguousarray(rgb.transpose(1, 2, 0)))
        np.save(out_h, ndsm.astype(np.float16))
        return t["key"], "ok"
    except Exception as e:
        return t["key"], f"FAILED {type(e).__name__}: {e}"

def download_all(fn, tiles, tag, passes=3):
    # several passes: tiles that FAILED (timeouts, rate limits) are retried after a pause; finished tiles are skipped
    t0, todo = time.time(), list(tiles)
    for ps in range(passes):
        failed = []
        with ThreadPoolExecutor(CFG["workers"]) as ex:
            for i, (t, (k, st)) in enumerate(zip(todo, ex.map(fn, todo))):
                if st.startswith("FAILED"):
                    failed.append(t)
                if st != "cached" or i % 20 == 0:
                    print(f"[{tag} pass {ps + 1}: {i + 1}/{len(todo)}] {k}: {st}  ({time.time() - t0:.0f}s)")
        if not failed:
            break
        todo = failed
        if ps < passes - 1:
            print(f"{len(failed)} {tag} tiles failed - retrying them in 60 s"); time.sleep(60)
    print(f"{tag}: {len(tiles) - len(failed) if failed else len(tiles)} / {len(tiles)} tiles ready or skipped as nodata")

download_all(prepare, [t for t in TILES if t["split"] in NEED_SPLITS], "CH")
READY = [t for t in TILES if os.path.exists(f"{DATA}/{t['key']}_ndsm.npy")]
SPLIT = {s: [t for t in READY if t["split"] == s] for s in ("train", "val", "test")}
print({s: len(v) for s, v in SPLIT.items()})
""")

code(r"""
# 5b. USA: NAIP RGB + USGS 3DEP LiDAR nDSM = DSM - DTM (2 m) via Microsoft Planetary Computer (anonymous SAS tokens)
_missing = [n for n in ("TILES", "prepare",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
# Regions with USGS 3DEP LiDAR (2017+) height-above-ground on Planetary Computer and matching NAIP (surveyed 2026-09-27)
US_REGIONS = {
    # train
    "houston": (-95.370, 29.760, "train"), "san_antonio": (-98.490, 29.420, "train"), "new_orleans": (-90.070, 29.950, "train"),
    "el_paso": (-106.440, 31.760, "train"), "san_juan_pr": (-66.060, 18.400, "train"), "charlotte": (-80.840, 35.230, "train"),
    "nashville": (-86.780, 36.160, "train"), "smoky_mountains": (-83.500, 35.650, "train"), "baton_rouge": (-91.150, 30.450, "train"),
    "dallas": (-96.800, 32.780, "train"), "austin": (-97.740, 30.270, "train"), "las_vegas": (-115.140, 36.170, "train"),
    "memphis": (-90.050, 35.150, "train"), "richmond": (-77.440, 37.540, "train"), "mayaguez_pr": (-67.140, 18.200, "train"),
    # validation (also calibrates the uncertainty intervals)
    "tucson": (-110.970, 32.220, "val"), "pittsburgh": (-79.990, 40.440, "val"),
    # test
    "sacramento": (-121.490, 38.580, "test"), "ponce_pr": (-66.610, 18.010, "test"), "st_louis": (-90.200, 38.630, "test"),
}
import rasterio, requests
from rasterio.warp import reproject, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from pyproj import Transformer
PC = "https://planetarycomputer.microsoft.com/api"
import threading
_tok, _tok_lock = {}, threading.Lock()
def sign(href, coll):
  with _tok_lock:
    # anonymous SAS token, cached 40 min, one request at a time; Planetary Computer rate-limits tokens (HTTP 429) -> wait, retry
    if coll not in _tok or time.time() - _tok[coll][1] > 2400:
        for attempt in range(10):
            try:
                r = requests.get(f"{PC}/sas/v1/token/{coll}", timeout=60)
                j = r.json() if r.ok else {}
            except Exception as e:  # network hiccup / non-JSON reply
                r, j = None, {}
            if "token" in j:
                _tok[coll] = (j["token"], time.time()); break
            wait = int((r.headers.get("Retry-After", 0) if r is not None else 0) or 0) or min(300, 30 * (attempt + 1))
            print(f"  SAS token for {coll}: HTTP {getattr(r, 'status_code', '-')} -> waiting {wait}s (rate limit; normal, it recovers)"); time.sleep(wait)
        else:
            raise RuntimeError(f"Planetary Computer refused SAS tokens for {coll} for ~25 min; wait 15 min and re-run this cell")
    return f"{href}?{_tok[coll][0]}"
def pc_search(coll, bbox, limit=20):
    r = requests.post(f"{PC}/stac/v1/search", json={"collections": [coll], "bbox": bbox, "limit": limit}, timeout=120)
    r.raise_for_status(); return r.json()["features"]
def year(it):
    p = it["properties"]; return int((p.get("datetime") or p.get("start_datetime") or "2000")[:4])

US_TILES = []
US_JSON = f"{RUN}/us_tiles.json"
if CFG["use_us"]:
    if os.path.exists(US_JSON):
        US_TILES = json.load(open(US_JSON))
    else:
        from pyproj import CRS as _CRS
        for name, (lon, lat, split) in US_REGIONS.items():
            try:
                bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
                hag = [h for h in pc_search("3dep-lidar-hag", bbox, 100) if year(h) >= 2017]
                naip = pc_search("naip", bbox, 100)
                if not hag or not naip:
                    print(f"{name:16s} no recent LiDAR / NAIP coverage"); continue
                n_ok = 0
                for h_it in sorted(hag, key=year, reverse=True):
                    with rasterio.open(sign(h_it["assets"]["data"]["href"], "3dep-lidar-hag")) as ds:
                        c = _CRS.from_wkt(ds.crs.to_wkt()); hc = c.sub_crs_list[0] if c.is_compound else c
                        crs = f"EPSG:{hc.to_epsg()}" if hc.to_epsg() else hc.to_wkt(); b = ds.bounds
                    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
                    cx, cy = min(max(cx, b.left + 600), b.right - 600), min(max(cy, b.bottom + 600), b.top - 600)  # keep inside this LiDAR tile
                    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    for k, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]):
                        left, top = cx - 500 + dx * 1000, cy + 500 + dy * 1000
                        if left < b.left or left + 1000 > b.right or top > b.top or top - 1000 < b.bottom:
                            continue
                        (w, s_), (e, n_) = to_ll.transform(left, top - 1000), to_ll.transform(left + 1000, top)
                        cover = [x for x in naip if x["bbox"][0] <= w and x["bbox"][1] <= s_ and x["bbox"][2] >= e and x["bbox"][3] >= n_]
                        cover = sorted(cover or [x for x in naip if not (x["bbox"][2] < w or x["bbox"][0] > e or x["bbox"][3] < s_ or x["bbox"][1] > n_)], key=lambda x: abs(year(x) - year(h_it)))
                        if not cover or abs(year(cover[0]) - year(h_it)) > 4:
                            continue
                        US_TILES.append(dict(key=f"us_{name}_{n_ok}", region=name, split=split, crs=crs, left=left, top=top, hag=h_it["assets"]["data"]["href"], naip=[x["assets"]["image"]["href"] for x in cover[:4]], years=[year(cover[0]), year(h_it)]))
                        n_ok += 1
                        if n_ok >= CFG["us_tiles_per_region"]:
                            break
                    if n_ok >= CFG["us_tiles_per_region"]:
                        break
                print(f"{name:16s} {split:5s} tiles: {n_ok}")
            except Exception as e:
                print(f"{name:16s} skipped: {type(e).__name__}: {e}")
        if len(US_TILES) >= 60:
            json.dump(US_TILES, open(US_JSON, "w"), indent=1)   # cached only when complete enough; a failed search is retried next run
        else:
            print(f"only {len(US_TILES)} US tiles found (network / rate limit?) - list NOT cached, re-run this cell in 10-15 min")

def prepare_us(t):
    out_rgb, out_h = f"{DATA}/{t['key']}_rgb.npy", f"{DATA}/{t['key']}_ndsm.npy"
    if os.path.exists(out_rgb) and os.path.exists(out_h):
        return t["key"], "cached"
    try:
        tr = from_origin(t["left"], t["top"], CFG["gsd"], CFG["gsd"])
        # target = LiDAR DSM - DTM (same 3DEP project). The Planetary Computer "hag" raster was checked and rejected:
        # it puts open desert ground at ~3 m (median), while DSM - DTM gives ~0.5 m there.
        hs = []
        for kind in ("dsm", "dtm"):
            a = np.full((N, N), np.nan, np.float32)
            href = t["hag"].replace("/hag/", f"/{kind}/").replace("-hag-", f"-{kind}-")
            with rasterio.open(sign(href, f"3dep-lidar-{kind}")) as ds:
                # WarpedVRT reads only the 1 km output window (memory ~ output size), never the whole remote tile
                with WarpedVRT(ds, crs=t["crs"], transform=tr, width=N, height=N, resampling=Resampling.bilinear, src_nodata=ds.nodata, nodata=np.nan, dtype="float32") as v:
                    a = v.read(1).astype(np.float32)
            hs.append(a)
        h = hs[0] - hs[1]
        rgb = np.zeros((3, N, N), np.uint8); got = np.zeros((N, N), bool)
        for href in t["naip"]:
            tmp = np.zeros((3, N, N), np.uint8)
            with rasterio.open(sign(href, "naip")) as ds:
                with WarpedVRT(ds, crs=t["crs"], transform=tr, width=N, height=N, resampling=Resampling.average, src_nodata=0, nodata=0) as v:
                    tmp = v.read((1, 2, 3))
            m = (tmp.max(axis=0) > 0) & ~got; rgb[:, m] = tmp[:, m]; got |= m
            if got.mean() > 0.99:
                break
        ndsm = np.clip(h, 0.0, CFG["max_height_m"]); ndsm[~np.isfinite(h) | ~got] = np.nan
        if np.isfinite(ndsm).mean() < 0.8:
            return t["key"], "skipped (coverage)"
        np.save(out_rgb, np.ascontiguousarray(rgb.transpose(1, 2, 0))); np.save(out_h, ndsm.astype(np.float16))
        return t["key"], "ok"
    except Exception as e:
        return t["key"], f"FAILED {type(e).__name__}: {e}"

download_all(prepare_us, [t for t in US_TILES if t["split"] in NEED_SPLITS], "US")
for t in TILES:
    t.setdefault("source", "swiss"); t["sharp"] = True
for t in US_TILES:
    t["source"] = "us"; t["sharp"] = False   # 2 m LiDAR target upsampled to 0.5 m: no edge (gradient) loss
READY = [t for t in TILES + US_TILES if os.path.exists(f"{DATA}/{t['key']}_ndsm.npy")]
SPLIT = {s: [t for t in READY if t["split"] == s] for s in ("train", "val", "test")}
print({s: (len([t for t in v if t["source"] == "swiss"]), len([t for t in v if t["source"] == "us"])) for s, v in SPLIT.items()}, "(swiss, us) |", ram())
if TRAINING_DONE:
    assert len(SPLIT["test"]) >= 10 and len(SPLIT["val"]) >= 5, f"only {len(SPLIT['val'])} val / {len(SPLIT['test'])} test tiles on disk: re-run cells 5 and 5b (check FAILED lines)"
else:
    _n_sw = sum(t["source"] == "swiss" for t in SPLIT["train"])
    assert _n_sw >= 60, (f"only {_n_sw} Swiss training tiles are on disk in {DATA} (after a disconnect the local disk is empty): re-run cell 5 and check its output for FAILED lines "
                         "before training - the Swiss 0.5 m LiDAR is the core of the training set")
    _n_us = sum(t["source"] == "us" for t in SPLIT["train"])
    assert not CFG["use_us"] or _n_us >= 30, (f"only {_n_us} US training tiles on disk: the US download failed (see FAILED lines above). "
                         "Re-run cell 5b (after 10-15 min if Planetary Computer was rate-limiting) - training without them would silently differ from the resumed run")
""")

code(r"""
# 6. Quick look at one training pair
_missing = [n for n in ("SPLIT",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import matplotlib.pyplot as plt
t = (SPLIT["train"] or SPLIT["val"] or SPLIT["test"])[0]
rgb = np.load(f"{DATA}/{t['key']}_rgb.npy", mmap_mode="r"); h = np.load(f"{DATA}/{t['key']}_ndsm.npy", mmap_mode="r")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(rgb[::4, ::4]); ax[0].set_title(f"{t['region']} {t['key']} RGB 0.5 m")
im = ax[1].imshow(np.asarray(h[::4, ::4], dtype=np.float32), cmap="viridis", vmin=0, vmax=30); ax[1].set_title("nDSM (m) from LiDAR"); plt.colorbar(im, ax=ax[1])
plt.show()
""")

code(r"""
# 7. Dataset with augmentations that match how DepthWizard feeds the model
_missing = [n for n in ("SPLIT",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def to_input(rgb_u8):
    x = (rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

class Crops(Dataset):
    def __init__(self, tiles, n, train=True):
        self.tiles, self.n, self.train = tiles, n, train
        self.sw = [t for t in tiles if t.get("source", "swiss") == "swiss"]
        self.us = [t for t in tiles if t.get("source") == "us"]
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        pool = self.us if (self.us and self.sw and rng.random() < CFG["us_sample_prob"]) else (self.sw or self.us)
        t = pool[rng.integers(len(pool))]
        rgb = np.load(f"{DATA}/{t['key']}_rgb.npy", mmap_mode="r"); h = np.load(f"{DATA}/{t['key']}_ndsm.npy", mmap_mode="r")
        C = CFG["crop"]
        s = rng.uniform(0.8, 1.25) if self.train else 1.0      # GSD jitter 0.4-0.63 m: robustness to other sensors
        c = int(round(C * s)); H, W = h.shape
        r0, c0 = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
        x = np.asarray(rgb[r0:r0 + c, c0:c0 + c]); y = np.asarray(h[r0:r0 + c, c0:c0 + c], dtype=np.float32)
        if c != C:
            x = cv2.resize(x, (C, C), interpolation=cv2.INTER_AREA if c > C else cv2.INTER_CUBIC)
            m = cv2.resize(np.isfinite(y).astype(np.uint8), (C, C), interpolation=cv2.INTER_NEAREST).astype(bool)
            y = cv2.resize(np.nan_to_num(y), (C, C), interpolation=cv2.INTER_LINEAR); y[~m] = np.nan
        if self.train:
            k = rng.integers(4); x, y = np.rot90(x, k), np.rot90(y, k)             # nadir: orientation-free
            if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
            x = x.astype(np.float32)
            x = (x - x.mean()) * rng.uniform(0.85, 1.15) + x.mean() * rng.uniform(0.85, 1.15)   # contrast / brightness
            g = x.mean(axis=2, keepdims=True); x = g + (x - g) * rng.uniform(0.8, 1.2)          # saturation
            x = np.clip(x, 0, 255).astype(np.uint8)
            if rng.random() < 0.3:   # 2 m imagery upsampled to 0.5 m, exactly what DepthWizard does for 2 m GeoTIFFs
                x = cv2.resize(cv2.resize(x, (C // 4, C // 4), interpolation=cv2.INTER_AREA), (C, C), interpolation=cv2.INTER_CUBIC)
            elif rng.random() < 0.2:
                x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.2))
        y = np.ascontiguousarray(y)
        return to_input(np.ascontiguousarray(x)), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(np.isfinite(y)), torch.tensor(bool(t.get("sharp", True)))

if TRAINING_DONE:
    train_dl = None
    print("training complete: no training data loader needed")
else:
    train_dl = DataLoader(Crops(SPLIT["train"], CFG["iters"] * CFG["batch"]), batch_size=CFG["batch"], num_workers=2, pin_memory=True, drop_last=True)
    xb, yb, mb, sb = next(iter(train_dl)); print(xb.shape, yb.shape, float(yb[mb].mean()), "m mean target", "sharp", sb.tolist())
    del xb, yb, mb, sb
""")

code(r"""
# 8. Model, loss, optimiser
_missing = [n for n in ("SPLIT", "BASE_W", "to_input",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
from depth_anything_v2.dpt import DepthAnythingV2
dev = "cuda"
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(BASE_W, map_location="cpu"), strict=True)
INIT = CFG.get("init_from") or ""
if INIT and os.path.exists(INIT):
    model.load_state_dict(torch.load(INIT, map_location="cpu"), strict=True); print("warm start from", INIT)
else:
    print("no v1 weights found at", INIT or "(unset)", "- starting from the zero-shot baseline")
model.to(dev)

def grad_loss(p, y, m):
    tot = 0.0
    for s in (1, 2, 4):
        if s > 1:
            w = F.avg_pool2d(m.float()[:, None], s)[:, 0]
            p_ = F.avg_pool2d((p * m)[:, None], s)[:, 0] / w.clamp_min(1e-6); y_ = F.avg_pool2d((y * m)[:, None], s)[:, 0] / w.clamp_min(1e-6); m_ = w > 0.99
        else:
            p_, y_, m_ = p, y, m
        d = p_ - y_
        mx = m_[:, :, 1:] & m_[:, :, :-1]; my = m_[:, 1:, :] & m_[:, :-1, :]
        if mx.any() and my.any():
            tot = tot + (d[:, :, 1:] - d[:, :, :-1]).abs()[mx].mean() + (d[:, 1:, :] - d[:, :-1, :]).abs()[my].mean()
    return tot / 3

def loss_fn(p, y, m, sharp):
    w = 1.0 + (CFG["building_weight"] - 1.0) * (y >= 2.5).float()      # buildings / trees count more than ground
    l = (F.smooth_l1_loss(p, y, beta=1.0, reduction="none") * w)[m].sum() / w[m].sum().clamp_min(1.0)
    if sharp.any():                                                      # edge term only on sharp (0.5 m LiDAR) targets
        l = l + CFG["grad_weight"] * grad_loss(p[sharp], y[sharp], m[sharp])
    return l

opt = torch.optim.AdamW([
    {"params": model.pretrained.parameters(), "lr": CFG["lr_encoder"]},
    {"params": model.depth_head.parameters(), "lr": CFG["lr_head"]},
], weight_decay=CFG["weight_decay"])
def lr_factor(it):
    if it < CFG["warmup"]:
        return (it + 1) / CFG["warmup"]
    q = (it - CFG["warmup"]) / max(1, CFG["iters"] - CFG["warmup"])
    return 0.5 * (1 + math.cos(math.pi * q)) * 0.95 + 0.05
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
scaler = torch.amp.GradScaler("cuda")
print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
""")

code(r"""
# 9. Full-tile inference exactly like DepthWizard (518 px tiles, 25 % overlap, Hann feathering) + STREAMING metrics
#    One tile in memory at a time: exact running sums + a 1 cm residual histogram (median / NMAD exact to 0.5 cm),
#    so evaluating 100 tiles uses the same RAM as evaluating one.
_missing = [n for n in ("model",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
@torch.no_grad()
def predict_tile(net, rgb, tile=518, overlap=0.25):
    net.eval(); H, W = rgb.shape[:2]; step = int(tile * (1 - overlap))
    rows = list(range(0, H - tile + 1, step)); cols = list(range(0, W - tile + 1, step))
    if rows[-1] + tile < H: rows.append(H - tile)
    if cols[-1] + tile < W: cols.append(W - tile)
    w1 = np.hanning(tile + 2)[1:-1]; win = np.outer(w1, w1) + 1e-3
    acc = np.zeros((H, W)); ws = np.zeros((H, W))
    for r in rows:
        batch = [to_input(np.ascontiguousarray(rgb[r:r + tile, c:c + tile])) for c in cols]
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(torch.stack(batch).to(dev)).float().cpu().numpy()
        for c, o in zip(cols, out):
            acc[r:r + tile, c:c + tile] += win * o; ws[r:r + tile, c:c + tile] += win
    return (acc / ws).astype(np.float32)

@torch.no_grad()
def predict_tile_tta(net, rgb):
    # mean and spread (std) of 4 orientation variants: identity, h-flip, v-flip, rot180 (all exact inverses)
    outs = []
    for f in (lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a[::-1, ::-1]):
        outs.append(f(predict_tile(net, np.ascontiguousarray(f(rgb)))))
    st = np.stack(outs)
    return st.mean(0).astype(np.float32), st.std(0).astype(np.float32)

import gc
EDGES = np.linspace(-150.0, 150.0, 30001)   # 1 cm residual bins (|residual| <= 120 m since nDSM is clipped to [0, 120])
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])

class Stats:
    # streaming nDSM error statistics: ME, RMSE, MAE, NMAD, Pearson r, and RMSE / ME for objects (>= 2.5 m) and ground
    def __init__(self):
        self.n = 0
        self.sums = np.zeros(8)                      # d, d^2, |d|, p, g, p^2, g^2, p*g   (float64)
        self.hist = np.zeros(len(CENTRES), np.int64)
        self.sub = {"objects_ge2.5m": np.zeros(3), "ground_lt2.5m": np.zeros(3)}   # n, sum d, sum d^2
    def add(self, pred, gt):
        m = np.isfinite(gt) & np.isfinite(pred)
        p = pred[m].astype(np.float64); g = gt[m].astype(np.float64); d = p - g
        self.n += d.size
        self.sums += (d.sum(), (d * d).sum(), np.abs(d).sum(), p.sum(), g.sum(), (p * p).sum(), (g * g).sum(), (p * g).sum())
        self.hist += np.histogram(np.clip(d, -149.995, 149.995), bins=EDGES)[0]
        for k, sel in (("objects_ge2.5m", g >= 2.5), ("ground_lt2.5m", g < 2.5)):
            dd = d[sel]; self.sub[k] += (dd.size, dd.sum(), (dd * dd).sum())
        return self
    def metrics(self):
        n = self.n
        if n == 0:
            return {"n": 0}
        sd, sd2, sad, sp, sg, spp, sgg, spg = self.sums
        med = CENTRES[np.searchsorted(np.cumsum(self.hist), n / 2)]
        dev = np.abs(CENTRES - med); o = np.argsort(dev, kind="stable")
        mad = dev[o][np.searchsorted(np.cumsum(self.hist[o]), n / 2)]
        r = (n * spg - sp * sg) / math.sqrt(max(1e-30, (n * spp - sp * sp) * (n * sgg - sg * sg)))
        out = dict(n=int(n), ME=float(sd / n), RMSE=float(math.sqrt(sd2 / n)), MAE=float(sad / n), NMAD=float(1.4826 * mad), r=float(r))
        for k, (nn, s1, s2) in self.sub.items():
            if nn > 100:
                out[f"RMSE_{k}"] = float(math.sqrt(s2 / nn)); out[f"ME_{k}"] = float(s1 / nn)
        return out

def metrics(pred, gt):
    return Stats().add(pred, gt).metrics()

def evaluate(net, tiles, center=None, keep_first=False, affine=False):
    # streams over tiles; returns ({"all" | "source:<s>" | "region:<r>": Stats}, first (t, pred, gt) if keep_first).
    # affine=True fits pred -> LiDAR per tile first (the zero-shot baseline's oracle best case)
    groups, first = {}, None
    for t in tiles:
        rgb = np.load(f"{DATA}/{t['key']}_rgb.npy"); gt = np.load(f"{DATA}/{t['key']}_ndsm.npy").astype(np.float32)
        if center:
            H = rgb.shape[0]; a = (H - center) // 2; rgb, gt = rgb[a:a + center, a:a + center], gt[a:a + center, a:a + center]
        p = predict_tile(net, rgb)
        if affine:
            m = np.isfinite(gt)
            coef = np.linalg.lstsq(np.stack([p[m], np.ones(int(m.sum()), np.float32)], 1), gt[m], rcond=None)[0]
            p = (p * coef[0] + coef[1]).astype(np.float32)
        for k in ("all", f"source:{t.get('source', 'swiss')}", f"region:{t['region']}"):
            groups.setdefault(k, Stats()).add(p, gt)
        if keep_first and first is None:
            first = (t, p, gt)
        del rgb, p, gt
    gc.collect()
    return groups, first
""")

code(r"""
# 10. Train (resumes from the last checkpoint if the session restarted)
_missing = [n for n in ("model", "evaluate", "train_dl",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import time
LAST, BEST = f"{CKPT}/last.pt", f"{CKPT}/best.pth"
start, best = 0, float("inf"); history = []
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location="cpu")
    start, best, history = ck["it"] + 1, ck["best"], ck["history"]
    KEYS = ("iters", "batch", "tiles_per_region", "use_us", "us_sample_prob", "building_weight", "lr_encoder", "lr_head", "model_version")
    diff = {k: (ck.get("cfg", {}).get(k), CFG[k]) for k in KEYS if ck.get("cfg", {}).get(k) != CFG[k]}
    assert not diff, f"checkpoint was written with different settings {diff}: use a new CFG['run_name'] for a fresh run"
    if start < CFG["iters"]:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        print("resumed at iteration", start)
    else:
        print(f"training already complete ({start} iterations, best val RMSE {best:.3f} m): going straight to evaluation")
    del ck; gc.collect()
if start == 0:
    json.dump({s: [t["key"] for t in v] for s, v in SPLIT.items()}, open(f"{CKPT}/split.json", "w"))   # the tiles this run trains on
    vm0 = evaluate(model, SPLIT["val"], center=1036)[0]["all"].metrics()
    history.append(dict(it=0, loss=None, **{f"val_{k}": v for k, v in vm0.items()}))
    print(f"  VAL before training (warm-start weights): RMSE {vm0['RMSE']:.2f} m  MAE {vm0['MAE']:.2f}  r {vm0['r']:.3f}  <- training must beat this")
it = start; t0 = time.time(); run = 0.0
loader = iter(train_dl) if start < CFG["iters"] else None
while it < CFG["iters"]:
    x, y, m, sh = next(loader)
    x, y, m, sh = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), m.to(dev, non_blocking=True), sh.to(dev)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        p = model(x)
    loss = loss_fn(p.float(), y, m, sh)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if it > start else loss.item()
    if it % 50 == 0:
        print(f"it {it:5d}  loss {run:.3f}  lr {sched.get_last_lr()[1]:.2e}  {(time.time() - t0) / max(1, it - start + 1):.2f}s/it")
    if (it + 1) % CFG["val_every"] == 0 or it + 1 == CFG["iters"]:
        vm = evaluate(model, SPLIT["val"], center=1036)[0]["all"].metrics()
        history.append(dict(it=it + 1, loss=run, **{f"val_{k}": v for k, v in vm.items()}))
        print(f"  VAL it {it + 1}: RMSE {vm['RMSE']:.2f} m  MAE {vm['MAE']:.2f}  ME {vm['ME']:+.2f}  r {vm['r']:.3f}")
        if vm["RMSE"] < best:
            best = vm["RMSE"]; torch.save(model.state_dict(), BEST); print("  -> new best, saved")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), it=it, best=best, history=history, cfg=CFG), LAST)
    if it % 1000 == 0:
        print("  ", ram())
    it += 1
del loader; gc.collect()   # stops the data-loader worker processes and frees their memory
json.dump({"iters": CFG["iters"], "best_val_rmse": best}, open(f"{CKPT}/done.json", "w"))
print("done. best val RMSE:", round(best, 3), "|", ram())
""")

code(r"""
# 11. Test on held-out REGIONS: fine-tuned v2 vs v1 vs zero-shot baseline (the baseline gets an oracle per-tile affine fit = its best case)
#     Streaming evaluation: one tile and one model in memory at a time (fits a free Colab runtime however many tiles).
_missing = [n for n in ("BEST", "evaluate",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import matplotlib.pyplot as plt
for _n in ("model", "opt", "sched", "scaler", "train_dl", "loader"):   # training objects are not needed any more
    globals().pop(_n, None)
gc.collect(); torch.cuda.empty_cache()

def load_net(path):
    net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    net.load_state_dict(torch.load(path, map_location="cpu")); return net.to(dev).eval()

def free(net):
    del net; gc.collect(); torch.cuda.empty_cache()

ft = load_net(BEST)
G_ft, FIRST = evaluate(ft, SPLIT["test"], keep_first=True)
test_ft = G_ft["all"].metrics()
by_source = {k.split(":", 1)[1]: v.metrics() for k, v in G_ft.items() if k.startswith("source:")}
per_region = {k.split(":", 1)[1]: v.metrics() for k, v in G_ft.items() if k.startswith("region:")}
print("by source:", {k: round(v["RMSE"], 2) for k, v in by_source.items()}, "|", ram())
V1_COMPARE = None
if INIT and os.path.exists(INIT):
    v1 = load_net(INIT); G_v1, _ = evaluate(v1, SPLIT["test"]); free(v1)
    V1_COMPARE = {}
    for src in ("swiss", "us"):
        a, b = G_ft.get(f"source:{src}"), G_v1.get(f"source:{src}")
        if a is not None and b is not None:
            ma, mb = a.metrics(), b.metrics()
            V1_COMPARE[src] = {"v2_RMSE": ma["RMSE"], "v1_RMSE": mb["RMSE"], "v2_objects_RMSE": ma.get("RMSE_objects_ge2.5m"), "v1_objects_RMSE": mb.get("RMSE_objects_ge2.5m")}
    print("v2 vs v1 on held-out test tiles:", {k: {kk: round(vv, 2) for kk, vv in v.items() if vv is not None} for k, v in V1_COMPARE.items()})
    sw = V1_COMPARE.get("swiss")
    if sw and sw["v2_RMSE"] > sw["v1_RMSE"]:
        print("VERDICT: v2 is WORSE than v1 on Swiss held-out LiDAR - do not install it; keep v1.")
    elif sw:
        print("VERDICT: v2 beats v1 on Swiss held-out LiDAR - download and install it; DepthWizard re-validation on its own test tiles decides the final switch.")

# Uncertainty: calibrate |error| quantiles per TTA-spread bin on VALIDATION tiles, check coverage on TEST tiles.
# Every 2nd pixel in each direction (millions of samples, a quarter of the memory).
def tta_errors(tiles, center=1036, step=2):
    S, E = [], []
    for t in tiles:
        rgb = np.load(f"{DATA}/{t['key']}_rgb.npy"); gt = np.load(f"{DATA}/{t['key']}_ndsm.npy").astype(np.float32)
        a = (rgb.shape[0] - center) // 2; rgb, gt = rgb[a:a + center, a:a + center], gt[a:a + center, a:a + center]
        mu, sd = predict_tile_tta(ft, rgb)
        mu, sd, gt = mu[::step, ::step], sd[::step, ::step], gt[::step, ::step]; m = np.isfinite(gt)
        S.append(sd[m].astype(np.float32)); E.append(np.abs(mu - gt)[m].astype(np.float32))
        del rgb, mu, sd, gt
    return np.concatenate(S), np.concatenate(E)
s_val, e_val = tta_errors(SPLIT["val"])
edges = np.unique(np.quantile(s_val, np.linspace(0, 1, 9)))
QS = (0.5, 0.8, 0.9)
bins_q = []
for i in range(len(edges) - 1):
    sel = (s_val >= edges[i]) & (s_val <= edges[i + 1])
    bins_q.append([float(np.quantile(e_val[sel], q)) if sel.any() else float("nan") for q in QS])
del s_val, e_val; gc.collect()
s_te, e_te = tta_errors(SPLIT["test"])
bi = np.clip(np.searchsorted(edges, s_te, side="right") - 1, 0, len(bins_q) - 1)
coverage = {f"{int(q * 100)}%": float(np.mean(e_te <= np.array([b[j] for b in bins_q])[bi])) for j, q in enumerate(QS)}
err_spread_corr = float(np.corrcoef(s_te, e_te)[0, 1])
del s_te, e_te, bi; gc.collect()
UNC = {"method": "4-way test-time augmentation (identity, h-flip, v-flip, rot180); per-pixel spread binned by validation-set quantiles; |error| quantiles per bin", "spread_bin_edges_m": [float(v) for v in edges], "abs_error_quantiles_m": {f"{int(q * 100)}%": [b[j] for b in bins_q] for j, q in enumerate(QS)}, "test_coverage": coverage, "test_spread_error_correlation": err_spread_corr}
print("uncertainty: test coverage", {k: round(v, 3) for k, v in coverage.items()}, "| spread-error corr", round(err_spread_corr, 3), "|", ram())
base = load_net(BASE_W); G_b, _ = evaluate(base, SPLIT["test"], affine=True); free(base)
test_b = G_b["all"].metrics()
print("TEST (held-out regions) nDSM vs LiDAR, metres")
print(" fine-tuned          :", {k: round(v, 3) for k, v in test_ft.items()})
print(" zero-shot + oracle  :", {k: round(v, 3) for k, v in test_b.items()})
for k, v in per_region.items():
    print(f"  {k:12s} RMSE {v['RMSE']:.2f}  MAE {v['MAE']:.2f}  ME {v['ME']:+.2f}  objects RMSE {v.get('RMSE_objects_ge2.5m', float('nan')):.2f}")
print(ram())
t, p, g = FIRST; rgb = np.load(f"{DATA}/{t['key']}_rgb.npy")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(rgb[::4, ::4]); ax[0].set_title(f"test {t['region']} {t['key']}")
ax[1].imshow(p[::4, ::4], vmin=0, vmax=30); ax[1].set_title("predicted nDSM (m)")
ax[2].imshow(g[::4, ::4], vmin=0, vmax=30); ax[2].set_title("LiDAR nDSM (m)"); plt.show()
del rgb, p, g, FIRST; gc.collect()
""")

code(r"""
# 12. Export weights + measured report and download
_missing = [n for n in ("ft", "UNC",) if n not in globals()]
assert not _missing, f"Runtime was reset or earlier cells were skipped (missing {_missing}): use Runtime -> Run all. Checkpoints in Drive are kept; training resumes."
import zipfile, hashlib, datetime
W = f"{ROOT}/depth_anything_v2_ndsm_s_v2.pth"
torch.save({k: v.detach().cpu() for k, v in ft.state_dict().items()}, W)
sha = hashlib.sha256(open(W, "rb").read()).hexdigest()
report = dict(
    created=datetime.datetime.utcnow().isoformat() + "Z", weights_file="depth_anything_v2_ndsm_s.pth", sha256=sha, model_version=CFG["model_version"],
    warm_start=bool(INIT and os.path.exists(INIT)), test_metrics_by_source=by_source, uncertainty_calibration=UNC, v1_comparison=V1_COMPARE, run_name=CFG["run_name"],
    base_model="depth-anything/Depth-Anything-V2-Small (Apache-2.0)", upstream_commit="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
    output_quantity="metric_ndsm_metres", training_gsd_m=CFG["gsd"], input_size=CFG["crop"], config=CFG,
    data="swisstopo SWISSIMAGE (RGB, 0.5 m from 0.1 m COG overviews) + swissSURFACE3D raster - swissALTI3D (0.5 m) = nDSM; (c) swisstopo OGD" + ("; USA: NAIP RGB (resampled to 0.5 m) + USGS 3DEP LiDAR DSM - DTM (2 m, bilinear to 0.5 m) via Microsoft Planetary Computer (public domain)" if US_TILES else ""),
    excluded_validation_tiles=DW_VALIDATION_TILES, exclusion_km=CFG["exclusion_km"],
    tiles=(json.load(open(f"{CKPT}/split.json")) if os.path.exists(f"{CKPT}/split.json")
           else {s: [t["key"] + " (" + t["region"] + ")" for t in TILES + US_TILES if t["split"] == s] for s in ("train", "val", "test")}),
    tiles_note="split.json = the tiles this run trained / validated on" if os.path.exists(f"{CKPT}/split.json") else "planned split from the cached tile lists (tiles that failed to download were not used)",
    history=history, test_metrics_finetuned=test_ft, test_metrics_zeroshot_oracle_affine=test_b, test_metrics_by_region=per_region,
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
)
json.dump(report, open(f"{ROOT}/training_report.json", "w"), indent=1)
Z = "/content/depthwizard_ndsm_model_v2.zip"
with zipfile.ZipFile(Z, "w") as z:
    z.write(W, "depth_anything_v2_ndsm_s.pth"); z.write(f"{ROOT}/training_report.json", "training_report.json")
print("sha256", sha, "| zip MB", round(os.path.getsize(Z) / 1e6, 1))
try:
    from google.colab import files; files.download(Z)
except Exception as e:
    print("download manually from the Files panel:", Z, e)
""")


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "cells": [],
    }
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).with_name("DepthWizard_FineTune_nDSM_v2_Colab.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, len(CELLS), "cells")


if __name__ == "__main__":
    main()
