"""Generates notebooks/DepthWizard_v2_Evaluate_Export_Colab.ipynb: evaluate the TRAINED v2 model and export it,
designed never to run out of RAM on a free Colab runtime (12.7 GB).

Why a separate notebook: the v2 training notebook's evaluation kept the whole training setup alive and grew in RAM
tile after tile (glibc heap fragmentation of large numpy buffers, 3 parallel downloads with their own GDAL caches,
TTA errors concatenated over all tiles). This one:
  * needs only the trained weights (best.pth from Drive, or uploaded) - no training data, no optimiser state;
  * downloads ONE tile at a time into a small local cache, never in parallel, with a 64 MB GDAL cache;
  * holds one model on the GPU at a time and one tile in RAM at a time, and calls malloc_trim after every tile so
    freed memory really goes back to the system;
  * keeps statistics in constant memory (running sums + 1 cm histograms), the uncertainty calibration included;
  * saves every tile's result to Drive, so after any crash / disconnect Run all resumes exactly where it stopped;
  * refuses to continue (with a clear message) if RAM ever passes a safety limit, instead of letting Colab crash.

Run:  python notebooks/build_eval_v2_notebook.py
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
# DepthWizard v2 — evaluate the trained model and export it (RAM-safe)

Use this **after** the v2 training run finished (it produced `best.pth`). It does **no training**.

**What it does**
1. Loads the trained v2 weights, from your Google Drive (`depthwizard_ft/v2b/ckpt/best.pth`) **or** from a file you upload to `/content/`.
2. Scores v2 on the **held-out test regions** (Swiss: Basel, Lugano, Thurgau, Jura-Ajoie, Davos · US: Sacramento, Ponce PR, St Louis). These are the same regions and rules as the training notebook, and none of them were used in training.
3. Compares it with **v1** (if you upload `depth_anything_v2_ndsm_s.pth`) and with the **zero-shot** model (given its best-case per-tile scale fit).
4. Calibrates the **uncertainty** (4-way flip test-time augmentation) on the validation regions and checks its coverage on the test regions.
5. Prints a **verdict** (install v2 or keep v1) and exports `depthwizard_ndsm_model_v2.zip` for `scripts/install_finetuned_model.py`.

**How to run**
1. Runtime → Change runtime type → **T4 GPU**.
2. If the weights are **not** in this account's Drive: upload `best.pth` (or `depth_anything_v2_ndsm_s_v2.pth`) to `/content/` in the Files panel. Optional: also upload v1 `depth_anything_v2_ndsm_s.pth` for the v2-vs-v1 comparison.
3. Runtime → **Run all**. It takes about 1–1.5 h on a T4.
4. If anything disconnects, just **Run all** again. Every finished tile is saved in Drive and skipped next time.

**Memory design**
* One tile (about 16 MB) and one model are in memory at a time.
* Downloads are sequential.
* Statistics are running sums and histograms, never growing arrays.
* Freed memory is returned to the system after every tile (`malloc_trim`).
* If RAM ever goes above 10 GB, the notebook stops itself with a message instead of crashing. Then **Runtime → Restart session → Run all**, and it resumes.
""")

code(r"""
# 1. Runtime + memory helpers
import os, sys, gc, json, math, time, ctypes, subprocess
os.environ.update(GDAL_CACHEMAX="64", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3",
                  VSI_CACHE="FALSE", OMP_NUM_THREADS="2", MALLOC_TRIM_THRESHOLD_="0")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "pyproj", "huggingface_hub", "requests", "psutil"], check=True)
import numpy as np, torch, psutil
torch.set_num_threads(2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU - this will be very slow. Runtime -> Change runtime type -> T4 GPU, then Run all.")
RAM_LIMIT_GB = 10.0     # of 12.7 GB on free Colab: stop cleanly before the kernel is killed

def ram_gb():
    return psutil.Process().memory_info().rss / 1e9

def trim():
    # return freed heap memory to the OS (glibc keeps large freed numpy buffers otherwise: the classic 'RAM creeps up' problem)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if dev == "cuda":
        torch.cuda.empty_cache()

def guard(where=""):
    trim()
    r = ram_gb()
    if r > RAM_LIMIT_GB:
        raise SystemExit(f"RAM {r:.1f} GB > {RAM_LIMIT_GB} GB at {where}. Nothing is lost: Runtime -> Restart session, then Run all (finished tiles are skipped).")
    return f"RAM {r:.1f} GB"
print(guard("start"))
""")

code(r"""
# 2. Settings, weights and resume folders
USE_DRIVE = True
RUN_NAME = "v2b"                 # the training run's name (training notebook CFG['run_name'])
MODEL_VERSION = "2.0.0"
GSD, TILE_M = 0.5, 1000          # 0.5 m, 1 km tiles = 2000 x 2000 px (same as training)
N = int(TILE_M / GSD)
MAX_H = 120.0
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/depthwizard_ft" if USE_DRIVE else "/content/depthwizard_ft"
RUN = f"{ROOT}/{RUN_NAME}"
EVAL = f"{RUN}/eval_v2"          # per-tile results (resume) + tile lists
CACHE = "/content/eval_tiles"    # local tile cache (~0.8 GB disk, not RAM)
os.makedirs(EVAL, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

def first_existing(paths):
    return next((p for p in paths if p and os.path.exists(p)), None)
V2_W = first_existing([f"{RUN}/ckpt/best.pth", "/content/best.pth", "/content/depth_anything_v2_ndsm_s_v2.pth", f"{ROOT}/depth_anything_v2_ndsm_s_v2.pth"])
V1_W = first_existing(["/content/depth_anything_v2_ndsm_s.pth", f"{ROOT}/depth_anything_v2_ndsm_s.pth"])
assert V2_W, ("v2 weights not found. Either run this in the Google account whose Drive has depthwizard_ft/" + RUN_NAME +
              "/ckpt/best.pth, or upload best.pth to /content/ (Files panel), then Run all again.")
print("v2 weights:", V2_W, f"({os.path.getsize(V2_W) / 1e6:.0f} MB)")
print("v1 weights:", V1_W or "not uploaded (v2-vs-v1 comparison skipped)")
# a training checkpoint (last.pt) also holds optimiser state (~3x larger): extract only the model weights, once
_sd = torch.load(V2_W, map_location="cpu")
if isinstance(_sd, dict) and "model" in _sd and "opt" in _sd:
    torch.save(_sd["model"], "/content/v2_model_only.pth"); V2_W = "/content/v2_model_only.pth"; print("extracted model weights from a training checkpoint")
del _sd
DW_VALIDATION_TILES = [(2682, 1247), (2621, 1202)]   # DepthWizard's own test tiles: never touched here either
SWISS = {"aarau": (8.044, 47.392, "val"), "fribourg": (7.161, 46.803, "val"), "entlebuch": (8.063, 46.990, "val"), "engelberg": (8.405, 46.820, "val"),
         "basel": (7.590, 47.559, "test"), "lugano": (8.952, 46.004, "test"), "thurgau": (9.100, 47.550, "test"), "jura_ajoie": (7.080, 47.420, "test"), "davos": (9.830, 46.800, "test")}
US = {"tucson": (-110.970, 32.220, "val"), "pittsburgh": (-79.990, 40.440, "val"),
      "sacramento": (-121.490, 38.580, "test"), "ponce_pr": (-66.610, 18.010, "test"), "st_louis": (-90.200, 38.630, "test")}
TILES_PER_REGION = 4             # training notebook used up to 6 (CH) / 5 (US); 4 keeps the run ~1 h and is plenty of pixels
print(guard("settings"))
""")

code(r"""
# 3. Depth Anything V2 code (same pinned commit as DepthWizard) + zero-shot baseline weights
if not os.path.exists("/content/Depth-Anything-V2"):
    subprocess.run(["git", "clone", "-q", "https://github.com/DepthAnything/Depth-Anything-V2", "/content/Depth-Anything-V2"], check=True)
    subprocess.run(["git", "-C", "/content/Depth-Anything-V2", "checkout", "-q", "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"], check=True)
sys.path.insert(0, "/content/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2
from huggingface_hub import hf_hub_download
BASE_W = hf_hub_download("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth")

def load_net(path):
    net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    sd = torch.load(path, map_location="cpu"); net.load_state_dict(sd, strict=True); del sd
    return net.to(dev).eval()

def free(net):
    net.cpu(); del net; trim()
print(guard("code"))
""")

code(r"""
# 4. Tile lists for the validation + test regions only (cached in Drive; same pairing rules as training)
import re, urllib.request, requests, threading
from pyproj import Transformer
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/{}/items?bbox={}&limit=100"
COLL = {"rgb": "ch.swisstopo.swissimage-dop10", "dsm": "ch.swisstopo.swisssurface3d-raster", "dtm": "ch.swisstopo.swissalti3d"}
to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

def stac_items(coll, bbox):
    url, out = STAC.format(coll, ",".join(f"{v:.5f}" for v in bbox)), []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        out += d["features"]; url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
    return out

def asset(item, suffix):
    return next((a["href"] for a in item["assets"].values() if a["href"].endswith(suffix)), None)

PC = "https://planetarycomputer.microsoft.com/api"
_tok, _lock = {}, threading.Lock()
def sign(href, coll):
    with _lock:
        if coll not in _tok or time.time() - _tok[coll][1] > 2400:
            for attempt in range(10):
                try:
                    r = requests.get(f"{PC}/sas/v1/token/{coll}", timeout=60); j = r.json() if r.ok else {}
                except Exception:
                    r, j = None, {}
                if "token" in j:
                    _tok[coll] = (j["token"], time.time()); break
                wait = min(300, 30 * (attempt + 1)); print(f"  token {coll}: waiting {wait}s (rate limit)"); time.sleep(wait)
            else:
                raise RuntimeError("Planetary Computer token refused for 25 min: wait 15 min, Run all again")
    return f"{href}?{_tok[coll][0]}"

def pc_search(coll, bbox, limit=100):
    r = requests.post(f"{PC}/stac/v1/search", json={"collections": [coll], "bbox": bbox, "limit": limit}, timeout=120); r.raise_for_status()
    return r.json()["features"]

def year(it):
    p = it["properties"]; return int((p.get("datetime") or p.get("start_datetime") or "2000")[:4])

LIST = f"{EVAL}/tiles.json"
if os.path.exists(LIST):
    TILES = json.load(open(LIST))
else:
    TILES = []
    for name, (lon, lat, split) in SWISS.items():
        bbox = (lon - 0.013, lat - 0.009, lon + 0.013, lat + 0.009)
        by = {k: {} for k in COLL}
        for k, c in COLL.items():
            for it in stac_items(c, bbox):
                m = re.search(r"_(\d{4})_(\d{4})-(\d{4})$", it["id"])
                if m:
                    by[k].setdefault((int(m.group(2)), int(m.group(3))), []).append((int(m.group(1)), it))
        keys = sorted(set(by["rgb"]) & set(by["dsm"]) & set(by["dtm"]))
        cx, cy = to_lv95.transform(lon, lat); keys.sort(key=lambda k: math.hypot(k[0] + 0.5 - cx / 1000, k[1] + 0.5 - cy / 1000))
        n_ok = 0
        for key in keys:
            if any(math.hypot(key[0] - e, key[1] - n) <= 5.0 for e, n in DW_VALIDATION_TILES):
                continue
            ys, dsm_it = max(by["dsm"][key], key=lambda t: t[0]); yt, dtm_it = max(by["dtm"][key], key=lambda t: t[0])
            yr, rgb_it = min(by["rgb"][key], key=lambda t: abs(t[0] - ys))
            u = (asset(rgb_it, "_0.1_2056.tif"), asset(dsm_it, "_0.5_2056_5728.tif"), asset(dtm_it, "_0.5_2056_5728.tif"))
            if abs(yr - ys) <= 3 and all(u):
                TILES.append(dict(key=f"{key[0]}-{key[1]}", region=name, split=split, source="swiss", rgb=u[0], dsm=u[1], dtm=u[2])); n_ok += 1
            if n_ok >= TILES_PER_REGION:
                break
        print(f"{name:12s} {split:5s} {n_ok} tiles")
    from pyproj import CRS as _CRS
    import rasterio
    for name, (lon, lat, split) in US.items():
        try:
            bbox = [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05]
            hag = [h for h in pc_search("3dep-lidar-hag", bbox) if year(h) >= 2017]; naip = pc_search("naip", bbox)
            n_ok = 0
            for h_it in sorted(hag, key=year, reverse=True):
                href = h_it["assets"]["data"]["href"]
                with rasterio.open(sign(href, "3dep-lidar-hag")) as ds:
                    c = _CRS.from_wkt(ds.crs.to_wkt()); hc = c.sub_crs_list[0] if c.is_compound else c
                    crs = f"EPSG:{hc.to_epsg()}" if hc.to_epsg() else hc.to_wkt()
                # the DSM / DTM files of the same tile id can cover a DIFFERENT (smaller) area than the HAG file
                # (Sacramento: a 368 m strip) -> place windows inside the overlap of the files actually read
                bs = []
                for kind in ("dsm", "dtm"):
                    with rasterio.open(sign(href.replace("/hag/", f"/{kind}/").replace("-hag-", f"-{kind}-"), f"3dep-lidar-{kind}")) as ds:
                        bs.append(ds.bounds)
                b = rasterio.coords.BoundingBox(max(x.left for x in bs), max(x.bottom for x in bs), min(x.right for x in bs), min(x.top for x in bs))
                if b.right - b.left < 1200 or b.top - b.bottom < 1200:
                    continue
                cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
                cx, cy = min(max(cx, b.left + 600), b.right - 600), min(max(cy, b.bottom + 600), b.top - 600)
                to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                for dx, dy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
                    left, top = cx - 500 + dx * 1000, cy + 500 + dy * 1000
                    if left < b.left or left + 1000 > b.right or top > b.top or top - 1000 < b.bottom:
                        continue
                    (w, s_), (e, n_) = to_ll.transform(left, top - 1000), to_ll.transform(left + 1000, top)
                    cover = sorted([x for x in naip if x["bbox"][0] <= w and x["bbox"][1] <= s_ and x["bbox"][2] >= e and x["bbox"][3] >= n_], key=lambda x: abs(year(x) - year(h_it)))
                    if not cover or abs(year(cover[0]) - year(h_it)) > 4:
                        continue
                    TILES.append(dict(key=f"us_{name}_{n_ok}", region=name, split=split, source="us", crs=crs, left=left, top=top, hag=h_it["assets"]["data"]["href"], naip=[x["assets"]["image"]["href"] for x in cover[:3]]))
                    n_ok += 1
                    if n_ok >= TILES_PER_REGION:
                        break
                if n_ok >= TILES_PER_REGION:
                    break
            print(f"{name:12s} {split:5s} {n_ok} tiles")
        except Exception as ex:
            print(f"{name:12s} skipped: {type(ex).__name__}: {ex}")
    if sum(t["split"] == "test" for t in TILES) >= 15:
        json.dump(TILES, open(LIST, "w"), indent=1)
    else:
        print("tile search incomplete (network?) - list not cached; Run all again in 10 min")
from collections import Counter
print(Counter((t["split"], t["source"]) for t in TILES), "|", guard("lists"))
""")

code(r"""
# 5. One tile at a time: download (sequential, 1 km window only) -> small local .npy cache -> load
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin

def fetch(t):
    fr, fh = f"{CACHE}/{t['key']}_rgb.npy", f"{CACHE}/{t['key']}_ndsm.npy"
    if os.path.exists(fr) and os.path.exists(fh):
        return True
    for attempt in range(3):
        try:
            if t["source"] == "swiss":
                with rasterio.open("/vsicurl/" + t["rgb"]) as ds:
                    rgb = ds.read((1, 2, 3), out_shape=(3, N, N), resampling=Resampling.average)
                hs = []
                for u in (t["dsm"], t["dtm"]):
                    with rasterio.open("/vsicurl/" + u) as ds:
                        hs.append(ds.read(1, out_shape=(N, N), resampling=Resampling.average, masked=True).astype("float32").filled(np.nan))
                h = hs[0] - hs[1]; del hs
            else:
                tr = from_origin(t["left"], t["top"], GSD, GSD); hs = []
                for kind in ("dsm", "dtm"):
                    href = t["hag"].replace("/hag/", f"/{kind}/").replace("-hag-", f"-{kind}-")
                    with rasterio.open(sign(href, f"3dep-lidar-{kind}")) as ds, WarpedVRT(ds, crs=t["crs"], transform=tr, width=N, height=N, resampling=Resampling.bilinear, src_nodata=ds.nodata, nodata=np.nan, dtype="float32") as v:
                        hs.append(v.read(1).astype(np.float32))
                h = hs[0] - hs[1]; del hs
                rgb = np.zeros((3, N, N), np.uint8); got = np.zeros((N, N), bool)
                for href in t["naip"]:
                    with rasterio.open(sign(href, "naip")) as ds, WarpedVRT(ds, crs=t["crs"], transform=tr, width=N, height=N, resampling=Resampling.average, src_nodata=0, nodata=0) as v:
                        tmp = v.read((1, 2, 3))
                    m = (tmp.max(axis=0) > 0) & ~got; rgb[:, m] = tmp[:, m]; got |= m; del tmp
                    if got.mean() > 0.99:
                        break
                h[~got] = np.nan; del got
            ndsm = np.clip(h, 0.0, MAX_H); ndsm[~np.isfinite(h)] = np.nan; del h
            cov = float(np.isfinite(ndsm).mean())
            if cov < 0.8 or rgb.max() == 0:
                print(f"  {t['key']}: skipped (LiDAR / image coverage {cov:.0%})")
                return False
            np.save(fr, np.ascontiguousarray(rgb.transpose(1, 2, 0))); np.save(fh, ndsm.astype(np.float16))
            del rgb, ndsm
            return True
        except Exception as ex:
            print(f"  {t['key']}: {type(ex).__name__}: {ex} (retry {attempt + 1}/3)"); time.sleep(20)
        finally:
            trim()
    return False

def load(t):
    return np.load(f"{CACHE}/{t['key']}_rgb.npy"), np.load(f"{CACHE}/{t['key']}_ndsm.npy").astype(np.float32)

ok_tiles = []
for i, t in enumerate(TILES):
    if fetch(t):
        ok_tiles.append(t)
    if i % 5 == 0:
        print(f"[{i + 1}/{len(TILES)}] {t['key']} | {guard('download')}")
SPLIT = {s: [t for t in ok_tiles if t["split"] == s] for s in ("val", "test")}
print({s: len(v) for s, v in SPLIT.items()}, "|", guard("downloads"))
assert len(SPLIT["test"]) >= 10, "too few test tiles downloaded (network?): Run all again - finished tiles are kept"
""")

code(r"""
# 6. Inference exactly like DepthWizard (518 px tiles, 25 % overlap, Hann feathering) + constant-memory statistics
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
TILE, STEP = 518, int(518 * 0.75)
W1 = np.hanning(TILE + 2)[1:-1].astype(np.float32); WIN = np.outer(W1, W1) + 1e-3

@torch.no_grad()
def predict(net, rgb):
    H, W = rgb.shape[:2]
    rows = list(range(0, H - TILE + 1, STEP)); cols = list(range(0, W - TILE + 1, STEP))
    if rows[-1] + TILE < H: rows.append(H - TILE)
    if cols[-1] + TILE < W: cols.append(W - TILE)
    acc = np.zeros((H, W), np.float32); ws = np.zeros((H, W), np.float32)
    for r in rows:
        x = np.stack([((rgb[r:r + TILE, c:c + TILE].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for c in cols])
        xt = torch.from_numpy(x).to(dev); del x
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            out = net(xt).float().cpu().numpy()
        del xt
        for c, o in zip(cols, out):
            acc[r:r + TILE, c:c + TILE] += WIN * o; ws[r:r + TILE, c:c + TILE] += WIN
        del out
    acc /= ws; del ws
    return acc

FLIPS = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a[::-1, ::-1]]
@torch.no_grad()
def predict_tta(net, rgb):
    # running mean / variance over the 4 exact-inverse orientations (Welford): never 4 full maps at once
    mean = None
    for k, f in enumerate(FLIPS):
        p = np.ascontiguousarray(f(predict(net, np.ascontiguousarray(f(rgb)))))
        if mean is None:
            mean = p.copy(); m2 = np.zeros_like(p)
        else:
            d = p - mean; mean += d / (k + 1); m2 += d * (p - mean); del d
        del p
    return mean, np.sqrt(m2 / len(FLIPS))

EDGES = np.linspace(-150.0, 150.0, 30001); CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
class Stats:
    def __init__(self):
        self.n = 0; self.sums = np.zeros(8); self.hist = np.zeros(len(CENTRES), np.int64); self.sub = np.zeros((2, 3))
    def add(self, p, g):
        m = np.isfinite(g) & np.isfinite(p); p = p[m].astype(np.float64); g = g[m].astype(np.float64); d = p - g
        self.n += d.size
        self.sums += (d.sum(), (d * d).sum(), np.abs(d).sum(), p.sum(), g.sum(), (p * p).sum(), (g * g).sum(), (p * g).sum())
        self.hist += np.histogram(np.clip(d, -149.995, 149.995), bins=EDGES)[0]
        for i, sel in enumerate((g >= 2.5, g < 2.5)):
            dd = d[sel]; self.sub[i] += (dd.size, dd.sum(), (dd * dd).sum())
        return self
    def merge(self, o):
        self.n += o.n; self.sums += o.sums; self.hist += o.hist; self.sub += o.sub; return self
    def save(self, path):
        np.savez_compressed(path, n=self.n, sums=self.sums, hist=self.hist, sub=self.sub)
    @staticmethod
    def load(path):
        z = np.load(path); s = Stats(); s.n = int(z["n"]); s.sums = z["sums"]; s.hist = z["hist"]; s.sub = z["sub"]; return s
    def metrics(self):
        n = self.n
        if n == 0:
            return {"n": 0}
        sd, sd2, sad, sp, sg, spp, sgg, spg = self.sums
        cum = np.cumsum(self.hist); med = CENTRES[np.searchsorted(cum, n / 2)]
        dev_ = np.abs(CENTRES - med); o = np.argsort(dev_, kind="stable"); mad = dev_[o][np.searchsorted(np.cumsum(self.hist[o]), n / 2)]
        r = (n * spg - sp * sg) / math.sqrt(max(1e-30, (n * spp - sp * sp) * (n * sgg - sg * sg)))
        out = dict(n=int(n), ME=float(sd / n), RMSE=float(math.sqrt(sd2 / n)), MAE=float(sad / n), NMAD=float(1.4826 * mad), r=float(r))
        for i, k in enumerate(("objects_ge2.5m", "ground_lt2.5m")):
            nn, s1, s2 = self.sub[i]
            if nn > 100:
                out[f"RMSE_{k}"] = float(math.sqrt(s2 / nn)); out[f"ME_{k}"] = float(s1 / nn)
        return out
print(guard("functions"))
""")

code(r"""
# 7. Evaluate each model on the TEST tiles, one model and one tile at a time; every tile's result is saved (resume)
def evaluate(tag, weights, tiles, affine=False):
    d = f"{EVAL}/{tag}"; os.makedirs(d, exist_ok=True)
    todo = [t for t in tiles if not os.path.exists(f"{d}/{t['key']}.npz")]
    if todo:
        net = load_net(weights)
        for i, t in enumerate(todo):
            rgb, gt = load(t)
            p = predict(net, rgb); del rgb
            if affine:   # zero-shot baseline, oracle per-tile scale + offset (its best possible case)
                m = np.isfinite(gt); A = np.stack([p[m], np.ones(int(m.sum()), np.float32)], 1)
                coef = np.linalg.lstsq(A, gt[m], rcond=None)[0]; del A, m
                p = p * np.float32(coef[0]) + np.float32(coef[1])
            Stats().add(p, gt).save(f"{d}/{t['key']}.npz"); del p, gt
            print(f"  {tag} [{i + 1}/{len(todo)}] {t['key']} | {guard(tag)}")
        free(net)
    groups = {}
    for t in tiles:
        s = Stats.load(f"{d}/{t['key']}.npz")
        for k in ("all", f"source:{t['source']}", f"region:{t['region']}"):
            groups.setdefault(k, Stats()).merge(s)
    return {k: v.metrics() for k, v in groups.items()}

R_V2 = evaluate("v2", V2_W, SPLIT["test"])
print("v2 test:", {k: round(v["RMSE"], 2) for k, v in R_V2.items() if not k.startswith("region")})
R_V1 = evaluate("v1", V1_W, SPLIT["test"]) if V1_W else None
if R_V1: print("v1 test:", {k: round(v["RMSE"], 2) for k, v in R_V1.items() if not k.startswith("region")})
R_ZS = evaluate("zeroshot_oracle", BASE_W, SPLIT["test"], affine=True)
print("zero-shot (oracle scale) test:", {k: round(v["RMSE"], 2) for k, v in R_ZS.items() if not k.startswith("region")})
print(guard("evaluation"))
""")

code(r"""
# 8. Uncertainty: 4-way TTA spread -> calibrated 50 / 80 / 90 % error intervals (validation), coverage (test).
#    Constant memory: per spread bin, a 1 cm histogram of |error|; per tile saved to Drive (resume).
SPREAD_EDGES = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 1e9], np.float32)   # m, fixed (no data-dependent edges)
ERR_EDGES = np.linspace(0, 150, 15001)
QS = (0.5, 0.8, 0.9)
CENTER, STEP_PX = 1036, 2          # centre crop, every 2nd pixel: plenty of samples, 1/4 of the work

def spread_err_hist(tag, tiles):
    d = f"{EVAL}/{tag}"; os.makedirs(d, exist_ok=True)
    todo = [t for t in tiles if not os.path.exists(f"{d}/{t['key']}.npy")]
    if todo:
        net = load_net(V2_W)
        for i, t in enumerate(todo):
            rgb, gt = load(t); a = (rgb.shape[0] - CENTER) // 2
            rgb, gt = np.ascontiguousarray(rgb[a:a + CENTER, a:a + CENTER]), gt[a:a + CENTER, a:a + CENTER]
            mu, sd = predict_tta(net, rgb); del rgb
            mu, sd, gt = mu[::STEP_PX, ::STEP_PX], sd[::STEP_PX, ::STEP_PX], gt[::STEP_PX, ::STEP_PX]; m = np.isfinite(gt)
            b = np.clip(np.searchsorted(SPREAD_EDGES, sd[m], side="right") - 1, 0, len(SPREAD_EDGES) - 2)
            e = np.abs(mu - gt)[m]
            H = np.zeros((len(SPREAD_EDGES) - 1, len(ERR_EDGES) - 1), np.int64)
            for k in range(H.shape[0]):
                H[k] = np.histogram(e[b == k], bins=ERR_EDGES)[0]
            np.save(f"{d}/{t['key']}.npy", H); del mu, sd, gt, m, b, e, H
            print(f"  {tag} [{i + 1}/{len(todo)}] {t['key']} | {guard(tag)}")
        free(net)
    return sum(np.load(f"{d}/{t['key']}.npy") for t in tiles)

Hv = spread_err_hist("tta_val", SPLIT["val"])
qtab = np.full((Hv.shape[0], len(QS)), np.nan)
for k in range(Hv.shape[0]):
    c = np.cumsum(Hv[k])
    if c[-1] > 1000:
        qtab[k] = [ERR_EDGES[1:][np.searchsorted(c, q * c[-1])] for q in QS]
Ht = spread_err_hist("tta_test", SPLIT["test"])
coverage = {}
for j, q in enumerate(QS):
    inside = total = 0
    for k in range(Ht.shape[0]):
        if np.isfinite(qtab[k, j]):
            c = np.cumsum(Ht[k]); total += int(c[-1]); inside += int(c[np.searchsorted(ERR_EDGES[1:], qtab[k, j])])
    coverage[f"{int(q * 100)}%"] = inside / max(total, 1)
UNC = {"method": "4-way TTA (identity, h-flip, v-flip, rot180): per-pixel spread; |error| quantiles per fixed spread bin, calibrated on validation regions",
       "spread_bin_edges_m": [float(v) for v in SPREAD_EDGES[:-1]] + ["inf"],
       "abs_error_quantiles_m": {f"{int(q * 100)}%": [None if not np.isfinite(v) else round(float(v), 2) for v in qtab[:, j]] for j, q in enumerate(QS)},
       "test_coverage": {k: round(v, 3) for k, v in coverage.items()}}
del Hv, Ht
print("uncertainty coverage on test (target 0.5 / 0.8 / 0.9):", UNC["test_coverage"], "|", guard("uncertainty"))
""")

code(r"""
# 9. Verdict, report, export
import zipfile, hashlib, datetime
def row(r, src):
    m = (r or {}).get(f"source:{src}")
    return None if not m else {"RMSE": round(m["RMSE"], 3), "MAE": round(m["MAE"], 3), "r": round(m["r"], 3), "objects_RMSE": round(m.get("RMSE_objects_ge2.5m", float("nan")), 3)}
print(f"{'':22s}{'Swiss test':>28s}{'US test':>28s}")
for name, r in (("v2 (this model)", R_V2), ("v1", R_V1), ("zero-shot, oracle scale", R_ZS)):
    if r: print(f"{name:22s}{str(row(r, 'swiss')):>28s}  {str(row(r, 'us')):>28s}")
verdict = "no v1 uploaded: compare v2 with v1's published Swiss test RMSE 3.86 m (training_report.json of v1)"
if R_V1:
    sw2, sw1 = R_V2["source:swiss"]["RMSE"], R_V1["source:swiss"]["RMSE"]
    us2, us1 = (R_V2.get("source:us") or {}).get("RMSE"), (R_V1.get("source:us") or {}).get("RMSE")
    better_sw, better_us = sw2 < sw1, (us2 is not None and us1 is not None and us2 < us1)
    if better_sw and better_us:
        verdict = f"INSTALL v2: better than v1 on BOTH held-out sets (Swiss {sw1:.2f} -> {sw2:.2f} m, US {us1:.2f} -> {us2:.2f} m)"
    elif better_us and not better_sw:
        verdict = f"MIXED: v2 better on US ({us1:.2f} -> {us2:.2f}) but worse on Swiss ({sw1:.2f} -> {sw2:.2f}); re-validate in DepthWizard before switching"
    elif better_sw:
        verdict = f"MIXED: v2 better on Swiss ({sw1:.2f} -> {sw2:.2f}) but not on US; re-validate in DepthWizard before switching"
    else:
        verdict = f"KEEP v1: v2 is not better on the held-out regions (Swiss {sw1:.2f} -> {sw2:.2f} m)"
print("\nVERDICT:", verdict)
report = dict(created=datetime.datetime.utcnow().isoformat() + "Z", model_version=MODEL_VERSION, weights_file="depth_anything_v2_ndsm_s.pth",
              base_model="depth-anything/Depth-Anything-V2-Small (Apache-2.0)", upstream_commit="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
              output_quantity="metric_ndsm_metres", training_gsd_m=GSD, input_size=518, run_name=RUN_NAME,
              test_metrics_finetuned=R_V2["all"], test_metrics_by_source={k.split(":")[1]: v for k, v in R_V2.items() if k.startswith("source:")},
              test_metrics_by_region={k.split(":")[1]: v for k, v in R_V2.items() if k.startswith("region:")},
              test_metrics_v1=R_V1["all"] if R_V1 else None, test_metrics_v1_by_source={k.split(":")[1]: v for k, v in R_V1.items() if k.startswith("source:")} if R_V1 else None,
              test_metrics_zeroshot_oracle_affine=R_ZS["all"], uncertainty_calibration=UNC, verdict=verdict,
              tiles={s: [t["key"] + " (" + t["region"] + ")" for t in SPLIT[s]] for s in ("val", "test")},
              excluded_validation_tiles=DW_VALIDATION_TILES, evaluation_notebook="DepthWizard_v2_Evaluate_Export_Colab.ipynb",
              gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__)
os.makedirs("/content/export", exist_ok=True)
W = "/content/export/depth_anything_v2_ndsm_s.pth"   # not /content/: that name may be the uploaded v1 file
sd = torch.load(V2_W, map_location="cpu"); torch.save({k: v.contiguous() for k, v in sd.items()}, W); del sd
report["sha256"] = hashlib.sha256(open(W, "rb").read()).hexdigest()
json.dump(report, open("/content/training_report.json", "w"), indent=1)
json.dump(report, open(f"{EVAL}/training_report.json", "w"), indent=1)
Z = "/content/depthwizard_ndsm_model_v2.zip"
with zipfile.ZipFile(Z, "w") as z:
    z.write(W, "depth_anything_v2_ndsm_s.pth"); z.write("/content/training_report.json", "training_report.json")
print("zip:", Z, round(os.path.getsize(Z) / 1e6, 1), "MB | sha256", report["sha256"][:16], "|", guard("export"))
try:
    from google.colab import files; files.download(Z)
except Exception as ex:
    print("download it from the Files panel:", Z, ex)
""")


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "cells": []}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    out = Path(__file__).with_name("DepthWizard_v2_Evaluate_Export_Colab.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, len(CELLS), "cells")


if __name__ == "__main__":
    main()
