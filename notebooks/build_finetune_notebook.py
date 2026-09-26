"""Generates notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb (keeps the notebook reviewable as plain Python).

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
# DepthWizard — fine-tune Depth Anything V2 Small to metric height above ground (nDSM)

**What this does (end to end, ~2–3 h on a free T4):**
1. Builds a training set from swisstopo open data: SWISSIMAGE RGB at **0.5 m** + target **nDSM = swissSURFACE3D − swissALTI3D** (airborne LiDAR, metres), for ~150 one-km tiles in 36 regions (cities, farmland, forest, Jura, Alps).
2. Splits **by region** into train / val / test. The two DepthWizard validation tiles (Zürich 2682-1247, Emmental 2621-1202) and everything within 5 km of them are **never downloaded**.
3. Fine-tunes Depth Anything V2 Small (Apache-2.0) to output metres, with the same 518 px / 0.5 m tiling DepthWizard uses at inference.
4. Evaluates on the held-out **test regions** against LiDAR and against the zero-shot baseline (given an *oracle* per-tile affine fit, i.e. its best case).
5. Exports `depth_anything_v2_ndsm_s.pth` + `training_report.json` (measured numbers, tile lists, config) as `depthwizard_ndsm_model.zip`.

**Before running:** Runtime → Change runtime type → **T4 GPU** (or better). Then Runtime → Run all.
If the session disconnects, set `USE_DRIVE = True` (checkpoints + dataset survive in Google Drive) and re-run: finished steps are skipped.

**After it finishes:** copy `depthwizard_ndsm_model.zip` into the DepthWizard folder and run
`python scripts/install_finetuned_model.py depthwizard_ndsm_model.zip`.

Data © swisstopo (Open Government Data, attribution required). Model: Depth Anything V2 Small, Apache-2.0.
""")

code(r"""
# 1. Runtime check + dependencies
import subprocess, sys, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", torch.cuda.get_device_name(0), "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "pyproj", "huggingface_hub"], check=True)
""")

code(r"""
# 2. Settings (edit here)
USE_DRIVE = False          # True: keep dataset + checkpoints in Google Drive (survives disconnects)
CFG = dict(
    gsd=0.5,               # training GSD (m/px) = DepthWizard Mode B inference GSD
    crop=518,              # DA-V2 native tile size (multiple of 14)
    batch=6,
    iters=8000,            # ~1-1.5 h on a T4; more helps (e.g. 15000 on A100/L4)
    warmup=300,
    lr_encoder=5e-6,       # DA-V2 metric fine-tuning recipe: small LR for the ViT, 10x for the DPT head
    lr_head=5e-5,
    weight_decay=0.01,
    grad_weight=0.5,       # multi-scale gradient-matching term (sharp roof / canopy edges)
    val_every=500,
    max_height_m=120.0,    # nDSM clipped to [0, 120] m (tallest Swiss structures/trees are below)
    max_year_gap=3,        # RGB and LiDAR acquisitions at most 3 years apart
    exclusion_km=5.0,      # no tile within 5 km of a DepthWizard validation tile
    tiles_per_region=4,
    workers=8,
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
DATA, CKPT = f"{ROOT}/tiles", f"{ROOT}/ckpt"
os.makedirs(DATA, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
print("working dir:", ROOT, "| regions:", len(REGIONS))
""")

code(r"""
# 3. Depth Anything V2 code (pinned upstream commit, same as DepthWizard's vendored copy) + baseline weights
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

TILES_JSON = f"{ROOT}/tiles.json"
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
    json.dump(TILES, open(TILES_JSON, "w"), indent=1)
from collections import Counter
print("total:", len(TILES), Counter(t["split"] for t in TILES))
assert not any(near_validation(*map(int, t["key"].split("-"))) for t in TILES)
""")

code(r"""
# 5. Download + prepare tiles: RGB read at 0.5 m from the COG overviews, nDSM = DSM - DTM (both LN02, datum cancels)
import rasterio, numpy as np, time
from rasterio.enums import Resampling
from concurrent.futures import ThreadPoolExecutor
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
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

t0 = time.time()
with ThreadPoolExecutor(CFG["workers"]) as ex:
    for i, (k, st) in enumerate(ex.map(prepare, TILES)):
        if st != "cached" or i % 20 == 0:
            print(f"[{i + 1}/{len(TILES)}] {k}: {st}  ({time.time() - t0:.0f}s)")
READY = [t for t in TILES if os.path.exists(f"{DATA}/{t['key']}_ndsm.npy")]
SPLIT = {s: [t for t in READY if t["split"] == s] for s in ("train", "val", "test")}
print({s: len(v) for s, v in SPLIT.items()})
""")

code(r"""
# 6. Quick look at one training pair
import matplotlib.pyplot as plt
t = SPLIT["train"][0]
rgb = np.load(f"{DATA}/{t['key']}_rgb.npy", mmap_mode="r"); h = np.load(f"{DATA}/{t['key']}_ndsm.npy", mmap_mode="r")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(rgb[::4, ::4]); ax[0].set_title(f"{t['region']} {t['key']} RGB 0.5 m")
im = ax[1].imshow(np.asarray(h[::4, ::4], dtype=np.float32), cmap="viridis", vmin=0, vmax=30); ax[1].set_title("nDSM (m) from LiDAR"); plt.colorbar(im, ax=ax[1])
plt.show()
""")

code(r"""
# 7. Dataset with augmentations that match how DepthWizard feeds the model
import cv2, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def to_input(rgb_u8):
    x = (rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

class Crops(Dataset):
    def __init__(self, tiles, n, train=True):
        self.tiles, self.n, self.train = tiles, n, train
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        t = self.tiles[rng.integers(len(self.tiles))]
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
        return to_input(np.ascontiguousarray(x)), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(np.isfinite(y))

train_dl = DataLoader(Crops(SPLIT["train"], CFG["iters"] * CFG["batch"]), batch_size=CFG["batch"], num_workers=2, pin_memory=True, drop_last=True)
xb, yb, mb = next(iter(train_dl)); print(xb.shape, yb.shape, float(yb[mb].mean()), "m mean target")
""")

code(r"""
# 8. Model, loss, optimiser
from depth_anything_v2.dpt import DepthAnythingV2
dev = "cuda"
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(BASE_W, map_location="cpu"), strict=True)
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

def loss_fn(p, y, m):
    return F.smooth_l1_loss(p[m], y[m], beta=1.0) + CFG["grad_weight"] * grad_loss(p, y, m)

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
# 9. Full-tile inference exactly like DepthWizard (518 px tiles, 25 % overlap, Hann feathering) + metrics
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

def metrics(pred, gt):
    m = np.isfinite(gt) & np.isfinite(pred); d = (pred - gt)[m]; g = gt[m]
    out = dict(n=int(m.sum()), ME=float(d.mean()), RMSE=float(np.sqrt((d ** 2).mean())), MAE=float(np.abs(d).mean()),
               NMAD=float(1.4826 * np.median(np.abs(d - np.median(d)))), r=float(np.corrcoef(pred[m], g)[0, 1]))
    for name, sel in (("objects_ge2.5m", g >= 2.5), ("ground_lt2.5m", g < 2.5)):
        if sel.sum() > 100:
            out[f"RMSE_{name}"] = float(np.sqrt((d[sel] ** 2).mean())); out[f"ME_{name}"] = float(d[sel].mean())
    return out

def evaluate(net, tiles, center=None):
    res = []
    for t in tiles:
        rgb = np.load(f"{DATA}/{t['key']}_rgb.npy"); gt = np.load(f"{DATA}/{t['key']}_ndsm.npy").astype(np.float32)
        if center:
            H = rgb.shape[0]; a = (H - center) // 2; rgb, gt = rgb[a:a + center, a:a + center], gt[a:a + center, a:a + center]
        res.append((t, predict_tile(net, rgb), gt))
    allp = np.concatenate([p.ravel() for _, p, _ in res]); allg = np.concatenate([g.ravel() for _, _, g in res])
    return metrics(allp, allg), res
""")

code(r"""
# 10. Train (resumes from the last checkpoint if the session restarted)
import time
LAST, BEST = f"{CKPT}/last.pt", f"{CKPT}/best.pth"
start, best = 0, float("inf"); history = []
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location="cpu")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start, best, history = ck["it"] + 1, ck["best"], ck["history"]; print("resumed at iteration", start)
it = start; t0 = time.time(); run = 0.0
loader = iter(train_dl)
while it < CFG["iters"]:
    x, y, m = next(loader)
    x, y, m = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        p = model(x)
    loss = loss_fn(p.float(), y, m)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if it > start else loss.item()
    if it % 50 == 0:
        print(f"it {it:5d}  loss {run:.3f}  lr {sched.get_last_lr()[1]:.2e}  {(time.time() - t0) / max(1, it - start + 1):.2f}s/it")
    if (it + 1) % CFG["val_every"] == 0 or it + 1 == CFG["iters"]:
        vm, _ = evaluate(model, SPLIT["val"], center=1036)
        history.append(dict(it=it + 1, loss=run, **{f"val_{k}": v for k, v in vm.items()}))
        print(f"  VAL it {it + 1}: RMSE {vm['RMSE']:.2f} m  MAE {vm['MAE']:.2f}  ME {vm['ME']:+.2f}  r {vm['r']:.3f}")
        if vm["RMSE"] < best:
            best = vm["RMSE"]; torch.save(model.state_dict(), BEST); print("  -> new best, saved")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), it=it, best=best, history=history), LAST)
    it += 1
print("done. best val RMSE:", round(best, 3))
""")

code(r"""
# 11. Test on held-out REGIONS: fine-tuned vs zero-shot baseline (baseline gets an oracle per-tile affine fit = its best case)
import matplotlib.pyplot as plt
ft = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]); ft.load_state_dict(torch.load(BEST, map_location="cpu")); ft.to(dev)
base = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384]); base.load_state_dict(torch.load(BASE_W, map_location="cpu")); base.to(dev)
test_ft, res_ft = evaluate(ft, SPLIT["test"])
_, res_b = evaluate(base, SPLIT["test"])
aligned = []
for (t, p, g) in res_b:
    m = np.isfinite(g); A = np.stack([p[m], np.ones(m.sum())], 1); coef = np.linalg.lstsq(A, g[m], rcond=None)[0]
    aligned.append(p * coef[0] + coef[1])
test_b = metrics(np.concatenate([a.ravel() for a in aligned]), np.concatenate([g.ravel() for _, _, g in res_b]))
per_region = {}
for (t, p, g) in res_ft:
    per_region.setdefault(t["region"], []).append((p, g))
per_region = {k: metrics(np.concatenate([p.ravel() for p, _ in v]), np.concatenate([g.ravel() for _, g in v])) for k, v in per_region.items()}
print("TEST (held-out regions) nDSM vs LiDAR, metres")
print(" fine-tuned          :", {k: round(v, 3) for k, v in test_ft.items()})
print(" zero-shot + oracle  :", {k: round(v, 3) for k, v in test_b.items()})
for k, v in per_region.items():
    print(f"  {k:12s} RMSE {v['RMSE']:.2f}  MAE {v['MAE']:.2f}  ME {v['ME']:+.2f}  objects RMSE {v.get('RMSE_objects_ge2.5m', float('nan')):.2f}")
t, p, g = res_ft[0]; rgb = np.load(f"{DATA}/{t['key']}_rgb.npy")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(rgb[::4, ::4]); ax[0].set_title(f"test {t['region']} {t['key']}")
ax[1].imshow(p[::4, ::4], vmin=0, vmax=30); ax[1].set_title("predicted nDSM (m)")
ax[2].imshow(g[::4, ::4], vmin=0, vmax=30); ax[2].set_title("LiDAR nDSM (m)"); plt.show()
""")

code(r"""
# 12. Export weights + measured report and download
import zipfile, hashlib, datetime
W = f"{ROOT}/depth_anything_v2_ndsm_s.pth"
torch.save({k: v.detach().cpu() for k, v in ft.state_dict().items()}, W)
sha = hashlib.sha256(open(W, "rb").read()).hexdigest()
report = dict(
    created=datetime.datetime.utcnow().isoformat() + "Z", weights_file="depth_anything_v2_ndsm_s.pth", sha256=sha,
    base_model="depth-anything/Depth-Anything-V2-Small (Apache-2.0)", upstream_commit="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
    output_quantity="metric_ndsm_metres", training_gsd_m=CFG["gsd"], input_size=CFG["crop"], config=CFG,
    data="swisstopo SWISSIMAGE (RGB, 0.5 m from 0.1 m COG overviews) + swissSURFACE3D raster - swissALTI3D (0.5 m) = nDSM; (c) swisstopo OGD",
    excluded_validation_tiles=DW_VALIDATION_TILES, exclusion_km=CFG["exclusion_km"],
    tiles={s: [t["key"] + " (" + t["region"] + ")" for t in v] for s, v in SPLIT.items()},
    history=history, test_metrics_finetuned=test_ft, test_metrics_zeroshot_oracle_affine=test_b, test_metrics_by_region=per_region,
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
)
json.dump(report, open(f"{ROOT}/training_report.json", "w"), indent=1)
Z = "/content/depthwizard_ndsm_model.zip"
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
    out = Path(__file__).with_name("DepthWizard_FineTune_nDSM_Colab.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, len(CELLS), "cells")


if __name__ == "__main__":
    main()
