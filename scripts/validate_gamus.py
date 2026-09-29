"""DepthWizard's height model on GAMUS - the dataset ISRO's SIH-DepthWizard-2026 page recommends
(https://huggingface.co/datasets/earthflow/GAMUS, CC BY 4.0; test split: Washington DC, New York, Philadelphia).

GAMUS gives 1024 x 1024 RGB tiles with height above ground (AGL, metres; about 0.33 m GSD, the dataset's nominal
resolution - the files carry no metadata). DepthWizard was never trained on GAMUS, so this is an out-of-distribution
test. A fixed random sample of test tiles (seed 0, equal per city) is run through exactly the app's tiled inference
(518 px tiles, 25 % overlap, resampled to 0.5 m for the model and back) and compared with the AGL map:
  * DepthWizard fine-tuned model (installed version, metres directly);
  * zero-shot Depth Anything V2 Small with an ORACLE per-tile affine fit to the reference (its best possible case).
Reference AGL is clipped at 0 (small negative values are LiDAR noise). Metrics: RMSE, MAE, Pearson r, and RMSE on
object pixels (AGL >= 2.5 m).
    python scripts/validate_gamus.py [--per-city 30]   -> data/gamus/results.json + printed table
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HF = "https://huggingface.co/datasets/earthflow/GAMUS/resolve/main/"
OUT = ROOT / "data" / "gamus"
GSD = 0.33


def sample(per_city: int, split: str = "test") -> list[str]:
    idx = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/earthflow/GAMUS", timeout=60))
    # image files are <stem>_RGB.h5 (DC, PHL) or <stem>_IMG.h5 (NYC)
    stems = sorted({Path(s["rfilename"]).name.rsplit("_", 1)[0] + "|" + Path(s["rfilename"]).name.rsplit("_", 1)[1] for s in idx["siblings"] if s["rfilename"].startswith(f"images/{split}/")})
    rng = random.Random(0)
    out = []
    for city in ("DC", "NYC", "PHL"):
        cs = [s for s in stems if s.startswith(city + "_")]
        out += rng.sample(cs, min(per_city, len(cs)))
    return out


def fetch(stem_img: str, split: str = "test") -> tuple[Path, Path] | None:
    stem, img_suffix = stem_img.split("|")
    d = OUT / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    pr, ph = d / f"{stem}_RGB.h5", d / f"{stem}_AGL.h5"
    for rel, p in ((f"images/{split}/{stem}_{img_suffix}", pr), (f"heights/{split}/{stem}_AGL.h5", ph)):
        for k in range(5):
            if p.exists() and p.stat().st_size > 1000:
                break
            try:
                tmp = p.with_suffix(".part")
                urllib.request.urlretrieve(HF + rel, tmp)
                tmp.replace(p)
            except Exception:  # noqa: BLE001
                time.sleep(5 * (k + 1))
        if not (p.exists() and p.stat().st_size > 1000):
            print("  download failed, tile skipped:", stem, flush=True)
            return None
    return pr, ph


class Acc:
    def __init__(self):
        self.n = 0
        self.s = np.zeros(8)
        self.obj = np.zeros(2)

    def add(self, p, g):
        m = np.isfinite(p) & np.isfinite(g)
        p, g = p[m].astype(np.float64), g[m].astype(np.float64)
        d = p - g
        self.n += d.size
        self.s += (d.sum(), (d * d).sum(), np.abs(d).sum(), p.sum(), g.sum(), (p * p).sum(), (g * g).sum(), (p * g).sum())
        o = g >= 2.5
        self.obj += (o.sum(), (d[o] ** 2).sum())

    def metrics(self):
        n = self.n
        sd, sd2, sad, sp, sg, spp, sgg, spg = self.s
        r = (n * spg - sp * sg) / np.sqrt(max(1e-30, (n * spp - sp * sp) * (n * sgg - sg * sg)))
        return {"n": int(n), "ME": round(sd / n, 3), "RMSE": round(float(np.sqrt(sd2 / n)), 3), "MAE": round(sad / n, 3), "r": round(float(r), 3),
                "RMSE_objects_ge2.5m": round(float(np.sqrt(self.obj[1] / max(self.obj[0], 1))), 3)}


def main() -> int:
    import h5py

    from backend.config.settings import load_settings
    from core.calib.fusion import plan_upsample, stitch_tiles, tiled_relative
    from core.inference.predictor import build_metric_predictor, build_predictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=30)
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--gsd", type=float, default=GSD, help="assumed GSD (m); choose it on --split val, report on test")
    a = ap.parse_args()
    s = load_settings()
    fz = s.fusion
    stems = sample(a.per_city, a.split)
    with ThreadPoolExecutor(4) as ex:
        pairs = [x for x in ex.map(lambda st: fetch(st, a.split), stems) if x]
    print(len(pairs), "GAMUS test tiles", flush=True)
    ft, zs = build_metric_predictor(s), build_predictor(s)

    def infer(pred, rgb):
        up = plan_upsample(rgb.shape[0], rgb.shape[1], a.gsd / fz.inference_gsd_m, tile=fz.tile_px, overlap=fz.overlap, max_tiles=fz.max_tiles)
        tp = tiled_relative(lambda x: pred.predict(x).relative_depth, rgb, upsample=up, tile=fz.tile_px, overlap=fz.overlap, inference_gsd_m=a.gsd / up)
        return stitch_tiles(tp).astype(np.float64)

    acc = {"finetuned": {"all": Acc()}, "zeroshot_oracle": {"all": Acc()}}
    for i, (pr, ph) in enumerate(pairs):
        city = pr.name.split("_")[0]
        rgb = h5py.File(pr, "r")["image"][...]
        agl = np.clip(h5py.File(ph, "r")["image"][...].astype(np.float64), 0.0, None)
        p = np.clip(infer(ft, rgb), 0.0, None)
        for k in ("all", city):
            acc["finetuned"].setdefault(k, Acc()).add(p, agl)
        z = infer(zs, rgb)
        A = np.c_[z.ravel(), np.ones(z.size)]
        coef = np.linalg.lstsq(A, agl.ravel(), rcond=None)[0]
        z = z * coef[0] + coef[1]
        for k in ("all", city):
            acc["zeroshot_oracle"].setdefault(k, Acc()).add(z, agl)
        if i % 10 == 0:
            print(f"[{i + 1}/{len(pairs)}] {pr.name[:-7]}  fine-tuned so far {acc['finetuned']['all'].metrics()['RMSE']} m", flush=True)
    res = {"model": f"{ft.card.name}@{ft.card.version}", "split": a.split, "tiles": len(pairs), "gsd_assumed_m": a.gsd,
           "results": {m: {k: v.metrics() for k, v in d.items()} for m, d in acc.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{a.split}_gsd{a.gsd:g}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n{'GAMUS test':12s} {'model':28s} {'RMSE':>6s} {'MAE':>6s} {'r':>6s} {'objects RMSE':>13s}")
    for k in ("all", "DC", "NYC", "PHL"):
        for m, name in (("finetuned", res["model"]), ("zeroshot_oracle", "zero-shot + oracle scale")):
            v = res["results"][m].get(k)
            if v:
                print(f"{k:12s} {name:28s} {v['RMSE']:6.2f} {v['MAE']:6.2f} {v['r']:6.3f} {v['RMSE_objects_ge2.5m']:13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
