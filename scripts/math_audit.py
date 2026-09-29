"""Mathematical audit study for Building Intelligence + Hazard Screening (docs/math_audit.md).

Runs the old-vs-new comparisons that justify every method change:
  1. building-height estimators on synthetic roofs (known truth, Monte Carlo)
  2. building-volume estimators on the same roofs
  3. the same estimators on a real scene against independent LiDAR (swissSURFACE3D - swissALTI3D), if available
  4. binary-cell flooded-area error against the analytic area
  5. slope/aspect against analytic planes (old vs new implementation)

Usage:  python scripts/math_audit.py [--job <job_id>] [--trials 400] [--out docs/math_audit_results.md]
Deterministic (seeded). Nothing here feeds back into the pipeline: it only measures.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.terrain import robust as R  # noqa: E402

GSD = 0.5  # m, matches the demo scenes
BLUR_PX = 1.0  # Gaussian sigma of the simulated model smoothing (px)
NOISE_M = 0.8  # per-pixel model noise sigma (m)
CORE_PX = 2  # erosion radius used by the core estimators (1 m at 0.5 m GSD)


# ---------------------------------------------------------------- synthetic roofs
def make_case(rng: np.random.Generator, case: str, h: float, side_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (observed nDSM, true nDSM, footprint mask) on a padded scene."""
    n = int(round(side_m / GSD))
    pad = 12
    H = W = n + 2 * pad
    true = np.zeros((H, W))
    fp = np.zeros((H, W), bool)
    fp[pad:pad + n, pad:pad + n] = True
    true[fp] = h
    if case == "sloped_roof":  # gable roof, eaves h-2.5, ridge h+2.5 across the width: median = h exactly
        cols = np.arange(W)[None, :].repeat(H, 0)
        u = np.abs((cols - pad) - (n - 1) / 2.0) / ((n - 1) / 2.0)  # 0 at ridge, 1 at eaves
        true = np.where(fp, h + 2.5 - 5.0 * u, 0.0)
    if case == "multi_level":  # podium (60 % of area) at h, tower (40 %) at 2h -> median = podium height h
        k = int(round(0.4 * n))
        true[pad:pad + n, pad + n - k:pad + n] = 2.0 * h
    obs = ndimage.gaussian_filter(true, BLUR_PX)
    obs = obs + rng.normal(0.0, NOISE_M, obs.shape) if case != "uniform" else obs
    idx = np.flatnonzero(fp)
    if case == "salt_pepper":
        sel = rng.choice(idx, int(0.05 * idx.size), replace=False)
        obs.flat[sel] = np.where(rng.random(sel.size) < 0.5, 0.0, h + 15.0)
    if case == "trees":  # canopy over one edge strip: 15 % of the footprint at 30-70 % of the roof height
        k = max(1, int(round(0.15 * n)))
        blk = np.zeros_like(fp)
        blk[pad:pad + n, pad:pad + k] = True
        obs[blk] = rng.uniform(0.3 * h, 0.7 * h, int(blk.sum()))
    if case == "rooftop_equipment":  # 6 % of roof pixels +2..4 m (HVAC, lift housings)
        sel = rng.choice(idx, int(0.06 * idx.size), replace=False)
        obs.flat[sel] += rng.uniform(2.0, 4.0, sel.size)
    if case == "partial_nodata":
        sel = rng.choice(idx, int(0.2 * idx.size), replace=False)
        obs.flat[sel] = np.nan
    return obs, true, fp


def core_mask(fp: np.ndarray, r: int) -> np.ndarray:
    d = ndimage.distance_transform_edt(np.pad(fp, 1))[1:-1, 1:-1]
    return fp & (d > r)


def estimators() -> dict:
    def core(fn):
        def f(obs, fp):
            c = core_mask(fp, CORE_PX) & np.isfinite(obs)
            v = obs[c] if c.sum() >= 10 else obs[fp & np.isfinite(obs)]
            return fn(v)
        return f

    def full(fn):
        return lambda obs, fp: fn(obs[fp & np.isfinite(obs)])

    return {
        "mean": full(R.mean),
        "median (table, old)": full(R.median),
        "P85 (extrusion, old)": full(lambda v: R.percentile(v, 85)),
        "P90": full(lambda v: R.percentile(v, 90)),
        "trimmed mean 10-90": full(R.trimmed_mean),
        "Huber": full(R.huber),
        "core median": core(R.median),
        "core Huber": core(R.huber),
    }


def err_stats(e: np.ndarray) -> dict:
    ok = np.isfinite(e)
    e2 = e[ok]
    if not e2.size:
        return {"bias": np.nan, "mae": np.nan, "rmse": np.nan, "nmad": np.nan, "fail": 1.0}
    return {"bias": float(e2.mean()), "mae": float(np.abs(e2).mean()), "rmse": float(np.sqrt((e2**2).mean())), "nmad": R.nmad(e2), "fail": float(1 - ok.mean())}


CASES = ["uniform", "gaussian", "salt_pepper", "trees", "rooftop_equipment", "partial_nodata", "sloped_roof", "multi_level"]
BANDS = {"low (3-10 m)": (3, 10), "medium (10-25 m)": (10, 25), "high (25-60 m)": (25, 60)}


def synthetic_study(trials: int, seed: int = 7) -> tuple[dict, dict, dict, dict]:
    rng = np.random.default_rng(seed)
    est = estimators()
    per_case = {c: {k: [] for k in est} for c in CASES}
    per_case_mean = {c: {k: [] for k in est} for c in CASES}  # truth = V/A (mean of the true roof)
    per_band = {b: {k: [] for k in est} for b in BANDS}
    vol = {c: {"A x median (old)": [], "A x P85": [], "integral sum h dA (new)": []} for c in CASES}
    runtime = {k: 0.0 for k in est}
    for c in CASES:
        for t in range(trials):
            band = list(BANDS)[t % 3]
            h = rng.uniform(*BANDS[band])
            side = rng.uniform(6.0, 40.0)
            obs, true, fp = make_case(rng, c, h, side)
            h_true = float(np.median(true[fp]))
            h_true_mean = float(np.mean(true[fp]))
            for k, f in est.items():
                t0 = time.perf_counter()
                v = f(obs, fp)
                runtime[k] += time.perf_counter() - t0
                per_case[c][k].append(v - h_true)
                per_case_mean[c][k].append(v - h_true_mean)
                if c in ("gaussian", "trees", "salt_pepper", "rooftop_equipment"):
                    per_band[band][k].append(v - h_true)
            a_px = GSD * GSD
            v_true = float(true[fp].sum() * a_px)
            ok = fp & np.isfinite(obs)
            area = fp.sum() * a_px
            integ = float(np.nansum(obs[ok]) * a_px * fp.sum() / max(1, ok.sum()))
            vol[c]["A x median (old)"].append((area * np.median(obs[ok]) - v_true) / v_true)
            vol[c]["A x P85"].append((area * np.percentile(obs[ok], 85) - v_true) / v_true)
            vol[c]["integral sum h dA (new)"].append((integ - v_true) / v_true)
    n_evals = trials * len(CASES)
    runtime = {k: 1e6 * v / n_evals for k, v in runtime.items()}
    return per_case, per_band, vol, runtime, per_case_mean


# ---------------------------------------------------------------- real scene vs LiDAR
def _job_footprints(job: Path, ndsm: np.ndarray, terr: np.ndarray, tr):
    """The job's own pipeline footprints (building_labels.tif + buildings.json) when present; otherwise re-extract
    from the stored texture (older jobs; the JPEG texture differs slightly from the pipeline's RGB input)."""
    import json

    import rasterio

    if (job / "building_labels.tif").exists():
        with rasterio.open(job / "building_labels.tif") as d:
            labels = d.read(1)
        return json.loads((job / "buildings.json").read_text(encoding="utf-8")), labels
    from PIL import Image

    from core.terrain.lod1 import extract_lod1_buildings

    rgb = np.array(Image.open(job / "texture.jpg").convert("RGB"))
    rgb = rgb if rgb.shape[:2] == ndsm.shape else None
    data = extract_lod1_buildings(ndsm.astype(np.float32), terr.astype(np.float32), rgb=rgb, gsd_m=abs(tr.a), transform=tr, return_labels=True)
    return data, data.pop("_labels")


def real_study(job_id: str) -> dict | None:

    import rasterio

    job = ROOT / "data" / "jobs" / job_id
    ref_dsm = ROOT / "assets" / "reference" / "swisssurface3d_urban_2682-1247_dsm_0.5m.tif"
    ref_dtm = ROOT / "assets" / "reference" / "swissalti3d_urban_2682-1247_dtm_0.5m.tif"
    if not (job / "ndsm.tif").exists() or not ref_dsm.exists():
        return None

    def rd(p):
        with rasterio.open(p) as d:
            a = d.read(1).astype(np.float64)
            if d.nodata is not None:
                a[a == d.nodata] = np.nan
            return a, d.transform

    ndsm, tr = rd(job / "ndsm.tif")
    terr, _ = rd(job / "terrain.tif")
    s_dsm, tr_ref = rd(ref_dsm)
    s_dtm, _ = rd(ref_dtm)
    if tuple(tr)[:6] != tuple(tr_ref)[:6] or ndsm.shape != s_dsm.shape:
        return None
    ref_ndsm = s_dsm - s_dtm  # same datum (LN02) cancels: height above ground needs no datum transform

    data, labels = _job_footprints(job, ndsm, terr, tr)
    est = estimators()
    truths = ("median over footprint", "mean over footprint", "median over footprint core")
    errs = {t: {k: [] for k in est} for t in truths}
    vol = {"A x median (old)": [], "integral sum h dA (new)": []}
    slices = ndimage.find_objects(labels)
    a_px = abs(tr.a * tr.e - tr.b * tr.d)
    n_used = 0
    for b in data["buildings"]:
        sl = slices[b["id"] - 1]
        if sl is None:
            continue
        fp = labels[sl] == b["id"]
        ref = ref_ndsm[sl][fp]
        ref = ref[np.isfinite(ref)]
        if ref.size < 0.9 * fp.sum():
            continue
        truth = float(np.median(ref))
        if truth < 2.5:  # the model card's object threshold; below it the reference itself is not a building roof
            continue
        n_used += 1
        refc = ref_ndsm[sl][core_mask(fp, CORE_PX)]
        refc = refc[np.isfinite(refc)]
        tv = {truths[0]: truth, truths[1]: float(ref.mean()), truths[2]: float(np.median(refc)) if refc.size >= 10 else truth}
        pred = ndsm[sl]
        for k, f in est.items():
            v = f(pred, fp)
            for t in truths:
                errs[t][k].append(v - tv[t])
        ok = fp & np.isfinite(pred)
        v_ref = float(ref.sum() * a_px)
        vol["A x median (old)"].append((fp.sum() * a_px * np.median(pred[ok]) - v_ref) / v_ref)
        vol["integral sum h dA (new)"].append((np.nansum(pred[ok]) * a_px * fp.sum() / ok.sum() - v_ref) / v_ref)
    return {"n": n_used, "n_extracted": len(data["buildings"]), "height": {t: {k: err_stats(np.array(v)) for k, v in e.items()} for t, e in errs.items()}, "volume": {k: err_stats(np.array(v)) for k, v in vol.items()}}


# ---------------------------------------------------------------- flooded area & slope/aspect
def _clip_halfplane(poly, nx: float, ny: float, c: float):
    """Sutherland-Hodgman: keep the part of a convex polygon with nx*x + ny*y <= c."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = nx * p[0] + ny * p[1] - c, nx * q[0] + ny * q[1] - c
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _shoelace(poly) -> float:
    n = len(poly)
    return 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) if n >= 3 else 0.0


def flood_area_study() -> list[dict]:
    """Inclined plane T = s (x cos th + y sin th) on an L x L domain, water level W. Exact wet area = area of the
    square clipped by the half-plane T <= W (exact polygon clipping). Binary = count of cells whose centre value
    <= W times the cell area. Oblique boundary (th = 27 deg) and non-aligned W: no cell-edge coincidences."""
    rows = []
    L, th = 400.0, np.radians(27.0)
    nx, ny = np.cos(th), np.sin(th)
    for gsd in (0.5, 2.0, 10.0, 30.0):
        n = int(round(L / gsd))
        xc = (np.arange(n) + 0.5) * gsd
        X, Y = np.meshgrid(xc, xc)
        Lg = n * gsd
        for s in (0.01, 0.2):
            T = s * (X * nx + Y * ny)
            for frac in (0.1337, 0.5213):
                W = s * Lg * (nx + ny) * frac
                exact = _shoelace(_clip_halfplane([(0, 0), (Lg, 0), (Lg, Lg), (0, Lg)], nx, ny, W / s))
                binary = float((T <= W).sum()) * gsd * gsd
                rows.append({"gsd": gsd, "slope": s, "exact_m2": exact, "binary_m2": binary, "rel_err": (binary - exact) / exact})
    return rows


def slope_aspect_study() -> list[dict]:
    from core.validate.coregister import HORN_KX, HORN_KY

    def old(z, gx, gy):  # the pre-audit implementation, verbatim (aspect = atan2(p, -q), divides by column norms)
        p = ndimage.correlate(z, HORN_KX, mode="nearest") / gx
        q = ndimage.correlate(z, HORN_KY, mode="nearest") / gy
        return np.degrees(np.arctan(np.hypot(p, q))), (np.degrees(np.arctan2(p, -q)) + 360.0) % 360.0
    from core.dsm.derive import slope_aspect_affine as new
    from affine import Affine

    rows = []
    n = 9
    col, row = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    for (name, gx, gy, rot, sh) in (("square 1 m", 1.0, 1.0, 0.0, 0.0), ("non-square 2x0.5 m", 2.0, 0.5, 0.0, 0.0), ("rotated 30 deg", 1.0, 1.0, 30.0, 0.0), ("sheared 20 deg", 1.0, 1.0, 0.0, 20.0)):
        tr = Affine.translation(1000, 2000) * Affine.rotation(rot) * Affine.shear(sh, 0) * Affine.scale(gx, -gy)
        x = tr.a * col + tr.b * row + tr.c
        y = tr.d * col + tr.e * row + tr.f
        for label, a, b, exp_aspect in (("faces W (rises E)", 0.2, 0.0, 270.0), ("faces S (rises N)", 0.0, 0.2, 180.0), ("faces E", -0.2, 0.0, 90.0), ("faces N", 0.0, -0.2, 0.0), ("faces SW", 0.1, 0.1, 225.0)):
            z = a * x + b * y
            exp_slope = np.degrees(np.arctan(np.hypot(a, b)))
            so, ao = old(z, np.hypot(tr.a, tr.d), np.hypot(tr.b, tr.e))
            sn, an = new(z, tr)
            rows.append({"grid": name, "plane": label, "exp_slope": exp_slope, "old_slope": float(so[4, 4]), "new_slope": float(sn[4, 4]), "exp_aspect": exp_aspect, "old_aspect": float(ao[4, 4]), "new_aspect": float(an[4, 4])})
    return rows


# ---------------------------------------------------------------- ground estimation from a ring (prompt §2.1)
def _ransac_plane(x, y, z, rng, thr=0.5, iters=200):
    """Plane from the largest consensus set (|residual| <= thr, 0.5 m = 3.3 sigma of the simulated 0.15 m ground
    noise), refitted by least squares on the inliers. Coordinates are centred for conditioning."""
    x0, y0 = x.mean(), y.mean()
    X = np.column_stack([x - x0, y - y0, np.ones_like(x)])
    best = None
    for _ in range(iters):
        i = rng.choice(x.size, 3, replace=False)
        try:
            p = np.linalg.solve(X[i], z[i])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(X @ p - z) <= thr
        if best is None or inl.sum() > best.sum():
            best = inl
    p, *_ = np.linalg.lstsq(X[best], z[best], rcond=None)
    return lambda xx, yy: p[0] * (xx - x0) + p[1] * (yy - y0) + p[2]


def ground_study(trials: int, seed: int = 11) -> dict:
    """Constant-height building (h in 6-30 m) on a plane T = a x + b y + c; a 3 m ring around the footprint observes
    the DSM (ground + 0.15 m noise) contaminated by a neighbouring roof (25 % of the ring, +8..15 m, one side) and
    trees (10 %, +3..10 m). Height estimate = median over the footprint of DSM_i - g(x_i, y_i)."""
    rng = np.random.default_rng(seed)
    slopes = (0.0, 0.05, 0.15, 0.30)
    methods = ("A ring median", "B ring trimmed mean", "C ring P10", "D ring LSQ plane", "E ring RANSAC plane", "F pixelwise terrain layer (pipeline)")
    res = {sl: {m: {"med": [], "pix": []} for m in methods} for sl in slopes}
    n_r, n_c, ring_px = 24, 32, 6  # 12 x 16 m footprint at 0.5 m, 3 m ring
    H, W = n_r + 2 * ring_px, n_c + 2 * ring_px
    rows, cols = np.mgrid[0:H, 0:W]
    x, y = (cols + 0.5) * GSD, -(rows + 0.5) * GSD
    fp = np.zeros((H, W), bool)
    fp[ring_px:ring_px + n_r, ring_px:ring_px + n_c] = True
    ring = ~fp
    sides = [rows < ring_px, rows >= H - ring_px, cols < ring_px, cols >= W - ring_px]
    for sl in slopes:
        for _ in range(trials):
            th = rng.uniform(0, 2 * np.pi)
            T = 400.0 + sl * (np.cos(th) * x + np.sin(th) * y)
            h = rng.uniform(6, 30)
            dsm = T + rng.normal(0, 0.15, T.shape)
            dsm[fp] = T[fp] + h + rng.normal(0, 0.3, int(fp.sum()))
            nb = ring & sides[int(rng.integers(4))]
            nb &= rng.random(nb.shape) < 0.25 * ring.sum() / max(1, nb.sum())
            dsm[nb] += rng.uniform(8, 15)
            trees = ring & ~nb & (rng.random(ring.shape) < 0.10)
            dsm[trees] += rng.uniform(3, 10, int(trees.sum()))
            zr, xr, yr = dsm[ring], x[ring], y[ring]
            lo, hi = np.percentile(zr, [10, 90])
            X = np.column_stack([xr - xr.mean(), yr - yr.mean(), np.ones_like(xr)])
            p, *_ = np.linalg.lstsq(X, zr, rcond=None)
            g = {
                methods[0]: np.full(T.shape, np.median(zr)),
                methods[1]: np.full(T.shape, zr[(zr >= lo) & (zr <= hi)].mean()),
                methods[2]: np.full(T.shape, lo),
                methods[3]: p[0] * (x - xr.mean()) + p[1] * (y - yr.mean()) + p[2],
                methods[4]: _ransac_plane(xr, yr, zr, rng)(x, y),
                methods[5]: T,
            }
            for m, gg in g.items():
                hh = (dsm - gg)[fp]
                res[sl][m]["med"].append(float(np.median(hh)) - h)
                res[sl][m]["pix"].append(float(np.sqrt(np.mean((hh - h) ** 2))))
    return {sl: {m: (err_stats(np.array(v["med"])), float(np.mean(v["pix"]))) for m, v in d.items()} for sl, d in res.items()}


# ---------------------------------------------------------------- real-scene flood: old vs new exposure, runtime
def real_flood_study(job_id: str) -> dict | None:
    import json
    import shutil
    import tempfile

    import rasterio

    from core.disaster.flood import run_flood_screening
    from core.geo.grid import Grid
    from core.geo.raster_io import write_raster

    job = ROOT / "data" / "jobs" / job_id
    if not (job / "terrain.tif").exists() or not (job / "ndsm.tif").exists():
        return None
    with rasterio.open(job / "ndsm.tif") as d:
        ndsm = d.read(1).astype(np.float32)
        tr, crs = d.transform, d.crs.to_string()
        ndsm[ndsm == d.nodata] = np.nan
    with rasterio.open(job / "terrain.tif") as d:
        terr = d.read(1).astype(np.float32)
        terr[terr == d.nodata] = np.nan
    t0 = time.perf_counter()
    data, labels = _job_footprints(job, ndsm, terr, tr)
    t_extract = time.perf_counter() - t0
    out = {"n_buildings": data["count"], "extract_s": t_extract, "levels": []}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy(job / "terrain.tif", d / "terrain.tif")
        (d / "buildings.json").write_text(json.dumps(data), encoding="utf-8")
        write_raster(d / "building_labels.tif", labels, Grid(labels.shape[1], labels.shape[0], tr, crs, "int32", 0), dtype="int32")
        for W in np.nanpercentile(terr, [5, 25, 50]):
            W = float(round(W, 2))
            t0 = time.perf_counter()
            new = run_flood_screening(d, W, {})
            t_new = time.perf_counter() - t0
            old_aff = sum(1 for b in data["buildings"] if b["base_elev_m"] < W)  # pre-audit rule: W > median ground
            out["levels"].append({"W": W, "old_affected": old_aff, "new_affected": new["affectedBuildingsCount"], "contact": new["contactBuildingsCount"], "partial": sum(1 for b in new["buildings"] if b.get("wet_fraction") is not None and b["wet_fraction"] < 0.5), "isolated_m2": new["isolatedAreaM2"], "wet_m2": new["affectedAreaM2"], "runtime_s": t_new})
    return out


# ---------------------------------------------------------------- report
def fmt(v, d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="b1c4eab4e781")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "docs" / "math_audit_results.md"))
    a = ap.parse_args()
    L = ["# Math audit — measured results", "", f"Generated by `scripts/math_audit.py` (seed 7, {a.trials} trials per case, GSD {GSD} m, model blur sigma {BLUR_PX} px, noise sigma {NOISE_M} m, core erosion {CORE_PX} px). Errors are estimate − truth in metres; truth = median of the true roof over the footprint.", ""]
    per_case, per_band, vol, rt, per_case_mean = synthetic_study(a.trials)
    keys = list(estimators())
    L += ["## 1. Building height estimators — synthetic roofs", "", "RMSE (m) per case; last column = mean RMSE over all cases.", "", "| Estimator | " + " | ".join(CASES) + " | all |", "|---|" + "---:|" * (len(CASES) + 1)]
    for k in keys:
        rm = [err_stats(np.array(per_case[c][k]))["rmse"] for c in CASES]
        L.append(f"| {k} | " + " | ".join(fmt(v) for v in rm) + f" | **{fmt(float(np.mean(rm)))}** |")
    L += ["", "Pooled over all cases:", "", "| Estimator | MAE | RMSE | Bias | NMAD | Runtime (µs) | Failure rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        s = err_stats(np.concatenate([np.array(per_case[c][k]) for c in CASES]))
        L.append(f"| {k} | {fmt(s['mae'])} | {fmt(s['rmse'])} | {fmt(s['bias'])} | {fmt(s['nmad'])} | {fmt(rt[k], 0)} | {fmt(s['fail'], 3)} |")
    L += ["", "Same roofs, truth = **volume-preserving height V/A** (mean of the true roof over the footprint). RMSE (m):", "", "| Estimator | " + " | ".join(CASES) + " | all |", "|---|" + "---:|" * (len(CASES) + 1)]
    for k in keys:
        rm = [err_stats(np.array(per_case_mean[c][k]))["rmse"] for c in CASES]
        L.append(f"| {k} | " + " | ".join(fmt(v) for v in rm) + f" | **{fmt(float(np.mean(rm)))}** |")
    L += ["", "By height band (noise + contamination cases):", "", "| Estimator | " + " | ".join(BANDS) + " |", "|---|" + "---:|" * len(BANDS)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(fmt(err_stats(np.array(per_band[b][k]))["rmse"]) for b in BANDS) + " |")
    L += ["", "## 2. Building volume — synthetic (relative error of V vs exact ∫h dA)", "", "| Case | " + " | ".join(vol[CASES[0]]) + " |", "|---|" + "---:|" * 3]
    for c in CASES:
        L.append(f"| {c} | " + " | ".join(f"{fmt(100 * err_stats(np.array(v))['bias'], 1)} % ± {fmt(100 * err_stats(np.array(v))['rmse'], 1)}" for v in vol[c].values()) + " |")
    L.append("\n(bias % ± RMSE %)")
    real = real_study(a.job)
    L += ["", "## 3. Real scene — Zürich 2682-1247 vs independent LiDAR", ""]
    if real:
        L += [f"Footprints from the pipeline's own extraction ({real['n_extracted']} buildings); {real['n']} used (≥ 90 % valid reference pixels and reference median ≥ 2.5 m). Reference = swissSURFACE3D − swissALTI3D (0.5 m), never used for calibration.", "", "The footprints are derived from the prediction, so the truth definition matters; every estimator is scored against three reasonable definitions."]
        for t, hs in real["height"].items():
            L += ["", f"Truth = reference nDSM **{t}**:", "", "| Estimator | MAE | RMSE | Bias | NMAD | n |", "|---|---:|---:|---:|---:|---:|"]
            for k, s in hs.items():
                L.append(f"| {k} | {fmt(s['mae'])} | {fmt(s['rmse'])} | {fmt(s['bias'])} | {fmt(s['nmad'])} | {real['n']} |")
        L += ["", "Volume (relative error vs reference ∫h dA over the same footprint):", "", "| Method | Bias | MAE | RMSE |", "|---|---:|---:|---:|"]
        for k, s in real["volume"].items():
            L.append(f"| {k} | {fmt(100 * s['bias'], 1)} % | {fmt(100 * s['mae'], 1)} % | {fmt(100 * s['rmse'], 1)} % |")
    else:
        L.append("Not run: job or reference rasters missing.")
    gs = ground_study(max(60, a.trials // 3))
    L += ["", "## 2b. Building ground estimation from a ring (sloped terrain, contaminated ring)", "", "Height error = median over the footprint of (DSM − ground model) − true h. `pix RMSE` = RMS error of the per-pixel heights h_i (the profile a scalar ground cannot represent).", "", "| Terrain slope | Method | Bias | RMSE | NMAD | pix RMSE |", "|---:|---|---:|---:|---:|---:|"]
    for sl, dm in gs.items():
        for m, (st, pix) in dm.items():
            L.append(f"| {sl} | {m} | {fmt(st['bias'])} | {fmt(st['rmse'])} | {fmt(st['nmad'])} | {fmt(pix)} |")
    rf = real_flood_study(a.job)
    L += ["", "## 3b. Real scene — building flood exposure, old rule vs new (Zürich terrain layer)", ""]
    if rf:
        L += [f"{rf['n_buildings']} buildings (job footprints). Old rule: affected iff W > median footprint ground. New: exposure depth W − P10(footprint ground) > 0 on the exact label raster.", "", "| W (m) | old affected | new affected | in contact (any wet cell) | new, < 50 % wet (missed by old) | wet area (m²) | isolated (m²) | screening runtime (s) |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rf["levels"]:
            L.append(f"| {r['W']} | {r['old_affected']} | {r['new_affected']} | {r['contact']} | {r['partial']} | {r['wet_m2']:.0f} | {r['isolated_m2']:.0f} | {r['runtime_s']:.2f} |")
    L += ["", "## 4. Flooded area — binary cell count vs analytic (inclined plane)", "", "| GSD (m) | slope | exact (m²) | binary (m²) | rel. error |", "|---:|---:|---:|---:|---:|"]
    for r in flood_area_study():
        L.append(f"| {r['gsd']} | {r['slope']} | {r['exact_m2']:.1f} | {r['binary_m2']:.1f} | {100 * r['rel_err']:+.3f} % |")
    L += ["", "## 5. Slope / aspect — analytic planes (Horn), old vs new", "", "| Grid | Plane | slope exact | old | new | aspect exact | old | new |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    try:
        for r in slope_aspect_study():
            L.append(f"| {r['grid']} | {r['plane']} | {r['exp_slope']:.4f} | {r['old_slope']:.4f} | {r['new_slope']:.4f} | {r['exp_aspect']:.1f} | {r['old_aspect']:.1f} | {r['new_aspect']:.1f} |")
    except ImportError:
        L.append("(new implementation not present yet)")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
