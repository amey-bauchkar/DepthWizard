"""Landslide hazard screening (Disaster Management): susceptibility, rainfall trigger and recent scars.
NOT a slope-stability analysis and NOT validated against a landslide inventory yet (see docs/landslide.md).

1. Susceptibility: Landslide Hazard Evaluation Factor (LHEF), BIS IS 14496 (Part 2): 1998, the Indian standard for
   macro-zonation. Total Estimated Hazard (TEHD) = sum of six factor ratings:
       slope morphometry   facet slope (mean terrain slope in a FACET_M window)     0.5 .. 2.0   from DepthWizard
       relative relief     max - min terrain within RELIEF_RADIUS_M               0.3 .. 1.0   from DepthWizard
       land use / cover    vegetation / canopy / built share in a FACET_M window  0.65 .. 2.0  from the image + nDSM
       lithology           rock / soil type                                        0.2 .. 2.0   USER (GSI Bhukosh maps)
       structure           discontinuities vs slope                                0.3 .. 2.0   USER
       hydrogeology        ground-water condition                                  0.0 .. 1.0   USER or rainfall
   Classes: TEHD < 3.5 very low, 3.5-5 low, 5-6 moderate, 6-7.5 high, > 7.5 very high.
   Factors the user does not give are set to the middle of their range, and the result also reports the class with
   those factors at their minimum and maximum, so the effect of the unknown geology is visible.
2. Rainfall trigger: Himalayan intensity-duration threshold (Dahal & Hasegawa 2008, Nepal):
       I_th(D) = 73.90 * D^-0.79   (I mm/h, D hours)
   exceeded when the mean intensity over any duration D in TRIGGER_DURATIONS_H is above I_th(D). Rain comes from the
   user or from the Open-Meteo archive/forecast for the scene centre (free, no key).
3. Recent scars (optional, online): Sentinel-2 L2A, new bare ground (NDVI drop) on slopes >= SCAR_MIN_SLOPE_DEG
   between two dates, same rule as the flood-scar mapping (scripts/glof_observed_extent.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

from core.dsm.derive import slope_layers
from core.geo.grid import Grid
from core.geo.raster_io import write_raster

METHOD = "lhef-is14496-2 (terrain + image factors; geology from the user) + Himalayan rainfall threshold"
FACET_M, RELIEF_RADIUS_M, WORK_M = 30.0, 250.0, 5.0
CLASSES = [(3.5, 1, "VERY LOW"), (5.0, 2, "LOW"), (6.0, 3, "MODERATE"), (7.5, 4, "HIGH"), (math.inf, 5, "VERY HIGH")]
SLOPE_RATING = [(15.0, 0.5), (25.0, 0.8), (35.0, 1.2), (45.0, 1.7), (90.0, 2.0)]  # IS 14496-2: very gentle .. escarpment
RELIEF_RATING = [(100.0, 0.3), (300.0, 0.6), (math.inf, 1.0)]
LANDCOVER = {"populated": 0.65, "thick_vegetation": 0.8, "moderate_vegetation": 1.2, "sparse_vegetation": 1.5, "barren": 2.0}
USER_FACTORS = {  # name: (min, max, choices shown in the UI)
    "lithology": (0.2, 2.0, {"massive hard rock (granite, quartzite)": 0.3, "weathered hard rock": 0.8, "schist / phyllite / shale": 1.3,
                             "old well-compacted debris": 0.8, "young loose debris / soil": 1.5, "highly weathered rock or loose soil": 2.0}),
    "structure": (0.3, 2.0, {"discontinuities favourable to stability": 0.3, "moderately favourable": 0.8, "unfavourable (dip out of slope)": 1.5, "highly unfavourable": 2.0}),
    "hydrogeology": (0.0, 1.0, {"dry": 0.0, "damp": 0.2, "wet": 0.5, "dripping": 0.8, "flowing": 1.0}),
}
TRIGGER = (73.90, -0.79)
TRIGGER_DURATIONS_H = (1, 3, 6, 12, 24, 48, 72, 120)
SCAR_MIN_SLOPE_DEG = 20.0
PREVIEW = {1: (0, 0, 0, 0), 2: (0, 0, 0, 0), 3: (250, 204, 21, 55), 4: (234, 88, 12, 150), 5: (185, 28, 28, 190)}  # low shown clear: the map is about where to act


def _rate(v: np.ndarray, table: list[tuple[float, float]]) -> np.ndarray:
    out = np.full(v.shape, np.nan)
    lo = -np.inf
    for hi, r in table:
        out[(v > lo) & (v <= hi)] = r
        lo = hi
    return out


def classify(tehd: np.ndarray) -> np.ndarray:
    cls = np.zeros(tehd.shape, np.uint8)
    lo = -np.inf
    for hi, k, _ in CLASSES:
        cls[(tehd > lo) & (tehd <= hi)] = k
        lo = hi
    return cls


def trigger_threshold(duration_h: float) -> float:
    a, b = TRIGGER
    return a * duration_h ** b


def rainfall_trigger(hourly_mm: list[float]) -> dict[str, Any]:
    """Worst ratio of mean intensity to the threshold over the durations (moving windows over the hourly series)."""
    x = np.nan_to_num(np.asarray(hourly_mm, dtype=np.float64))
    rows = []
    for D in TRIGGER_DURATIONS_H:
        if len(x) < D:
            continue
        s = np.convolve(x, np.ones(D), "valid")
        k = int(np.argmax(s))
        I, th = s[k] / D, trigger_threshold(D)
        rows.append({"durationH": D, "rainMm": round(float(s[k]), 1), "intensityMmH": round(float(I), 2), "thresholdMmH": round(th, 2), "ratio": round(float(I / th), 2), "endIndex": k + D - 1})
    worst = max(rows, key=lambda r: r["ratio"]) if rows else None
    return {"rule": "I > 73.90 D^-0.79 (Dahal & Hasegawa 2008, Nepal Himalaya)", "exceeded": bool(worst and worst["ratio"] >= 1.0), "worst": worst, "byDuration": rows}


def fetch_rain(lat: float, lon: float, past_days: int = 5, forecast_days: int = 3) -> dict[str, Any]:
    """Hourly precipitation (mm) for the last past_days and next forecast_days from Open-Meteo (no key needed)."""
    import urllib.request

    u = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&hourly=precipitation&past_days={past_days}&forecast_days={forecast_days}&timezone=UTC"
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "DepthWizard/1.0"}), timeout=30) as r:
        j = json.load(r)
    return {"source": "Open-Meteo (open-meteo.com, CC BY 4.0), model blend incl. ECMWF / GFS", "time": j["hourly"]["time"], "mm": j["hourly"]["precipitation"]}


def _landcover(job_dir: Path, shape_w: tuple[int, int], f: int, n_facet: int) -> tuple[np.ndarray, dict[str, float]]:
    """Land-cover rating on the working grid from the input RGB (excess green) and nDSM (canopy) + buildings."""
    from core.disaster.flood import _footprint_sets

    with rasterio.open(job_dir / "input.tif") as ds:
        rgb = ds.read([1, 2, 3], out_shape=(3, shape_w[0] * f, shape_w[1] * f)).astype(np.float64) if ds.count >= 3 else None
    if rgb is None:
        return np.full(shape_w, 1.2), {"note": 1.0}
    s = rgb.sum(axis=0) + 1e-6
    veg = (2 * rgb[1] - rgb[0] - rgb[2]) / s > 0.03
    canopy = np.zeros_like(veg)
    if (job_dir / "ndsm.tif").exists():
        with rasterio.open(job_dir / "ndsm.tif") as ds:
            nd = ds.read(1, out_shape=veg.shape, masked=True).filled(0)
        canopy = veg & (nd > 2.5)
    bl = json.loads((job_dir / "buildings.json").read_text(encoding="utf-8")) if (job_dir / "buildings.json").exists() else {}
    with rasterio.open(job_dir / "input.tif") as ds:
        full_shape = (ds.height, ds.width)
    lab, _ = _footprint_sets(job_dir, bl, full_shape)
    built = np.zeros_like(veg) if lab is None else ndimage.zoom((lab > 0).astype(np.float32), (veg.shape[0] / full_shape[0], veg.shape[1] / full_shape[1]), order=0) > 0.5

    def frac(m: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(m.reshape(shape_w[0], f, shape_w[1], f).mean(axis=(1, 3)), n_facet)

    fv, fc, fb = frac(veg), frac(canopy), frac(built)
    r = np.where(fv < 0.1, LANDCOVER["barren"], np.where(fv < 0.4, LANDCOVER["sparse_vegetation"], np.where(fc >= 0.6, LANDCOVER["thick_vegetation"], LANDCOVER["moderate_vegetation"])))
    r = np.where(fb >= 0.2, LANDCOVER["populated"], r)
    shares = {k: round(float((r == v).mean()), 3) for k, v in LANDCOVER.items()}
    return r, shares


def run_landslide_screening(job_dir: Path, result: dict[str, Any], *, lithology: float | None = None, structure: float | None = None, hydrogeology: float | None = None,
                            rain_mm_hourly: list[float] | None = None, fetch_rainfall: bool = False, scars: dict[str, str] | None = None) -> dict[str, Any]:
    tp = job_dir / "terrain.tif"
    if not tp.exists():
        raise FileNotFoundError("Landslide screening requires the terrain layer (terrain.tif)")
    if not result.get("metric"):
        raise ValueError("Landslide screening needs metric terrain (calibration tier T or A)")
    user = {"lithology": lithology, "structure": structure, "hydrogeology": hydrogeology}
    for k, v in user.items():
        lo, hi, _ = USER_FACTORS[k]
        if v is not None and not (math.isfinite(v) and lo <= v <= hi):
            raise ValueError(f"{k} rating must be between {lo:g} and {hi:g} (IS 14496-2)")

    with rasterio.open(tp) as ds:
        px = abs(ds.transform.a)
        f = max(1, int(round(WORK_M / px)))
        shape_w = (ds.height // f, ds.width // f)
        T = ds.read(1, out_shape=shape_w, masked=True, resampling=rasterio.enums.Resampling.average).astype(np.float64).filled(np.nan)
        tr_w = ds.transform * ds.transform.scale(ds.width / shape_w[1], ds.height / shape_w[0])
        crs = ds.crs
    valid = np.isfinite(T)
    Tf = np.where(valid, T, np.nanmedian(T))
    cell = abs(tr_w.a)
    n_facet = max(1, int(round(FACET_M / cell)))
    slope, _ = slope_layers(Tf, tr_w)
    ok = np.isfinite(slope) & valid  # normalised mean: missing edge cells are not read as flat ground
    slope_f = ndimage.uniform_filter(np.where(ok, slope, 0.0), n_facet) / np.maximum(ndimage.uniform_filter(ok.astype(float), n_facet), 1e-6)
    rr = max(1, int(round(RELIEF_RADIUS_M / cell)))
    relief = ndimage.maximum_filter(Tf, size=2 * rr + 1) - ndimage.minimum_filter(Tf, size=2 * rr + 1)
    r_slope, r_relief = _rate(slope_f, SLOPE_RATING), _rate(relief, RELIEF_RATING)
    r_lc, lc_shares = _landcover(job_dir, shape_w, f, n_facet)

    trig = None
    rain_src = None
    if rain_mm_hourly:
        trig, rain_src = rainfall_trigger(rain_mm_hourly), "user"
    elif fetch_rainfall and crs is not None:
        from pyproj import Transformer

        cx, cy = tr_w * (shape_w[1] / 2, shape_w[0] / 2)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)
        try:
            rain = fetch_rain(lat, lon)
            trig, rain_src = rainfall_trigger(rain["mm"]), rain["source"]
            if trig["worst"]:
                trig["worst"]["endTime"] = rain["time"][trig["worst"]["endIndex"]]
            trig["window"] = [rain["time"][0], rain["time"][-1]]
            trig["totalMm"] = round(float(np.nansum(np.asarray(rain["mm"], float))), 1)
        except Exception as e:  # noqa: BLE001 - offline: no trigger, stated
            trig, rain_src = {"error": f"rainfall not available ({e.__class__.__name__}); enter it manually"}, None
    if user["hydrogeology"] is None and trig and trig.get("worst"):
        # rainfall above the threshold -> saturated slopes; stated as an assumption in the output
        user["hydrogeology"] = 1.0 if trig["exceeded"] else (0.5 if trig["worst"]["ratio"] >= 0.5 else 0.2)
        hydro_from = "rainfall trigger (flowing if exceeded, wet if >= half of it, else damp)"
    else:
        hydro_from = "user" if user["hydrogeology"] is not None else None

    known = r_slope + r_relief + r_lc
    mid = {k: (v if v is not None else (USER_FACTORS[k][0] + USER_FACTORS[k][1]) / 2) for k, v in user.items()}
    lo = {k: (v if v is not None else USER_FACTORS[k][0]) for k, v in user.items()}
    hi = {k: (v if v is not None else USER_FACTORS[k][1]) for k, v in user.items()}
    tehd, tehd_lo, tehd_hi = known + sum(mid.values()), known + sum(lo.values()), known + sum(hi.values())
    cls, cls_lo, cls_hi = classify(tehd), classify(tehd_lo), classify(tehd_hi)
    cls[~valid] = 255

    scar_out = None
    if scars:
        scar_out = _scars(job_dir, shape_w, tr_w, crs, slope, scars)

    g = Grid(width=shape_w[1], height=shape_w[0], transform=tr_w, crs=crs.to_string() if crs else None, dtype="uint8", nodata=255, units="class", metric=False, vertical_reference=None, tier=result.get("calibration_tier") or "T")
    write_raster(job_dir / "landslide_hazard.tif", cls, g, tags={"KIND": "landslide_hazard_class", "CLASSES": "1 very low, 2 low, 3 moderate, 4 high, 5 very high (IS 14496-2 TEHD)", "METHOD": METHOD}, dtype="uint8")
    _preview(job_dir / "landslide_preview.png", cls, scar_out["mask"] if scar_out else None, job_dir)
    a = cell * cell
    n_valid = int(valid.sum())
    unknown = [k for k, v in user.items() if v is None]
    summary = {
        "scenario": "landslide", "method": METHOD, "standard": "BIS IS 14496 (Part 2): 1998, Landslide Hazard Evaluation Factor",
        "classAreaPct": {name: round(100 * float((cls == k).sum()) / n_valid, 1) for _, k, name in CLASSES},
        "classAreaPctIfGeologyBest": {name: round(100 * float(((cls_lo == k) & valid).sum()) / n_valid, 1) for _, k, name in CLASSES} if unknown else None,
        "classAreaPctIfGeologyWorst": {name: round(100 * float(((cls_hi == k) & valid).sum()) / n_valid, 1) for _, k, name in CLASSES} if unknown else None,
        "highOrWorseAreaM2": round(float(((cls >= 4) & (cls < 255)).sum()) * a, 0),
        "factors": {"slopeMorphometry": "terrain layer, facet mean slope", "relativeRelief": f"terrain max - min within {RELIEF_RADIUS_M:g} m", "landCover": lc_shares,
                    "user": {k: v for k, v in user.items()}, "unknown": unknown, "hydrogeologyFrom": hydro_from, "choices": {k: v[2] for k, v in USER_FACTORS.items()}},
        "meanSlopeDeg": round(float(np.nanmean(slope_f[valid])), 1), "reliefM": round(float(np.nanpercentile(relief[valid], 50)), 0),
        "rainfall": {"source": rain_src, **trig} if trig else None,
        "scars": {k: v for k, v in scar_out.items() if k != "mask"} if scar_out else None,
        "buildingsHighOrWorse": _buildings_in(job_dir, cls, f),
        "rasterResult": "landslide_hazard.tif", "previewResult": "landslide_preview.png",
        "warnings": [
            "Landslide susceptibility screening (IS 14496-2 hazard zonation factors), not a slope-stability analysis or a forecast.",
            "Not yet validated against a landslide inventory: treat the classes as relative, and confirm on the ground.",
            f"Slope and relief come from the {WORK_M:g} m terrain layer (the DEM's 30 m detail); road cuts and retaining walls are not seen.",
        ] + ([f"Unknown factors set to mid-range: {', '.join(unknown)}. Enter them from GSI Bhukosh geology maps or a site visit; the best/worst-case class shares show their effect."] if unknown else []),
    }
    (job_dir / "disaster_landslide.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _buildings_in(job_dir: Path, cls: np.ndarray, f: int) -> int:
    p = job_dir / "buildings.json"
    if not p.exists():
        return 0
    n = 0
    for b in json.loads(p.read_text(encoding="utf-8")).get("buildings", []):
        bb = b.get("pixel_bbox")
        if bb:
            r, c = int((bb[1] + bb[3]) / 2 // f), int((bb[0] + bb[2]) / 2 // f)
            if 0 <= r < cls.shape[0] and 0 <= c < cls.shape[1] and 4 <= cls[r, c] < 255:
                n += 1
    return n


def _scars(job_dir: Path, shape_w, tr_w, crs, slope: np.ndarray, dates: dict[str, str]) -> dict[str, Any]:
    """New bare ground on steep slopes between two Sentinel-2 date windows ({'before': 'a/b', 'after': 'c/d'})."""
    from rasterio.warp import transform_bounds

    from core.geo import sentinel2 as S2  # the same reader and rule as the flood-scar validation

    b = rasterio.transform.array_bounds(shape_w[0], shape_w[1], tr_w)
    bbox = list(transform_bounds(crs, "EPSG:4326", *b))  # array_bounds -> (west, south, east, north)
    tok = S2.token()
    got = {}
    for tag in ("before", "after"):
        for it in S2.search(bbox, dates[tag])[:6]:
            nd, mw, ok = S2.indices(it, tok, crs, tr_w, shape_w)
            if ok.mean() > 0.9:
                got[tag] = (it["id"], nd, ok)
                break
        if tag not in got:
            return {"error": f"no cloud-free Sentinel-2 image in the {tag} window {dates[tag]}", "mask": None}
    (ib, nb, okb), (ia, na, oka) = got["before"], got["after"]
    m = okb & oka & (na < 0.25) & ((nb - na) > 0.15) & (slope >= SCAR_MIN_SLOPE_DEG)
    m = ndimage.binary_opening(m)
    lab, n = ndimage.label(m)
    a = abs(tr_w.a * tr_w.e)
    return {"before": ib, "after": ia, "rule": f"NDVI drop > 0.15 to < 0.25 on slopes >= {SCAR_MIN_SLOPE_DEG:g} deg (Sentinel-2 L2A, Copernicus)",
            "count": int(n), "areaM2": round(float(m.sum()) * a, 0), "mask": m}


def _preview(path: Path, cls: np.ndarray, scar: np.ndarray | None, job_dir: Path) -> None:
    with rasterio.open(job_dir / "terrain.tif") as ds:
        full = (ds.width, ds.height)
    rgba = np.zeros(cls.shape + (4,), np.uint8)
    for k, c in PREVIEW.items():
        rgba[cls == k] = c
    if scar is not None:
        edge = scar & ~ndimage.binary_erosion(scar)
        rgba[scar] = (217, 70, 239, 150)
        rgba[edge] = (217, 70, 239, 255)
    from core.geo.atomic import write_atomic

    img = Image.fromarray(rgba, "RGBA").resize(full, Image.NEAREST)
    write_atomic(path, lambda t: img.save(t, format="PNG"))
