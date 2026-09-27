"""Before / after 3D change screening from two single-image height results (disaster screening, NOT damage grading).

Inputs: two READY Mode B jobs covering the same place at two dates ("before" defines the grid). Method:

1. Bring the "after" nDSM and image onto the "before" grid (rasterio reproject: any CRS / GSD / extent).
2. Co-register: phase correlation of the two images (upsampled to 0.1 px) removes the residual shift between two
   orthorectified acquisitions; it is applied only when it is small and the correlation is unambiguous.
3. dH = nDSM_after - nDSM_before (height above ground: independent of the shared DEM).
4. Thresholds come from THIS pair, not from assumptions: most of any scene is unchanged, so the robust spread
   (NMAD) of dH over all pixels is the noise of the difference (model error at both dates, residual misregistration,
   different view angles). A pixel changes when |dH| > max(3 NMAD, 2.5 m) even after allowing TOL_M of horizontal
   displacement (a roof seen from two view angles moves by h * |tan a1 - tan a2|); a building changes when the
   difference of its footprint medians exceeds max(3 NMAD_buildings, 2.5 m), where NMAD_buildings is the spread of
   that same difference over all buildings of the scene.
5. Buildings are the "before" LoD-1 footprints. "Major height loss" = significant loss AND at least half of the
   before height gone (consistent with collapse or removal); it is a screening flag to verify on the imagery.

Known false alarms: very different view angles (building lean), seasonal vegetation (tree crowns), shadows, and
construction between the dates. Every threshold and noise level is reported with the result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

MIN_CHANGE_M = 2.5  # never flag less than the model's own ground-level RMSE (model card)
K_PIXEL = 3.0
K_BUILDING = 3.0
MAJOR_LOSS_FRACTION = 0.5
MIN_BLOB_M2 = 8.0  # pixel-change blobs smaller than this are speckle
MAX_SHIFT_M = 15.0
TOL_M = 3.0  # pixel changes tolerate this much horizontal displacement (building lean between two view angles)
CLASSES = ("MAJOR_HEIGHT_LOSS", "HEIGHT_LOSS", "HEIGHT_GAIN", "NO_SIGNIFICANT_CHANGE", "NOT_COMPARABLE")
PIXEL_CLASS = {"loss": -1, "none": 0, "gain": 1, "nodata": -128}


def nmad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def _grid(result: dict[str, Any]) -> tuple[Affine, str, int, int]:
    g = result["grid"]
    return Affine.from_gdal(*g["transform"]), g["crs"], int(g["width"]), int(g["height"])


def overlap_fraction(pre: dict[str, Any], post: dict[str, Any]) -> float:
    """Share of the 'before' footprint covered by the 'after' footprint (bounding boxes, in the 'before' CRS)."""
    try:
        t1, c1, w1, h1 = _grid(pre)
        t2, c2, w2, h2 = _grid(post)
    except (KeyError, TypeError):
        return 0.0
    a = (t1.c, t1.f + h1 * t1.e, t1.c + w1 * t1.a, t1.f)
    b0 = (t2.c, t2.f + h2 * t2.e, t2.c + w2 * t2.a, t2.f)
    b = transform_bounds(c2, c1, *b0) if c1 != c2 else b0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return float(ix * iy / area) if area > 0 else 0.0


def _read_float(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1, masked=True).astype(np.float64).filled(np.nan)


def _rgb(job_dir: Path, result: dict[str, Any]) -> np.ndarray:
    """The ingested image on the job grid (input_preview.png is written at full grid resolution)."""
    im = np.asarray(Image.open(job_dir / result["artifacts"]["input_preview"]).convert("RGB"))
    _t, _c, w, h = _grid(result)
    if im.shape[:2] != (h, w):
        im = np.asarray(Image.fromarray(im).resize((w, h), Image.BILINEAR))
    return im


def _warp(src: np.ndarray, src_res: dict[str, Any], dst_res: dict[str, Any], resampling: Resampling) -> np.ndarray:
    ts, cs, _ws, _hs = _grid(src_res)
    td, cd, wd, hd = _grid(dst_res)
    out = np.full((hd, wd), np.nan, np.float64)
    reproject(src.astype(np.float64), out, src_transform=ts, src_crs=cs, dst_transform=td, dst_crs=cd, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def register(ref: np.ndarray, mov: np.ndarray, gsd_m: float) -> dict[str, Any]:
    """Sub-pixel translation that moves `mov` onto `ref` (phase correlation on a central window, high-passed)."""
    from skimage.registration import phase_cross_correlation

    h, w = ref.shape
    s = min(h, w, 1024)
    r0, c0 = (h - s) // 2, (w - s) // 2
    a, b = ref[r0:r0 + s, c0:c0 + s], mov[r0:r0 + s, c0:c0 + s]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.mean() < 0.5:
        return {"applied": False, "reason": "too little overlap in the central window", "dy_px": 0.0, "dx_px": 0.0}
    fill = lambda x: np.where(np.isfinite(x), x, np.nanmedian(x))  # noqa: E731
    hp = lambda x: fill(x) - ndimage.gaussian_filter(fill(x), 8)  # noqa: E731 - illumination / season independent
    ha, hb = hp(a), hp(b)
    shift, _err, _ = phase_cross_correlation(ha, hb, upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    mag_m = math.hypot(dy, dx) * gsd_m
    # quality: normalised cross-correlation of the high-passed images without / with the shift (inner window only)
    m = max(4, int(math.ceil(max(abs(dy), abs(dx)))) + 2)
    ncc = lambda u, v: float(np.corrcoef(u[m:-m, m:-m].ravel(), v[m:-m, m:-m].ravel())[0, 1])  # noqa: E731
    ncc0, ncc1 = ncc(ha, hb), ncc(ha, ndimage.shift(hb, (dy, dx), order=1, mode="nearest"))
    applied = mag_m <= MAX_SHIFT_M and ncc1 >= ncc0 and ncc1 >= 0.2
    why = "applied" if applied else (f"not applied: shift {mag_m:.1f} m > {MAX_SHIFT_M} m" if mag_m > MAX_SHIFT_M else f"not applied: correlation {ncc1:.2f} too weak (images too different to align)")
    return {"applied": bool(applied), "dy_px": round(dy, 2), "dx_px": round(dx, 2), "shift_m": round(mag_m, 2), "ncc_before": round(ncc0, 3), "ncc_after": round(ncc1, 3), "reason": why}


def _despeckle(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def building_changes(labels: np.ndarray, pre: np.ndarray, post: np.ndarray, px_class: np.ndarray, b_data: dict[str, Any], px_area: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    slices = ndimage.find_objects(labels)
    recs = []
    for b in b_data.get("buildings", []):
        bid = int(b["id"])
        sl = slices[bid - 1] if 0 < bid <= len(slices) else None
        if sl is None:
            continue
        fp = labels[sl] == bid
        a, c = pre[sl][fp], post[sl][fp]
        ok = np.isfinite(a) & np.isfinite(c)
        cover = float(ok.mean()) if fp.any() else 0.0
        rec: dict[str, Any] = {"id": bid, "coords": b.get("coords"), "area_m2": b.get("area_m2"), "cover": round(cover, 3)}
        if ok.sum() >= 4:
            hb, ha = float(np.median(a[ok])), float(np.median(c[ok]))
            rec.update(height_before_m=round(hb, 2), height_after_m=round(ha, 2), dh_m=round(ha - hb, 2),
                       volume_change_m3=round(float(np.sum(c[ok] - a[ok])) * px_area, 1),
                       loss_pixel_fraction=round(float((px_class[sl][fp] == PIXEL_CLASS["loss"]).mean()), 3))
        recs.append(rec)
    good = np.array([r["dh_m"] for r in recs if "dh_m" in r and r["cover"] >= 0.8])
    s_b = nmad(good) if good.size >= 10 else float("nan")
    thr = max(K_BUILDING * s_b, MIN_CHANGE_M) if math.isfinite(s_b) else MIN_CHANGE_M * 2
    for r in recs:
        if "dh_m" not in r or r["cover"] < 0.5:
            r["class"] = "NOT_COMPARABLE"
            continue
        dh, hb = r["dh_m"], r["height_before_m"]
        r["z"] = round(dh / s_b, 2) if math.isfinite(s_b) and s_b > 0 else None
        if dh <= -thr and r["height_after_m"] <= (1 - MAJOR_LOSS_FRACTION) * hb:
            r["class"] = "MAJOR_HEIGHT_LOSS"
        elif dh <= -thr:
            r["class"] = "HEIGHT_LOSS"
        elif dh >= thr:
            r["class"] = "HEIGHT_GAIN"
        else:
            r["class"] = "NO_SIGNIFICANT_CHANGE"
    return recs, {"nmad_buildings_m": round(s_b, 3) if math.isfinite(s_b) else None, "threshold_buildings_m": round(thr, 2), "n_buildings_for_noise": int(good.size)}


LOSS_RGB, GAIN_RGB = (225, 0, 145), (0, 175, 235)  # magenta / cyan: distinct from red-tile roofs, vegetation, concrete


def _preview(dh: np.ndarray, px_class: np.ndarray, path: Path, *, loss_outline: np.ndarray | None = None, gain_outline: np.ndarray | None = None, vmax: float = 15.0) -> None:
    """Transparent overlay: magenta = height loss, cyan = gain (significant pixels, alpha grows with |dH|), plus solid
    outlines of the buildings flagged at building level."""
    h, w = dh.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    t = np.clip(np.abs(np.nan_to_num(dh)) / vmax, 0, 1)
    # gains are drawn fainter: after a disaster most of them are disagreements between the two height maps (see caveats)
    for mask, rgb, a0, a1 in ((px_class == PIXEL_CLASS["loss"], LOSS_RGB, 110, 110), (px_class == PIXEL_CLASS["gain"], GAIN_RGB, 60, 70)):
        rgba[mask, :3] = rgb
        rgba[mask, 3] = (a0 + a1 * t[mask]).astype(np.uint8)
    for mask, rgb in ((gain_outline, GAIN_RGB), (loss_outline, LOSS_RGB)):
        if mask is not None:
            rgba[mask, :3] = rgb
            rgba[mask, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def _outline(labels: np.ndarray, ids: list[int], width_px: int) -> np.ndarray | None:
    if not ids:
        return None
    m = np.isin(labels, np.asarray(ids, dtype=labels.dtype))
    return ndimage.binary_dilation(m, iterations=width_px) & ~ndimage.binary_erosion(m, iterations=1)


def _write_tif(path: Path, arr: np.ndarray, t: Affine, crs: str, dtype: str, nodata: float, tags: dict[str, str]) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype=dtype, crs=crs, transform=t, nodata=nodata, compress="deflate", tiled=True) as ds:
        ds.write(arr.astype(dtype), 1)
        ds.update_tags(**tags)


def screen_change(pre_dir: Path, pre_res: dict[str, Any], post_dir: Path, post_res: dict[str, Any], *, prefix: str, labels: dict[str, str] | None = None) -> dict[str, Any]:
    pre_dir, post_dir = Path(pre_dir), Path(post_dir)
    for r in (pre_res, post_res):
        if r.get("mode") != "B" or not (r.get("grid") or {}).get("crs"):
            raise ValueError("change screening needs two georeferenced (Mode B) results")
    if not (pre_dir / "ndsm.tif").exists() or not (post_dir / "ndsm.tif").exists():
        raise ValueError("both results need an nDSM (height above ground) layer")
    ov = overlap_fraction(pre_res, post_res)
    if ov < 0.1:
        raise ValueError(f"the two results barely overlap ({100 * ov:.0f} % of the 'before' area): choose images of the same place")
    t, crs, w, h = _grid(pre_res)
    gsd = float(pre_res.get("gsd_m") or abs(t.a))
    px_area = abs(t.a * t.e - t.b * t.d)
    pre = _read_float(pre_dir / "ndsm.tif")
    post = _warp(_read_float(post_dir / "ndsm.tif"), post_res, pre_res, Resampling.bilinear)
    g_pre = _rgb(pre_dir, pre_res).astype(np.float64).mean(axis=2)
    rgb_post = np.stack([_warp(c.astype(np.float64), post_res, pre_res, Resampling.bilinear) for c in np.moveaxis(_rgb(post_dir, post_res), 2, 0)], axis=-1)
    reg = register(g_pre, rgb_post.mean(axis=2), gsd)
    if reg["applied"]:
        sh = (reg["dy_px"], reg["dx_px"])
        post = ndimage.shift(post, sh, order=1, mode="constant", cval=np.nan)
        rgb_post = np.stack([ndimage.shift(rgb_post[..., i], sh, order=1, mode="constant", cval=np.nan) for i in range(3)], axis=-1)
    valid = np.isfinite(pre) & np.isfinite(post)
    if valid.mean() < 0.05:
        raise ValueError("no overlapping valid heights after alignment")
    dh = np.where(valid, post - pre, np.nan)
    s_px = nmad(dh[valid])
    thr_px = max(K_PIXEL * s_px, MIN_CHANGE_M)
    min_px = max(1, int(round(MIN_BLOB_M2 / px_area)))
    # displacement-tolerant differencing: a roof seen from two view angles moves by up to h * |tan(a1) - tan(a2)|;
    # a pixel lost height only if NOTHING within TOL_M in the after heights is as tall (and vice versa for gains)
    rr = max(1, int(round(TOL_M / gsd)))
    yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    disk = (yy * yy + xx * xx) <= rr * rr
    post_max = ndimage.maximum_filter(np.where(np.isfinite(post), post, -1e9), footprint=disk)
    pre_max = ndimage.maximum_filter(np.where(np.isfinite(pre), pre, -1e9), footprint=disk)
    loss = _despeckle(valid & (post_max - pre < -thr_px) & (pre >= MIN_CHANGE_M), min_px)
    gain = _despeckle(valid & (post - pre_max > thr_px) & (post >= MIN_CHANGE_M), min_px)
    px_class = np.full((h, w), PIXEL_CLASS["none"], np.int8)
    px_class[loss] = PIXEL_CLASS["loss"]
    px_class[gain] = PIXEL_CLASS["gain"]
    px_class[~valid] = PIXEL_CLASS["nodata"]
    # buildings: the "before" footprints
    recs: list[dict[str, Any]] = []
    bstats: dict[str, Any] = {}
    outlines: dict[str, np.ndarray | None] = {}
    bj = (pre_res.get("artifacts") or {}).get("buildings_json")
    if bj and (pre_dir / bj).exists():
        from core.disaster.flood import _footprint_sets  # one footprint definition for every product

        b_data = json.loads((pre_dir / bj).read_text(encoding="utf-8"))
        lab, method = _footprint_sets(pre_dir, b_data, (h, w))
        if lab is not None:
            recs, bstats = building_changes(lab, pre, post, px_class, b_data, px_area)
            bstats["footprints"] = method
            ow = max(2, int(round(1.5 / gsd)))
            outlines["loss_outline"] = _outline(lab, [r["id"] for r in recs if r["class"] in ("MAJOR_HEIGHT_LOSS", "HEIGHT_LOSS")], ow)
            outlines["gain_outline"] = _outline(lab, [r["id"] for r in recs if r["class"] == "HEIGHT_GAIN"], ow)
    counts = {c: sum(r["class"] == c for r in recs) for c in CLASSES}
    names = {"dh_tif": f"{prefix}_dh.tif", "class_tif": f"{prefix}_class.tif", "overlay_png": f"{prefix}_overlay.png", "after_png": f"{prefix}_after.png", "json": f"{prefix}.json", "buildings_geojson": f"{prefix}_buildings.geojson", "buildings_csv": f"{prefix}_buildings.csv"}
    tags = {"QUANTITY": "nDSM(after) - nDSM(before), metres", "BEFORE": labels.get("before", "") if labels else "", "AFTER": labels.get("after", "") if labels else "", "METHOD": "DepthWizard change screening v1"}
    _write_tif(pre_dir / names["dh_tif"], np.where(valid, dh, -9999.0), t, crs, "float32", -9999.0, tags)
    _write_tif(pre_dir / names["class_tif"], px_class, t, crs, "int8", PIXEL_CLASS["nodata"], {**tags, "CLASSES": "-1 height loss, 0 no significant change, 1 height gain, -128 no data"})
    _preview(dh, px_class, pre_dir / names["overlay_png"], **outlines)
    Image.fromarray(np.nan_to_num(rgb_post).clip(0, 255).astype(np.uint8), "RGB").save(pre_dir / names["after_png"], optimize=True)
    lost = [r for r in recs if r["class"] in ("MAJOR_HEIGHT_LOSS", "HEIGHT_LOSS")]
    summary = {
        "method": "DepthWizard change screening v1 (nDSM difference, pair-calibrated robust thresholds)",
        "labels": labels or {}, "overlap_fraction": round(ov, 3), "valid_fraction": round(float(valid.mean()), 3),
        "compared_area_km2": round(float(valid.sum()) * px_area / 1e6, 3), "registration": reg,
        "noise": {"nmad_pixels_m": round(s_px, 3), "threshold_pixels_m": round(thr_px, 2), **bstats},
        "pixels": {"loss_area_m2": round(float(loss.sum()) * px_area), "gain_area_m2": round(float(gain.sum()) * px_area),
                   "loss_volume_m3": round(float(np.nansum(np.where(loss, dh, 0.0))) * px_area), "gain_volume_m3": round(float(np.nansum(np.where(gain, dh, 0.0))) * px_area)},
        "buildings": {"n": len(recs), "counts": counts, "lost_footprint_m2": round(sum(r.get("area_m2") or 0 for r in lost)),
                      "volume_change_lost_m3": round(sum(r.get("volume_change_m3") or 0 for r in lost))},
        "rules": {"pixel": f"|dH| > max({K_PIXEL} x NMAD, {MIN_CHANGE_M} m), tolerant to {TOL_M} m of horizontal displacement (lean between view angles); loss needs >= {MIN_CHANGE_M} m before, gain >= {MIN_CHANGE_M} m after; blobs < {MIN_BLOB_M2} m2 removed",
                  "building": f"median dH over the 'before' footprint vs max({K_BUILDING} x NMAD over all buildings, {MIN_CHANGE_M} m); MAJOR = loss AND after <= {MAJOR_LOSS_FRACTION:.0%} of before"},
        "caveats": ["Screening, not a damage assessment: verify flagged buildings on the imagery.",
                    "Different view angles (building lean), seasonal vegetation, snow, shadows and construction between the dates cause false alarms; tree canopies mistaken for buildings are the most common one.",
                    "A height GAIN on an existing building is usually a disagreement between the two single-image height maps (different sun or view angle), not real growth; genuine gains are new structures, tents or debris piles.",
                    "Single-image heights have metre-level errors; changes smaller than the reported thresholds are not detectable, and missed collapses (recall) are not measured by this screening."],
        "artifacts": names,
    }
    order = {c: i for i, c in enumerate(CLASSES)}
    recs.sort(key=lambda r: (order[r["class"]], r.get("dh_m") if r.get("dh_m") is not None else 0.0))
    (pre_dir / names["json"]).write_text(json.dumps({"summary": summary, "buildings": recs}, default=float), encoding="utf-8")
    _building_exports(pre_dir, names, recs, pre_res)
    return {"summary": summary, "buildings": recs}


def _building_exports(pre_dir: Path, names: dict[str, str], recs: list[dict[str, Any]], pre_res: dict[str, Any]) -> None:
    from pyproj import Transformer

    bj = json.loads((pre_dir / pre_res["artifacts"]["buildings_json"]).read_text(encoding="utf-8")) if (pre_res.get("artifacts") or {}).get("buildings_json") else None
    if not bj:
        return
    from core.terrain.buildings import _to_crs

    to_ll = Transformer.from_crs(pre_res["grid"]["crs"], "EPSG:4326", always_xy=True)
    feats, rows = [], ["id,class,height_before_m,height_after_m,dh_m,z,area_m2,volume_change_m3,loss_pixel_fraction,cover,lon,lat"]
    for r in recs:
        if not r.get("coords"):
            continue
        ring = [to_ll.transform(x, y) for x, y in _to_crs(bj, r["coords"])]
        lon = sum(p[0] for p in ring[:-1]) / max(1, len(ring) - 1)
        lat = sum(p[1] for p in ring[:-1]) / max(1, len(ring) - 1)
        props = {k: v for k, v in r.items() if k != "coords"}
        feats.append({"type": "Feature", "id": r["id"], "properties": props, "geometry": {"type": "Polygon", "coordinates": [[[round(x, 7), round(y, 7)] for x, y in ring]]}})
        rows.append(",".join("" if r.get(k) is None else str(r.get(k)) for k in ("id", "class", "height_before_m", "height_after_m", "dh_m", "z", "area_m2", "volume_change_m3", "loss_pixel_fraction", "cover")) + f",{lon:.6f},{lat:.6f}")
    (pre_dir / names["buildings_geojson"]).write_text(json.dumps({"type": "FeatureCollection", "name": "depthwizard_building_change", "features": feats}), encoding="utf-8")
    (pre_dir / names["buildings_csv"]).write_text("\n".join(rows) + "\n", encoding="utf-8")
