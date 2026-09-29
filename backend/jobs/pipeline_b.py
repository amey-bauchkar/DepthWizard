"""Mode B pipeline stages: GeoTIFF -> relative structure -> DEM-anchored fusion -> DSM + derivatives.

Outputs (all on the job grid, GeoTIFFs with provenance tags):
  relative.tif       zero-shot relative structure [0,1] (tier R), whole-image prediction
  object_rel.tif     relative object layer (>=0, unitless) from the morphological ground filter (ground mask source)
  dsm.tif            DEM + model detail below one DEM posting (tier T), anchor-refined gain/offset (tier A);
                     DEM only when no detail could be calibrated (flagged NO_OBJECT_SCALE)
  terrain.tif        ground estimate, metres, declared vertical CRS   <- NOT a certified DTM
  ndsm.tif           dsm - terrain (height above the terrain layer, metres)
  slope.tif/aspect.tif, flags.tif, previews, calib_report.json, result.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from backend.config.settings import Settings
from backend.logging_setup import JobLogger
from core.calib.anchors import robust_offset, split_holdout
from core.calib.dem import grid_bounds, load_dem_on_grid, select_dem
from core.calib.fusion import TiledPrediction, default_detail_scales, fit_anchor_gain, fuse_detail, highpass, plan_upsample, stitch_tiles, tiled_relative
from core.calib.terrain import object_layer_from_relative, terrain_layer
from core.calib.tier import decide
from core.dsm.derive import build_flags, elevation_preview, flags_preview, hillshade_preview, slope_layers, slope_preview
from core.dsm.rdsm import make_rdsm, write_preview
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.geo.vertical import VerticalTransformUnsafeError, register_bundled_grids, transform_heights_xy
from core.ingest.geotiff import GeoIngestResult, ingest_geotiff
from core.inference.predictor import METRIC_QUANTITY, BasePredictor
from core.terrain.heightfield import build_heightfield, write_heightfield, write_texture

METHOD_VERSION = "tlcsm-1.0 (tiled inference + DEM-preserving detail fusion)"


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def stage_ingest_geotiff(job_dir: Path, input_path: Path, settings: Settings, log: JobLogger) -> GeoIngestResult:
    t0 = time.perf_counter()
    res = ingest_geotiff(input_path, job_dir, max_dim=settings.ingest.max_image_dim)
    _dump(job_dir / "meta.json", res.meta.to_dict())
    Image.fromarray(res.rgb, "RGB").save(job_dir / "input_preview.png")
    log.event("UPLOADED", "ingested GeoTIFF", crs=res.grid.crs, gsd_m=res.meta.gsd_m, reprojected=res.meta.reprojected, width=res.grid.width, height=res.grid.height, ms=round((time.perf_counter() - t0) * 1000, 1))
    return res


def stage_tiled_inference(job_dir: Path, rgb: np.ndarray, gsd_m: float | None, predictor: BasePredictor, settings: Settings, log: JobLogger, progress_cb: Any | None = None) -> TiledPrediction:
    """Overlapping-tile inference at the configured inference GSD (Mode B) or at native resolution (Mode A)."""
    f = settings.fusion
    h, w = rgb.shape[:2]
    desired = (gsd_m / f.inference_gsd_m) if gsd_m else 1.0
    desired = float(min(max(desired, f.min_upsample), f.max_upsample))
    up = plan_upsample(h, w, desired, tile=f.tile_px, overlap=f.overlap, max_tiles=f.max_tiles)
    tp = tiled_relative(
        lambda x: predictor.predict(x).relative_depth,
        rgb,
        upsample=up,
        tile=f.tile_px,
        overlap=f.overlap,
        inference_gsd_m=(gsd_m / up) if gsd_m else None,
        progress_cb=progress_cb,
    )
    tp.quantity = predictor.card.output_quantity
    val = (predictor.card.extra or {}).get("validation") or {}
    ft = val.get("finetuned") if isinstance(val, dict) else None
    tp.model_info = {"name": predictor.card.name, "version": predictor.card.version, "object_rmse_m": (ft or {}).get("RMSE_objects_ge2.5m"), "object_me_m": (ft or {}).get("ME_objects_ge2.5m"), "ground_rmse_m": (ft or {}).get("RMSE_ground_lt2.5m"), "test_regions": val.get("test_regions") if isinstance(val, dict) else None}
    _dump(job_dir / "tiled_inference.json", tp.summary())
    log.event("INFERENCE", "tiled inference complete", **tp.summary())
    return tp


def load_anchors(path: Path) -> list[dict[str, Any]]:
    """CSV with header: id,x,y,z,type[,sigma]. Comment lines: '# crs=EPSG:xxxx' (horizontal CRS of x,y; default = job CRS)
    and '# vcrs=<EGM2008|EGM96|ellipsoidal|EPSG:code>' (vertical reference of z; default = output vertical CRS)."""
    rows: list[dict[str, Any]] = []
    crs = None
    vcrs = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "vcrs=" in line:
                vcrs = line.split("vcrs=")[1].strip()
            elif "crs=" in line:
                crs = line.split("crs=")[1].strip()
            continue
        if line.lower().startswith("id,"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append({"id": parts[0], "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]), "type": parts[4].lower(), "sigma": float(parts[5]) if len(parts) > 5 and parts[5] else 1.0, "crs": crs, "vcrs": vcrs})
        except ValueError:
            continue  # malformed row: skipped (counted as n_total - n_in_grid)
    return rows


def _anchor_calibration(job_dir: Path, grid: Grid, terrain: np.ndarray, dem: np.ndarray, detail: np.ndarray | None, anchors: list[dict[str, Any]], settings: Settings, log: JobLogger, out_vcrs: str = "EGM2008") -> dict[str, Any]:
    """Tier A: robust terrain offset from ground anchors (median, blunder flags, hold-out) and a robust
    gain/offset refinement of the fused DSM from all anchors (core.calib.fusion.fit_anchor_gain)."""
    from pyproj import Transformer

    c = settings.calib
    pts = []
    for a in anchors:
        x, y = a["x"], a["y"]
        if a.get("crs") and a["crs"] != grid.crs:
            t = Transformer.from_crs(a["crs"], grid.crs, always_xy=True)
            x, y = t.transform(x, y)
        col, row = grid.crs_to_pixel(x, y)
        ci, ri = int(round(col)), int(round(row))
        if 0 <= ci < grid.width and 0 <= ri < grid.height and np.isfinite(terrain[ri, ci]):
            d = float(detail[ri, ci]) if detail is not None and np.isfinite(detail[ri, ci]) else 0.0
            pts.append({**a, "x_grid": x, "y_grid": y, "col": ci, "row": ri, "terrain": float(terrain[ri, ci]), "dem": float(dem[ri, ci]), "detail": d})
    # anchor heights -> output vertical reference (C-1 guarded transformer; a failure here is a hard error, never silent)
    vcrs_in = anchors[0].get("vcrs") if anchors else None
    datum_note = "anchor heights taken as already in the output vertical reference"
    if pts and vcrs_in and vcrs_in != out_vcrs:
        xs = np.array([p["x_grid"] for p in pts]); ys = np.array([p["y_grid"] for p in pts]); zs = np.array([p["z"] for p in pts])
        zz, info = transform_heights_xy(xs, ys, zs, grid.crs, vcrs_in, out_vcrs)
        for p_, z_ in zip(pts, zz):
            p_["z_input"] = p_["z"]; p_["z"] = float(z_)
        datum_note = f"anchor heights converted {vcrs_in} -> {out_vcrs} ({info.get('pipeline', '')})"
    ground = [p for p in pts if p["type"].startswith("g")]
    objects = [p for p in pts if not p["type"].startswith("g")]
    rep: dict[str, Any] = {"n_total": len(anchors), "n_in_grid": len(pts), "n_ground": len(ground), "n_object": len(objects), "accepted": False, "offset_m": 0.0, "datum": datum_note, "input_vcrs": vcrs_in or out_vcrs}
    if len(ground) >= c.anchor_min_n:
        fit_idx, hold_idx = split_holdout(len(ground), c.anchor_holdout_fraction, seed=0)
        res_all = np.array([p["z"] - p["terrain"] for p in ground])
        fit = robust_offset(res_all[fit_idx], min_n=min(c.anchor_min_n, len(fit_idx)))
        sigma = float(np.median([p["sigma"] for p in ground]))
        tol = max(c.anchor_accept_k * max(sigma, 0.1), c.anchor_accept_nmad_m)
        accepted = fit.nmad <= tol
        rep.update({"offset_m": fit.offset, "fit_nmad_m": fit.nmad, "accept_tolerance_m": tol, "blunders": [ground[fit_idx[i]]["id"] for i in fit.blunder_idx], "n_used": int(len(fit_idx)), "anchor_sigma_m": sigma, "accepted": bool(accepted), "reason": "accepted" if accepted else f"offset fit NMAD {fit.nmad:.2f} m > tolerance {tol:.2f} m"})
        if len(hold_idx):
            hr = res_all[hold_idx] - fit.offset
            rep["holdout"] = {"n": int(len(hold_idx)), "ME": float(hr.mean()), "NMAD": float(1.4826 * np.median(np.abs(hr - np.median(hr)))), "RMSE": float(np.sqrt((hr**2).mean()))}
            rep["holdout_nmad"] = rep["holdout"]["NMAD"]
        rep["used_ids"] = [ground[i]["id"] for i in fit_idx]
        rep["holdout_ids"] = [ground[i]["id"] for i in hold_idx]
    else:
        rep["reason"] = f"need >= {c.anchor_min_n} ground anchors inside the grid (have {len(ground)})"
    # DSM refinement from ALL anchors (ground + object): gain on the fused detail, significant offset only
    if detail is not None and pts:
        g = fit_anchor_gain(np.array([p["z"] for p in pts]), np.array([p["dem"] for p in pts]), np.array([p["detail"] for p in pts]), min_n=c.anchor_min_n, k_range=(0.0, settings.fusion.anchor_gain_max))
        rep["gain_fit"] = g
    _dump(job_dir / "anchors_used.json", {"used_ids": rep.get("used_ids", []), "holdout_ids": rep.get("holdout_ids", []), "points": pts})
    log.event("CALIBRATION", "anchor calibration", **{k: v for k, v in rep.items() if k not in ("blunders", "used_ids", "holdout_ids", "gain_fit")}, gain_fit_accepted=(rep.get("gain_fit") or {}).get("accepted"))
    return rep


# Typical 1-sigma point errors MEASURED by DepthWizard itself (never assumed): DEM-based layers vs NASA ICESat-2
# checkpoints over the six Sikkim scenes (docs/validation_india.md = CartoDEM, docs/validation_india_copernicus.md).
MEASURED_DEM_RMSE = {
    "cartodem": {"dem_m": 7.98, "terrain_m": 7.28, "dsm_m": 10.75, "source": "measured: DepthWizard (model v2, g = 1, f = 0.75) with CartoDEM vs ICESat-2, 6 Sikkim scenes (scripts/select_dem_trust.py)"},
    "bundled": {"dem_m": 10.51, "terrain_m": 8.54, "dsm_m": 11.10, "source": "measured: DepthWizard with Copernicus GLO-30 vs ICESat-2, 6 Sikkim scenes (docs/validation_india_copernicus.md)"},
}
# These are steep-Himalaya figures and are shown for every scene (the conservative choice). On the held-out Swiss
# test tiles (docs/validation_results.md) the same method measured terrain 3.2-5.3 m and DSM 5.5-6.4 m RMSE.
for _v in MEASURED_DEM_RMSE.values():
    _v["source"] += "; steep Himalayan terrain - on the Swiss LiDAR test tiles terrain was 3.2-5.3 m, DSM 5.5-6.4 m"


def point_uncertainty(tier: str, metric: bool, model_info: dict[str, Any], dem_name: str | None) -> dict[str, Any]:
    """Per-layer typical error (RMSE, metres) shown next to every point reading; a layer is absent when no measurement backs it."""
    out: dict[str, Any] = {}
    if metric and model_info.get("object_rmse_m"):
        out["ndsm"] = {"ground_m": model_info.get("ground_rmse_m"), "object_m": model_info["object_rmse_m"], "object_threshold_m": 2.5,
                       "source": f"model card {model_info.get('name')}@{model_info.get('version')}: RMSE on held-out Swiss LiDAR regions"}
    d = MEASURED_DEM_RMSE.get(dem_name or "")
    if d and tier in ("T", "A"):
        out["terrain"] = {"value_m": d["terrain_m"], "source": d["source"]}
        out["dsm"] = {"value_m": d["dsm_m"], "source": d["source"]}
        out["dem"] = {"value_m": d["dem_m"], "source": d["source"] + " (input DEM vs ICESat-2 ground)"}
    return out


def _ground_from_surface(dsm: np.ndarray, gsd_m: float, window_m: float) -> np.ndarray:
    """Morphological ground estimate of a surface: grey opening with a window wider than buildings, smoothed."""
    w = max(3, int(round(window_m / max(gsd_m, 1e-6))) | 1)
    fill = float(np.nanmedian(dsm)) if np.isfinite(dsm).any() else 0.0
    z = np.where(np.isfinite(dsm), dsm, fill)
    return ndimage.gaussian_filter(ndimage.grey_opening(z, size=(w, w)), w / 4.0).astype(np.float32)


def stage_calibrate_and_compose(job_dir: Path, ing: GeoIngestResult, rel_depth: np.ndarray, prov: dict[str, Any], settings: Settings, log: JobLogger, *, tiled: TiledPrediction | None = None, user_dem: Path | None = None, user_dem_vcrs: str = "EGM2008", anchors_path: Path | None = None, footprints_path: Path | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    c = settings.calib
    grid = ing.grid
    gsd = grid.pixel_size or (1.0, 1.0)
    gsd_m = float((gsd[0] * gsd[1]) ** 0.5)
    valid = np.isfinite(rel_depth) & (ing.valid_mask if ing.valid_mask is not None else np.ones_like(rel_depth, bool))
    # --- relative structure (tier R) ---
    rel, _g, rstats = make_rdsm(rel_depth, valid, method=settings.rdsm.method, percentiles=tuple(settings.rdsm.percentiles), nodata=settings.rdsm.nodata, orientation=settings.rdsm.orientation)  # type: ignore[arg-type]
    rel_f = np.where(rel == settings.rdsm.nodata, np.nan, rel).astype(np.float32)
    rg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "relative", False, None, "R")
    write_raster(job_dir / "relative.tif", rel_f, rg, {"MODEL": f"{prov['model_name']}@{prov['model_version']}", "OUTPUT_QUANTITY": "relative_height_normalised"})
    write_preview(job_dir / "rdsm_preview.png", rel, settings.rdsm.nodata, mode="ramp")
    write_preview(job_dir / "depth_preview.png", rel, settings.rdsm.nodata, mode="gray")
    # --- relative object layer (ground mask for the terrain layer) ---
    obj = object_layer_from_relative(rel_f, valid, gsd_m=gsd_m, ground_window_m=c.ground_window_m)
    write_raster(job_dir / "object_rel.tif", obj.object_rel, rg, {"OUTPUT_QUANTITY": "relative_object_layer", "METHOD": obj.params["method"]})
    # --- DEM discovery + datum ---
    register_bundled_grids(settings.proj_grids_dir)
    out_vcrs = c.output_vertical_crs
    bounds = grid_bounds(grid)
    dem_src, dem_select = select_dem(bounds, settings.dem_dir, grid, user_dem=user_dem, user_dem_vcrs=user_dem_vcrs, priority=tuple(c.dem_priority), cartodem_vcrs=c.cartodem_vertical_crs)
    report: dict[str, Any] = {"method_version": METHOD_VERSION, "output_vertical_crs": out_vcrs, "gsd_m": gsd_m, "object_layer": obj.params, "relative_stats": rstats.to_dict(), "dem": None, "terrain": None, "fusion": None, "anchors": None, "consistency": None, "tiled_inference": tiled.summary() if tiled else None}
    datum_ok = True
    dem = dem_valid = None
    terrain_res = None
    fusion = None
    anchor_rep = None
    if dem_src is not None:
        try:
            dem, dem_valid, dem_prov = load_dem_on_grid(dem_src, grid, out_vcrs)
            report["dem"] = {**dem_prov, "selection": dem_select}
        except VerticalTransformUnsafeError as e:
            datum_ok = False
            report["dem"] = {"source": dem_src.to_dict(), "error": e.to_dict()}
            log.event("CALIBRATION", "vertical datum transform refused", level=30, code=e.code, detail=e.detail)
    detail_t = None
    # fine-tuned metric nDSM model (tier H heights): tiles are already metres above ground -> stitched directly
    metric = tiled is not None and tiled.quantity == METRIC_QUANTITY
    ndsm_model = None
    ground_mask = obj.ground_mask
    if metric:
        ndsm_model = np.clip(stitch_tiles(tiled), 0.0, None).astype(np.float32)
        ndsm_model[~valid] = np.nan
        ground_mask = np.isfinite(ndsm_model) & (ndsm_model < settings.fusion.metric_ground_max_m)
        report["metric_model"] = {"tiles": tiled.summary(), "ndsm_p50_p90_p99_m": [float(v) for v in np.nanpercentile(ndsm_model, [50, 90, 99])], "ground_fraction": float(ground_mask[valid].mean()) if valid.any() else 0.0}
    if dem is not None and datum_ok:
        v2 = valid & dem_valid
        terrain_res = terrain_layer(dem, dem_valid, ground_mask & valid, gsd_m=gsd_m, dem_posting_m=c.dem_posting_m, sigma_cells=c.sigma_cells, w_min=c.w_min)
        report["terrain"] = {**terrain_res.params, **terrain_res.stats}
        # consistency: mean offset between the terrain layer and the DEM where ground
        d = (terrain_res.terrain - dem)[v2 & ground_mask]
        report["consistency"] = {"ME": float(np.nanmean(d)) if d.size else 0.0, "NMAD": float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))) if d.size else 0.0, "n": int(d.size)}
        # --- DEM-preserving detail fusion (tier T object layer) ---
        if metric:
            d1, _d2 = default_detail_scales(c.dem_posting_m, gsd_m)
            detail_t = np.where(dem_valid & valid, highpass(np.nan_to_num(ndsm_model), d1), 0.0).astype(np.float32)
            n = len(tiled.tiles)
            report["fusion"] = {"accepted": True, "metric_model": True, "reason": "accepted (metric model: scale known)", "tiles_with_detail": n, "n_tiles": n, "gain_median": 1.0, "composition": settings.fusion.metric_composition, "method": "fine-tuned metric nDSM model; DSM = DEM + high-pass(nDSM) below one DEM posting (gain 1, zero-mean detail at the DEM scale); terrain = DSM - nDSM" if settings.fusion.metric_composition == "highpass" else "fine-tuned metric nDSM model; DSM = ground-weighted DEM terrain (ground = model nDSM < threshold) + nDSM"}
        elif tiled is not None and settings.fusion.enabled:
            d1, d2 = default_detail_scales(c.dem_posting_m, gsd_m)
            fusion = fuse_detail(np.where(dem_valid, dem, np.nan), tiled, detail_scale_px=d1, band_high_px=d2, max_gain=settings.fusion.max_tile_gain)
            accepted = fusion.stats["tiles_with_detail"] > 0 and fusion.stats["detail_std"] > 1e-3
            report["fusion"] = {"accepted": bool(accepted), "reason": "accepted" if accepted else "no tile showed a positive model/DEM agreement in the DEM-resolved band", "params": fusion.params, **fusion.stats, "per_tile_gain": [round(g, 3) for g in fusion.gains], "method": "per-tile least squares of model vs DEM band-pass (1-4 DEM postings); gain applied to model detail below one posting; zero-mean detail at the DEM scale"}
            if accepted:
                detail_t = np.where(dem_valid & valid, fusion.detail, 0.0).astype(np.float32)
        if anchors_path is not None and anchors_path.exists():
            anchor_rep = _anchor_calibration(job_dir, grid, terrain_res.terrain, dem, detail_t, load_anchors(anchors_path), settings, log, out_vcrs=out_vcrs)
            report["anchors"] = anchor_rep
    tier = decide(georeferenced=True, dem_found=dem_src is not None, terrain_stats=terrain_res.stats if terrain_res else None, scale_fit=report["fusion"], anchor_fit=anchor_rep, out_vcrs=out_vcrs, datum_ok=datum_ok, consistency=report["consistency"], cfg={"datum_sanity_m": c.datum_sanity_m}, is_metric=metric)
    report["tier"] = tier.to_dict()
    # --- compose ---
    artifacts: dict[str, str] = {"input_preview": "input_preview.png", "depth_preview": "depth_preview.png", "rdsm_preview": "rdsm_preview.png", "relative_tif": "relative.tif", "object_rel_tif": "object_rel.tif", "meta": "meta.json", "prep": "prep.json", "prediction": "prediction.json", "calib_report": "calib_report.json"}
    if tiled is not None:
        artifacts["tiled_inference"] = "tiled_inference.json"
    layers: dict[str, Any] = {"relative": {"units": "relative", "tier": "R", "preview": "rdsm_preview.png"}}
    _tex_path, tex_size = write_texture(job_dir / "texture.jpg", ing.rgb, max_dim=settings.terrain.max_texture_dim)
    artifacts["texture"] = "texture.jpg"

    scale = None
    scale_source = None
    if terrain_res is not None and tier.tier in ("T", "A"):
        gain_fit = (anchor_rep or {}).get("gain_fit") or {}
        k, c_off = 1.0, 0.0
        terr = terrain_res.terrain.astype(np.float32)
        if anchor_rep and anchor_rep.get("accepted"):
            terr = terr + float(anchor_rep["offset_m"])
        if detail_t is not None:
            scale_source = "fine-tuned metric nDSM model" if metric else "tiled_dem_band_fusion (unvalidated)"
            if gain_fit.get("accepted"):
                k, c_off = float(gain_fit["detail_gain"]), float(gain_fit["offset_m"])
                scale_source += " + anchor gain"
            scale = k if metric else (float(fusion.stats["gain_median"]) * k if fusion else None)
        if metric and settings.fusion.metric_composition == "terrain_plus_ndsm":
            # terrain from the DEM with the model's ground mask, objects from the model: DSM = terrain + nDSM
            terrain = terr.copy()
            dsm = (terrain + np.nan_to_num(ndsm_model)).astype(np.float32)
        else:
            fz = settings.fusion
            g_d = fz.metric_detail_gain if metric else 1.0
            dsm = (dem + c_off + (g_d * k * detail_t if detail_t is not None else 0.0)).astype(np.float32)
            if metric:
                # terrain = DEM - f * low-pass(nDSM) (+ anchor offset): f = share of the smoothed object height the DEM
                # itself contains (1 = a full surface model); never above the DSM
                lp = np.nan_to_num(ndsm_model) - (np.nan_to_num(detail_t) if detail_t is not None else 0.0)
                terrain = np.fmin(dem + c_off - fz.metric_dem_object_fraction * k * lp, dsm).astype(np.float32)
            else:
                # DEM-based ground layer (anchor offset in tier A) combined with the morphological ground of the
                # fused DSM; never above the surface itself
                ground_from_dsm = _ground_from_surface(dsm, gsd_m, c.ground_window_m)
                terrain = np.fmin(np.fmin(terr, ground_from_dsm), dsm).astype(np.float32)
        dsm[~(valid & dem_valid)] = np.nan
        terrain[~(valid & dem_valid)] = np.nan
        ndsm = np.clip(dsm - terrain, 0.0, None).astype(np.float32)
        tg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, out_vcrs, tier.tier)
        write_raster(job_dir / "dem.tif", np.where(dem_valid, dem, np.nan), tg, {"OUTPUT_QUANTITY": "input_dem_on_job_grid", "DEM_SOURCE": dem_src.product if dem_src else "", "NOTE": "bilinear resample + datum transform only; baseline for validation"})
        write_raster(job_dir / "terrain.tif", terrain, tg, {"OUTPUT_QUANTITY": "terrain_ground_estimate", "NOT_A_DTM": "true", "DEM_SOURCE": dem_src.product if dem_src else "", "METHOD": "min(ground-weighted DEM normalized convolution [+ anchor offset], morphological ground of the DSM, DSM)"})
        write_raster(job_dir / "dsm.tif", dsm, tg, {"OUTPUT_QUANTITY": "dsm_surface_elevation" if detail_t is not None else "dem_only_no_object_detail", "OBJECT_SCALE_SOURCE": scale_source or "none", "DEM_SOURCE": dem_src.product if dem_src else "", "METHOD": METHOD_VERSION, "MODEL": f"{prov['model_name']}@{prov['model_version']}", "MODEL_SHA256": prov.get("model_sha256") or "n/a"})
        dlo, dhi = float(np.nanpercentile(dsm, 1)), float(np.nanpercentile(dsm, 99))
        leg = elevation_preview(job_dir / "dsm_preview.png", dsm, lo=dlo, hi=dhi)
        elevation_preview(job_dir / "terrain_preview.png", terrain, lo=dlo, hi=dhi)
        hillshade_preview(job_dir / "hillshade_preview.png", dsm, gsd[0], gsd[1])
        slope, aspect = slope_layers(dsm, grid.transform)  # surface (DSM) slope; terrain slope is derived in core.disaster
        sg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, tier.tier)
        write_raster(job_dir / "slope.tif", slope, sg, {"OUTPUT_QUANTITY": "surface_slope_degrees", "SURFACE": "dsm (includes buildings and canopy)", "METHOD": "Horn 3x3 in pixel space, mapped to CRS axes by J^-T of the affine; edges flagged"})
        write_raster(job_dir / "aspect.tif", aspect, sg, {"OUTPUT_QUANTITY": "surface_aspect_degrees", "CONVENTION": "downslope (facing) azimuth, clockwise from grid north, [0,360); NaN where flat"})
        sleg = slope_preview(job_dir / "slope_preview.png", slope)
        low_support = terrain_res.support < 0.25
        flags = build_flags(dsm.shape, valid=np.isfinite(dsm), raw_fallback=terrain_res.raw_fallback, dem_void=~dem_valid, no_object_scale=detail_t is None, low_support=low_support)
        write_raster(job_dir / "flags.tif", flags, Grid(grid.width, grid.height, grid.transform, grid.crs, "uint16", 0, "relative", False, None, tier.tier), {"BITS": "BORDER=1,TERRAIN_RAW_DEM=2,DEM_VOID=4,NODATA=8,NO_OBJECT_SCALE=16,LOW_SUPPORT=32"}, dtype="uint16")
        flags_preview(job_dir / "flags_preview.png", flags)
        write_raster(job_dir / "ndsm.tif", ndsm, Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, tier.tier), {"OUTPUT_QUANTITY": "height_above_terrain_layer", "SCALE_SOURCE": scale_source or "none (DEM relief only)"})
        nleg = elevation_preview(job_dir / "ndsm_preview.png", ndsm, lo=0.0, hi=max(1.0, float(np.nanpercentile(ndsm, 99))))
        artifacts.update({"dem_tif": "dem.tif", "terrain_tif": "terrain.tif", "dsm_tif": "dsm.tif", "ndsm_tif": "ndsm.tif", "slope_tif": "slope.tif", "aspect_tif": "aspect.tif", "flags_tif": "flags.tif", "dsm_preview": "dsm_preview.png", "terrain_preview": "terrain_preview.png", "ndsm_preview": "ndsm_preview.png", "hillshade_preview": "hillshade_preview.png", "slope_preview": "slope_preview.png", "flags_preview": "flags_preview.png"})
        dsm_label = "ABSOLUTE · DEM + calibrated model detail" if detail_t is not None else "ABSOLUTE · DEM relief only (no model detail calibrated)"
        layers.update({
            "dsm": {"units": "metres", "vertical_crs": out_vcrs, "tier": tier.tier, "preview": "dsm_preview.png", "legend": leg, "label": dsm_label},
            "terrain": {"units": "metres", "vertical_crs": out_vcrs, "tier": tier.tier, "preview": "terrain_preview.png", "legend": leg, "label": "ABSOLUTE · ground estimate (not a certified DTM)"},
            "ndsm": {"units": "metres", "tier": tier.tier, "preview": "ndsm_preview.png", "legend": nleg, "label": f"METRIC · height above terrain layer · detail: {scale_source or 'none'}"},
            "slope": {"units": "degrees", "preview": "slope_preview.png", "legend": sleg},
            "hillshade": {"preview": "hillshade_preview.png"},
            "flags": {"preview": "flags_preview.png"},
        })
        # heightfields for the viewer (display copies; the GeoTIFFs above are never modified)
        for name, arr in (("dsm", dsm), ("terrain", terrain), ("ndsm", ndsm)):
            hf, _hv, hmeta = build_heightfield(
                np.where(np.isfinite(arr), arr, -9999.0),
                -9999.0,
                max_mesh_dim=settings.terrain.max_mesh_dim,
                units="metres",
                metric=True,
                tier=tier.tier,
                vertical_reference=out_vcrs if name != "ndsm" else "height_above_terrain",
                texture_size=tex_size,
                viewer_smooth_sigma=0.0 if name != "terrain" else 1.0,
                guide_rgb=ing.rgb if name in ("dsm", "ndsm") and detail_t is not None else None,
                edge_sharpen=name in ("dsm", "ndsm") and detail_t is not None,
            )
            if name == "ndsm":  # edge sharpening can overshoot below ground in the display copy; heights are >= 0
                hf = np.where(np.isfinite(hf), np.maximum(hf, 0.0), hf).astype(np.float32)
                hmeta.min = max(hmeta.min, 0.0)
            write_heightfield(job_dir / f"heightfield_{name}.f32", hf)
            _dump(job_dir / f"heightfield_{name}.json", hmeta.to_dict())
            layers[name]["heightfield"] = f"heightfield_{name}.f32"
            layers[name]["heightfield_meta"] = f"heightfield_{name}.json"
        # LoD-1 building blocks (visualisation; heights sampled from ndsm, bases from terrain)
        if detail_t is not None:
            try:
                from core.terrain import footprints as fpm
                from core.terrain.lod1 import extract_lod1_buildings, save_lod1_buildings

                # outlines: an uploaded footprint GeoJSON > bundled open footprints covering the scene > detection
                fp, fp_info = None, None
                if footprints_path is not None and footprints_path.exists():
                    rings, rcrs = fpm.load_geojson(footprints_path)
                    fp_info = {"source": "uploaded footprints", "file": footprints_path.name, "licence": "as supplied by the user"}
                else:
                    hit = fpm.discover(bounds, settings.footprints_dir)
                    if hit is not None:
                        rings, rcrs = fpm.load_geojson(hit[0])
                        fp_info = {"source": hit[1]["source"], "file": hit[1]["file"], "licence": hit[1]["licence"], "coverage": hit[1]["coverage"]}
                if fp_info is not None:
                    fp = fpm.to_labels(rings, rcrs, grid.transform, grid.crs, (grid.height, grid.width))
                    reg = fpm.coregister(fp[0], ing.rgb, ndsm, gsd_m)  # outlines from other imagery can sit metres off the roofs
                    if reg["applied"]:
                        fp = fpm.to_labels(rings, rcrs, fpm.shifted(grid.transform, reg), grid.crs, (grid.height, grid.width))
                    fp_info["coregistration"] = reg
                lod1_data = extract_lod1_buildings(ndsm, terrain, rgb=ing.rgb, gsd_m=gsd_m, transform=grid.transform, return_labels=True, footprints=fp,
                                                   object_filter=os.environ.get("DW_LOD1_OBJECT_FILTER", "1") != "0")  # "0": raw detector (building study)
                lod1_data["footprints"] = fp_info or {"source": "detected from the image (RGB + nDSM rules, object filter)",
                                                      "note": "approximate: tree crowns and rock can still be counted as buildings, and small houses missed; see docs/building_detection_validation.md"}
                labels = lod1_data.pop("_labels")
                # exact footprint pixel sets (value = building id): hazard screening intersects these, not polygons
                write_raster(job_dir / "building_labels.tif", labels, Grid(grid.width, grid.height, grid.transform, grid.crs, "int32", 0, "relative", False, None, tier.tier), {"OUTPUT_QUANTITY": "building_id", "NOTE": "0 = no building; ids match buildings.json"}, dtype="int32")
                artifacts["building_labels_tif"] = "building_labels.tif"
                mi = tiled.model_info if tiled is not None else {}
                lod1_data["height_error"] = ({"typical_m": mi.get("object_rmse_m"), "bias_m": mi.get("object_me_m"), "source": f"{mi.get('name')}@{mi.get('version')} model card: RMSE for objects >= 2.5 m on held-out LiDAR regions {mi.get('test_regions')}", "calibrated": True}
                                             if metric and mi.get("object_rmse_m") else {"typical_m": None, "source": "zero-shot detail scaled against the DEM: building-height error not calibrated", "calibrated": False})
                lod1_data["vertical_reference"] = out_vcrs
                lod1_data["grid"] = {"crs": grid.crs, "transform": list(grid.transform.to_gdal()), "width": grid.width, "height": grid.height}
                save_lod1_buildings(job_dir / "buildings.json", lod1_data)
                artifacts["buildings_json"] = "buildings.json"
                log.event("LOD1", f"extracted {lod1_data['count']} LoD-1 building blocks ({lod1_data.get('segmentation_method', 'unknown')})")
            except Exception as e:  # noqa: BLE001 - visual extra; never fails the job
                log.event("WARN", f"LoD-1 extraction skipped: {type(e).__name__}: {e}", level=30)
    if metric and tier.tier == "H":
        # tier H: fine-tuned model heights above ground in metres; no DEM -> no absolute elevation, no DSM
        scale_source = "fine-tuned metric nDSM model"
        ndsm = np.where(valid, np.nan_to_num(ndsm_model), np.nan).astype(np.float32)
        hg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, "H")
        write_raster(job_dir / "ndsm.tif", ndsm, hg, {"OUTPUT_QUANTITY": "height_above_ground", "SCALE_SOURCE": scale_source, "MODEL": tiled.quantity})
        nleg = elevation_preview(job_dir / "ndsm_preview.png", ndsm, lo=0.0, hi=max(1.0, float(np.nanpercentile(ndsm, 99))))
        slope, aspect = slope_layers(ndsm, grid.transform)
        write_raster(job_dir / "slope.tif", slope, hg, {"OUTPUT_QUANTITY": "slope_degrees_of_ndsm"})
        sleg = slope_preview(job_dir / "slope_preview.png", slope)
        hillshade_preview(job_dir / "hillshade_preview.png", ndsm, gsd[0], gsd[1])
        artifacts.update({"ndsm_tif": "ndsm.tif", "ndsm_preview": "ndsm_preview.png", "slope_tif": "slope.tif", "slope_preview": "slope_preview.png", "hillshade_preview": "hillshade_preview.png"})
        layers.update({"ndsm": {"units": "metres", "tier": "H", "preview": "ndsm_preview.png", "legend": nleg, "label": "METRIC · height above ground (fine-tuned model) · no absolute elevation"}, "slope": {"units": "degrees", "preview": "slope_preview.png", "legend": sleg}, "hillshade": {"preview": "hillshade_preview.png"}})
        hf, _hv, hmeta = build_heightfield(np.where(np.isfinite(ndsm), ndsm, -9999.0), -9999.0, max_mesh_dim=settings.terrain.max_mesh_dim, units="metres", metric=True, tier="H", vertical_reference="height_above_ground", texture_size=tex_size)
        write_heightfield(job_dir / "heightfield_ndsm.f32", hf)
        _dump(job_dir / "heightfield_ndsm.json", hmeta.to_dict())
        layers["ndsm"]["heightfield"] = "heightfield_ndsm.f32"
        layers["ndsm"]["heightfield_meta"] = "heightfield_ndsm.json"
        scale = 1.0
    # relative heightfield always available (tier R view)
    _hf, _hv, hmeta = build_heightfield(rel, settings.rdsm.nodata, max_mesh_dim=settings.terrain.max_mesh_dim, texture_size=tex_size)
    write_heightfield(job_dir / "heightfield_relative.f32", _hf)
    _dump(job_dir / "heightfield_relative.json", hmeta.to_dict())
    layers["relative"].update({"heightfield": "heightfield_relative.f32", "heightfield_meta": "heightfield_relative.json"})
    _dump(job_dir / "calib_report.json", report)
    result = {
        "mode": "B",
        "mode_label": ing.meta.mode_label,
        "metric": bool(tier.metric_vertical),
        "metric_horizontal": True,
        "absolute_elevation": bool(tier.absolute_elevation),
        "calibration_tier": tier.tier,
        "quality": tier.quality,
        "quality_triggers": tier.triggers,
        "flags": tier.flags,
        "notes": tier.notes,
        "units": "metres" if tier.tier in ("T", "A", "H") else "relative",
        "vertical_reference": out_vcrs if tier.tier in ("T", "A") else None,
        "object_scale_source": scale_source,
        "object_scale_m_per_unit": scale,
        "method_version": METHOD_VERSION,
        "grid": grid.to_dict(),
        "gsd_m": gsd_m,
        "dem": report["dem"]["source"] if report.get("dem") and "source" in report["dem"] else None,
        "layers": layers,
        "artifacts": artifacts,
        "heightfield": json.loads((job_dir / ("heightfield_dsm.json" if "dsm" in layers and "heightfield" in layers["dsm"] else "heightfield_ndsm.json" if "ndsm" in layers and "heightfield" in layers["ndsm"] else "heightfield_relative.json")).read_text(encoding="utf-8")),
        "timings_ms": {"calibration_ms": (time.perf_counter() - t0) * 1000.0},
    }
    result["uncertainty"] = point_uncertainty(tier.tier, metric, tiled.model_info if tiled is not None else {}, dem_src.name if dem_src is not None else None)
    _dump(job_dir / "result.json", result)
    log.event("CALIBRATION", "calibration complete", tier=tier.tier, quality=tier.quality, scale_source=scale_source, ms=round((time.perf_counter() - t0) * 1000, 1))
    return result
