"""Mode B pipeline stages: GeoTIFF -> relative structure -> two-layer calibration -> DSM + derivatives.

Outputs (all on the job grid, COG-style GeoTIFFs with tags):
  relative.tif       zero-shot relative structure [0,1] (tier R) — also used as the Mode-A style preview
  object_rel.tif     relative object layer (>=0, unitless) from the morphological ground filter
  terrain.tif        terrain layer, metres, declared vertical CRS (tier T)   <- NOT a certified DTM
  ndsm.tif           object layer in metres (only when an object scale exists; scale source recorded)
  dsm.tif            terrain + ndsm (tier T/A) or terrain only (flagged NO_OBJECT_SCALE)
  slope.tif/aspect.tif, flags.tif, previews, calib_report.json, result.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.config.settings import Settings
from backend.logging_setup import JobLogger
from core.calib.anchors import MIN_ANCHORS, ransac_scale, robust_offset, split_holdout
from core.geo.vertical import transform_heights_xy
from core.calib.dem import discover_dem, grid_bounds, load_dem_on_grid
from core.calib.terrain import compose_dsm, fit_object_scale_to_dem_residual, object_layer_from_relative, terrain_layer
from core.calib.tier import decide
from core.dsm.derive import build_flags, elevation_preview, flags_preview, hillshade_preview, slope_layers, slope_preview
from core.dsm.rdsm import make_rdsm, write_preview
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.geo.vertical import VerticalTransformUnsafeError, register_bundled_grids
from core.ingest.geotiff import GeoIngestResult, ingest_geotiff
from core.inference.predictor import BasePredictor
from core.terrain.heightfield import build_heightfield, write_heightfield, write_texture


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def stage_ingest_geotiff(job_dir: Path, input_path: Path, settings: Settings, log: JobLogger) -> GeoIngestResult:
    t0 = time.perf_counter()
    res = ingest_geotiff(input_path, job_dir, max_dim=settings.ingest.max_image_dim)
    _dump(job_dir / "meta.json", res.meta.to_dict())
    Image.fromarray(res.rgb, "RGB").save(job_dir / "input_preview.png")
    log.event("UPLOADED", "ingested GeoTIFF", crs=res.grid.crs, gsd_m=res.meta.gsd_m, reprojected=res.meta.reprojected, width=res.grid.width, height=res.grid.height, ms=round((time.perf_counter() - t0) * 1000, 1))
    return res


def load_anchors(path: Path) -> list[dict[str, Any]]:
    """CSV with header: id,x,y,z,type[,sigma]. Comment lines: '# crs=EPSG:xxxx' (horizontal CRS of x,y; default = job CRS)
    and '# vcrs=<EGM2008|EGM96|ellipsoidal|EPSG:code>' (vertical reference of z; default = output vertical CRS)."""
    rows: list[dict[str, Any]] = []
    crs = None
    vcrs = None
    for line in path.read_text(encoding="utf-8").splitlines():
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
        rows.append({"id": parts[0], "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]), "type": parts[4].lower(), "sigma": float(parts[5]) if len(parts) > 5 and parts[5] else 1.0, "crs": crs, "vcrs": vcrs})
    return rows


def _anchor_calibration(job_dir: Path, grid: Grid, terrain: np.ndarray, object_rel: np.ndarray, anchors: list[dict[str, Any]], settings: Settings, log: JobLogger, out_vcrs: str = "EGM2008") -> dict[str, Any]:
    """Tier A: robust offset from ground anchors, RANSAC object scale from object anchors, hold-out reporting."""
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
            pts.append({**a, "x_grid": x, "y_grid": y, "col": ci, "row": ri, "terrain": float(terrain[ri, ci]), "obj_rel": float(object_rel[ri, ci]) if np.isfinite(object_rel[ri, ci]) else 0.0})
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
    rep: dict[str, Any] = {"n_total": len(anchors), "n_in_grid": len(pts), "n_ground": len(ground), "n_object": len(objects), "accepted": False, "offset_m": 0.0, "scale_m_per_unit": None, "datum": datum_note, "input_vcrs": vcrs_in or out_vcrs}
    if len(ground) < c.anchor_min_n:
        rep["reason"] = f"need >= {c.anchor_min_n} ground anchors inside the grid (have {len(ground)})"
        return rep
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
    if len(objects) >= MIN_ANCHORS:
        x = np.array([p["obj_rel"] for p in objects])
        y = np.array([p["z"] - p["terrain"] - fit.offset for p in objects])
        try:
            sf = ransac_scale(x, y, residual_threshold=max(1.5, 2 * sigma))
            d = sf.to_dict()
            inl_frac = sf.n_inliers / max(sf.n_total, 1)
            p99 = float(np.nanquantile(object_rel[np.isfinite(object_rel)], 0.99)) if np.isfinite(object_rel).any() else 0.0
            implied = sf.scale * p99
            r_xy = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
            ok = (
                sf.scale > 0
                and inl_frac >= c.anchor_scale_min_inlier_fraction
                and sf.n_inliers >= min(c.anchor_min_n, len(objects))
                and 2.0 <= implied <= 80.0
                and (r_xy >= 0.2 or inl_frac >= 0.7)
            )
            d.update({
                "inlier_fraction": inl_frac,
                "implied_p99_object_height_m": implied,
                "r_object_layer_vs_anchor_height": r_xy,
                "accepted": bool(ok),
                "reason": "accepted" if ok else f"rejected: (r={r_xy:.2f}, scale={sf.scale:.2f}, inliers {sf.n_inliers}/{sf.n_total}, implied p99 height {implied:.1f} m)",
            })
            rep["scale_m_per_unit"] = sf.scale if ok else None
            rep["scale_fit"] = d
        except ValueError as e:
            rep["scale_fit"] = {"error": str(e), "accepted": False}
    used = [ground[i]["id"] for i in fit_idx]
    held = [ground[i]["id"] for i in hold_idx]
    _dump(job_dir / "anchors_used.json", {"used_ids": used, "holdout_ids": held, "points": pts})
    log.event("CALIBRATION", "anchor calibration", **{k: v for k, v in rep.items() if k not in ("blunders",)})
    return rep


def stage_calibrate_and_compose(job_dir: Path, ing: GeoIngestResult, rel_depth: np.ndarray, prov: dict[str, Any], settings: Settings, log: JobLogger, *, user_dem: Path | None = None, user_dem_vcrs: str = "EGM2008", anchors_path: Path | None = None) -> dict[str, Any]:
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
    # --- object layer (relative or direct metric) ---
    is_metric_head = prov.get("output_quantity") == "metric_ndsm_metres"
    obj = object_layer_from_relative(rel_f, valid, gsd_m=gsd_m, ground_window_m=c.ground_window_m)
    if is_metric_head:
        # Model outputs direct height in metres (h >= 0.0)
        obj.object_rel = np.maximum(0.0, np.where(valid, rel_depth, 0.0)).astype(np.float32)
    write_raster(job_dir / "object_rel.tif", obj.object_rel, rg, {"OUTPUT_QUANTITY": "metric_object_layer" if is_metric_head else "relative_object_layer", "METHOD": "direct_metric_head" if is_metric_head else obj.params["method"]})
    # --- DEM discovery + datum ---
    register_bundled_grids(settings.proj_grids_dir)
    out_vcrs = c.output_vertical_crs
    bounds = grid_bounds(grid)
    dem_src = discover_dem(bounds, settings.dem_dir, user_dem=user_dem, user_dem_vcrs=user_dem_vcrs)
    report: dict[str, Any] = {"method_version": "tlcsm-0.3 (metric head)" if is_metric_head else "tlcsm-0.2 (baseline object layer)", "output_vertical_crs": out_vcrs, "gsd_m": gsd_m, "object_layer": obj.params, "relative_stats": rstats.to_dict(), "dem": None, "terrain": None, "scale_fit": None, "anchors": None, "consistency": None}
    datum_ok = True
    dem = dem_valid = None
    terrain_res = None
    scale_fit = None
    anchor_rep = None
    if dem_src is not None:
        try:
            dem, dem_valid, dem_prov = load_dem_on_grid(dem_src, grid, out_vcrs)
            report["dem"] = dem_prov
        except VerticalTransformUnsafeError as e:
            datum_ok = False
            report["dem"] = {"source": dem_src.to_dict(), "error": e.to_dict()}
            log.event("CALIBRATION", "vertical datum transform refused", level=30, code=e.code, detail=e.detail)
    if dem is not None and datum_ok:
        v2 = valid & dem_valid
        terrain_res = terrain_layer(dem, dem_valid, obj.ground_mask & valid, gsd_m=gsd_m, dem_posting_m=c.dem_posting_m, sigma_cells=c.sigma_cells, w_min=c.w_min)
        report["terrain"] = {**terrain_res.params, **terrain_res.stats}
        # consistency: mean offset between the terrain layer and the DEM where ground
        d = (terrain_res.terrain - dem)[v2 & obj.ground_mask]
        report["consistency"] = {"ME": float(np.nanmean(d)) if d.size else 0.0, "NMAD": float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))) if d.size else 0.0, "n": int(d.size)}
        if anchors_path is not None and anchors_path.exists():
            anchor_rep = _anchor_calibration(job_dir, grid, terrain_res.terrain, obj.object_rel, load_anchors(anchors_path), settings, log, out_vcrs=out_vcrs)
            report["anchors"] = anchor_rep
        if not is_metric_head and not (anchor_rep and anchor_rep.get("accepted") and anchor_rep.get("scale_m_per_unit")):
            scale_fit = fit_object_scale_to_dem_residual(dem, terrain_res.terrain, obj.object_rel, v2, gsd_m=gsd_m, dem_posting_m=c.dem_posting_m, min_object_fraction=c.min_object_fraction, min_implied_p99_m=c.min_implied_p99_m, max_implied_p99_m=c.max_implied_p99_m, min_r=c.min_r)
            report["scale_fit"] = scale_fit.to_dict()
    tier = decide(georeferenced=True, dem_found=dem_src is not None, terrain_stats=terrain_res.stats if terrain_res else None, scale_fit=report["scale_fit"], anchor_fit=anchor_rep, out_vcrs=out_vcrs, datum_ok=datum_ok, consistency=report["consistency"], cfg={"datum_sanity_m": c.datum_sanity_m}, is_metric_head=is_metric_head)
    report["tier"] = tier.to_dict()
    # --- compose ---
    artifacts: dict[str, str] = {"input_preview": "input_preview.png", "depth_preview": "depth_preview.png", "rdsm_preview": "rdsm_preview.png", "relative_tif": "relative.tif", "object_rel_tif": "object_rel.tif", "meta": "meta.json", "prep": "prep.json", "prediction": "prediction.json", "calib_report": "calib_report.json"}
    layers: dict[str, Any] = {"relative": {"units": "relative", "tier": "R", "preview": "rdsm_preview.png"}}
    
    # Texture for viewer is always generated once at the start of composition
    tex_path, tex_size = write_texture(job_dir / "texture.jpg", ing.rgb, max_dim=settings.terrain.max_texture_dim)
    artifacts["texture"] = "texture.jpg"

    scale = None
    scale_source = None
    if is_metric_head:
        scale = 1.0
        scale_source = "direct_metric_neural_head"

    if terrain_res is not None and tier.tier in ("T", "A"):
        if anchor_rep and anchor_rep.get("accepted"):
            terrain_res.terrain = terrain_res.terrain + float(anchor_rep["offset_m"])
            if not is_metric_head and anchor_rep.get("scale_m_per_unit"):
                scale, scale_source = float(anchor_rep["scale_m_per_unit"]), "anchors_ransac"
        if scale is None and scale_fit and scale_fit.accepted:
            scale, scale_source = scale_fit.scale, "dem_residual_fit (unvalidated)"
        tg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, out_vcrs, tier.tier)
        write_raster(job_dir / "terrain.tif", terrain_res.terrain, tg, {"OUTPUT_QUANTITY": "terrain_layer_low_frequency", "NOT_A_DTM": "true", "DEM_SOURCE": dem_src.product if dem_src else ""})
        lo, hi = float(np.nanpercentile(terrain_res.terrain, 1)), float(np.nanpercentile(terrain_res.terrain, 99))
        dsm, ndsm = compose_dsm(terrain_res.terrain, obj.object_rel, scale, valid & dem_valid)
        write_raster(job_dir / "dsm.tif", dsm, tg, {"OUTPUT_QUANTITY": "dsm_surface_elevation" if ndsm is not None else "terrain_only_no_object_scale", "OBJECT_SCALE_SOURCE": scale_source or "none", "MODEL": f"{prov['model_name']}@{prov['model_version']}", "MODEL_SHA256": prov.get("model_sha256") or "n/a"})
        dlo, dhi = float(np.nanpercentile(dsm, 1)), float(np.nanpercentile(dsm, 99))
        leg = elevation_preview(job_dir / "dsm_preview.png", dsm, lo=dlo, hi=dhi)
        elevation_preview(job_dir / "terrain_preview.png", terrain_res.terrain, lo=dlo, hi=dhi)
        hillshade_preview(job_dir / "hillshade_preview.png", dsm, gsd[0], gsd[1])
        slope, aspect = slope_layers(dsm, gsd[0], gsd[1])
        sg = Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, tier.tier)
        write_raster(job_dir / "slope.tif", slope, sg, {"OUTPUT_QUANTITY": "slope_degrees", "METHOD": "Horn 3x3, edges flagged"})
        write_raster(job_dir / "aspect.tif", aspect, sg, {"OUTPUT_QUANTITY": "aspect_degrees_from_north"})
        sleg = slope_preview(job_dir / "slope_preview.png", slope)
        low_support = terrain_res.support < 0.25
        flags = build_flags(dsm.shape, valid=np.isfinite(dsm), raw_fallback=terrain_res.raw_fallback, dem_void=~dem_valid, no_object_scale=ndsm is None, low_support=low_support)
        write_raster(job_dir / "flags.tif", flags, Grid(grid.width, grid.height, grid.transform, grid.crs, "uint16", 0, "relative", False, None, tier.tier), {"BITS": "BORDER=1,TERRAIN_RAW_DEM=2,DEM_VOID=4,NODATA=8,NO_OBJECT_SCALE=16,LOW_SUPPORT=32"}, dtype="uint16")
        flags_preview(job_dir / "flags_preview.png", flags)
        artifacts.update({"terrain_tif": "terrain.tif", "dsm_tif": "dsm.tif", "slope_tif": "slope.tif", "aspect_tif": "aspect.tif", "flags_tif": "flags.tif", "dsm_preview": "dsm_preview.png", "terrain_preview": "terrain_preview.png", "hillshade_preview": "hillshade_preview.png", "slope_preview": "slope_preview.png", "flags_preview": "flags_preview.png"})
        layers.update({"dsm": {"units": "metres", "vertical_crs": out_vcrs, "tier": tier.tier, "preview": "dsm_preview.png", "legend": leg, "label": "ABSOLUTE" if ndsm is not None else "ABSOLUTE (terrain only)"}, "terrain": {"units": "metres", "vertical_crs": out_vcrs, "tier": "T", "preview": "terrain_preview.png", "legend": leg, "label": "ABSOLUTE · DEM-derived low-frequency terrain (not a DTM)"}, "slope": {"units": "degrees", "preview": "slope_preview.png", "legend": sleg}, "hillshade": {"preview": "hillshade_preview.png"}, "flags": {"preview": "flags_preview.png"}})
        if ndsm is not None:
            write_raster(job_dir / "ndsm.tif", ndsm, Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, tier.tier), {"OUTPUT_QUANTITY": "height_above_local_ground", "SCALE_SOURCE": scale_source or ""})
            nleg = elevation_preview(job_dir / "ndsm_preview.png", ndsm, lo=0.0, hi=float(np.nanpercentile(ndsm, 99)))
            artifacts["ndsm_tif"] = "ndsm.tif"
            artifacts["ndsm_preview"] = "ndsm_preview.png"
            layers["ndsm"] = {"units": "metres", "tier": tier.tier, "preview": "ndsm_preview.png", "legend": nleg, "label": f"METRIC RELATIVE · height above local ground · scale: {scale_source}"}
        # heightfields for the viewer: dsm (absolute, metres) and terrain (metres)
        _smooth = 0.5 if ndsm is not None else 1.5
        names_to_build = [("dsm", dsm), ("terrain", terrain_res.terrain)]
        if ndsm is not None:
            names_to_build.append(("ndsm", ndsm))
        for name, arr in names_to_build:
            hf, hv, hmeta = build_heightfield(
                np.where(np.isfinite(arr), arr, -9999.0),
                -9999.0,
                max_mesh_dim=settings.terrain.max_mesh_dim,
                units="metres",
                metric=True,
                tier=tier.tier,
                vertical_reference=out_vcrs if name != "ndsm" else "height_above_ground",
                texture_size=tex_size,
                viewer_smooth_sigma=_smooth if name != "ndsm" else 0.0,
                guide_rgb=ing.rgb if name in ("dsm", "ndsm") else None,
                edge_sharpen=name in ("dsm", "ndsm"),
            )
            write_heightfield(job_dir / f"heightfield_{name}.f32", hf)
            _dump(job_dir / f"heightfield_{name}.json", hmeta.to_dict())
            layers[name]["heightfield"] = f"heightfield_{name}.f32"
            layers[name]["heightfield_meta"] = f"heightfield_{name}.json"

        # LoD-1 3D vector building extraction (Solution C)
        # Uses AI-powered RGB segmentation + nDSM height fusion when RGB is available
        if ndsm is not None:
            try:
                from core.terrain.lod1 import extract_lod1_buildings, save_lod1_buildings
                buildings_path = job_dir / "buildings.json"
                # Cache: skip extraction if buildings.json already exists with data
                if buildings_path.exists():
                    try:
                        cached = json.loads(buildings_path.read_text(encoding="utf-8"))
                        if cached.get("count", 0) > 0:
                            lod1_data = cached
                            log.event("LOD1", f"using cached {cached['count']} buildings")
                        else:
                            raise ValueError("empty cache")
                    except Exception:
                        lod1_data = extract_lod1_buildings(ndsm, terrain_res.terrain, rgb=ing.rgb, gsd_m=gsd_m)
                        save_lod1_buildings(buildings_path, lod1_data)
                else:
                    lod1_data = extract_lod1_buildings(ndsm, terrain_res.terrain, rgb=ing.rgb, gsd_m=gsd_m)
                    save_lod1_buildings(buildings_path, lod1_data)
                artifacts["buildings_json"] = "buildings.json"
                seg_method = lod1_data.get("segmentation_method", "unknown")
                log.event("LOD1", f"extracted {lod1_data['count']} LoD-1 building instances ({seg_method})")
            except Exception as e:  # noqa: BLE001
                log.event("WARN", f"LoD-1 extraction skipped: {e}")
    elif tier.tier == "H":
        # Tier H: metric nDSM directly available from neural head
        ndsm = np.where(valid, np.maximum(obj.object_rel * 1.0, 0.0), np.nan).astype(np.float32)
        write_raster(job_dir / "ndsm.tif", ndsm, Grid(grid.width, grid.height, grid.transform, grid.crs, "float32", -9999.0, "metres", True, None, "H"), {"OUTPUT_QUANTITY": "height_above_local_ground", "SCALE_SOURCE": scale_source or "direct_metric_neural_head"})
        nleg = elevation_preview(job_dir / "ndsm_preview.png", ndsm, lo=0.0, hi=float(np.nanpercentile(ndsm, 99)))
        artifacts["ndsm_tif"] = "ndsm.tif"
        artifacts["ndsm_preview"] = "ndsm_preview.png"
        layers["ndsm"] = {"units": "metres", "tier": "H", "preview": "ndsm_preview.png", "legend": nleg, "label": "METRIC RELATIVE · height above local ground (direct metric neural head)"}
        hf, hv, hmeta = build_heightfield(
            np.where(np.isfinite(ndsm), ndsm, -9999.0),
            -9999.0,
            max_mesh_dim=settings.terrain.max_mesh_dim,
            units="metres",
            metric=True,
            tier="H",
            vertical_reference="height_above_ground",
            texture_size=tex_size,
            viewer_smooth_sigma=0.0,
        )
        write_heightfield(job_dir / "heightfield_ndsm.f32", hf)
        _dump(job_dir / "heightfield_ndsm.json", hmeta.to_dict())
        layers["ndsm"]["heightfield"] = "heightfield_ndsm.f32"
        layers["ndsm"]["heightfield_meta"] = "heightfield_ndsm.json"
    
    # relative heightfield always available (tier R view)
    hf, hv, hmeta = build_heightfield(rel, settings.rdsm.nodata, max_mesh_dim=settings.terrain.max_mesh_dim, texture_size=tex_size)
    write_heightfield(job_dir / "heightfield_relative.f32", hf)
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
        "units": "metres" if tier.tier in ("T", "A") else "relative",
        "vertical_reference": out_vcrs if tier.tier in ("T", "A") else None,
        "object_scale_source": scale_source,
        "object_scale_m_per_unit": scale,
        "grid": grid.to_dict(),
        "gsd_m": gsd_m,
        "dem": report["dem"]["source"] if report.get("dem") and "source" in report["dem"] else None,
        "layers": layers,
        "artifacts": artifacts,
        "heightfield": json.loads((job_dir / ("heightfield_dsm.json" if "dsm" in layers and "heightfield" in layers["dsm"] else "heightfield_relative.json")).read_text()),
        "timings_ms": {"calibration_ms": (time.perf_counter() - t0) * 1000.0},
    }
    _dump(job_dir / "result.json", result)
    log.event("CALIBRATION", "two-layer calibration complete", tier=tier.tier, quality=tier.quality, scale_source=scale_source, ms=round((time.perf_counter() - t0) * 1000, 1))
    return result
