"""Elevation-based flood screening (Disaster Management). Static "bathtub" thresholding on the terrain layer — NOT a
hydraulic or hydrodynamic simulation. Mathematics: docs/math_audit.md §5-8; constants: core/screening_params.py.

For water level W (metres, same vertical datum as the terrain layer T) and valid cells Omega (T finite, not NoData):

    I(x)       = 1  if T(x) <= W                         inundation indicator (SCENARIO)
    D(x)       = max(W - T(x), 0)                         screened water depth above the terrain layer (SCENARIO)
    A_wet      = sum_x I(x) * |det J|                     wet area; |det J| = exact cell area of the affine grid
    connected  = wet cells 8-connected (within the wet set) to an open boundary cell: a wet cell on the raster edge or
                 next to a NoData cell (NoData is unknown ground, not a wall). Wet cells without such a path are
                 "isolated": below W, but with no surface path for water in this scene.

Two water models (FLOOD_MODELS):
    "level"  a still water surface at elevation W (lakes, reservoirs, flat plains, coasts): the rules above.
    "river"  a river stage h in metres ABOVE THE CHANNEL: the surface T is replaced by HAND (height above nearest
             drainage, core.disaster.hand), so wet = HAND <= h and depth = h - HAND. The water surface then follows the
             valley gradient; this is the model that makes sense in hills, where one flat level cannot.

Depth is terrain-relative (never DSM-relative: roofs are not ground). Per building B (exact footprint pixel set):

    f_wet      = |{i in B : T_i <= W}| / |B_valid|        wet footprint fraction
    D_mean     = mean over wet footprint cells of (W - T_i)
    D_max      = max over the footprint of D_i
    D_exp      = max(W - Q10(T_B), 0)                     exposure depth on the low side of the footprint
    class      = bin(D_exp)  by EXPOSURE_BINS_M           (policy bins, returned with every result)

Uncertainty (flood_probability.tif). The surface S (terrain or HAND) is read with a measured error sigma (the job's
terrain RMSE against laser checkpoints, result["uncertainty"]["terrain"]), taken as Gaussian and independent of W:

    P_wet(x)   = Phi((W - S(x)) / sigma)                  probability that the cell is below the water surface
    likely     = P_wet >= FLOOD_P_LIKELY,  possible = P_wet >= FLOOD_P_POSSIBLE   (areas and building counts)

Per building the same rule is applied to Q10 of the footprint. The deterministic map is the P = 0.5 contour.
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
from scipy.special import ndtr, ndtri

from core.disaster.cache import cached
from core.dsm.derive import pixel_area_m2
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.screening_params import EXPOSURE_BINS_M, EXPOSURE_GROUND_QUANTILE, EXPOSURE_LABELS, FLOOD_CONNECTIVITY

PREVIEW_MAX_DEPTH_M = 5.0  # display only: the preview colour ramp saturates here (4 colours, linear in depth)
PREVIEW_COLOURS = ["#add8e6", "#00bfff", "#0000cd", "#000080"]
METHOD = "bathtub-threshold-2 (terrain layer, exact footprints, 8-connectivity diagnostic)"
METHOD_RIVER = "hand-stage-1 (height above nearest drainage on the terrain layer, exact footprints)"
FLOOD_MODELS = ("level", "river")
FIELD_MAX_DIM = 1600  # display only: longest side (px) of the surface sent to the browser for live slider previews
RIVER_STAGE_MAX_M = 100.0  # POLICY: largest river rise accepted (the 2023 Teesta GLOF was ~20 m at Chungthang)
FLOOD_P_LIKELY, FLOOD_P_POSSIBLE = 0.9, 0.1
"""POLICY. Probability bands reported as 'likely' and 'possibly' flooded."""
FLOOD_BAND_OBSERVED = {"likely": (0.49, 0.57), "possible": (0.23, 0.44), "unlikely_dry": (0.94, 0.97)}
"""MEASURED (docs/hazard_validation.md, two real floods: Teesta GLOF 2023, Sunkoshi 2024; river model). Share of each
band that the observed flood scar covered ('unlikely_dry': share of the P < 0.1 area that stayed dry). The
probabilities are over-confident where the DEM over-floods, so the UI reports these measured rates, not P itself."""
RIVER_RELIEF_M = 40.0
"""POLICY. Default model choice: if the terrain layer's P95 - P5 relief exceeds this, the scene is hilly and the
river-stage model is suggested (a flat level would flood whole hillsides from one valley end)."""
QUANTITY_CATEGORY = {
    "waterLevel_m": "SCENARIO", "affectedAreaM2": "SCENARIO", "affectedAreaPct": "SCENARIO", "maxDepth_m": "SCENARIO",
    "meanDepth_m": "SCENARIO", "isolatedAreaM2": "SCENARIO", "totalAreaM2": "DERIVED", "terrain": "REFERENCE",
    "flood_depth_m": "SCENARIO", "wet_fraction": "SCENARIO", "base_elev_m": "REFERENCE", "height_m": "PREDICTED",
}


def exposure_rules() -> list[dict[str, Any]]:
    """The single definition of the exposure classes, serialised into every response (the UI must not re-derive it)."""
    lo = 0.0
    rules = []
    for label, hi in zip(EXPOSURE_LABELS, (*EXPOSURE_BINS_M, None)):
        rules.append({"label": label, "gt_m": lo, "le_m": hi})
        lo = hi if hi is not None else lo
    return rules


def classify_exposure(depth_m: float) -> str:
    """NONE if d <= 0 (or not finite); LOW if 0 < d <= b0; MODERATE if b0 < d <= b1; HIGH if b1 < d <= b2; else VERY HIGH."""
    if not math.isfinite(depth_m) or depth_m <= 0:
        return "NONE"
    for label, hi in zip(EXPOSURE_LABELS, EXPOSURE_BINS_M):
        if depth_m <= hi:
            return label
    return EXPOSURE_LABELS[-1]


def connectivity_structure(n: int = FLOOD_CONNECTIVITY) -> np.ndarray:
    if n == 8:
        return np.ones((3, 3), bool)
    if n == 4:
        return ndimage.generate_binary_structure(2, 1)
    raise ValueError("connectivity must be 4 or 8")


def boundary_connected(wet: np.ndarray, valid: np.ndarray, connectivity: int = FLOOD_CONNECTIVITY) -> np.ndarray:
    """Wet cells connected (within the wet set) to an open boundary cell: raster edge or adjacent to NoData."""
    st = connectivity_structure(connectivity)
    open_b = np.zeros_like(wet)
    open_b[0, :] = open_b[-1, :] = True
    open_b[:, 0] = open_b[:, -1] = True
    open_b |= ndimage.binary_dilation(~valid, structure=st)
    lab, n = ndimage.label(wet, structure=st)
    if n == 0:
        return np.zeros_like(wet)
    seeds = np.unique(lab[wet & open_b])
    keep = np.zeros(n + 1, bool)
    keep[seeds[seeds > 0]] = True
    return keep[lab]


def _footprint_sets(job_dir: Path, b_data: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """Exact label raster if the job has one; else rasterise the stored pixel polygons (cell-centre rule)."""
    lp = job_dir / "building_labels.tif"
    if lp.exists():
        with rasterio.open(lp) as ds:
            lab = ds.read(1)
        if lab.shape == shape:
            return lab.astype(np.int32), "exact_label_raster"
    shapes = [({"type": "Polygon", "coordinates": [b["pixel_coords"]]}, int(b["id"])) for b in b_data.get("buildings", []) if len(b.get("pixel_coords") or []) >= 4]
    if shapes:
        from rasterio.features import rasterize

        return rasterize(shapes, out_shape=shape, transform=rasterio.Affine.identity(), fill=0, dtype="int32"), "polygon_cell_centre"
    return None, "scalar_ground_legacy"


def load_terrain(job_dir: Path) -> tuple[np.ndarray, np.ndarray, Any, str | None, float]:
    """terrain.tif as float64 (NaN outside valid cells), valid mask, transform, CRS, cell area. Cached, read-only."""
    p = job_dir / "terrain.tif"

    def build():
        with rasterio.open(p) as ds:
            t = ds.read(1).astype(np.float64)
            nodata, tr, crs = ds.nodata, ds.transform, ds.crs.to_string() if ds.crs else None
        valid = np.isfinite(t)
        if nodata is not None:
            valid &= t != nodata
        return np.where(valid, t, np.nan), valid, tr, crs, pixel_area_m2(tr)

    return cached("terrain", [p], build)


def load_buildings(job_dir: Path, shape: tuple[int, int]) -> tuple[dict[str, Any], np.ndarray | None, str | None, list]:
    """buildings.json, its footprint label raster (_footprint_sets), method and per-label slices. Cached, read-only."""
    bp, lp = job_dir / "buildings.json", job_dir / "building_labels.tif"

    def build():
        if not bp.exists():
            return {}, None, None, []
        b_data = json.loads(bp.read_text(encoding="utf-8"))
        labels, method = _footprint_sets(job_dir, b_data, shape)
        return b_data, labels, method, (ndimage.find_objects(labels) if labels is not None else [])

    return cached("buildings", [bp, lp], build, extra=tuple(shape))


def _hand_mem(job_dir: Path, terrain: np.ndarray, grid: Grid, a_px: float, drainage_area_m2: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """_hand_cached, kept in memory while terrain.tif is unchanged (HAND is derived from it alone)."""
    h, d, meta = cached("hand", [job_dir / "terrain.tif"], lambda: _hand_cached(job_dir, terrain, grid, a_px, drainage_area_m2), extra=(drainage_area_m2,))
    return h, d, dict(meta)


def building_exposure(terrain: np.ndarray, wet: np.ndarray, W: float, labels: np.ndarray | None, b_data: dict[str, Any], sigma_m: float | None = None, band_counts: dict[str, int] | None = None, slices: list | None = None) -> list[dict[str, Any]]:
    """Per-building exposure. With labels: pixelwise over the footprint. Without: legacy scalar rule on base_elev_m.
    With sigma_m: wetProbability = Phi((W - ground) / sigma) per building, and band_counts gets likely / possible."""
    if slices is None:
        slices = ndimage.find_objects(labels) if labels is not None else []
    out = []
    for b in b_data.get("buildings", []):
        bid = int(b.get("id"))
        base = b.get("base_elev_m")
        rec = {"id": bid, "base_elev_m": round(base, 2) if base is not None else None, "height_m": b.get("height_median_m", b.get("height_m")), "area_m2": b.get("area_m2"), "coords": b.get("coords"), "pixel_bbox": b.get("pixel_bbox")}
        sl = slices[bid - 1] if labels is not None and 0 < bid <= len(slices) else None
        if sl is not None:
            fp = labels[sl] == bid
            T = terrain[sl][fp]
            ok = np.isfinite(T)
            if not ok.any():
                continue
            T = T[ok].astype(np.float64)
            w = wet[sl][fp][ok]
            d = np.where(w, W - T, 0.0)
            q = float(np.percentile(T, EXPOSURE_GROUND_QUANTILE))
            d_exp = max(W - q, 0.0) if w.any() else 0.0  # wet mask may be the connected subset
            rec.update({"ground_p10_m": round(q, 2), "flood_depth_m": round(d_exp, 2), "wet_fraction": round(float(w.mean()), 3), "mean_depth_wet_m": round(float(d[w].mean()), 2) if w.any() else 0.0, "max_depth_m": round(float(d.max()), 2), "contact": bool(w.any())})
        elif base is not None:  # legacy: one ground value per building (pre-audit behaviour, flagged)
            d_exp = max(W - float(base), 0.0)
            rec.update({"flood_depth_m": round(d_exp, 2), "wet_fraction": None, "contact": d_exp > 0})
        else:
            continue
        rec["exposure"] = classify_exposure(rec["flood_depth_m"])
        g = rec.get("ground_p10_m", rec.get("base_elev_m"))
        if sigma_m and g is not None:
            pw = float(ndtr((W - float(g)) / sigma_m))
            rec["wetProbability"] = round(pw, 3)
            if band_counts is not None:
                band_counts["likely"] += pw >= FLOOD_P_LIKELY
                band_counts["possible"] += pw >= FLOOD_P_POSSIBLE
        if rec["contact"]:
            out.append(rec)
    return out


def terrain_sigma(result: dict[str, Any]) -> tuple[float | None, str | None]:
    """Measured vertical error of the terrain layer (1 sigma, metres) and its source, or (None, None)."""
    t = (result.get("uncertainty") or {}).get("terrain") or {}
    v = t.get("value_m")
    if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
        return float(v), str(t.get("source") or "terrain uncertainty of this job")
    return None, None


P_WET_Z_SATURATED = 8.0  # |z| beyond this: Phi(z) is 0 or 1 to < 1e-15 (exact for the percent raster and the bands)


def _p_wet(surface: np.ndarray, valid: np.ndarray, W: float, sigma_m: float) -> np.ndarray:
    """P(wet) = Phi((W - S) / sigma) on valid cells with a defined surface, else 0. Phi is evaluated only where it is
    not saturated (most of a hilly scene lies many sigma above the water), which made this the slowest step."""
    ok = valid & np.isfinite(surface)
    z = np.full(surface.shape, -np.inf)
    z[ok] = (W - surface[ok]) / sigma_m
    pw = np.zeros(surface.shape)
    mid = np.abs(z) <= P_WET_Z_SATURATED
    pw[mid] = ndtr(z[mid])
    pw[z > P_WET_Z_SATURATED] = 1.0
    return pw


def terrain_relief(job_dir: Path) -> dict[str, Any]:
    """P5 / P95 of the terrain layer and the suggested flood model (RIVER_RELIEF_M)."""
    return dict(cached("relief", [job_dir / "terrain.tif"], lambda: _terrain_relief(job_dir)))


def _terrain_relief(job_dir: Path) -> dict[str, Any]:
    with rasterio.open(job_dir / "terrain.tif") as ds:
        t = ds.read(1, out_shape=(max(1, ds.height // 4), max(1, ds.width // 4)), masked=True).astype(np.float64).filled(np.nan)
    t = t[np.isfinite(t)]
    if not t.size:
        return {"p5_m": None, "p95_m": None, "relief_m": None, "suggestedModel": "level", "reliefThreshold_m": RIVER_RELIEF_M}
    p5, p95 = float(np.percentile(t, 5)), float(np.percentile(t, 95))
    return {"p5_m": round(p5, 2), "p95_m": round(p95, 2), "relief_m": round(p95 - p5, 1), "suggestedModel": "river" if p95 - p5 > RIVER_RELIEF_M else "level", "reliefThreshold_m": RIVER_RELIEF_M}


def _mapped_channels(grid: Grid) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Mapped rivers covering the scene (assets/waterways, scripts/fetch_waterways.py), rasterised for stream burning."""
    import os

    from backend.config.settings import REPO_ROOT
    from core.geo.raster_io import grid_bounds_wgs84
    from core.terrain.footprints import discover

    wdir = Path(os.environ.get("DW_WATERWAYS_DIR", REPO_ROOT / "assets" / "waterways"))
    try:
        hit = discover(grid_bounds_wgs84(grid), wdir)
    except Exception:  # noqa: BLE001 - no CRS etc.: DEM channels only
        hit = None
    if hit is None:
        return None, {"source": "DEM flow accumulation only (no mapped rivers for this area)"}
    feats = json.loads(hit[0].read_text(encoding="utf-8")).get("features", [])
    if not feats:
        return None, {"source": "DEM flow accumulation only (no mapped rivers in this area)", "checked": hit[1]["source"]}
    from pyproj import Transformer
    from rasterio.features import rasterize

    to = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    shapes = [({"type": "LineString", "coordinates": [to.transform(x, y) for x, y in f["geometry"]["coordinates"]]}, 1) for f in feats if f.get("geometry", {}).get("type") == "LineString"]
    burn = rasterize(shapes, out_shape=(grid.height, grid.width), transform=grid.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
    names = sorted({f["properties"].get("name") for f in feats if f["properties"].get("name")})
    return (burn if burn.any() else None), {"source": f"{hit[1]['source']} burned into the terrain + DEM flow accumulation", "licence": hit[1]["licence"], "mapped": names, "lines": len(shapes)}


def _hand_cached(job_dir: Path, terrain: np.ndarray, grid: Grid, a_px: float, drainage_area_m2: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """HAND + drainage raster for this job (cached as hand.tif / hand_drainage.tif for one drainage area)."""
    from core.disaster.hand import hand_on_grid

    hp, dp, mp = job_dir / "hand.tif", job_dir / "hand_drainage.tif", job_dir / "hand.json"
    if hp.exists() and dp.exists() and mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("drainage_area_m2") == drainage_area_m2 and "channels" in meta:
            with rasterio.open(hp) as ds:
                h = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
            with rasterio.open(dp) as ds:
                d = ds.read(1) > 0
            return h, d, meta
    burn, channels = _mapped_channels(grid)
    out = hand_on_grid(terrain, math.sqrt(a_px), drainage_area_m2=drainage_area_m2, burn=burn)
    h, d, meta = out["hand"], out["drainage"], out["stats"]
    meta["channels"] = channels
    g = Grid(width=grid.width, height=grid.height, transform=grid.transform, crs=grid.crs, dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference="height above nearest drainage", tier=grid.tier)
    write_raster(hp, np.where(np.isfinite(h), h, -9999.0).astype(np.float32), g, tags={"KIND": "hand", "QUANTITY": "height above nearest drainage (m)", "METHOD": METHOD_RIVER})
    gd = Grid(width=grid.width, height=grid.height, transform=grid.transform, crs=grid.crs, dtype="uint8", nodata=0, units="relative", metric=False, vertical_reference=None, tier=grid.tier)
    write_raster(dp, d.astype(np.uint8), gd, tags={"KIND": "drainage", "DRAINAGE_AREA_M2": repr(drainage_area_m2)}, dtype="uint8")
    mp.write_text(json.dumps(meta), encoding="utf-8")
    return h, d, meta


def run_flood_screening(job_dir: Path, water_level_m: float, result: dict[str, Any], *, connected_only: bool = False, model: str = "level", drainage_area_m2: float | None = None, write_outputs: bool = True) -> dict[str, Any]:
    """Flood screening. model="level": still water level W on terrain.tif (connected_only=False: every valid cell with
    T <= W is wet, an upper bound; True: only cells connected to an open boundary). model="river": W is a river stage
    in metres above the channel, applied to HAND.

    write_outputs=False (live slider preview): the same numbers, but no rasters, preview or disaster_flood.json are
    written, so the files on disk (used by downloads, road access and the report) stay those of the last full run."""
    if not math.isfinite(water_level_m):
        raise ValueError("water level must be a finite number of metres")
    if model not in FLOOD_MODELS:
        raise ValueError(f"unknown flood model {model!r}; expected one of {FLOOD_MODELS}")
    if model == "river" and water_level_m < 0:
        raise ValueError("a river stage is a height above the channel and cannot be negative")
    if model == "river" and water_level_m > RIVER_STAGE_MAX_M:
        raise ValueError(f"a river rise of {water_level_m:g} m above the channel is not plausible (max {RIVER_STAGE_MAX_M:g} m): "
                         "was an elevation given? Use the still-water-level model for absolute elevations")
    terrain_path = job_dir / "terrain.tif"
    if not terrain_path.exists():
        raise FileNotFoundError("Disaster analysis requires a valid terrain/elevation surface (terrain.tif).")

    terrain, valid, tr, crs, a_px = load_terrain(job_dir)
    grid = Grid(width=terrain.shape[1], height=terrain.shape[0], transform=tr, crs=crs, dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference=result.get("vertical_reference"), tier=result.get("calibration_tier") or "T")
    W = float(water_level_m)
    hand_meta: dict[str, Any] | None = None
    surface = terrain  # what the water level is compared with: terrain ("level") or HAND ("river")
    undefined = np.zeros_like(valid)
    if model == "river":
        from core.disaster.hand import DRAINAGE_AREA_M2

        da = float(drainage_area_m2) if drainage_area_m2 else DRAINAGE_AREA_M2
        if not (math.isfinite(da) and da > 0):
            raise ValueError("drainage area must be a positive number of square metres")
        surface, _drain, hand_meta = _hand_mem(job_dir, terrain, grid, a_px, da)
        undefined = valid & ~np.isfinite(surface)
        vref = "metres above the river channel (height above nearest drainage)"
        below = valid & np.isfinite(surface) & (surface <= W)
        connected = below  # every HAND cell drains to a channel by construction
        wet = below
    else:
        vref = result.get("vertical_reference") or "unknown (vertical datum of terrain.tif)"
        below = valid & (terrain <= W)
        connected = boundary_connected(below, valid)
        wet = connected if connected_only else below

    depth = np.where(wet, W - surface, 0.0)
    depth[~valid] = np.nan
    n_wet, n_valid = int(wet.sum()), int(valid.sum())

    sigma_m, sigma_src = terrain_sigma(result)
    band_counts = {"likely": 0, "possible": 0}
    uncertainty: dict[str, Any] | None = None
    if sigma_m:
        if write_outputs:
            pw = _p_wet(surface, valid, W, sigma_m)
            n_likely, n_possible = int((pw >= FLOOD_P_LIKELY).sum()), int((pw >= FLOOD_P_POSSIBLE).sum())
            gp = Grid(width=grid.width, height=grid.height, transform=grid.transform, crs=grid.crs, dtype="uint8", nodata=255, units="percent", metric=False, vertical_reference=None, tier=grid.tier)
            write_raster(job_dir / "flood_probability.tif", np.where(valid, np.rint(100 * pw), 255).astype(np.uint8), gp, tags={
                "KIND": "flood_probability", "QUANTITY": "P(wet) in percent = Phi((W - S) / sigma)", "SIGMA_M": repr(sigma_m), "FLOOD_MODEL": model}, dtype="uint8")
        else:  # live preview: Phi is monotone, so P >= p  <=>  S <= W - sigma * Phi^-1(p); no per-cell Phi needed
            ok = valid & np.isfinite(surface)
            n_likely = int((ok & (surface <= W - sigma_m * float(ndtri(FLOOD_P_LIKELY)))).sum())
            n_possible = int((ok & (surface <= W - sigma_m * float(ndtri(FLOOD_P_POSSIBLE)))).sum())
        uncertainty = {"sigmaM": sigma_m, "source": sigma_src, "rule": "P(wet) = Phi((W - S) / sigma), S = " + ("HAND" if model == "river" else "terrain"),
                       "likelyThreshold": FLOOD_P_LIKELY, "possibleThreshold": FLOOD_P_POSSIBLE,
                       "likelyAreaM2": n_likely * a_px, "possibleAreaM2": n_possible * a_px,
                       "rasterResult": "flood_probability.tif", "measuredReliability": FLOOD_BAND_OBSERVED,
                       "note": (f"Checked against two real floods: {FLOOD_BAND_OBSERVED['likely'][0]:.0%}-{FLOOD_BAND_OBSERVED['likely'][1]:.0%} of the 'likely' area and "
                                f"{FLOOD_BAND_OBSERVED['possible'][0]:.0%}-{FLOOD_BAND_OBSERVED['possible'][1]:.0%} of the 'possible' area really flooded; "
                                f"{FLOOD_BAND_OBSERVED['unlikely_dry'][0]:.0%}-{FLOOD_BAND_OBSERVED['unlikely_dry'][1]:.0%} of the rest stayed dry.")}

    buildings_result: list[dict[str, Any]] = []
    b_data, labels, footprint_method, slices = load_buildings(job_dir, terrain.shape)
    if b_data:
        buildings_result = building_exposure(surface, wet, W, labels, b_data, sigma_m, band_counts, slices=slices)
    affected = [b for b in buildings_result if b["exposure"] != "NONE"]

    if write_outputs:
        write_raster(job_dir / "flood_depth.tif", np.where(valid, depth, -9999.0).astype(np.float32), grid, tags={
            "KIND": "flood_depth", "QUANTITY": "max(W - terrain, 0), metres above the terrain layer",
            "SCENARIO_WATER_LEVEL_M": repr(W), "WATER_LEVEL_VERTICAL_REFERENCE": vref, "FLOOD_MODEL": model, "CONNECTED_ONLY": str(connected_only), "METHOD": METHOD,
            "WARNING": "Elevation-based screening only. Not a hydraulic simulation."})
        _generate_flood_preview(job_dir / "flood_preview.png", depth, wet, PREVIEW_MAX_DEPTH_M)

    counts = {lab: sum(b["exposure"] == lab for b in buildings_result) for lab in EXPOSURE_LABELS}
    relief = terrain_relief(job_dir)
    if model == "river":
        model_warnings = [
            "River-stage screening (HAND): the water surface is assumed parallel to the channel; no discharge, backwater, embankments or timing.",
            f"Channels: {hand_meta['channels']['source']}; unmapped channels start where at least {hand_meta['drainage_area_m2'] / 1e4:g} ha drain to a cell, on the DEM-derived terrain layer (the DEM's 30 m detail).",
        ] + (["Cells whose water leaves the scene before reaching a channel have no defined height above the channel and are never shown wet."] if hand_meta.get("undefined_fraction", 0) > 0 else [])
        if relief.get("suggestedModel") == "level":
            model_warnings.append("This scene is fairly flat: the still-water-level model may describe it better.")
    else:
        model_warnings = [
            "Elevation-based screening only.",
            f"Water level is read in {vref}, the terrain layer's vertical datum; convert heights from other datums first.",
            "Connectivity is a diagnostic, not a hydraulic model." if not connected_only else "Only cells with a surface path to the scene boundary are shown.",
        ]
        if relief.get("suggestedModel") == "river":
            model_warnings.append(f"This scene is hilly (relief {relief['relief_m']:g} m): one still water level floods whole hillsides from the lowest point; the river-stage model is more appropriate.")
    summary = {
        "scenario": "flood_screening",
        "model": model,
        "method": METHOD_RIVER if model == "river" else METHOD,
        "elevationSource": "hand.tif (from terrain.tif)" if model == "river" else "terrain.tif",
        "hand": hand_meta,
        "undefinedAreaM2": int(undefined.sum()) * a_px,
        "relief": relief,
        "waterLevel_m": W,
        # W is compared with T directly, so it is read in T's vertical datum; there is no conversion of W
        "waterLevelVerticalReference": vref,
        "connectedOnly": connected_only,
        "connectivity": FLOOD_CONNECTIVITY,
        "pixelAreaM2": a_px,
        "affectedAreaM2": n_wet * a_px,
        "affectedAreaPct": round(100.0 * n_wet / n_valid, 1) if n_valid else 0.0,
        "totalAreaM2": n_valid * a_px,
        "isolatedAreaM2": int((below & ~connected).sum()) * a_px,
        "affectedBuildingsCount": len(affected),
        "uncertainty": ({**uncertainty, "buildingsLikely": band_counts["likely"], "buildingsPossible": band_counts["possible"]} if uncertainty else None),
        "contactBuildingsCount": len(buildings_result),
        "exposureCounts": counts,
        "exposureRules": exposure_rules(),
        "exposureDepthDefinition": (f"h - P{EXPOSURE_GROUND_QUANTILE:g} of the footprint's height above the channel (depth on the low side of the building)" if model == "river"
                                    else f"W - P{EXPOSURE_GROUND_QUANTILE:g} of footprint ground elevation (depth on the low side of the building)"),
        "footprintMethod": footprint_method,
        "maxDepth_m": round(float(depth[wet].max()), 2) if n_wet else 0.0,
        "meanDepth_m": round(float(depth[wet].mean()), 2) if n_wet else 0.0,
        "rasterResult": "flood_depth.tif",
        "previewResult": "flood_preview.png",
        # the legend must be drawn from this: colour i sits at depth stops_m[i] (linear interpolation in between)
        "previewRamp": {"colours": PREVIEW_COLOURS, "stops_m": [round(PREVIEW_MAX_DEPTH_M * i / (len(PREVIEW_COLOURS) - 1), 2) for i in range(len(PREVIEW_COLOURS))], "saturates_above_m": PREVIEW_MAX_DEPTH_M},
        "qualityMask": "flags.tif",
        "buildings": affected,
        "quantityCategory": QUANTITY_CATEGORY,
        "warnings": model_warnings + ["Not a hydrodynamic simulation."] + (["Building footprints rasterised from simplified polygons (job predates exact footprints)."] if footprint_method == "polygon_cell_centre" else [])
          + (["Building exposure uses one ground value per building (legacy job)."] if footprint_method == "scalar_ground_legacy" else []),
    }
    if write_outputs:
        (job_dir / "disaster_flood.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary["live"] = True  # numbers only; the files on disk are those of the last full run
    return summary


def display_field(job_dir: Path, model: str, max_dim: int = FIELD_MAX_DIM) -> tuple[np.ndarray, dict[str, Any]]:
    """The surface the water level is compared with (HAND for "river", terrain for "level"), float32 with NaN where
    no cell can be wet, sampled at block centres to at most max_dim px, and the preview colour ramp. The browser
    colours it for every slider position (wet = S <= W, depth = W - S: the rule of run_flood_screening), so the
    overlay follows the slider without a server round trip; the full-resolution preview replaces it on release."""
    from core.disaster.hand import DRAINAGE_AREA_M2

    if model not in FLOOD_MODELS:
        raise ValueError(f"unknown flood model {model!r}; expected one of {FLOOD_MODELS}")
    if not (job_dir / "terrain.tif").exists():
        raise FileNotFoundError("Disaster analysis requires a valid terrain/elevation surface (terrain.tif).")
    terrain, _valid, tr, crs, a_px = load_terrain(job_dir)
    surface = terrain
    if model == "river":
        grid = Grid(width=terrain.shape[1], height=terrain.shape[0], transform=tr, crs=crs, dtype="float32", nodata=-9999.0, units="metres", metric=True, vertical_reference=None, tier="T")
        surface, _d, _m = _hand_mem(job_dir, terrain, grid, a_px, DRAINAGE_AREA_M2)
    f = max(1, math.ceil(max(surface.shape) / max_dim))
    arr = np.ascontiguousarray(surface[f // 2::f, f // 2::f], dtype=np.float32)
    ramp = {"colours": PREVIEW_COLOURS, "alpha": 200, "maxDepthM": PREVIEW_MAX_DEPTH_M}
    return arr, {"width": int(arr.shape[1]), "height": int(arr.shape[0]), "stride": f, "model": model, "rule": "wet = S <= W; depth = W - S", "ramp": ramp}


def _generate_flood_preview(path: Path, depth: np.ndarray, mask: np.ndarray, max_depth_ramp: float = 5.0) -> None:
    import os
    import time
    import uuid

    # Colors: shallow = light cyan, deep = dark blue (display only; the ramp saturates at max_depth_ramp metres)
    ramp = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] + [200] for c in PREVIEW_COLOURS], dtype=np.float32)
    h, w = depth.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask.any():
        idx = np.clip(depth[mask] / max_depth_ramp, 0, 1) * (len(ramp) - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.clip(i0 + 1, 0, len(ramp) - 1)
        fr = (idx - i0)[..., None]
        rgba[mask] = (ramp[i0] * (1 - fr) + ramp[i1] * fr).astype(np.uint8)

    tmp_path = path.with_name(f".{path.stem}_{uuid.uuid4().hex[:8]}.tmp.png")
    try:
        Image.fromarray(rgba, "RGBA").save(tmp_path, compress_level=1)  # lossless; fast encode (slider re-runs)
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except (PermissionError, OSError):
                if attempt == 4:
                    try:
                        if path.exists():
                            os.remove(path)
                        os.replace(tmp_path, path)
                    except Exception:
                        pass
                else:
                    time.sleep(0.04 * (attempt + 1))
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass

