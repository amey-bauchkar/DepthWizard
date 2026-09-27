"""Point-checkpoint validation (e.g. NASA ICESat-2 ground/canopy heights) — for regions without LiDAR rasters (India).

Checkpoint CSV: `id,lon,lat,h_ground[,h_canopy,gnd_photons,veg_photons,night,date,...]` with an optional
`# vcrs=<ellipsoidal|EGM96|EGM2008|EPSG:code>` comment line (vertical reference of h_ground; default ellipsoidal).
h_canopy is canopy height ABOVE ground (m), as in ICESat-2 ATL08 / SlideRule PhoREAL.

Each checkpoint represents a ~20 m ICESat-2 segment, so the raster is summarised in a disk of `radius_m` around it:
  terrain  -> median of the terrain layer            vs h_ground            (bare-earth elevation)
  dem      -> median of the input DEM (baseline)     vs h_ground
  canopy   -> 98th percentile of the nDSM            vs h_canopy            (height above ground of the top surface)
  surface  -> 98th percentile of the DSM             vs h_ground + h_canopy (absolute top-of-surface elevation)
Checkpoint heights are converted to the job's vertical reference with the C-1 datum guard before differencing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from core.geo.grid import Grid
from core.geo.vertical import transform_heights
from core.validate.metrics import metric_set


@dataclass
class Checkpoints:
    ids: list[str]
    lon: np.ndarray
    lat: np.ndarray
    h_ground: np.ndarray
    h_canopy: np.ndarray  # NaN when absent
    gnd_photons: np.ndarray  # NaN when absent
    vcrs: str
    source: str = ""


def load_checkpoints(path: str | Path) -> Checkpoints:
    vcrs, header, rows, notes = "ellipsoidal", None, [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "vcrs=" in s:
                vcrs = s.split("vcrs=")[1].strip()
            else:
                notes.append(s.lstrip("# "))
            continue
        parts = [p.strip() for p in s.split(",")]
        if header is None and not _is_number(parts[1] if len(parts) > 1 else ""):
            header = [p.lower() for p in parts]
            continue
        rows.append(parts)
    header = header or ["id", "lon", "lat", "h_ground"]
    col = {h: i for i, h in enumerate(header)}
    for need in ("lon", "lat", "h_ground"):
        if need not in col:
            raise ValueError(f"checkpoint CSV needs a '{need}' column (have {header})")

    def num(name: str) -> np.ndarray:
        if name not in col:
            return np.full(len(rows), np.nan)
        return np.array([float(r[col[name]]) if len(r) > col[name] and _is_number(r[col[name]]) else np.nan for r in rows])

    ids = [r[col["id"]] if "id" in col else f"P{i}" for i, r in enumerate(rows)]
    return Checkpoints(ids, num("lon"), num("lat"), num("h_ground"), num("h_canopy"), num("gnd_photons"), vcrs, " ".join(notes)[:300])


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


@dataclass
class PointValidation:
    n_checkpoints: int
    n_in_grid: int
    radius_m: float
    datum_handling: str
    filters: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    strata: dict[str, dict[str, Any]] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_points(grid: Grid, layers: dict[str, np.ndarray], cp: Checkpoints, *, out_vcrs: str = "EGM2008", radius_m: float = 10.0, min_ground_photons: int = 5, border_px: int = 4) -> PointValidation:
    """layers: any of {'terrain','dem','ndsm','dsm'} as float arrays on `grid` (NaN = nodata)."""
    if grid.crs is None or grid.transform is None:
        raise ValueError("point validation needs a georeferenced job")
    ok = np.isfinite(cp.lon) & np.isfinite(cp.lat) & np.isfinite(cp.h_ground)
    if np.isfinite(cp.gnd_photons).any():
        ok &= ~(cp.gnd_photons < min_ground_photons)
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(cp.lon, cp.lat)
    inv = ~grid.transform
    col, row = inv @ (np.asarray(x), np.asarray(y))
    col, row = np.asarray(col), np.asarray(row)
    ok &= (col >= border_px) & (row >= border_px) & (col < grid.width - border_px) & (row < grid.height - border_px)
    idx = np.flatnonzero(ok)
    note = f"checkpoint heights assumed to be in {out_vcrs}"
    hg = cp.h_ground.astype(float).copy()
    if cp.vcrs not in (out_vcrs, "same") and idx.size:
        hg[idx] = transform_heights(cp.lon[idx], cp.lat[idx], hg[idx], cp.vcrs, out_vcrs)
        note = f"checkpoint heights converted {cp.vcrs} -> {out_vcrs} (C-1 guarded, offline geoid grids)"
    gsd = grid.pixel_size or (1.0, 1.0)
    r_px = max(1.0, radius_m / gsd[0])
    H, W = grid.height, grid.width
    summ: dict[str, np.ndarray] = {k: np.full(idx.size, np.nan) for k in ("terrain", "dem", "canopy", "surface")}
    reducers = (("terrain", "terrain", np.median), ("dem", "dem", np.median), ("canopy", "ndsm", lambda v: np.percentile(v, 98)), ("surface", "dsm", lambda v: np.percentile(v, 98)))
    for j, i in enumerate(idx):
        r_lo, r_hi = max(0, int(row[i] - r_px)), min(H, int(row[i] + r_px) + 1)
        c_lo, c_hi = max(0, int(col[i] - r_px)), min(W, int(col[i] + r_px) + 1)
        yy, xx = np.mgrid[r_lo:r_hi, c_lo:c_hi]
        disk = (yy + 0.5 - row[i]) ** 2 + (xx + 0.5 - col[i]) ** 2 <= r_px**2
        for key, lname, fn in reducers:
            a = layers.get(lname)
            if a is None:
                continue
            w = a[r_lo:r_hi, c_lo:c_hi][disk]
            w = w[np.isfinite(w)]
            if w.size:
                summ[key][j] = float(fn(w))
    g = hg[idx]
    canopy = cp.h_canopy[idx]
    metrics: dict[str, dict[str, Any]] = {}
    if np.isfinite(summ["terrain"]).any():
        metrics["terrain_vs_ground"] = metric_set(summ["terrain"], g)
    if np.isfinite(summ["dem"]).any():
        metrics["input_dem_vs_ground"] = metric_set(summ["dem"], g)
    if np.isfinite(summ["canopy"]).any() and np.isfinite(canopy).any():
        metrics["ndsm_vs_canopy_height"] = metric_set(summ["canopy"], canopy)
    if np.isfinite(summ["surface"]).any() and np.isfinite(canopy).any():
        top = g + np.nan_to_num(canopy)
        metrics["dsm_vs_top_of_surface"] = metric_set(summ["surface"], top)
        if np.isfinite(summ["dem"]).any():
            metrics["input_dem_vs_top_of_surface"] = metric_set(summ["dem"], top)
    strata: dict[str, dict[str, Any]] = {}
    if np.isfinite(canopy).any():
        for name, sel in (("open_canopy_lt2m", canopy < 2.0), ("vegetated_or_built_ge5m", canopy >= 5.0)):
            if sel.sum() >= 5:
                strata[name] = {k: metric_set(summ[s][sel], ref[sel]) for k, s, ref in (("terrain_vs_ground", "terrain", g), ("input_dem_vs_ground", "dem", g), ("ndsm_vs_canopy_height", "canopy", canopy)) if np.isfinite(summ[s][sel]).any()}
    caveats = [
        "ICESat-2 segments are ~20 m long and ~11 m wide; raster values are summarised in a disk around each segment centre, so steep slopes add sampling error.",
        "ICESat-2 acquisitions span several years; construction, clearing or tree growth since then appears as error.",
        "h_canopy (ATL08-style) measures the top of vegetation AND of buildings above ground; it is a sparse, independent check, not a dense LiDAR reference.",
    ]
    return PointValidation(int(cp.lon.size), int(idx.size), radius_m, note, {"min_ground_photons": min_ground_photons, "border_px": border_px}, metrics, strata, caveats)
