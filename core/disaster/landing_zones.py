"""Helicopter landing-zone (HLZ) screening: candidate sites only, NOT a landing clearance. docs/landing_zones.md.

Rules (constants and sources in core.screening_params, FM 3-21.38 ch. 4):
    pad        disk of diameter D (aircraft size), round in CRS metres (exact under rotation/shear of the grid)
    clear      no invalid cell, no detected object (nDSM > object threshold), no building and (optionally) no wet
               cell inside the disk; the disk must lie inside the raster
    slope      least-squares plane z = a dx + b dy + c of the DSM over the disk; slope = atan |(a, b)|
    roughness  RMS residual about that plane
    approach   for each of 16 bearings, a corridor of width D from the pad edge to R + L. A cell at distance s from
               the pad centre (the touchdown point) violates when  H(q) - Z_pad > s / ratio  (10:1), where
               H = DSM, plus k * sigma_object on object cells.

The disk sums are computed for every centre at once with FFT correlations. The plane fit decouples because the disk
is centrally symmetric (sum dx = sum dy = 0):
    c = S_z / n,   [a b] = M^-1 [S_xz S_yz],   RSS = S_zz - c S_z - a S_xz - b S_yz
Elevations are centred on the scene median first, so that FFT round-off (~1e-13 of the largest term) stays far
below the residuals being measured.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio import Affine
from scipy.ndimage import binary_dilation, grey_dilation, maximum_filter
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.signal import fftconvolve

from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.screening_params import (
    HLZ_ADVISORY_MIN_SIZE,
    HLZ_APPROACH_LENGTH_M,
    HLZ_BEARINGS,
    HLZ_CORRIDOR_CELL_M,
    HLZ_CORRIDOR_WIDTH_FACTOR,
    HLZ_MAX_EVALUATED,
    HLZ_MAX_SITES,
    HLZ_MIN_PAD_PIXELS_ACROSS,
    HLZ_OBJECT_BUFFER_M,
    HLZ_OBJECT_MARGIN_SIGMA,
    HLZ_OBSTACLE_RATIO,
    HLZ_ROUGHNESS_MAX_M,
    HLZ_SITE_LATTICE_FRACTION,
    HLZ_SIZES,
    HLZ_SLOPE_ADVISORY_DEG,
    HLZ_SLOPE_ALL_DEG,
    HLZ_SLOPE_MARGIN_DEG,
)

METHOD = "hlz-screening-1"
_FFT_WORKERS = -1  # scipy.fft threads (all cores)

# reason bitmask written to landing_feasible.tif (0 is nodata in DepthWizard uint rasters, so feasible = 128)
R_SLOPE, R_OBJECT, R_ROUGH, R_WET, R_INVALID, R_FEASIBLE = 1, 2, 4, 8, 16, 128

WARNINGS = [
    "Candidate landing zones: screening only. Ground or air reconnaissance is required before use.",
    "Not detectable: wires, cables, poles, antennas, loose debris and any obstacle lower than the object threshold.",
    "Surface condition (soft ground, water, snow, dust, rotor-wash debris) is not assessed.",
    "Approach corridors are checked to a fixed length only; terrain rising further out is not checked.",
    "Measured on Swiss LiDAR (docs/landing_zones_results.md): about 1 in 4 approach bearings shown CLEAR were blocked by "
    "an obstacle the model under-reads, so treat bearings as candidates for aerial reconnaissance, not as cleared paths.",
]


def _jacobian(t: Affine) -> np.ndarray:
    return np.array([[t.a, t.b], [t.d, t.e]], dtype=np.float64)


def pad_kernel(transform: Affine, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Disk of radius R in CRS metres as a pixel-offset mask. Returns (K, dx, dy, k): K is (2k+1)^2 bool, and dx, dy
    are the world offsets of each pixel offset (dcol, drow) from the centre pixel. The mask is centrally symmetric."""
    J = _jacobian(transform)
    smin = float(np.linalg.svd(J, compute_uv=False)[-1])
    k = int(math.ceil(radius_m / smin))
    dc, dr = np.meshgrid(np.arange(-k, k + 1, dtype=np.float64), np.arange(-k, k + 1, dtype=np.float64))
    dx = J[0, 0] * dc + J[0, 1] * dr
    dy = J[1, 0] * dc + J[1, 1] * dr
    K = dx * dx + dy * dy <= radius_m * radius_m * (1.0 + 1e-12)
    return K, dx, dy, k


def _correlate(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out[p] = sum_o w[o] a[p + o] for an odd-sized kernel w centred on the offset (0, 0)."""
    return fftconvolve(a, w[::-1, ::-1], mode="same")


class _Correlator:
    """The same correlation as _correlate, for several images and kernels on ONE zero-padded FFT grid, so every image
    and every kernel is transformed once (screen() needs 6-8 correlations of 4-5 inputs with 3-4 kernels). With
    kernel w of size 2k+1 and flipped wf, the linear convolution satisfies full[p + k] = sum_o w[o] a[p + o];
    the grid is at least (H + 2 kmax) x (W + 2 kmax), so nothing wraps around."""

    def __init__(self, shape: tuple[int, int], kmax: int):
        self.h, self.w = shape
        self.s = (next_fast_len(self.h + 2 * kmax, real=True), next_fast_len(self.w + 2 * kmax, real=True))
        self._kernels: dict[str, tuple[np.ndarray, int]] = {}

    def data(self, a: np.ndarray) -> np.ndarray:
        return rfft2(a.astype(np.float64), s=self.s, workers=_FFT_WORKERS)

    def add_kernel(self, name: str, w: np.ndarray) -> None:
        self._kernels[name] = (rfft2(w[::-1, ::-1].astype(np.float64), s=self.s, workers=_FFT_WORKERS), w.shape[0] // 2)

    def corr(self, A: np.ndarray, name: str) -> np.ndarray:
        W, k = self._kernels[name]
        return irfft2(A * W, s=self.s, workers=_FFT_WORKERS)[k:k + self.h, k:k + self.w]


def count_in_pad(mask: np.ndarray, K: np.ndarray, k: int, *, outside: bool) -> np.ndarray:
    """Number of mask cells inside the disk around every centre. Cells beyond the raster count as `outside`."""
    m = np.pad(mask.astype(np.float64), k, constant_values=1.0 if outside else 0.0)
    return np.rint(_correlate(m, K.astype(np.float64))[k:-k, k:-k]).astype(np.int64)


def plane_fields(z: np.ndarray, valid: np.ndarray, K: np.ndarray, dx: np.ndarray, dy: np.ndarray, corr: _Correlator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares plane over the disk around every centre: (a, b, c, rms). a = dz/dx (east), b = dz/dy (north),
    c = plane elevation at the centre. The fit is meaningful only where the disk contains no invalid cell.
    corr: a correlator that already holds kernel "K" (= K); one is built otherwise."""
    zref = float(np.median(z[valid])) if valid.any() else 0.0
    zc = np.where(valid, z - zref, 0.0)
    Kf = K.astype(np.float64)
    n = float(Kf.sum())
    wx, wy = Kf * dx, Kf * dy
    sxx, syy, sxy = float((wx * dx).sum()), float((wy * dy).sum()), float((wx * dy).sum())
    det = sxx * syy - sxy * sxy
    if corr is None:
        corr = _Correlator(z.shape, K.shape[0] // 2)
        corr.add_kernel("K", Kf)
    corr.add_kernel("Kx", wx)
    corr.add_kernel("Ky", wy)
    Z, Z2 = corr.data(zc), corr.data(zc * zc)
    s_z, s_xz, s_yz, s_zz = corr.corr(Z, "K"), corr.corr(Z, "Kx"), corr.corr(Z, "Ky"), corr.corr(Z2, "K")
    a = (syy * s_xz - sxy * s_yz) / det
    b = (sxx * s_yz - sxy * s_xz) / det
    c = s_z / n
    rss = np.maximum(s_zz - c * s_z - a * s_xz - b * s_yz, 0.0)
    return a, b, c + zref, np.sqrt(rss / n)


def slope_limit_deg(size: int, user_max_deg: float | None) -> float:
    lim = HLZ_SLOPE_ADVISORY_DEG if size >= HLZ_ADVISORY_MIN_SIZE else HLZ_SLOPE_ALL_DEG
    return lim if user_max_deg is None else min(lim, float(user_max_deg))


def _pooled(H: np.ndarray, invalid: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool heights and any-pool invalidity by f x f blocks (conservative), then dilate one cell (3 x 3) so that a
    sampling lattice with spacing <= one cell in pixel space sees every cell that intersects the corridor."""
    if f > 1:
        h, w = H.shape
        ph, pw = -h % f, -w % f
        H = np.pad(H, ((0, ph), (0, pw)), constant_values=-np.inf).reshape((h + ph) // f, f, (w + pw) // f, f).max(axis=(1, 3))
        invalid = np.pad(invalid, ((0, ph), (0, pw)), constant_values=True).reshape((h + ph) // f, f, (w + pw) // f, f).any(axis=(1, 3))
    return grey_dilation(H, size=(3, 3), mode="constant", cval=-np.inf), binary_dilation(invalid, structure=np.ones((3, 3), bool))


def approach_status(Hd: np.ndarray, inv_d: np.ndarray, transform_c: Affine, x0: float, y0: float, z_pad: float, radius_m: float, width_m: float, *, length_m: float = HLZ_APPROACH_LENGTH_M, ratio: float = HLZ_OBSTACLE_RATIO, n_bearings: int = HLZ_BEARINGS) -> list[dict[str, Any]]:
    """Per-bearing corridor check on the pooled/dilated height grid Hd (transform_c: its pixel -> CRS affine).
    CLEAR: every sampled cell satisfies H - Z_pad <= s / ratio, and the whole corridor is inside the grid with data.
    BLOCKED: some cell violates (the first violating distance and its height above the pad are reported).
    UNVERIFIED: no violation among the observed cells, but part of the corridor is outside the grid or has no data."""
    Ji = np.linalg.inv(_jacobian(transform_c))
    step = float(np.linalg.svd(_jacobian(transform_c), compute_uv=False)[-1])
    s = np.linspace(radius_m, radius_m + length_m, int(math.ceil(length_m / step)) + 1)
    u = np.linspace(-width_m / 2.0, width_m / 2.0, int(math.ceil(width_m / step)) + 1)
    allowed = s / ratio
    rows_n, cols_n = Hd.shape
    out = []
    for i in range(n_bearings):
        th = 2.0 * math.pi * i / n_bearings  # clockwise from grid north (+y)
        ex, ey, nx, ny = math.sin(th), math.cos(th), math.cos(th), -math.sin(th)
        X = x0 + s[:, None] * ex + u[None, :] * nx - transform_c.c
        Y = y0 + s[:, None] * ey + u[None, :] * ny - transform_c.f
        col = np.floor(Ji[0, 0] * X + Ji[0, 1] * Y).astype(np.int64)
        row = np.floor(Ji[1, 0] * X + Ji[1, 1] * Y).astype(np.int64)
        inside = (col >= 0) & (col < cols_n) & (row >= 0) & (row < rows_n)
        cc, rr = np.clip(col, 0, cols_n - 1), np.clip(row, 0, rows_n - 1)
        rel = np.where(inside, Hd[rr, cc] - z_pad, -np.inf)
        viol = rel > allowed[:, None]
        hit = np.flatnonzero(viol.any(axis=1))
        bearing = round(360.0 * i / n_bearings, 1)
        if hit.size:
            j = int(hit[0])
            out.append({"bearingDeg": bearing, "status": "BLOCKED", "firstObstacle": {"distanceM": round(float(s[j]), 1), "heightAbovePadM": round(float(rel[j][viol[j]].max()), 1)}})
        elif not inside.all() or bool(inv_d[rr, cc][inside].any()):
            out.append({"bearingDeg": bearing, "status": "UNVERIFIED", "firstObstacle": None})
        else:
            out.append({"bearingDeg": bearing, "status": "CLEAR", "firstObstacle": None})
    return out


def _read(path: Path) -> tuple[np.ndarray, Affine, Any]:
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return a, ds.transform, ds.crs


def screen(dsm: np.ndarray, ndsm: np.ndarray, buildings: np.ndarray | None, wet: np.ndarray | None, transform: Affine, *, size: int, object_threshold_m: float, object_sigma_m: float, max_slope_deg: float | None = None, roughness_max_m: float = HLZ_ROUGHNESS_MAX_M, object_buffer_m: float = HLZ_OBJECT_BUFFER_M, slope_margin_deg: float = HLZ_SLOPE_MARGIN_DEG) -> dict[str, Any]:
    """Pure-array core (no I/O). Returns the reason raster, the plane fields and the ranked, corridor-checked sites."""
    if size not in HLZ_SIZES:
        raise ValueError(f"unknown landing-point size {size}; expected one of {sorted(HLZ_SIZES)}")
    D = HLZ_SIZES[size]["diameter_m"]
    R = D / 2.0
    J = _jacobian(transform)
    px = math.sqrt(abs(float(np.linalg.det(J))))
    if D / px < HLZ_MIN_PAD_PIXELS_ACROSS:
        raise ValueError(f"pixel size {px:.2f} m is too coarse for a {D:.0f} m pad (needs >= {HLZ_MIN_PAD_PIXELS_ACROSS} pixels across)")
    if max_slope_deg is not None and not (math.isfinite(max_slope_deg) and 0.0 < max_slope_deg <= HLZ_SLOPE_ADVISORY_DEG):
        raise ValueError(f"max slope must be in (0, {HLZ_SLOPE_ADVISORY_DEG:g}] degrees")
    slope_max = slope_limit_deg(size, max_slope_deg)
    t_all, t_max = HLZ_SLOPE_ALL_DEG - slope_margin_deg, slope_max - slope_margin_deg  # measured-slope thresholds

    valid = np.isfinite(dsm) & np.isfinite(ndsm)
    obj = valid & (np.where(valid, ndsm, 0.0) > object_threshold_m)
    if buildings is not None:
        obj |= buildings
    K, dx, dy, k = pad_kernel(transform, R)
    Ko, _, _, ko = pad_kernel(transform, R + object_buffer_m)  # detected obstacles keep a horizontal buffer
    corr = _Correlator(dsm.shape, max(k, ko))
    corr.add_kernel("K", K.astype(np.float64))
    corr.add_kernel("Ko", Ko.astype(np.float64))
    # invalid-or-outside cells in the disk = disk size - valid cells inside the raster (zero padding = outside)
    n_inv = int(K.sum()) - np.rint(corr.corr(corr.data(valid), "K")).astype(np.int64)
    n_obj = np.rint(corr.corr(corr.data(obj), "Ko")).astype(np.int64)
    n_wet = np.rint(corr.corr(corr.data(wet), "K")).astype(np.int64) if wet is not None else np.zeros_like(n_inv)
    a, b, c, rms = plane_fields(dsm, valid, K, dx, dy, corr)
    slope = np.degrees(np.arctan(np.hypot(a, b)))

    ok_data = n_inv == 0
    reasons = np.where(ok_data, 0, R_INVALID).astype(np.uint8)
    reasons |= np.where(n_obj > 0, R_OBJECT, 0).astype(np.uint8)
    reasons |= np.where(n_wet > 0, R_WET, 0).astype(np.uint8)
    reasons |= np.where(ok_data & (slope > t_max), R_SLOPE, 0).astype(np.uint8)
    reasons |= np.where(ok_data & (rms > roughness_max_m), R_ROUGH, 0).astype(np.uint8)
    feasible = reasons == 0
    reasons[feasible] = R_FEASIBLE

    # corridor heights: DSM, objects raised by k * sigma; pooled to ~HLZ_CORRIDOR_CELL_M (max) and dilated one cell
    # object cells are also spread horizontally by the buffer (square max filter on the pooled grid: conservative)
    f = max(1, int(HLZ_CORRIDOR_CELL_M // px))
    Hd, inv_d = _pooled(np.where(valid, dsm, -np.inf), ~valid, f)
    Ho, _ = _pooled(np.where(obj & valid, dsm + HLZ_OBJECT_MARGIN_SIGMA * object_sigma_m, -np.inf), np.zeros_like(valid), f)
    nb = int(math.ceil(object_buffer_m / (f * px)))
    if nb > 0:
        Ho = maximum_filter(Ho, size=2 * nb + 1, mode="constant", cval=-np.inf)
    Hd = np.maximum(Hd, Ho)
    t_c = transform @ Affine.scale(f)
    width = HLZ_CORRIDOR_WIDTH_FACTOR * D

    # candidate centres on a lattice, ranked (slope band, slope, roughness), non-maximum suppression at spacing D
    st = max(1, int(round(HLZ_SITE_LATTICE_FRACTION * D / px)))
    lat = feasible[::st, ::st]
    li, lj = np.nonzero(lat)
    rows, cols = li * st, lj * st
    band = (slope[rows, cols] > t_all).astype(np.int8)
    order = np.lexsort((rms[rows, cols], slope[rows, cols], band))
    Kb, _, _, kb = pad_kernel(transform @ Affine.scale(st), D * (1.0 - 1e-9))  # lattice offsets closer than D
    blocked = np.zeros(lat.shape, dtype=bool)
    sites: list[dict[str, Any]] = []
    n_blocked = n_eval = 0
    for idx in order:
        i, j = int(li[idx]), int(lj[idx])
        if blocked[i, j]:
            continue
        i0, i1, j0, j1 = max(0, i - kb), min(lat.shape[0], i + kb + 1), max(0, j - kb), min(lat.shape[1], j + kb + 1)
        blocked[i0:i1, j0:j1] |= Kb[i0 - i + kb:i1 - i + kb, j0 - j + kb:j1 - j + kb]
        r, cidx = int(rows[idx]), int(cols[idx])
        x0, y0 = transform @ (cidx + 0.5, r + 0.5)
        appr = approach_status(Hd, inv_d, t_c, x0, y0, float(c[r, cidx]), R, width)
        n_eval += 1
        st_set = {q["status"] for q in appr}
        if st_set == {"BLOCKED"}:
            n_blocked += 1
        else:
            sl = float(slope[r, cidx])
            flags = []
            if sl > t_all:
                flags.append("UPSLOPE_ADVISORY")
            if "UNVERIFIED" in st_set:
                flags.append("APPROACH_UNVERIFIED_BEYOND_SCENE")
            cls = "CANDIDATE" if sl <= t_all and "CLEAR" in st_set else "MARGINAL"
            ga, gb = float(a[r, cidx]), float(b[r, cidx])
            sites.append({
                "class": cls, "row": r, "col": cidx, "x": round(x0, 2), "y": round(y0, 2),
                "elevationM": round(float(c[r, cidx]), 2), "slopeDeg": round(sl, 2), "slopePct": round(100.0 * math.hypot(ga, gb), 1),
                "upslopeBearingDeg": round((math.degrees(math.atan2(ga, gb)) + 360.0) % 360.0, 1) if math.hypot(ga, gb) > 1e-6 else None,
                "roughnessM": round(float(rms[r, cidx]), 2), "approaches": appr,
                "clearBearings": [q["bearingDeg"] for q in appr if q["status"] == "CLEAR"], "flags": flags,
            })
        if len(sites) >= HLZ_MAX_SITES or n_eval >= HLZ_MAX_EVALUATED:
            break
    sites.sort(key=lambda s: (s["class"] != "CANDIDATE", s["slopeDeg"], s["roughnessM"]))
    for n, s_ in enumerate(sites, 1):
        s_["id"] = n
    return {"reasons": reasons, "feasible": feasible, "sites": sites, "blockedSites": n_blocked, "evaluated": n_eval, "slopeMaxDeg": slope_max, "diameterM": D, "objectMask": obj, "valid": valid, "slope": slope, "roughness": rms, "planeZ": c}


def run_landing_zone_screening(job_dir: Path, result: dict[str, Any], *, size: int, max_slope_deg: float | None = None, exclude_flooded: bool = False) -> dict[str, Any]:
    tier = result.get("calibration_tier")
    if tier not in ("T", "A") or not result.get("metric"):
        raise ValueError("Landing-zone screening needs metric terrain and heights (calibration tier T or A); this job is " + str(tier))
    unc = (result.get("uncertainty") or {}).get("ndsm") or {}
    if not isinstance(unc.get("object_m"), (int, float)) or not isinstance(unc.get("object_threshold_m"), (int, float)):
        raise ValueError("This job has no measured object-height uncertainty; reprocess it to enable landing-zone screening")
    for name in ("dsm.tif", "ndsm.tif"):
        if not (job_dir / name).exists():
            raise ValueError(f"Landing-zone screening requires {name}")
    dsm, transform, crs = _read(job_dir / "dsm.tif")
    ndsm, _, _ = _read(job_dir / "ndsm.tif")

    from core.disaster.flood import _footprint_sets  # one footprint definition for every hazard product

    bpath = job_dir / "buildings.json"
    b_data = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else {}
    lab, footprint_method = _footprint_sets(job_dir, b_data, dsm.shape)
    buildings = lab > 0 if lab is not None else None
    wet = None
    if exclude_flooded:
        if not (job_dir / "flood_depth.tif").exists():
            raise ValueError("Run flood screening first to exclude flooded ground")
        depth, _, _ = _read(job_dir / "flood_depth.tif")
        wet = np.nan_to_num(depth, nan=0.0) > 0.0

    out = screen(dsm, ndsm, buildings, wet, transform, size=size, object_threshold_m=float(unc["object_threshold_m"]), object_sigma_m=float(unc["object_m"]), max_slope_deg=max_slope_deg)
    D = out["diameterM"]
    posting = (result.get("dem") or {}).get("posting_m")
    coarse = isinstance(posting, (int, float)) and posting > D
    vcrs = ((result.get("layers") or {}).get("dsm") or {}).get("vertical_crs") or result.get("vertical_reference")
    lonlat = None
    if crs is not None:
        from pyproj import Transformer

        lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    inv = ~transform
    R = D / 2.0
    for s in out["sites"]:
        if coarse:
            s["flags"].append("TERRAIN_COARSER_THAN_PAD")
        # 2D-map geometry in pixel coordinates (exact for any affine grid): pad outline and non-blocked corridors
        s["padPixelRing"] = [[round(v, 2) for v in inv @ (s["x"] + R * math.sin(t), s["y"] + R * math.cos(t))] for t in np.linspace(0.0, 2.0 * math.pi, 33)]
        s["approachPixelLines"] = [
            {"bearingDeg": a["bearingDeg"], "status": a["status"], "line": [[round(v, 2) for v in inv @ (s["x"] + d * math.sin(math.radians(a["bearingDeg"])), s["y"] + d * math.cos(math.radians(a["bearingDeg"])))] for d in (R, R + HLZ_APPROACH_LENGTH_M)]}
            for a in s["approaches"] if a["status"] != "BLOCKED"
        ]
        if lonlat is not None:
            lon, lat = lonlat.transform(s["x"], s["y"])
            s["lon"], s["lat"] = round(lon, 7), round(lat, 7)

    grid = Grid(width=dsm.shape[1], height=dsm.shape[0], transform=transform, crs=crs.to_string() if crs else None, dtype="uint8", nodata=0, units="metres", metric=True, vertical_reference=None, tier=tier)
    write_raster(job_dir / "landing_feasible.tif", out["reasons"], grid, tags={"KIND": "hlz_reason_bitmask", "BITS": "SLOPE=1,OBJECT_IN_PAD=2,ROUGH=4,WET_IN_PAD=8,INVALID_OR_EDGE=16,FEASIBLE=128", "PAD_DIAMETER_M": repr(D), "WARNING": WARNINGS[0]}, dtype="uint8")
    _preview(job_dir / "landing_preview.png", out["feasible"], out["objectMask"], out["valid"])
    geo = _geojson(out["sites"], transform, lonlat, D) if lonlat is not None else None
    if geo is not None:
        (job_dir / "landing_zones.geojson").write_text(json.dumps(geo), encoding="utf-8")

    warnings = list(WARNINGS)
    warnings.insert(2, f"Object threshold for this job: {unc['object_threshold_m']:g} m. Obstacle heights carry a {HLZ_OBJECT_MARGIN_SIGMA:g}-sigma margin of {unc['object_m']:g} m.")
    if coarse:
        warnings.append(f"The terrain DEM posting ({posting:g} m) is coarser than the pad ({D:g} m): slope below that scale comes from the image model, not a measured terrain model.")
    if not exclude_flooded and (job_dir / "flood_depth.tif").exists():
        warnings.append("A flood result exists but flooded ground was not excluded.")
    a_px = abs(transform.a * transform.e - transform.b * transform.d)
    summary = {
        "scenario": "landing_zones",
        "method": METHOD,
        "size": size,
        "padDiameterM": D,
        "rules": {
            "slopeAllDeg": HLZ_SLOPE_ALL_DEG, "slopeAdvisoryDeg": HLZ_SLOPE_ADVISORY_DEG, "slopeMaxDegApplied": out["slopeMaxDeg"],
            "obstacleRatio": HLZ_OBSTACLE_RATIO, "approachLengthM": HLZ_APPROACH_LENGTH_M, "corridorWidthM": HLZ_CORRIDOR_WIDTH_FACTOR * D,
            "bearings": HLZ_BEARINGS, "objectThresholdM": float(unc["object_threshold_m"]), "objectMarginM": HLZ_OBJECT_MARGIN_SIGMA * float(unc["object_m"]),
            "roughnessMaxM": HLZ_ROUGHNESS_MAX_M, "objectBufferM": HLZ_OBJECT_BUFFER_M, "slopeMarginDeg": HLZ_SLOPE_MARGIN_DEG, "source": "US Army FM 3-21.38 (2006) ch. 4: sizes 4-3.d, slope 4-1.d, 10:1 obstacle ratio 4-1.i",
        },
        "sizes": [{"size": k, **v} for k, v in HLZ_SIZES.items()],
        "feasibleCentreAreaM2": round(int(out["feasible"].sum()) * a_px, 1),
        "nSites": len(out["sites"]),
        "nCandidate": sum(1 for s in out["sites"] if s["class"] == "CANDIDATE"),
        "blockedSites": out["blockedSites"],
        "evaluatedSites": out["evaluated"],
        "sites": out["sites"],
        "excludeFlooded": bool(exclude_flooded),
        "footprintMethod": footprint_method,
        "verticalCrs": vcrs,
        "crs": crs.to_string() if crs else None,
        "rasterResult": "landing_feasible.tif",
        "previewResult": "landing_preview.png",
        "vectorResult": "landing_zones.geojson" if geo is not None else None,
        "quantityCategory": {"slopeDeg": "DERIVED", "roughnessM": "DERIVED", "elevationM": "DERIVED", "approaches": "SCENARIO", "class": "SCENARIO"},
        "warnings": warnings,
    }
    (job_dir / "disaster_landing_zones.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _geojson(sites: list[dict[str, Any]], transform: Affine, lonlat: Any, D: float) -> dict[str, Any]:
    """RFC 7946 (WGS84 lon/lat): per site a pad polygon (64-gon, drawn in CRS metres then transformed), a centre point
    and one centre line per CLEAR bearing from the pad edge to the checked length."""
    R = D / 2.0
    feats = []
    ang = np.linspace(0.0, 2.0 * math.pi, 65)
    for s in sites:
        props = {k: v for k, v in s.items() if k not in ("approaches", "padPixelRing", "approachPixelLines")}
        ring = [list(lonlat.transform(s["x"] + R * math.sin(t), s["y"] + R * math.cos(t))) for t in ang]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {**props, "kind": "pad"}})
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": list(lonlat.transform(s["x"], s["y"]))}, "properties": {**props, "kind": "centre"}})
        for bdeg in s["clearBearings"]:
            t = math.radians(bdeg)
            p0 = lonlat.transform(s["x"] + R * math.sin(t), s["y"] + R * math.cos(t))
            p1 = lonlat.transform(s["x"] + (R + HLZ_APPROACH_LENGTH_M) * math.sin(t), s["y"] + (R + HLZ_APPROACH_LENGTH_M) * math.cos(t))
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [list(p0), list(p1)]}, "properties": {"kind": "approach", "site": s["id"], "bearingDeg": bdeg}})
    return {"type": "FeatureCollection", "features": feats}


def _preview(path: Path, feasible: np.ndarray, obj: np.ndarray, valid: np.ndarray) -> None:
    rgba = np.zeros((*feasible.shape, 4), dtype=np.uint8)
    rgba[obj & valid] = (90, 90, 90, 90)
    rgba[feasible] = (29, 122, 79, 120)
    Image.fromarray(rgba, "RGBA").save(path)
