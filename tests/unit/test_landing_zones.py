"""Golden tests for helicopter landing-zone screening (core/disaster/landing_zones.py, docs/landing_zones.md)."""
import json
import math

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from core.disaster.landing_zones import (
    R_FEASIBLE,
    R_INVALID,
    R_OBJECT,
    R_SLOPE,
    R_WET,
    _pooled,
    approach_status,
    count_in_pad,
    pad_kernel,
    plane_fields,
    run_landing_zone_screening,
    screen,
)
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.screening_params import HLZ_SIZES

X0, Y0 = 2_600_000.0, 1_200_000.0  # large projected coordinates on purpose


def north_up(px=0.5, py=None):
    return Affine.translation(X0, Y0) @ Affine.scale(px, -(py or px))


GRIDS = {
    "north_up": north_up(0.5),
    "rotated_30": Affine.translation(X0, Y0) @ Affine.rotation(30) @ Affine.scale(0.5, -0.5),
    "sheared": Affine.translation(X0, Y0) @ Affine.shear(12, 0) @ Affine.scale(0.5, -0.5),
    "non_square": north_up(0.5, 0.7),
}


def world_xy(t, shape):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    cx, cy = cols + 0.5, rows + 0.5
    return t.a * cx + t.b * cy + t.c, t.d * cx + t.e * cy + t.f


# ------------------------------------------------------------------ kernel and disk counts
@pytest.mark.parametrize("name", list(GRIDS))
def test_pad_kernel_is_round_in_world_metres_and_symmetric(name):
    t = GRIDS[name]
    K, dx, dy, _ = pad_kernel(t, 12.5)
    assert np.array_equal(K, K[::-1, ::-1])
    assert np.all(np.hypot(dx, dy)[K] <= 12.5 + 1e-9) and np.all(np.hypot(dx, dy)[~K] > 12.5)
    a_px = abs(t.a * t.e - t.b * t.d)
    assert abs(K.sum() * a_px / (math.pi * 12.5 ** 2) - 1) < 0.02


@pytest.mark.parametrize("name", list(GRIDS))
def test_fft_disk_count_equals_direct_count(name):
    t = GRIDS[name]
    rng = np.random.default_rng(1)
    m = rng.random((40, 45)) < 0.05
    K, _, _, k = pad_kernel(t, 3.0)
    got = count_in_pad(m, K, k, outside=True)
    pad = np.pad(m, k, constant_values=True)
    for r, c in [(0, 0), (20, 22), (39, 44), (5, 30), (k, k)]:
        assert got[r, c] == int(pad[r:r + 2 * k + 1, c:c + 2 * k + 1][K].sum())


# ------------------------------------------------------------------ plane fit
@pytest.mark.parametrize("name", list(GRIDS))
@pytest.mark.parametrize("offset", [0.0, 3000.0])
def test_plane_fit_is_exact_on_world_planes(name, offset):
    t = GRIDS[name]
    X, Y = world_xy(t, (90, 90))
    gx, gy = 0.07, -0.04
    z = offset + 400.0 + gx * (X - X0) + gy * (Y - Y0)
    K, dx, dy, k = pad_kernel(t, 8.0)
    a, b, c, rms = plane_fields(z, np.isfinite(z), K, dx, dy)
    sl = np.s_[k + 1:-k - 1, k + 1:-k - 1]
    assert np.abs(a[sl] - gx).max() < 1e-9 and np.abs(b[sl] - gy).max() < 1e-9
    assert np.abs(c[sl] - z[sl]).max() < 1e-6
    assert rms[sl].max() < 1e-5


def test_plane_fit_matches_lstsq_on_random_windows():
    t = GRIDS["rotated_30"]
    rng = np.random.default_rng(7)
    z = 3500.0 + np.cumsum(np.cumsum(rng.normal(0, 0.05, (80, 80)), 0), 1)
    K, dx, dy, k = pad_kernel(t, 6.0)
    a, b, c, rms = plane_fields(z, np.ones_like(z, bool), K, dx, dy)
    for _ in range(100):
        r, cc = rng.integers(k, 80 - k, 2)
        w = z[r - k:r + k + 1, cc - k:cc + k + 1][K]
        M = np.column_stack([dx[K], dy[K], np.ones(K.sum())])
        (ea, eb, ec), *_ = np.linalg.lstsq(M, w, rcond=None)
        erms = math.sqrt(np.mean((M @ [ea, eb, ec] - w) ** 2))
        assert abs(a[r, cc] - ea) < 1e-6 and abs(b[r, cc] - eb) < 1e-6 and abs(c[r, cc] - ec) < 1e-6 and abs(rms[r, cc] - erms) < 1e-6


# ------------------------------------------------------------------ pad clearance, slope rules
def flat(n=150, px=1.0, z=100.0):
    t = north_up(px)
    return np.full((n, n), z), np.zeros((n, n)), t


def run(dsm, ndsm, t, size=1, **kw):
    """Doctrine geometry only: the measured buffer and slope margin are switched off unless a test sets them."""
    kw.setdefault("object_buffer_m", 0.0)
    kw.setdefault("slope_margin_deg", 0.0)
    return screen(dsm, ndsm, kw.pop("buildings", None), kw.pop("wet", None), t, size=size, object_threshold_m=2.5, object_sigma_m=5.5, **kw)


def test_object_in_pad_exact_radius():
    dsm, ndsm, t = flat()
    b = np.zeros(dsm.shape, bool)
    b[75, 75] = True
    rs = run(dsm, ndsm, t, buildings=b)["reasons"]  # size 1: R = 12.5 m = 12.5 px
    assert rs[75, 75 + 12] & R_OBJECT and not rs[75, 75 + 13] & R_OBJECT
    ndsm2 = ndsm.copy()
    ndsm2[75, 75] = 3.0  # an nDSM object above the 2.5 m threshold acts the same
    assert run(dsm, ndsm2, t)["reasons"][75, 87] & R_OBJECT
    ndsm2[75, 75] = 2.4  # below the threshold: not detectable, by design
    assert not run(dsm, ndsm2, t)["reasons"][75, 87] & R_OBJECT


@pytest.mark.parametrize("name", list(GRIDS))
def test_shared_fft_counts_equal_the_reference_counts(name):
    """screen() counts invalid/outside and object cells on one shared FFT grid; they must equal count_in_pad."""
    t = GRIDS[name]
    rng = np.random.default_rng(5)
    n = 90
    dsm = 100.0 + rng.normal(0, 0.01, (n, n))
    dsm[rng.random((n, n)) < 0.002] = np.nan
    nd = np.where(rng.random((n, n)) < 0.002, 5.0, 0.0)
    out = run(dsm, nd, t, object_buffer_m=4.0, roughness_max_m=np.inf)
    valid = np.isfinite(dsm)
    K, _, _, k = pad_kernel(t, 12.5)
    Ko, _, _, ko = pad_kernel(t, 16.5)
    ref_inv = count_in_pad(~valid, K, k, outside=True) > 0
    ref_obj = count_in_pad(valid & (nd > 2.5), Ko, ko, outside=False) > 0
    assert np.array_equal((out["reasons"] & R_INVALID) > 0, ref_inv)
    assert np.array_equal((out["reasons"] & R_OBJECT) > 0, ref_obj)


def test_obstacle_buffer_widens_the_clear_radius_exactly():
    dsm, ndsm, t = flat(n=200)
    b = np.zeros(dsm.shape, bool)
    b[100, 100] = True
    rs = run(dsm, ndsm, t, buildings=b, object_buffer_m=10.0)["reasons"]  # 12.5 + 10 = 22.5 m
    assert rs[100, 100 + 22] & R_OBJECT and not rs[100, 100 + 23] & R_OBJECT


def test_slope_margin_tightens_both_doctrine_bands():
    n = 160
    t = north_up(1.0)
    X, _ = world_xy(t, (n, n))
    for deg, size, feasible in [(4.9, 1, True), (5.1, 1, False), (12.9, 3, True), (13.1, 3, False)]:
        dsm = 500.0 + math.tan(math.radians(deg)) * (X - X0)
        out = run(dsm, np.zeros_like(dsm), t, size=size, slope_margin_deg=2.0)
        assert bool(out["reasons"][n // 2, n // 2] == R_FEASIBLE) is feasible, (deg, size)
    dsm = 500.0 + math.tan(math.radians(6.0)) * (X - X0)  # 6 deg: MARGINAL (upslope advisory) for size 3 with the margin
    assert all("UPSLOPE_ADVISORY" in s["flags"] for s in run(dsm, np.zeros_like(dsm), t, size=3, slope_margin_deg=2.0)["sites"])


def test_nodata_edge_and_wet_rules():
    dsm, ndsm, t = flat()
    dsm[75, 75] = np.nan
    wet = np.zeros(dsm.shape, bool)
    wet[40, 40] = True
    rs = run(dsm, ndsm, t)["reasons"]
    assert rs[75, 80] & R_INVALID and rs[5, 75] & R_INVALID and rs[40, 45] == R_FEASIBLE
    assert run(dsm, ndsm, t, wet=wet)["reasons"][40, 45] & R_WET


@pytest.mark.parametrize("deg,size,expect_feasible,advisory", [
    (6.9, 1, True, False), (7.1, 1, False, False), (7.1, 3, True, True), (14.9, 4, True, True), (15.1, 4, False, False),
])
def test_doctrine_slope_bands(deg, size, expect_feasible, advisory):
    n = 160
    t = north_up(1.0)
    X, Y = world_xy(t, (n, n))
    dsm = 500.0 + math.tan(math.radians(deg)) * (X - X0)
    out = run(dsm, np.zeros_like(dsm), t, size=size)
    c = n // 2
    assert bool(out["reasons"][c, c] == R_FEASIBLE) is expect_feasible
    if not expect_feasible:
        assert out["reasons"][c, c] & R_SLOPE
    if expect_feasible:
        assert all(("UPSLOPE_ADVISORY" in s["flags"]) is advisory for s in out["sites"])


def test_sites_do_not_overlap_and_small_scene_is_never_clear():
    dsm, ndsm, t = flat(n=120, px=0.5)
    out = run(dsm, ndsm, t)
    D = HLZ_SIZES[1]["diameter_m"]
    xy = np.array([[s["x"], s["y"]] for s in out["sites"]])
    assert len(xy) >= 2
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(xy), 1)].min() >= D - 1e-6
    # a 60 m scene cannot contain a 300 m corridor: nothing may be CLEAR, every site is MARGINAL
    assert all(a["status"] != "CLEAR" for s in out["sites"] for a in s["approaches"])
    assert all(s["class"] == "MARGINAL" and "APPROACH_UNVERIFIED_BEYOND_SCENE" in s["flags"] for s in out["sites"])


# ------------------------------------------------------------------ approach corridors
def corridor(dsm, ndsm_obj_margin=None, px=1.0):
    t = north_up(px)
    H = dsm if ndsm_obj_margin is None else dsm + ndsm_obj_margin
    Hd, inv = _pooled(np.where(np.isfinite(H), H, -np.inf), ~np.isfinite(H), 1)
    n = dsm.shape[0]
    x0, y0 = t @ (n // 2 + 0.5, n // 2 + 0.5)
    return {a["bearingDeg"]: a for a in approach_status(Hd, inv, t, x0, y0, float(dsm[n // 2, n // 2]), 12.5, 25.0)}


@pytest.mark.parametrize("h,blocked", [(10.6, True), (9.4, False)])
def test_post_obeys_ten_to_one_from_touchdown_point(h, blocked):
    n = 800
    dsm = np.full((n, n), 100.0)
    dsm[n // 2 - 100, n // 2] = 100.0 + h  # 100 m north of the centre: allowed height 10 m
    res = corridor(dsm)
    assert (res[0.0]["status"] == "BLOCKED") is blocked
    assert all(a["status"] == "CLEAR" for b, a in res.items() if b != 0.0)
    if blocked:
        assert 97.0 <= res[0.0]["firstObstacle"]["distanceM"] <= 100.5


def test_terrain_wall_blocks_only_toward_the_wall():
    n = 800
    t = north_up(1.0)
    X, Y = world_xy(t, (n, n))
    y_c = (t @ (0, n // 2 + 0.5))[1]
    dsm = 100.0 + 0.2 * np.maximum(0.0, (Y - y_c) - 50.0)  # 20 % wall starting 50 m north
    res = corridor(dsm)
    assert res[0.0]["status"] == "BLOCKED" and res[22.5]["status"] == "BLOCKED"
    assert res[90.0]["status"] == "CLEAR" and res[180.0]["status"] == "CLEAR"


@pytest.mark.parametrize("grade,up_blocked", [(0.05, False), (0.12, True)])
def test_uniform_slope_blocks_upslope_only_above_ten_percent(grade, up_blocked):
    n = 800
    t = north_up(1.0)
    _, Y = world_xy(t, (n, n))
    res = corridor(100.0 + grade * (Y - Y0))
    assert (res[0.0]["status"] == "BLOCKED") is up_blocked
    assert res[180.0]["status"] == "CLEAR"


def test_corridor_leaving_scene_or_nodata_is_unverified():
    n = 400
    dsm = np.full((n, n), 100.0)
    res = corridor(dsm)  # 200 m to the edge < 312.5 m
    assert {a["status"] for a in res.values()} == {"UNVERIFIED"}
    n = 800
    dsm = np.full((n, n), 100.0)
    dsm[n // 2 - 200, n // 2] = np.nan
    assert corridor(dsm)[0.0]["status"] == "UNVERIFIED"


# ------------------------------------------------------------------ invariants
def smooth_scene(n=200, seed=3):
    rng = np.random.default_rng(seed)
    t = north_up(0.5)
    X, Y = world_xy(t, (n, n))
    z = 300.0 + 0.03 * (X - X0) + 2.0 * np.sin((Y - Y0) / 15.0) + rng.normal(0, 0.05, (n, n))
    nd = np.zeros((n, n))
    nd[60:70, 120:135] = 8.0
    return z + nd, nd, t


def test_vertical_translation_invariance():
    z, nd, t = smooth_scene()
    a, b = run(z, nd, t), run(z + 3000.0, nd, t)
    assert np.array_equal(a["reasons"], b["reasons"])
    assert [(s["x"], s["y"], s["class"], s["slopeDeg"]) for s in a["sites"]] == [(s["x"], s["y"], s["class"], s["slopeDeg"]) for s in b["sites"]]
    assert all(abs(q["elevationM"] - p["elevationM"] - 3000.0) < 0.011 for p, q in zip(a["sites"], b["sites"]))


def test_rotating_the_raster_keeps_the_world_result():
    z, nd, t = smooth_scene()
    n = z.shape[0]
    t_rot = t @ Affine(0, -1, n, 1, 0, 0)  # world-identical grid for np.rot90 data
    a, b = run(z, nd, t), run(np.rot90(z), np.rot90(nd), t_rot)
    assert np.array_equal(np.rot90(a["reasons"]), b["reasons"])


# ------------------------------------------------------------------ job-level entry point
def make_job(tmp_path, tier="T", uncertainty=True, n=160):
    t = north_up(1.0)
    g = Grid(width=n, height=n, transform=t, crs="EPSG:32645", dtype="float32", nodata=-9999.0, units="metres", metric=True)
    X, _ = world_xy(t, (n, n))
    dsm = 1500.0 + 0.02 * (X - X0)
    write_raster(tmp_path / "dsm.tif", dsm, g)
    write_raster(tmp_path / "ndsm.tif", np.zeros((n, n)), g)
    result = {"calibration_tier": tier, "metric": tier != "R", "dem": {"posting_m": 30.0}, "layers": {"dsm": {"vertical_crs": "EGM2008"}}}
    if uncertainty:
        result["uncertainty"] = {"ndsm": {"object_m": 5.5, "object_threshold_m": 2.5}}
    return result


def test_job_entry_point_writes_artifacts(tmp_path):
    result = make_job(tmp_path)
    out = run_landing_zone_screening(tmp_path, result, size=1)
    assert out["nSites"] >= 1 and out["rules"]["obstacleRatio"] == 10.0
    assert all("TERRAIN_COARSER_THAN_PAD" in s["flags"] and "lat" in s for s in out["sites"])  # 30 m DEM > 25 m pad
    with rasterio.open(tmp_path / "landing_feasible.tif") as ds:
        assert set(np.unique(ds.read(1))) <= {R_FEASIBLE, R_INVALID}
    gj = json.loads((tmp_path / "landing_zones.geojson").read_text())
    assert {f["properties"]["kind"] for f in gj["features"]} >= {"pad", "centre"}
    assert (tmp_path / "landing_preview.png").exists() and (tmp_path / "disaster_landing_zones.json").exists()


@pytest.mark.parametrize("kw,msg", [({"tier": "R"}, "tier"), ({"uncertainty": False}, "uncertainty")])
def test_job_entry_point_refuses(tmp_path, kw, msg):
    result = make_job(tmp_path, **kw)
    with pytest.raises(ValueError, match=msg):
        run_landing_zone_screening(tmp_path, result, size=1)


def test_refuses_pad_too_small_for_pixels_and_bad_slope():
    dsm, ndsm, t = flat(n=60, px=3.0)  # 25 m / 3 m < 10 pixels
    with pytest.raises(ValueError, match="too coarse"):
        run(dsm, ndsm, t)
    dsm, ndsm, t = flat()
    with pytest.raises(ValueError, match="max slope"):
        run(dsm, ndsm, t, max_slope_deg=20.0)

