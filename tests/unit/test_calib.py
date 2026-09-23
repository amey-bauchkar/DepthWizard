"""Unit tests for the Mode B calibration core: terrain layer, object layer, scale-fit gates, tier decision, DEM discovery."""
from pathlib import Path

import numpy as np
import pytest

from core.calib.dem import _cop_tile_bounds, discover_dem
from core.calib.terrain import compose_dsm, fit_object_scale_to_dem_residual, object_layer_from_relative, terrain_layer
from core.calib.tier import decide

GSD = 2.0
F = 15  # 30 m / 2 m


def _plane(h=300, w=300, gx=0.02, gy=-0.01, z0=400.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return z0 + gx * xx * GSD + gy * yy * GSD


def test_terrain_layer_preserves_tilted_plane():
    """Order-1 normalized convolution must not tilt or flatten a sloped DEM even with one-sided ground support."""
    dem = _plane()
    rng = np.random.default_rng(0)
    ground = rng.random(dem.shape) < 0.5
    ground[:, :150] = False  # asymmetric support: no ground at all on the left half
    t = terrain_layer(dem, np.ones_like(dem, bool), ground, gsd_m=GSD)
    d = t.terrain - dem
    assert np.nanmax(np.abs(d)) < 0.05, f"plane not preserved: max |d| = {np.nanmax(np.abs(d)):.3f} m"
    assert t.stats["method"].startswith("order1")


def test_terrain_layer_removes_object_contaminated_cells_but_never_raises():
    """DEM cells with no ground support and a +10 m bump (canopy/roof) are lowered towards the ground plane;
    the correction never raises the DEM by more than max_raise_m (DEM is an upper bound)."""
    dem = _plane()
    bump = np.zeros_like(dem)
    r0, c0 = 7 * F, 8 * F
    bump[r0 : r0 + 2 * F, c0 : c0 + 2 * F] = 10.0  # 2x2 DEM cells fully covered by objects
    ground = np.ones_like(dem, bool)
    ground[r0 : r0 + 2 * F, c0 : c0 + 2 * F] = False
    t = terrain_layer(dem + bump, np.ones_like(dem, bool), ground, gsd_m=GSD)
    inside = t.terrain[r0 + F // 2 : r0 + F, c0 + F // 2 : c0 + F] - dem[r0 + F // 2 : r0 + F, c0 + F // 2 : c0 + F]
    assert np.nanmean(inside) < 3.0, f"bump not removed: mean residual {np.nanmean(inside):.2f} m"
    assert np.nanmax(t.terrain - (dem + bump)) <= 0.5 + 1e-6
    assert t.stats["correction_min_m"] < -5.0


def test_terrain_layer_raw_fallback_when_no_support():
    dem = _plane()
    t = terrain_layer(dem, np.ones_like(dem, bool), np.zeros_like(dem, bool), gsd_m=GSD)
    assert t.stats["raw_fallback_fraction"] == 1.0
    assert np.allclose(t.terrain, dem, atol=1e-6)


def test_object_layer_nonnegative_and_ground_fraction():
    rng = np.random.default_rng(1)
    rel = rng.random((200, 200)).astype(np.float32) * 0.1
    rel[50:80, 50:80] += 0.6  # a "building"
    valid = np.ones_like(rel, bool)
    o = object_layer_from_relative(rel, valid, gsd_m=GSD, ground_window_m=60)
    assert np.nanmin(o.object_rel) >= 0
    assert o.object_rel[60:70, 60:70].mean() > 0.4
    assert 0.2 < o.params["ground_fraction"] <= 1.0


def test_scale_fit_recovers_scale_and_gates_reject_garbage():
    rng = np.random.default_rng(2)
    h = w = 20 * F
    terrain = _plane(h, w)
    obj = np.clip(rng.random((h, w)) - 0.5, 0, None) * 0.4  # ~50% object pixels, values 0..0.2
    true_scale = 25.0
    dem = terrain + true_scale * obj + rng.normal(0, 0.3, (h, w))
    valid = np.ones((h, w), bool)
    fit = fit_object_scale_to_dem_residual(dem, terrain, obj, valid, gsd_m=GSD)
    assert fit.accepted, fit.reason
    assert abs(fit.scale - true_scale) / true_scale < 0.15
    # garbage: residual unrelated to the object layer -> rejected (r gate) or implausible implied heights
    dem2 = terrain + rng.normal(0, 3.0, (h, w))
    fit2 = fit_object_scale_to_dem_residual(dem2, terrain, obj, valid, gsd_m=GSD)
    assert not fit2.accepted and fit2.scale is None
    # implausible: residual tracks the object layer but implies 300 m objects -> rejected by the height gate
    dem3 = terrain + 1500.0 * obj
    fit3 = fit_object_scale_to_dem_residual(dem3, terrain, obj, valid, gsd_m=GSD)
    assert not fit3.accepted and "implied p99" in fit3.reason


def test_compose_dsm_without_scale_is_terrain_only():
    terrain = _plane(60, 60)
    obj = np.full_like(terrain, 0.5)
    dsm, ndsm = compose_dsm(terrain, obj, None, np.ones_like(terrain, bool))
    assert ndsm is None and np.allclose(dsm, terrain)
    dsm2, ndsm2 = compose_dsm(terrain, obj, 10.0, np.ones_like(terrain, bool))
    assert np.allclose(ndsm2, 5.0) and np.allclose(dsm2 - terrain, 5.0)


@pytest.mark.parametrize(
    "kw,tier,quality,flag",
    [
        (dict(georeferenced=False, dem_found=False), "R", "UNVALIDATED", None),
        (dict(dem_found=False), "R", "WARNING", "NO_DEM"),
        (dict(datum_ok=False), "R", "INVALID", "DATUM_UNSAFE"),
        (dict(consistency={"ME": 40.0}), "R", "INVALID", "DATUM_SUSPECT"),
        (dict(scale_fit={"accepted": True, "n_cells": 100, "residual_nmad_m": 1.0}), "T", "LIMITED", "OBJECT_SCALE_DEM_FIT"),
        (dict(), "T", "WARNING", "NO_OBJECT_SCALE"),
        (dict(anchor_fit={"accepted": True, "n_used": 8, "offset_m": -1.0, "holdout_nmad": 0.5, "scale_fit": {"accepted": False, "reason": "r too low"}}), "A", "WARNING", "NO_OBJECT_SCALE"),
        (dict(anchor_fit={"accepted": True, "n_used": 8, "offset_m": -1.0, "holdout_nmad": 0.5, "scale_fit": {"accepted": True, "n_inliers": 9, "n_total": 10}}), "A", "GOOD", "OBJECT_SCALE_ANCHORS"),
    ],
)
def test_tier_decision(kw, tier, quality, flag):
    base = dict(georeferenced=True, dem_found=True, terrain_stats={"support_mean": 0.7, "raw_fallback_fraction": 0.0, "dem_void_fraction": 0.0}, scale_fit=None, anchor_fit=None, out_vcrs="EGM2008", datum_ok=True, consistency={"ME": 0.1}, cfg={"datum_sanity_m": 15.0})
    base.update(kw)
    d = decide(**base)
    assert d.tier == tier and d.quality == quality, (d.tier, d.quality, d.triggers)
    if flag:
        assert flag in d.flags


def test_copernicus_tile_bounds_and_discovery(tmp_path: Path):
    assert _cop_tile_bounds("Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif") == (8.0, 47.0, 9.0, 48.0)
    assert _cop_tile_bounds("Copernicus_DSM_COG_10_S02_00_W071_00_DEM.tif") == (-71.0, -2.0, -70.0, -1.0)
    assert _cop_tile_bounds("random.tif") is None
    (tmp_path / "Copernicus_DSM_COG_10_N28_00_E077_00_DEM.tif").write_bytes(b"")
    src = discover_dem((77.2, 28.5, 77.3, 28.7), tmp_path)
    assert src is not None and src.vertical_crs == "EGM2008" and len(src.files) == 1
    assert discover_dem((10.0, 10.0, 10.1, 10.1), tmp_path) is None
    user = tmp_path / "mydem.tif"; user.write_bytes(b"")
    u = discover_dem((10.0, 10.0, 10.1, 10.1), tmp_path, user_dem=user, user_dem_vcrs="EGM96")
    assert u is not None and u.vertical_crs == "EGM96" and u.name == "user"
