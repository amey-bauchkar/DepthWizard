"""Unit tests for the Mode B calibration core: terrain layer, object layer, detail fusion, anchor gain, tier decision, DEM discovery."""
from pathlib import Path

import numpy as np
import pytest

from core.calib.dem import _cop_tile_bounds, discover_dem
from core.calib.fusion import fit_anchor_gain, fuse_detail, plan_upsample, tiled_relative
from core.calib.terrain import object_layer_from_relative, terrain_layer
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


def _city(h=400, w=400, seed=3):
    """Sloped ground + buildings of several footprints (6-24 m) and heights; DEM = 30 m block mean of the truth."""
    rng = np.random.default_rng(seed)
    truth = _plane(h, w)
    for _ in range(70):
        s_ = int(rng.integers(3, 13))  # 6..24 m footprints at 2 m GSD
        r, c = int(rng.integers(0, h - s_)), int(rng.integers(0, w - s_))
        truth[r : r + s_, c : c + s_] += float(rng.uniform(6, 25))
    hh, ww = h // F, w // F
    blocks = truth[: hh * F, : ww * F].reshape(hh, F, ww, F).mean(axis=(1, 3))
    from scipy import ndimage

    dem = ndimage.zoom(blocks, (h / hh, w / ww), order=1)[:h, :w]
    return truth, dem


def _rgb_of(z):
    g = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def test_fusion_adds_calibrated_detail_and_preserves_dem_cell_means():
    """Model = affine map of the true surface per tile (the relative-depth ambiguity). The fused DSM must be
    closer to the truth than the DEM, and its 30 m block means must equal the DEM's."""
    truth, dem = _city()
    predict = lambda x: 0.37 * x.astype(np.float64).mean(axis=2) / 255.0 + 5.0  # noqa: E731
    tp = tiled_relative(predict, _rgb_of(truth), upsample=plan_upsample(400, 400, 1.0, tile=128, overlap=0.25, max_tiles=64), tile=128, overlap=0.25)
    fr = fuse_detail(dem, tp, detail_scale_px=F, band_high_px=4 * F)
    fused = dem + fr.detail
    inner = (slice(20, -20), slice(20, -20))
    rmse = lambda a: float(np.sqrt(np.mean((a - truth)[inner] ** 2)))  # noqa: E731
    assert rmse(fused) < 0.85 * rmse(dem), (rmse(fused), rmse(dem))
    assert fr.stats["tiles_with_detail"] >= 0.75 * fr.stats["n_tiles"] and fr.stats["gain_median"] > 0
    from scipy import ndimage

    assert float(np.abs(ndimage.gaussian_filter(fr.detail, F))[inner].mean()) < 0.35  # detail is zero-mean at DEM scale


def test_fusion_does_not_degrade_dem_with_uninformative_model():
    truth, dem = _city(seed=4)
    rng = np.random.default_rng(9)
    predict = lambda x: rng.random(x.shape[:2])  # noqa: E731  (noise: no relation to heights)
    tp = tiled_relative(predict, _rgb_of(truth), upsample=1.0, tile=128, overlap=0.25)
    fr = fuse_detail(dem, tp, detail_scale_px=F, band_high_px=4 * F)
    inner = (slice(20, -20), slice(20, -20))
    rmse = lambda a: float(np.sqrt(np.mean((a - truth)[inner] ** 2)))  # noqa: E731
    assert rmse(dem + fr.detail) < 1.03 * rmse(dem)


def test_anchor_gain_fit_gain_only_and_significant_offset():
    rng = np.random.default_rng(5)
    det = rng.normal(0, 3.0, 40)
    base = rng.normal(400, 5, 40)
    z = base + 2.0 * det + rng.normal(0, 1.5, 40)  # true gain 2, no datum offset
    g = fit_anchor_gain(z, base, det)
    assert g["accepted"] and abs(g["detail_gain"] - 2.0) < 0.3 and abs(g["offset_m"]) < 0.8
    z2 = z + 3.0  # a real datum shift is significant and kept
    g2 = fit_anchor_gain(z2, base, det)
    assert g2["accepted"] and abs(g2["offset_m"] - 3.0) < 0.8 and g2["offset_significant"]
    g3 = fit_anchor_gain(base + det + rng.normal(0, 8.0, 40), base, det)  # tier T already optimal -> gain near 1 or rejected
    assert (not g3["accepted"]) or abs(g3["detail_gain"] - 1.0) < 1.0
    assert not fit_anchor_gain(z[:3], base[:3], det[:3])["accepted"]


@pytest.mark.parametrize(
    "kw,tier,quality,flag",
    [
        (dict(georeferenced=False, dem_found=False), "R", "UNVALIDATED", None),
        (dict(dem_found=False), "R", "WARNING", "NO_DEM"),
        (dict(datum_ok=False), "R", "INVALID", "DATUM_UNSAFE"),
        (dict(consistency={"ME": 40.0}), "R", "INVALID", "DATUM_SUSPECT"),
        (dict(scale_fit={"accepted": True, "tiles_with_detail": 7, "n_tiles": 9, "gain_median": 3.0}), "T", "LIMITED", "OBJECT_SCALE_DEM_FIT"),
        (dict(), "T", "WARNING", "NO_OBJECT_SCALE"),
        (dict(anchor_fit={"accepted": True, "n_used": 8, "offset_m": -1.0, "holdout_nmad": 0.5, "gain_fit": {"accepted": False, "reason": "no detail"}}), "A", "WARNING", "NO_OBJECT_SCALE"),
        (dict(scale_fit={"accepted": True, "tiles_with_detail": 7, "n_tiles": 9, "gain_median": 3.0}, anchor_fit={"accepted": True, "n_used": 8, "offset_m": -1.0, "holdout_nmad": 0.5, "gain_fit": {"accepted": True, "detail_gain": 1.6, "offset_m": 0.0, "loo_rmse_m": 5.0, "tier_t_rmse_m": 6.0}}), "A", "GOOD", "OBJECT_SCALE_ANCHORS"),
        (dict(scale_fit={"accepted": True, "tiles_with_detail": 7, "n_tiles": 9, "gain_median": 3.0}, anchor_fit={"accepted": False, "gain_fit": {"accepted": False, "reason": "rejected: leave-one-out"}}), "T", "LIMITED", "OBJECT_SCALE_DEM_FIT"),
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


def test_tier_metric_model_ground_anchor_offset_alone_is_not_tier_a():
    """With a metric model, terrain = DSM - model heights; a ground-anchor terrain offset alone is not applied, so it
    must not upgrade the result to tier A (only an accepted anchor fit on the DSM does)."""
    base = dict(georeferenced=True, dem_found=True, terrain_stats={"support_mean": 0.7, "raw_fallback_fraction": 0.0, "dem_void_fraction": 0.0}, scale_fit={"accepted": True, "tiles_with_detail": 9, "n_tiles": 9, "gain_median": 1.0}, out_vcrs="EGM2008", datum_ok=True, consistency={"ME": 0.1}, cfg={"datum_sanity_m": 15.0}, is_metric=True)
    d = decide(anchor_fit={"accepted": True, "n_used": 8, "offset_m": -6.0, "gain_fit": {"accepted": False, "reason": "rejected: leave-one-out"}}, **base)
    assert d.tier == "T" and "METRIC_NDSM_MODEL" in d.flags and any("anchors supplied but not used" in t for t in d.triggers)
    d2 = decide(anchor_fit={"accepted": True, "n_used": 8, "offset_m": -6.0, "gain_fit": {"accepted": True, "detail_gain": 1.2, "offset_m": 0.0, "loo_rmse_m": 4.0, "tier_t_rmse_m": 5.0}}, **base)
    assert d2.tier == "A" and "OBJECT_SCALE_ANCHORS" in d2.flags
    d3 = decide(anchor_fit=None, **{**base, "dem_found": False})
    assert d3.tier == "H" and d3.absolute_elevation is False


@pytest.mark.parametrize("offset_kind,expected", [("geoid", "EGM96"), ("ellipsoidal", "ellipsoidal"), ("garbage", None)])
def test_cartodem_selection_and_datum_autocheck(tmp_path: Path, offset_kind, expected):
    """CartoDEM tiles (any file name) in <dem_dir>/cartodem are preferred; their vertical datum is decided by comparison
    with Copernicus (EGM2008): same heights -> geoid; heights shifted by the geoid undulation N -> ellipsoidal; else refused."""
    import rasterio
    from rasterio.transform import from_origin

    from core.calib.dem import select_dem
    from core.geo.grid import Grid
    from core.geo.raster_io import grid_bounds_wgs84
    from core.geo.vertical import register_bundled_grids, transform_heights

    register_bundled_grids()
    grid = Grid(200, 200, from_origin(300000.0, 3160000.0, 2.0, 2.0), "EPSG:32643", "float32", -9999.0, "metres", True, None, "R")
    w, s, e, n = grid_bounds_wgs84(grid)
    lon0, lat0 = int(np.floor(w)), int(np.floor(s))

    def write(path, add):
        tr = from_origin(w - 0.01, n + 0.01, 0.0003, 0.0003)
        nx, ny = int((e - w + 0.02) / 0.0003) + 1, int((n - s + 0.02) / 0.0003) + 1
        yy, xx = np.mgrid[0:ny, 0:nx]
        z = (250.0 + 0.5 * xx * 0.01 + add).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32", crs="EPSG:4326", transform=tr) as ds:
            ds.write(z, 1)

    write(tmp_path / f"Copernicus_DSM_COG_10_N{lat0:02d}_00_E{lon0:03d}_00_DEM.tif", 0.0)
    (tmp_path / "cartodem").mkdir()
    und = -float(transform_heights(np.array([(w + e) / 2]), np.array([(s + n) / 2]), np.array([0.0]), "ellipsoidal", "EGM2008")[0])
    write(tmp_path / "cartodem" / "cdnh43e_v3r1_anyname.tif", {"geoid": 0.4, "ellipsoidal": und, "garbage": 200.0}[offset_kind])
    src, rep = select_dem((w, s, e, n), tmp_path, grid)
    if expected is None:
        assert src is not None and src.name == "bundled" and rep["cartodem_datum_check"]["decision"] == "rejected"
    else:
        assert src is not None and src.name == "cartodem" and src.vertical_crs == expected, rep
    # CartoDEM absent -> Copernicus
    src2, _ = select_dem((w, s, e, n), tmp_path, grid, priority=("copernicus",))
    assert src2 is not None and src2.name == "bundled"
