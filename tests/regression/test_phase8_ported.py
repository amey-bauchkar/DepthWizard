"""Phase 8 Level-0 checks ported to pytest. Each test names its Phase 8 ID (validation/phase8/results/*.json)."""
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage

from core.geo.grid import Grid
from core.validate.coregister import estimate_shift_gradient, estimate_shift_grid, horn_gradients, slope_aspect_deg
from core.validate.metrics import affine_fit, metric_set


# ---- L0-02: GeoTIFF round trip & pixel-centre convention -------------------------------------
def test_l0_02_geotiff_roundtrip_pixel_centre(tmp_path):
    tr = from_origin(500000.0, 3000000.0, 1.0, 1.0)
    prof = dict(driver="GTiff", width=64, height=64, count=1, dtype="float32", crs=CRS.from_epsg(32643), transform=tr, nodata=-9999.0)
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    arr[0, 0] = -9999
    p = tmp_path / "utm.tif"
    with rasterio.open(p, "w", **prof) as ds:
        ds.write(arr, 1)
        ds.update_tags(VERTICAL_CRS="EPSG:3855", TIER="T")
    with rasterio.open(p) as ds:
        assert ds.crs.to_epsg() == 32643 and ds.transform == tr and ds.nodata == -9999.0
        assert ds.tags()["VERTICAL_CRS"] == "EPSG:3855"
        x, y = ds.xy(20, 10)
        assert (x, y) == (500010.5, 3000000 - 20.5)
        assert ds.index(500010.5, 3000000 - 20.5) == (20, 10)
        assert list(ds.bounds) == [500000.0, 3000000.0 - 64, 500000.0 + 64, 3000000.0]
        assert int((ds.read(1) == -9999).sum()) == 1
    g = Grid(64, 64, tr, "EPSG:32643", nodata=-9999.0, units="metres", metric=True, vertical_reference="EPSG:3855", tier="T")
    assert g.pixel_center_to_crs(10, 20) == (500010.5, 3000000 - 20.5)
    c, r = g.crs_to_pixel(500010.5, 3000000 - 20.5)
    assert (round(c), round(r)) == (10, 20)
    assert g.bounds == (500000.0, 3000000.0 - 64, 500000.0 + 64, 3000000.0)


def test_l0_02_rotation_and_no_crs():
    g = Grid(64, 64, Affine(0.9, 0.1, 500000.0, 0.1, -0.9, 3000000.0), "EPSG:32643")
    assert g.pixel_size is not None and abs(g.pixel_size[0] - (0.9**2 + 0.1**2) ** 0.5) < 1e-12
    a = Grid.pixel_space(320, 240)
    assert a.crs is None and a.transform is None and a.tier == "R" and a.metric is False and a.bounds is None


# ---- C-4 / L0-02c: geographic CRS anisotropy -> reproject to UTM -------------------------------
def test_c4_geographic_reprojected_to_utm(tmp_path):
    from core.geo.reproject import ensure_metric_grid, utm_epsg_for

    assert utm_epsg_for(77.21, 28.61) == 32643
    trg = from_origin(77.2, 28.6, 1e-5, 1e-5)
    prof = dict(driver="GTiff", width=64, height=64, count=1, dtype="float32", crs=CRS.from_epsg(4326), transform=trg, nodata=-9999.0)
    src = tmp_path / "wgs84.tif"
    with rasterio.open(src, "w", **prof) as ds:
        ds.write(np.random.default_rng(0).random((64, 64), dtype=np.float32), 1)
    grid, rec = ensure_metric_grid(str(src), str(tmp_path / "utm.tif"))
    assert rec.performed and rec.dst_crs == "EPSG:32643" and rec.anisotropy_before > 1.05  # ~0.98 x 1.11 m at 28.6N
    assert grid.crs == "EPSG:32643" and grid.pixel_size is not None
    px, py = grid.pixel_size
    assert abs(px - py) / py < 0.02  # isotropic metres after reprojection


# ---- C-1 / L0-01b: PROJ ballpark vertical transform must be rejected -------------------------
def test_c1_ballpark_vertical_transform_rejected(monkeypatch):
    from backend.errors import VerticalTransformUnsafeError
    from core.geo.vertical import grid_status, safe_vertical_transformer

    monkeypatch.delenv("DW_PROJ_GRIDS", raising=False)
    st = grid_status(["EGM2008"])[0]
    if st.found:
        t = safe_vertical_transformer("ellipsoidal", "EGM2008")
        _x, _y, h = t.transform(77.21, 28.61, 0.0)
        assert abs(-h - (-52.555)) < 0.05  # Delhi undulation, Phase 8 L0-01
    else:
        with pytest.raises(VerticalTransformUnsafeError):  # grids absent -> must NOT silently return 0 m
            safe_vertical_transformer("ellipsoidal", "EGM2008")


# ---- C-2 / L0-07: co-registration recovers a known synthetic shift ----------------------------
def _shift_fixture():
    rng = np.random.default_rng(3)
    N = 256
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    smooth = 100 + 6 * np.sin(xx / 30) + 4 * np.cos(yy / 25)
    ref = smooth + 0.3 * rng.normal(size=(N, N))
    pred = ndimage.shift(smooth, (-1.0, 2.0), order=3, mode="nearest") + 0.3 * rng.normal(size=(N, N))  # true (dx, dy) = (2, -1)
    return pred, ref


def test_c2_horn_gradient_sign_on_plane():
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    p, q = horn_gradients(0.1 * xx + 0.05 * yy, 1.0)
    assert p[10, 10] > 0 and q[10, 10] > 0  # correlate, not convolve (Phase 8 l0_07c divergence)
    assert abs(p[10, 10] - 0.1) < 1e-9 and abs(q[10, 10] - 0.05) < 1e-9


def test_c2_shift_recovery_grid_search():
    pred, ref = _shift_fixture()
    r = estimate_shift_grid(pred, ref, max_shift=4)
    assert abs(r.dx - 2.0) < 0.05 and abs(r.dy - (-1.0)) < 0.05  # Phase 8: (1.979, -1.020)
    assert r.rmse_after < r.rmse_before


def test_c2_shift_recovery_iterative_gradient():
    pred, ref = _shift_fixture()
    r = estimate_shift_gradient(pred, ref, smooth_sigma=2.0)
    assert r.converged and r.iterations <= 6
    assert abs(r.dx - 2.0) < 0.05 and abs(r.dy - (-1.0)) < 0.05  # Phase 8 l0_07d: (2.000, -1.012)


# ---- L0-06: Horn slope on an analytic plane --------------------------------------------------
def test_l0_06_slope_plane():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    slope, aspect = slope_aspect_deg(0.10 * xx + 0.05 * yy, 1.0)
    expect = np.degrees(np.arctan(np.hypot(0.10, 0.05)))
    assert np.abs(slope[2:-2, 2:-2] - expect).max() < 1e-9
    assert np.abs(slope - expect).max() > 1.0  # edge error exists -> edges must be flagged (Phase 8: 3.18 deg)
    assert aspect[2:-2, 2:-2].std() < 1e-9


# ---- L0-08: metric identities and correlation blindness --------------------------------------
def test_l0_08_metric_identities_and_correlation_blindness():
    rng = np.random.default_rng(1)
    ref = rng.normal(100, 5, 20000)
    e = rng.normal(0.5, 2.0, ref.size)
    m = metric_set(ref + e, ref)
    assert abs(m["RMSE"] - np.sqrt(m["ME"] ** 2 + e.var())) < 1e-6
    assert abs(m["NMAD"] - 2.0) < 0.1
    blind = metric_set(1.5 * ref + 10.0, ref)
    assert blind["pearson_r"] > 0.999999 and blind["ME"] > 50.0  # r == 1 with tens of metres of bias
    af = affine_fit(1.5 * ref + 10.0, ref)
    assert abs(af["scale"] - 1 / 1.5) < 1e-9 and af["label"] == "oracle_affine_alignment"


# ---- C-3 / L0-05: robust anchors --------------------------------------------------------------
def test_c3_robust_offset_and_scale_with_outliers():
    from core.calib.anchors import MIN_ANCHORS, ransac_scale, robust_offset, split_holdout

    rng = np.random.default_rng(7)
    res = np.full(30, 3.0) + rng.normal(0, 0.5, 30)
    res[:3] += 20.0  # gross blunders
    fit = robust_offset(res)
    assert abs(fit.offset - 3.0) < 0.2 and len(fit.blunder_idx) == 3 and set(fit.blunder_idx) == {0, 1, 2}
    x = rng.uniform(5, 40, 20)
    y = x / 1.2 + rng.normal(0, 0.3, 20)
    y[0] -= 12.0
    sf = ransac_scale(x, y)
    assert abs(sf.scale - 1 / 1.2) < 0.02 and sf.n_inliers >= 17
    with pytest.raises(ValueError):
        robust_offset(res[: MIN_ANCHORS - 1])
    fit_idx, hold_idx = split_holdout(30, 0.3, seed=0)
    assert len(hold_idx) == 9 and len(set(fit_idx) & set(hold_idx)) == 0
    assert len(split_holdout(6)[1]) == 0  # below min_n_for_split -> no hold-out, reported


# ---- L0-10 / viewer: composition nodata & heightfield/texture alignment -----------------------
def test_heightfield_texture_alignment_checkerboard(stub_settings, checkerboard, tmp_path):
    """Stub model: relative height = brightness. The four corner markers must land on the four heightfield corners
    with the same orientation as the texture (top-left bright, bottom-right dark)."""
    from core.dsm.rdsm import make_rdsm
    from core.inference.predictor import StubPredictor
    from core.terrain.heightfield import build_heightfield, write_texture

    pred = StubPredictor(stub_settings).predict(checkerboard)
    rel, grid, _ = make_rdsm(pred.relative_depth, method="minmax")
    tex_path, tex_size = write_texture(tmp_path / "t.jpg", checkerboard)
    heights, valid, meta = build_heightfield(rel, grid.nodata, max_mesh_dim=768, texture_size=tex_size)
    assert (meta.width, meta.height) == (320, 240) and (meta.texture_width, meta.texture_height) == (320, 240)
    assert heights[0, 0] == 1.0 and heights[-1, -1] == 0.0
    assert heights[0, -1] < 1.0 and heights[-1, 0] < 1.0
    # area-average path: downsample to <=100 and check corners still ordered the same way
    h2, v2, m2 = build_heightfield(rel, grid.nodata, max_mesh_dim=100, texture_size=tex_size)
    assert m2.downsample_factor == 4 and (m2.width, m2.height) == (80, 60)
    assert h2[0, 0] > h2[-1, -1] and m2.residual_vs_source["rmse"] >= 0
    assert abs(m2.texture_width / m2.texture_height - m2.width / m2.height) < 1e-9


def test_l0_10_nodata_propagation():
    a = np.ones((10, 10), np.float32)
    b = np.ones((10, 10), np.float32)
    a[0, 0] = np.nan
    b[5, 5] = np.nan
    assert np.isnan(a + b).sum() == 2


def test_heightfield_non_divisible_sizes_cover_full_footprint():
    """Regression for the Sprint 1 bug: 2048/4096 px inputs failed in build_heightfield when the size was not a
    multiple of the downsample factor (broadcast error). The mesh must cover the full footprint."""
    from core.terrain.heightfield import build_heightfield

    rng = np.random.default_rng(0)
    for (h, w, maxd) in ((2048, 2048, 768), (4096, 4096, 768), (1001, 733, 300), (50, 70, 8)):
        rel = rng.random((h, w), dtype=np.float32)
        rel[:5, :5] = -9999.0  # nodata corner
        zh, valid, meta = build_heightfield(rel, -9999.0, max_mesh_dim=maxd, texture_size=(w, h))
        f = meta.downsample_factor
        assert (meta.height, meta.width) == (-(-h // f), -(-w // f)) and max(meta.height, meta.width) <= maxd
        assert np.isfinite(zh[valid]).all() and meta.residual_vs_source["rmse"] >= 0.0
        assert 0.0 <= meta.min <= meta.max <= 1.0
