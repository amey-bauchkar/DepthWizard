"""Analytical golden fixtures + invariants for Building Intelligence (docs/math_audit.md §16, §22 A-F, L-O).

Every expected number is derived by hand in the comment next to it; none is copied from the implementation's output.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from core.dsm.derive import pixel_area_m2
from core.geo.grid import Grid
from core.terrain import robust as R
from core.terrain.building_stats import floors_range, footprint_stats, ground_plane
from core.terrain.lod1 import extract_lod1_buildings

NORTH_UP_1M = Affine(1.0, 0.0, 2_600_000.0, 0.0, -1.0, 1_200_000.0)  # realistic projected origin (LV95-like)


def fp_full(n_rows: int, n_cols: int) -> np.ndarray:
    return np.ones((n_rows, n_cols), bool)


# ---------------------------------------------------------------- A. flat terrain, constant height
def test_A_flat_constant_building():
    T = np.full((6, 5), 400.0)
    h = np.full((6, 5), 12.0)
    st = footprint_stats(h, T, fp_full(6, 5), transform=NORTH_UP_1M)
    assert st["area_m2"] == 30.0  # 30 cells x 1 m^2
    assert st["height_median_m"] == 12.0 and st["height_block_m"] == 12.0
    assert st["volume_m3"] == 360.0  # 30 m^2 x 12 m
    assert st["base_elev_m"] == 400.0 and st["roof_elev_m"] == 412.0
    assert st["ground_slope_deg"] == 0.0
    assert st["quality_flags"] == []


# ---------------------------------------------------------------- B. sloped terrain, constant height
@pytest.mark.parametrize("a,b", [(0.1, 0.0), (0.0, -0.2), (0.15, 0.08)])
def test_B_sloped_terrain_constant_building(a, b):
    rows, cols = np.mgrid[0:8, 0:10]
    x, y = cols + 0.5, -(rows + 0.5)  # pixel-centre CRS offsets for NORTH_UP_1M (origin cancels in the plane fit)
    T = 500.0 + a * x + b * y
    h = np.full(T.shape, 9.0)
    st = footprint_stats(h, T, fp_full(8, 10), transform=NORTH_UP_1M)
    # nDSM is terrain-relative per pixel, so the height is exact on any slope
    assert st["height_median_m"] == 9.0 and st["height_block_m"] == 9.0
    # recovered ground tilt = atan(hypot(a, b))
    assert st["ground_slope_deg"] == pytest.approx(math.degrees(math.atan(math.hypot(a, b))), abs=0.01)
    # roof elevation = median(T + h) = median(T) + 9 for constant h
    assert st["roof_elev_m"] == pytest.approx(float(np.median(T)) + 9.0, abs=0.01)
    assert st["ground_max_m"] - st["ground_min_m"] == pytest.approx(abs(a) * 9 + abs(b) * 7, abs=0.01)


# ---------------------------------------------------------------- C. sloped terrain, stepped building
def test_C_sloped_terrain_stepped_building():
    # 4 x 4 footprint; terrain rises east T = 100 + 0.5*col; tower (h=20) on the LOW side cols 0-2, podium (h=10) col 3
    T = np.tile(100.0 + 0.5 * np.arange(4), (4, 1))
    h = np.tile(np.array([20.0, 20.0, 20.0, 10.0]), (4, 1))
    st = footprint_stats(h, T, fp_full(4, 4), transform=NORTH_UP_1M)
    # roof surface T+h per column: 120, 120.5, 121, 111.5 (4 each) -> sorted median = (120 + 120.5)/2 = 120.25
    assert st["roof_elev_m"] == 120.25
    # the pre-audit rule median(T) + median(h) = 100.75 + 20 = 120.75 is 0.5 m off: medians are not additive
    assert float(np.median(T)) + float(np.median(h)) == 120.75
    # volume: 4 rows x (20+20+20+10) m x 1 m^2 = 280 m^3; A x median would give 16 x 20 = 320 (+14 %)
    assert st["volume_m3"] == 280.0
    assert st["height_block_m"] == 17.5  # V / A = 280 / 16
    assert st["height_median_m"] == 20.0  # dominant roof level
    assert st["height_p10_m"] == 10.0 and st["height_p90_m"] == 20.0  # the distribution exposes the second level


# ---------------------------------------------------------------- D. tree contamination
def test_D_tree_contamination_median_is_exact():
    h = np.full((10, 10), 12.0)
    h[:, :1] = 5.0
    h[:2, 1] = 5.0  # 12 canopy pixels at 5 m (12 %)
    st = footprint_stats(h, np.zeros_like(h), fp_full(10, 10), transform=NORTH_UP_1M)
    assert st["height_median_m"] == 12.0  # 88 of 100 samples are 12: exact under 12 % contamination
    assert st["volume_m3"] == 88 * 12 + 12 * 5  # the integral reports the observed surface honestly (1116 m^3)


# ---------------------------------------------------------------- E. NoData inside the footprint
def test_E_nodata_imputation_and_flag():
    h = np.full((10, 10), 8.0)
    h[:3, :] = np.nan  # 30 % missing
    st = footprint_stats(h, np.zeros_like(h), fp_full(10, 10), transform=NORTH_UP_1M)
    assert st["area_m2"] == 100.0  # the footprint still covers 100 m^2
    assert st["volume_m3"] == 800.0  # 70 valid x 8 m x (100/70) imputation = 800 exactly
    assert st["valid_fraction"] == 0.7
    assert "LOW_VALID_FRACTION" in st["quality_flags"]


def test_E_all_nodata_returns_none():
    h = np.full((3, 3), np.nan)
    assert footprint_stats(h, np.zeros_like(h), fp_full(3, 3), transform=NORTH_UP_1M) is None


# ---------------------------------------------------------------- F. building crossing the raster boundary
def test_F_building_truncated_by_raster_edge_is_flagged():
    n = np.zeros((40, 40), np.float32)
    n[10:26, 30:40] = 10.0  # 16 x 10 block touching the right edge
    t = np.full_like(n, 300.0)
    d = extract_lod1_buildings(n, t, gsd_m=1.0, min_area_m2=15.0, return_labels=True)
    assert d["count"] == 1
    b = d["buildings"][0]
    assert "TRUNCATED_BY_RASTER_EDGE" in b["quality_flags"]
    assert b["area_m2"] == 160.0 and b["height_median_m"] == 10.0
    assert int((d["_labels"] == b["id"]).sum()) == 160  # label raster == the statistics' pixel set


# ---------------------------------------------------------------- L/M. rotated raster, non-square pixels
def test_L_rotated_raster_preserves_pixel_area():
    for deg in (0, 17, 45, 90, 133):
        t = Affine.rotation(deg) * Affine.scale(0.5, -0.5)
        assert pixel_area_m2(t) == pytest.approx(0.25, rel=1e-12)


def test_M_non_square_and_sheared_pixel_area():
    assert pixel_area_m2(Affine(2.0, 0, 0, 0, -0.5, 0)) == pytest.approx(1.0)
    sh = Affine.shear(20, 0) * Affine.scale(1.0, -1.0)  # shear preserves area (det 1); norms product does not
    assert pixel_area_m2(sh) == pytest.approx(1.0)
    g = Grid(10, 10, sh)
    assert g.pixel_size[0] * g.pixel_size[1] > 1.05  # the old width*height assumption over-estimates


# ---------------------------------------------------------------- N. coordinate round trip
def test_N_pixel_crs_round_trip():
    rng = np.random.default_rng(0)
    for t in (NORTH_UP_1M, Affine.rotation(23) * Affine.scale(0.5, -0.7), Affine(0.3, 0.1, 5e5, 0.05, -0.4, 4.1e6)):
        g = Grid(100, 100, t, "EPSG:32643")
        for c, r in rng.uniform(0, 100, (20, 2)):
            x, y = g.pixel_center_to_crs(c, r)
            c2, r2 = g.crs_to_pixel(x, y)
            assert abs(c2 - c) < 1e-6 and abs(r2 - r) < 1e-6  # float64 at 1e6 m: ~1e-10 m, far inside 1e-6 px


# ---------------------------------------------------------------- O. vertical datum shift & invariants
def test_O_datum_translation_invariance():
    rng = np.random.default_rng(1)
    h = rng.uniform(3, 30, (12, 9))
    T = rng.uniform(200, 210, (12, 9))
    s0 = footprint_stats(h, T, fp_full(12, 9), transform=NORTH_UP_1M)
    s1 = footprint_stats(h, T + 47.3, fp_full(12, 9), transform=NORTH_UP_1M)  # e.g. EGM96 -> EGM2008-like offset
    for k in ("height_median_m", "height_block_m", "height_p10_m", "height_p90_m", "area_m2", "volume_m3", "ground_slope_deg"):
        assert s0[k] == s1[k], k
    assert s1["base_elev_m"] == pytest.approx(s0["base_elev_m"] + 47.3, abs=0.011)
    assert s1["roof_elev_m"] == pytest.approx(s0["roof_elev_m"] + 47.3, abs=0.011)


@pytest.mark.parametrize("est", [R.median, R.mean, R.trimmed_mean, R.huber, lambda v: R.percentile(v, 90)])
def test_estimator_scale_and_translation_equivariance(est):
    rng = np.random.default_rng(2)
    for _ in range(25):
        v = rng.gamma(2.0, 4.0, rng.integers(5, 300))
        c, k = rng.uniform(0.1, 10), rng.uniform(-50, 50)
        assert est(c * v) == pytest.approx(c * est(v), rel=1e-6, abs=1e-6)
        assert est(v + k) == pytest.approx(est(v) + k, rel=1e-6, abs=1e-6)


def test_area_scales_with_pixel_dimensions():
    h = np.full((5, 5), 10.0)
    a1 = footprint_stats(h, np.zeros_like(h), fp_full(5, 5), transform=Affine(1, 0, 0, 0, -1, 0))
    a2 = footprint_stats(h, np.zeros_like(h), fp_full(5, 5), transform=Affine(2, 0, 0, 0, -2, 0))
    assert a2["area_m2"] == 4 * a1["area_m2"] and a2["volume_m3"] == 4 * a1["volume_m3"]


def test_nodata_outside_footprint_does_not_change_estimate():
    rng = np.random.default_rng(3)
    h = rng.uniform(5, 15, (20, 20))
    fp = np.zeros((20, 20), bool)
    fp[5:15, 4:16] = True
    s0 = footprint_stats(h, np.zeros_like(h), fp, transform=NORTH_UP_1M)
    h2 = h.copy()
    h2[~fp] = np.nan
    s1 = footprint_stats(h2, np.zeros_like(h), fp, transform=NORTH_UP_1M)
    assert s0 == s1


def test_ground_plane_is_stable_at_projected_coordinates():
    # uncentred [x y 1] at 2.6e6 m would be ill-conditioned; centring must recover the plane to float precision
    rows, cols = np.mgrid[0:5, 0:5]
    x = 2_600_000.0 + cols * 0.5
    y = 1_200_000.0 - rows * 0.5
    z = 0.3 * (x - 2_600_000.0) - 0.1 * (y - 1_200_000.0) + 450.0
    a, b, _ = ground_plane(x.ravel(), y.ravel(), z.ravel())
    assert a == pytest.approx(0.3, abs=1e-9) and b == pytest.approx(-0.1, abs=1e-9)
    assert ground_plane(np.arange(5.0), np.arange(5.0), np.ones(5)) is None  # collinear -> not identifiable


def test_floors_range_boundaries():
    assert floors_range(2.49) == [0, 0]
    assert floors_range(2.5) == [1, 1]  # floor(2.5/3.5)=0 -> 1; ceil(2.5/3)=1
    assert floors_range(10.5) == [3, 4]  # floor(10.5/3.5)=3, ceil(10.5/3)=4
    assert floors_range(float("nan")) == [0, 0]


# ---------------------------------------------------------------- API / export consistency (§21, §27.12-13)
def test_exports_equal_api_records():
    import csv
    import io

    from core.terrain import buildings as B

    n = np.zeros((30, 30), np.float32)
    n[5:15, 5:17] = 12.0
    n[18:27, 3:12] = 7.0
    t = np.full_like(n, 410.0)
    tr = Affine(0.5, 0, 2_682_000.0, 0, -0.5, 1_248_000.0)
    d = extract_lod1_buildings(n, t, gsd_m=0.5, transform=tr, min_area_m2=5.0)
    d.update({"grid": {"crs": "EPSG:2056", "transform": list(tr.to_gdal()), "width": 30, "height": 30}, "height_error": {"typical_m": 2.0}})
    recs = B.records(d)
    assert len(recs) == 2
    rows = list(csv.DictReader(io.StringIO(B.to_csv(recs))))
    gj = B.to_geojson(d, recs)
    for r, row, f in zip(recs, rows, gj["features"]):
        for k in ("height_m", "height_block_m", "area_m2", "volume_m3", "roof_elev_m", "ground_elev_m"):
            assert float(row[k]) == float(r[k]) == float(f["properties"][k]), k
        # height is PREDICTED, volume DERIVED; block volume == reported volume (V = A * H_block)
        assert abs(r["volume_m3"] - r["area_m2"] * r["height_block_m"]) <= 0.5 + 0.005 * r["area_m2"]  # integer + 2 dp rounding
    assert B.summary(d, recs)["quantity_category"]["height_m"] == "PREDICTED"


def test_watershed_labels_partition_the_mask():
    # a building within min_distance of the border gets no watershed marker; it must still become an instance
    n = np.zeros((30, 30), np.float32)
    n[5:15, 5:17] = 12.0
    n[18:27, 3:12] = 7.0
    d = extract_lod1_buildings(n, np.zeros_like(n), gsd_m=0.5, min_area_m2=5.0, return_labels=True)
    assert d["count"] == 2
    assert int((d["_labels"] > 0).sum()) == 120 + 81
