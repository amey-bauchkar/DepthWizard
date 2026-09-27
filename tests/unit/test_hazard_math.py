"""Analytical golden fixtures + invariants for Hazard Screening (docs/math_audit.md §16, §22 G-K).

Every expected number is derived by hand in the comment next to it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import rasterio
from affine import Affine

from core.disaster.accessibility import run_accessibility_screening
from core.disaster.flood import boundary_connected, classify_exposure, exposure_rules, run_flood_screening
from core.dsm.derive import slope_aspect_affine
from core.geo.grid import Grid
from core.geo.raster_io import write_raster
from core.screening_params import EXPOSURE_BINS_M

T1 = Affine(1.0, 0.0, 600_000.0, 0.0, -1.0, 3_000_000.0)


def _write(job, name, arr, t=T1, dtype="float32", nodata=-9999.0):
    write_raster(job / name, arr, Grid(arr.shape[1], arr.shape[0], t, "EPSG:32643", dtype, nodata, "metres", True), dtype=dtype)


@pytest.fixture
def job(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    return d


# ---------------------------------------------------------------- G/H. threshold + depth matrices
def test_G_H_threshold_and_depth_exact(job):
    T = np.array([[10.0, 11.0, 12.0], [9.5, 10.0, 13.0], [-9999.0, 10.25, 10.5]])
    _write(job, "terrain.tif", T)
    s = run_flood_screening(job, 10.5, {})
    # wet: T <= 10.5 -> (0,0) 10, (1,0) 9.5, (1,1) 10, (2,1) 10.25, (2,2) 10.5 (boundary inclusive) -> 5 cells
    assert s["affectedAreaM2"] == 5.0 and s["totalAreaM2"] == 8.0  # one NoData cell excluded
    assert s["affectedAreaPct"] == 62.5
    with rasterio.open(job / "flood_depth.tif") as ds:
        D = ds.read(1)
    exp = np.array([[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [-9999.0, 0.25, 0.0]])
    assert np.allclose(D, exp)
    assert s["maxDepth_m"] == 1.0
    assert s["meanDepth_m"] == round((0.5 + 1.0 + 0.5 + 0.25 + 0.0) / 5, 2)  # 0.45


def test_flood_uses_det_area_on_non_square_grid(job):
    T = np.zeros((4, 4))
    _write(job, "terrain.tif", T, t=Affine(2.0, 0.0, 6e5, 0.0, -0.5, 3e6))
    s = run_flood_screening(job, 1.0, {})
    assert s["pixelAreaM2"] == 1.0 and s["affectedAreaM2"] == 16.0


def test_invalid_water_level_rejected(job):
    _write(job, "terrain.tif", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        run_flood_screening(job, float("nan"), {})


# ---------------------------------------------------------------- I. partial building inundation (exact footprint)
def test_I_partial_building_inundation(job):
    # terrain rises east 0.1 m per column; building = 10 cols x 10 rows starting at col 2
    T = np.tile(100.0 + 0.1 * np.arange(14), (12, 1))
    lab = np.zeros((12, 14), np.int32)
    lab[1:11, 2:12] = 1  # footprint ground values 100.2 .. 101.1 (10 each)
    _write(job, "terrain.tif", T)
    _write(job, "building_labels.tif", lab, dtype="int32", nodata=0)
    (job / "buildings.json").write_text(json.dumps({"buildings": [{"id": 1, "base_elev_m": 100.65, "height_median_m": 9.0, "area_m2": 100.0}]}))
    s = run_flood_screening(job, 100.65, {})
    b = s["buildings"][0]
    # wet footprint columns: T <= 100.65 -> 100.2, 100.3, 100.4, 100.5, 100.6 -> 5 of 10 columns
    assert b["wet_fraction"] == 0.5
    # depths on wet cells: .45 .35 .25 .15 .05 -> mean .25, max .45
    assert b["mean_depth_wet_m"] == 0.25 and b["max_depth_m"] == 0.45
    # P10 of 100 footprint values (10 x each of 100.2..101.1): index 9.9 -> 100.2 + 0.9*0.1 = 100.29 -> D_exp = .36
    assert b["ground_p10_m"] == 100.29 and b["flood_depth_m"] == 0.36 and b["exposure"] == "LOW"
    # the pre-audit rule (W - median ground = 100.65 - 100.65 = 0) reported this half-flooded building as NOT affected
    assert s["affectedBuildingsCount"] == 1 and s["footprintMethod"] == "exact_label_raster"


# ---------------------------------------------------------------- connectivity (isolated depression)
def test_isolated_depression_is_identified(job):
    T = np.full((9, 9), 5.0)
    T[2:7, 2:7] = 12.0  # ring wall at 12 m
    T[3:6, 3:6] = 1.0  # enclosed pit at 1 m (9 cells)
    _write(job, "terrain.tif", T)
    s = run_flood_screening(job, 6.0, {})
    # below 6 m: outer apron 81 - 25 = 56 cells (connected to the edge) + 9 pit cells (isolated)
    assert s["affectedAreaM2"] == 65.0 and s["isolatedAreaM2"] == 9.0
    s2 = run_flood_screening(job, 6.0, {}, connected_only=True)
    assert s2["affectedAreaM2"] == 56.0


def test_diagonal_gap_4_vs_8_connectivity():
    wet = np.zeros((5, 5), bool)
    wet[0, 0] = wet[1, 1] = wet[2, 2] = True  # a diagonal chain from the corner
    valid = np.ones_like(wet)
    assert boundary_connected(wet, valid, 8)[2, 2]  # 8: reached through corner contacts
    assert not boundary_connected(wet, valid, 4)[2, 2]  # 4: (2,2) is cut off


def test_connectivity_4_subset_of_8_random():
    rng = np.random.default_rng(5)
    for _ in range(30):
        wet = rng.random((30, 30)) < rng.uniform(0.3, 0.7)
        valid = rng.random((30, 30)) > 0.05
        wet &= valid
        c4, c8 = boundary_connected(wet, valid, 4), boundary_connected(wet, valid, 8)
        assert not (c4 & ~c8).any()


# ---------------------------------------------------------------- invariants
def test_flood_vertical_translation_invariance(tmp_path):
    rng = np.random.default_rng(6)
    T = rng.uniform(0, 20, (25, 25))
    out = []
    for c in (0.0, 1234.5):
        d = tmp_path / f"j{c}"
        d.mkdir()
        _write(d, "terrain.tif", T + c)
        s = run_flood_screening(d, 9.0 + c, {})
        with rasterio.open(d / "flood_depth.tif") as ds:
            out.append((s["affectedAreaM2"], s["isolatedAreaM2"], ds.read(1)))
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    assert np.allclose(out[0][2], out[1][2], atol=2e-4)  # float32 raster storage at ~1e3 m: ulp ~1e-4


# ---------------------------------------------------------------- exposure classification (single definition)
def test_exposure_boundaries():
    b0, b1, b2 = EXPOSURE_BINS_M
    assert classify_exposure(float("nan")) == "NONE"
    assert classify_exposure(-0.0) == "NONE" and classify_exposure(0.0) == "NONE"
    assert classify_exposure(1e-9) == "LOW" and classify_exposure(b0) == "LOW"
    assert classify_exposure(math.nextafter(b0, 9)) == "MODERATE" and classify_exposure(b1) == "MODERATE"
    assert classify_exposure(math.nextafter(b1, 9)) == "HIGH" and classify_exposure(b2) == "HIGH"
    assert classify_exposure(math.nextafter(b2, 9)) == "VERY HIGH" and classify_exposure(1e6) == "VERY HIGH"
    rules = exposure_rules()
    assert [r["label"] for r in rules] == ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    assert rules[0]["gt_m"] == 0.0 and rules[-1]["le_m"] is None


# ---------------------------------------------------------------- J/K. slope + aspect on analytic planes
@pytest.mark.parametrize("t", [T1, Affine(2.0, 0, 6e5, 0, -0.5, 3e6), Affine.translation(6e5, 3e6) * Affine.rotation(30) * Affine.scale(1, -1), Affine.translation(6e5, 3e6) * Affine.shear(20, 0) * Affine.scale(1, -1)])
@pytest.mark.parametrize("a,b,aspect", [(0.2, 0.0, 270.0), (0.0, 0.2, 180.0), (-0.2, 0.0, 90.0), (0.0, -0.2, 0.0), (0.1, 0.1, 225.0)])
def test_J_K_slope_aspect_analytic_planes(t, a, b, aspect):
    rows, cols = np.mgrid[0:9, 0:9]
    x = t.a * (cols + 0.5) + t.b * (rows + 0.5) + t.c
    y = t.d * (cols + 0.5) + t.e * (rows + 0.5) + t.f
    z = a * (x - t.c) + b * (y - t.f)
    s, asp = slope_aspect_affine(z, t)
    assert s[4, 4] == pytest.approx(math.degrees(math.atan(math.hypot(a, b))), abs=1e-9)
    # circular difference: north can come out as 359.999... ; 1e-6 deg covers float64 cancellation at 1e6 m coordinates
    assert abs((asp[4, 4] - aspect + 180.0) % 360.0 - 180.0) < 1e-6


def test_K_flat_surface_has_no_aspect():
    s, asp = slope_aspect_affine(np.full((5, 5), 7.0), T1)
    assert s[2, 2] == 0.0 and np.isnan(asp[2, 2])


# ---------------------------------------------------------------- accessibility semantics
def test_accessibility_uses_terrain_and_excludes_buildings(job):
    rows, cols = np.mgrid[0:12, 0:12]
    T = 100.0 + 0.1 * cols  # 5.71 deg everywhere
    lab = np.zeros((12, 12), np.int32)
    lab[3:7, 3:7] = 1  # 16 building cells
    _write(job, "terrain.tif", T)
    _write(job, "building_labels.tif", lab, dtype="int32", nodata=0)
    s = run_accessibility_screening(job, 10.0, {})
    # slope is undefined on the outer ring (3x3 stencil incomplete): 10 x 10 = 100 interior cells, 16 of them building
    assert s["buildingAreaM2"] == 16.0
    assert s["accessibleAreaM2"] == 84.0 and s["steepAreaM2"] == 0.0
    s2 = run_accessibility_screening(job, 5.0, {})  # 5 < 5.71 deg everywhere
    assert s2["accessibleAreaM2"] == 0.0 and s2["steepAreaM2"] == 84.0
    with pytest.raises(ValueError):
        run_accessibility_screening(job, 95.0, {})
