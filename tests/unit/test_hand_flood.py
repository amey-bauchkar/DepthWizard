"""River-stage flood screening (HAND), stream burning, and the landing-zone diagnosis."""
import numpy as np

from core.disaster.hand import hand_on_grid
from core.disaster.landing_zones import R_FEASIBLE, R_INVALID, R_OBJECT, R_SLOPE, advice, diagnose


def _valley(n=300, g=2.0, cross=0.2, down=0.03, x_ch=300.0):
    yy, xx = np.mgrid[0:n, 0:n] * g
    return 100 + cross * np.abs(xx - x_ch) - down * yy, xx


def test_hand_matches_the_height_above_a_sloping_valley_floor():
    z, xx = _valley()
    out = hand_on_grid(z, 2.0, cell_m=4.0, drainage_area_m2=4000)
    h = out["hand"]
    core = np.abs(xx - 300) < 200
    assert np.isfinite(h[core]).all()
    assert np.nanmax(np.abs(h - 0.2 * np.abs(xx - 300))[core]) < 1.0  # water surface parallel to the channel
    # a flat level cannot represent this: the valley floor itself drops 18 m over the scene
    assert np.ptp(z[:, 150]) > 15


def test_mapped_river_pulls_the_channel_off_the_dem_thalweg():
    z, xx = _valley(x_ch=300.0)
    burn = np.zeros(z.shape, bool)
    burn[:, 165] = True  # the mapped river runs 30 m (15 px) beside the DEM thalweg at x = 300 m
    out = hand_on_grid(z, 2.0, cell_m=2.0, drainage_area_m2=1e9, burn=burn)  # no DEM channels: mapped only
    assert out["drainage"][:, 165].all() and out["stats"]["burned_cells"] > 0
    assert np.nanmax(out["hand"][:, 165]) < 1e-6


def test_landing_zone_diagnosis_and_advice():
    r = np.full((10, 10), R_OBJECT, np.uint8)
    r[:, :3] |= R_SLOPE
    r[0, 0] = R_INVALID
    d = diagnose(r)
    assert d["centresWithData"] == 99 and d["obstaclePct"] == 100.0 and 29 < d["slopePct"] < 31
    assert d["obstacleOnlyPct"] > 60 and d["feasiblePct"] == 0.0
    txt = advice(d, 3, 50.0, 15.0, 0, 0)
    assert "building or tree" in txt and "size 2" in txt
    assert "every approach direction is blocked" in advice(d, 3, 50.0, 15.0, 0, 2)
    assert advice(d, 3, 50.0, 15.0, 3, 0) is None
    r2 = np.full((4, 4), R_FEASIBLE, np.uint8)
    assert diagnose(r2)["feasiblePct"] == 100.0


def test_site_confidence_follows_the_measured_error_rates():
    from core.disaster.landing_zones import site_confidence

    good = site_confidence(1.0, 7.0, 4, validated=True)
    assert good["label"] == "HIGH" and good["score"] > 0.99
    assert site_confidence(1.0, 7.0, 4, validated=False)["label"] == "MEDIUM"  # obstacle heights not laser-checked
    assert site_confidence(1.0, 7.0, 1, validated=True)["pApproachClear"] == 0.75  # 1 - measured false-clear rate
    edge = site_confidence(7.0, 7.0, 4, validated=True)
    assert abs(edge["pSlopeWithinLimit"] - 0.5) < 1e-9 and edge["label"] == "LOW"
    assert site_confidence(1.0, 7.0, 0, validated=True)["score"] == 0.0


def test_flood_probability_raster_and_ranges(tmp_path):
    import json

    import rasterio
    from rasterio.transform import from_origin

    from core.disaster.flood import run_flood_screening

    z = np.tile(np.arange(20, dtype=np.float32), (20, 1))  # terrain rising 1 m per cell, 0..19 m
    with rasterio.open(tmp_path / "terrain.tif", "w", driver="GTiff", width=20, height=20, count=1, dtype="float32", crs="EPSG:32645", transform=from_origin(0, 20, 1, 1), nodata=-9999) as ds:
        ds.write(z, 1)
    res = {"uncertainty": {"terrain": {"value_m": 2.0, "source": "test"}}, "vertical_reference": "EGM2008"}
    s = run_flood_screening(tmp_path, 10.0, res, model="level")
    u = s["uncertainty"]
    assert u["sigmaM"] == 2.0 and u["likelyAreaM2"] < s["affectedAreaM2"] < u["possibleAreaM2"]
    with rasterio.open(tmp_path / "flood_probability.tif") as ds:
        p = ds.read(1)
    assert p[0, 0] == 100 and p[0, 19] == 0 and p[0, 10] == 50  # Phi(0) at the water line
    assert json.loads((tmp_path / "disaster_flood.json").read_text())["uncertainty"]["measuredReliability"]
    assert run_flood_screening(tmp_path, 10.0, {}, model="level")["uncertainty"] is None  # no measured error: no range


def test_river_model_refuses_an_elevation_passed_as_a_rise(tmp_path):
    import pytest
    import rasterio
    from rasterio.transform import from_origin

    from core.disaster.flood import run_flood_screening

    with rasterio.open(tmp_path / "terrain.tif", "w", driver="GTiff", width=8, height=8, count=1, dtype="float32", crs="EPSG:32645", transform=from_origin(0, 8, 1, 1)) as ds:
        ds.write(np.full((8, 8), 1500, np.float32), 1)
    with pytest.raises(ValueError, match="was an elevation given"):
        run_flood_screening(tmp_path, 1537.0, {}, model="river")
