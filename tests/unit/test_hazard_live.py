"""Live slider previews: same numbers as a full run, no files written, cache invalidated when the job changes, and
the browser field reproduces the server's wet rule."""
import json
import os
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.disaster.accessibility import display_field as access_field
from core.disaster.accessibility import run_accessibility_screening
from core.disaster.flood import _p_wet, display_field, run_flood_screening


def _terrain(d, z):
    with rasterio.open(d / "terrain.tif", "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32", crs="EPSG:32645", transform=from_origin(0, z.shape[0], 1, 1), nodata=-9999) as ds:
        ds.write(z.astype(np.float32), 1)


def _scene(d):
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    z = 100 + 0.3 * np.abs(xx - 40) + 0.05 * yy
    z[0, 0] = -9999  # nodata cell
    _terrain(d, z)
    lab = np.zeros(z.shape, np.int32)
    lab[20:26, 36:44] = 1
    lab[40:46, 10:16] = 2
    with rasterio.open(d / "building_labels.tif", "w", driver="GTiff", width=80, height=60, count=1, dtype="int32", crs="EPSG:32645", transform=from_origin(0, 60, 1, 1), nodata=0) as ds:
        ds.write(lab, 1)
    bld = [{"id": 1, "base_elev_m": 101.0, "height_m": 6.0, "area_m2": 48.0, "coords": [], "pixel_bbox": [36, 20, 44, 26]},
           {"id": 2, "base_elev_m": 109.0, "height_m": 4.0, "area_m2": 36.0, "coords": [], "pixel_bbox": [10, 40, 16, 46]}]
    (d / "buildings.json").write_text(json.dumps({"buildings": bld}), encoding="utf-8")
    return {"uncertainty": {"terrain": {"value_m": 1.5, "source": "test"}}, "vertical_reference": "EGM2008"}


def test_live_run_matches_the_full_run_and_writes_nothing(tmp_path):
    res = _scene(tmp_path)
    for model, levels in (("level", (101.0, 103.5, 110.0)), ("river", (0.5, 2.0, 6.0))):
        for W in levels:
            for f in ("flood_depth.tif", "flood_probability.tif", "flood_preview.png", "disaster_flood.json"):
                (tmp_path / f).unlink(missing_ok=True)
            live = run_flood_screening(tmp_path, W, res, model=model, write_outputs=False)
            assert not any((tmp_path / f).exists() for f in ("flood_depth.tif", "flood_probability.tif", "flood_preview.png", "disaster_flood.json"))
            full = run_flood_screening(tmp_path, W, res, model=model)
            assert live.pop("live") is True
            assert live == full, (model, W)


def test_cached_terrain_is_reloaded_when_the_file_changes(tmp_path):
    res = _scene(tmp_path)
    a = run_flood_screening(tmp_path, 103.0, res, model="level")["affectedAreaM2"]
    time.sleep(0.02)
    z = np.full((60, 80), 50.0, np.float32)  # re-processed job: everything is now below the level
    _terrain(tmp_path, z)
    os.utime(tmp_path / "terrain.tif")
    b = run_flood_screening(tmp_path, 103.0, res, model="level")["affectedAreaM2"]
    assert b == 60 * 80 and a < b


def test_browser_field_reproduces_the_wet_rule(tmp_path):
    res = _scene(tmp_path)
    for model, W in (("level", 104.0), ("river", 3.0)):
        arr, meta = display_field(tmp_path, model, max_dim=4000)
        assert meta["stride"] == 1 and arr.shape == (60, 80) and arr.dtype == np.float32
        full = run_flood_screening(tmp_path, W, res, model=model)
        wet_px = int((arr <= W).sum())  # NaN (no surface) is never wet
        assert wet_px * full["pixelAreaM2"] == full["affectedAreaM2"]
    arr, meta = display_field(tmp_path, "river", max_dim=40)
    assert meta["stride"] == 2 and arr.shape == (30, 40)


def test_p_wet_is_identical_to_phi_everywhere():
    from scipy.special import ndtr

    rng = np.random.default_rng(0)
    s = rng.uniform(-50, 150, (200, 300))
    s[rng.random(s.shape) < 0.05] = np.nan
    valid = rng.random(s.shape) > 0.02
    for W, sig in ((10.0, 0.5), (60.0, 7.3), (-100.0, 2.0)):
        ref = np.where(valid & np.isfinite(s), ndtr((W - np.nan_to_num(s, nan=np.inf)) / sig), 0.0)
        new = _p_wet(s, valid, W, sig)
        assert np.array_equal(np.rint(100 * ref), np.rint(100 * new))
        assert np.abs(ref - new).max() < 1e-14


def test_accessibility_live_and_field(tmp_path):
    res = _scene(tmp_path)
    live = run_accessibility_screening(tmp_path, 10.0, res, write_outputs=False)
    assert not (tmp_path / "accessibility.tif").exists()
    full = run_accessibility_screening(tmp_path, 10.0, res)
    assert live.pop("live") is True and live == full
    arr, meta = access_field(tmp_path, max_dim=4000)
    assert (arr[20:26, 36:44] == -1).all()  # building footprint
    ok = np.isfinite(arr) & (arr >= 0)
    assert int((ok & (arr <= 10.0)).sum()) * full["pixelAreaM2"] == full["accessibleAreaM2"]
