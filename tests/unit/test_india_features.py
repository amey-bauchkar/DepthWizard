"""ISRO product ingest, landslide factors and rainfall trigger, road-access graph."""
import zipfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.disaster.landslide import CLASSES, classify, rainfall_trigger, trigger_threshold
from core.disaster.roads import _reach_exit, build_graph
from core.ingest.isro import convert


def _band(path, value, shape=(20, 20), px=1.0):
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint16", crs="EPSG:32645", transform=from_origin(600000, 3000000, px, px)) as ds:
        ds.write(np.full(shape, value, np.uint16), 1)


def test_cartosat_mx_bands_are_reordered_to_rgb(tmp_path):
    d = tmp_path / "C3_MX"
    d.mkdir()
    for n, v in ((1, 100), (2, 200), (3, 300), (4, 400)):  # B1 blue .. B4 NIR
        _band(d / f"BAND{n}.tif", v)
    (d / "BAND_META.txt").write_text("SatID=CARTOSAT-3\nSensor=MX\n")
    z = tmp_path / "prod.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in d.iterdir():
            zf.write(f, f"C3_MX/{f.name}")
    info = convert(z, tmp_path / "rgb.tif")
    assert info["sensor"] == "cartosat"
    with rasterio.open(tmp_path / "rgb.tif") as ds:
        r, g, b = (int(ds.read(i)[5, 5]) for i in (1, 2, 3))
        assert ds.crs.to_epsg() == 32645
    assert r > g > b  # red (300) brightest, blue (100) darkest: not swapped


def test_liss4_has_no_blue_band_and_is_simulated(tmp_path):
    d = tmp_path / "RS2A_LISS4"
    d.mkdir()
    for n, v in ((2, 200), (3, 300), (4, 600)):
        _band(d / f"BAND{n}.tif", v)
    info = convert(d, tmp_path / "rgb.tif")
    assert info["sensor"] == "liss4" and any("simulated" in n for n in info["notes"])


def test_pan_sharpening_uses_the_pan_grid(tmp_path):
    d = tmp_path / "prod"
    (d / "PAN").mkdir(parents=True)
    for n, v in ((1, 100), (2, 200), (3, 300)):
        _band(d / f"BAND{n}.tif", v, (10, 10), 2.0)
    _band(d / "PAN" / "BAND_PAN.tif", 500, (20, 20), 1.0)
    info = convert(d, tmp_path / "rgb.tif")
    assert (info["width"], info["height"]) == (20, 20) and any("Brovey" in n for n in info["notes"])


def test_rainfall_threshold_and_trigger():
    assert abs(trigger_threshold(24) * 24 - 144) < 3  # ~144 mm/day in the Nepal Himalaya
    assert not rainfall_trigger([2.0] * 72)["exceeded"]  # 144 mm over 3 days: below the 72 h threshold
    wet = rainfall_trigger([0.0] * 10 + [15.0] * 12 + [0.0] * 10)  # 180 mm in 12 h
    assert wet["exceeded"] and wet["worst"]["durationH"] in (6, 12)


def test_lhef_classes_follow_is_14496():
    t = np.array([3.0, 4.0, 5.5, 7.0, 8.0])
    assert classify(t).tolist() == [1, 2, 3, 4, 5]
    assert [c[2] for c in CLASSES] == ["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"]


def test_flooded_link_cuts_the_village_off_but_not_the_other_side():
    # road from x=-5 (outside: exit) through the scene to a village at x=15; water at x=8..9
    feats = [{"properties": {"motorable": True, "name": "NH"}, "geometry": {"coordinates": [[-5, 5], [5, 5], [15, 5]]}}]
    cut = np.zeros((10, 20), bool)
    cut[:, 8:10] = True
    G = build_graph(feats, lambda x, y: (x, y), (10, 20), cut, np.zeros_like(cut))
    before, after = _reach_exit(G, False, True), _reach_exit(G, True, True)
    assert (15.0, 5.0) in before and (15.0, 5.0) not in after
    assert (5.0, 5.0) in after  # the near side keeps its exit
    Gb = build_graph([{**feats[0], "properties": {"motorable": True, "bridge": True}}], lambda x, y: (x, y), (10, 20), cut, np.zeros_like(cut))
    assert (15.0, 5.0) in _reach_exit(Gb, True, True)  # a bridge clears shallow water


def test_confidence_interval_lookup_is_monotone_and_complete():
    from core.inference.confidence import interval_from_spread, load_calibration

    cal = load_calibration()
    assert cal is not None
    s = np.array([[0.0, 0.3, 1.0], [5.0, 50.0, np.nan]], np.float32)
    iv = interval_from_spread(s, cal)
    assert np.isnan(iv[1, 2]) and np.isfinite(iv[:, :2]).all() and np.isfinite(iv[0]).all()
    v = iv.ravel()[:5]
    assert np.all(np.diff(v) >= 0)  # more disagreement -> wider interval, also beyond the last populated bin
