"""Open building footprints for LoD-1: rasterisation, coverage discovery, co-registration, and the LoD-1 footprint path."""
import json

import numpy as np
from rasterio.transform import from_origin

from core.terrain import footprints as fpm
from core.terrain.building_filter import FEATURES, filter_labels, load_model
from core.terrain.lod1 import extract_lod1_buildings

GSD, N, X0, Y0, CRS = 0.5, 200, 500000.0, 3000000.0, "EPSG:32645"
TR = from_origin(X0, Y0, GSD, GSD)


def _rect(c0, r0, w, h):
    x0, y0 = X0 + c0 * GSD, Y0 - r0 * GSD
    return [[x0, y0], [x0 + w * GSD, y0], [x0 + w * GSD, y0 - h * GSD], [x0, y0 - h * GSD], [x0, y0]]


def _scene(shift=(0, 0)):
    """nDSM + RGB with three flat grey 'roofs'; footprints drawn `shift` pixels (drow, dcol) away from them."""
    boxes = [(30, 30, 24, 16), (100, 60, 20, 20), (60, 130, 30, 18)]
    nd = np.zeros((N, N), np.float32)
    rgb = np.full((N, N, 3), (90, 120, 70), np.uint8)  # green fields
    for c0, r0, w, h in boxes:
        nd[r0:r0 + h, c0:c0 + w] = 7.0
        rgb[r0:r0 + h, c0:c0 + w] = (170, 160, 150)
    rings = [_rect(c0 - shift[1], r0 - shift[0], w, h) for c0, r0, w, h in boxes]
    return nd, rgb, rings


def test_to_labels_rasterises_each_footprint_with_its_own_id():
    nd, rgb, rings = _scene()
    lab, polys = fpm.to_labels(rings, CRS, TR, CRS, (N, N))
    assert sorted(np.unique(lab).tolist()) == [0, 1, 2, 3] and set(polys) == {1, 2, 3}
    assert (lab == 1).sum() == 24 * 16 and polys[1][0] == (30.0, 30.0)


def test_discover_needs_coverage(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({"files": [{"file": "a.geojson", "bounds_wgs84": [88.0, 27.0, 88.1, 27.1], "source": "s", "licence": "l", "count": 3}]}))
    assert fpm.discover((88.02, 27.02, 88.05, 27.05), tmp_path)[1]["coverage"] == 1.0
    assert fpm.discover((88.08, 27.08, 88.2, 27.2), tmp_path) is None  # mostly outside the file's coverage


def test_coregister_recovers_a_known_offset_and_refuses_when_aligned():
    nd, rgb, rings = _scene(shift=(6, -8))  # outlines 3 m up and 4 m right of the roofs
    lab, _ = fpm.to_labels(rings, CRS, TR, CRS, (N, N))
    reg = fpm.coregister(lab, rgb, nd, GSD)
    assert reg["applied"] and (reg["drow"], reg["dcol"]) == (6, -8)
    lab2, _ = fpm.to_labels(rings, CRS, fpm.shifted(TR, reg), CRS, (N, N))
    assert np.all(nd[lab2 > 0] == 7.0)  # every footprint pixel now on a roof
    nd, rgb, rings = _scene()
    lab, _ = fpm.to_labels(rings, CRS, TR, CRS, (N, N))
    assert not fpm.coregister(lab, rgb, nd, GSD)["applied"]


def test_lod1_from_footprints_keeps_known_buildings_and_flags_low_ones(tmp_path):
    nd, rgb, rings = _scene()
    nd[130:148, 60:90] = 1.0  # the model reads the third building (cols 60-90, rows 130-148) at 1 m
    fp = fpm.to_labels(rings, CRS, TR, CRS, (N, N))
    out = extract_lod1_buildings(nd, np.full((N, N), 100.0, np.float32), rgb=rgb, gsd_m=GSD, transform=TR, footprints=fp, return_labels=True)
    assert out["segmentation_method"] == "reference_footprints" and out["count"] == 3
    low = [b for b in out["buildings"] if "LOW_PREDICTED_HEIGHT" in b.get("quality_flags", [])]
    assert len(low) == 1 and abs(low[0]["height_m"] - 1.0) < 0.1
    assert sorted(np.unique(out["_labels"]).tolist()) == [0, 1, 2, 3]


def test_object_filter_model_is_shipped_and_applies():
    m = load_model()
    assert m is not None and m["features"] == list(FEATURES) and len(m["coef"]) == len(FEATURES)
    assert "leave_one_scene_out" in m and m["threshold"] == 0.5
    nd, rgb, rings = _scene()
    lab, _ = fpm.to_labels(rings, CRS, TR, CRS, (N, N))
    out, rep = filter_labels(lab, rgb, nd, GSD)
    assert rep["applied"] and rep["candidates"] == 3 and 0 <= rep["kept"] <= 3
    assert set(np.unique(out).tolist()) <= {0, 1, 2, 3}
