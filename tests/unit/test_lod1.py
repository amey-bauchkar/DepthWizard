import numpy as np
from core.terrain.lod1 import extract_lod1_buildings, rdp_simplify


def test_rdp_simplify_reduces_collinear_points():
    # Straight line with intermediate points
    pts = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.0, 3.0]]
    simplified = rdp_simplify(pts, epsilon=0.1)
    assert len(simplified) < len(pts)
    assert simplified[0] == [0.0, 0.0]
    assert simplified[-1] == [3.0, 3.0]


def test_extract_lod1_buildings_synthetic_block():
    ndsm = np.zeros((100, 100), dtype=np.float32)
    terrain = np.full((100, 100), 400.0, dtype=np.float32)

    # 30x30 building block of 15m height
    ndsm[20:50, 20:50] = 15.0

    res = extract_lod1_buildings(ndsm, terrain, gsd_m=1.0, min_height_m=2.5, min_area_m2=20.0)
    assert res["count"] >= 1
    b = res["buildings"][0]
    assert abs(b["height_m"] - 15.0) < 1.0
    assert abs(b["base_elev_m"] - 400.0) < 1.0
    assert len(b["coords"]) >= 4
