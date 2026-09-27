"""Before / after change screening on a synthetic pair with a known misregistration, collapses and a new structure."""
import json

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from scipy import ndimage

from core.change.detect import nmad, overlap_fraction, screen_change

N, GSD, X0, Y0, CRS = 240, 0.5, 400000.0, 3100000.0, "EPSG:32637"


def _texture(seed: int) -> np.ndarray:
    g = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(N, N)), 2.0)
    g = (g - g.min()) / (g.max() - g.min()) * 200 + 25
    return np.dstack([g, g * 0.9, g * 0.8]).astype(np.uint8)


def _scene():
    lab = np.zeros((N, N), np.int32)
    h = np.zeros((N, N))
    bld, k = [], 0
    rng = np.random.default_rng(3)
    for r0 in range(20, 200, 45):
        for c0 in range(20, 200, 45):
            k += 1
            lab[r0:r0 + 20, c0:c0 + 20] = k
            hk = float(rng.uniform(6, 15))
            h[r0:r0 + 20, c0:c0 + 20] = hk
            ex = (N - 1) * GSD
            ring = [[c * GSD - ex / 2, ex / 2 - r * GSD] for c, r in ((c0, r0), (c0 + 20, r0), (c0 + 20, r0 + 20), (c0, r0 + 20), (c0, r0))]
            bld.append({"id": k, "height_m": hk, "base_elev_m": 100.0, "area_m2": 400 * GSD * GSD, "coords": ring})
    return lab, h, bld


def _job(d, ndsm, rgb, *, buildings=None, labels=None):
    d.mkdir(parents=True)
    tr = from_origin(X0, Y0, GSD, GSD)
    with rasterio.open(d / "ndsm.tif", "w", driver="GTiff", width=N, height=N, count=1, dtype="float32", crs=CRS, transform=tr, nodata=-9999.0) as ds:
        ds.write(ndsm.astype("float32"), 1)
    Image.fromarray(rgb).save(d / "input_preview.png")
    res = {"mode": "B", "grid": {"transform": list(tr.to_gdal()), "crs": CRS, "width": N, "height": N}, "gsd_m": GSD, "artifacts": {"input_preview": "input_preview.png"}}
    if buildings is not None:
        (d / "buildings.json").write_text(json.dumps({"buildings": buildings, "grid": res["grid"], "gsd_m": GSD, "extent": [(N - 1) * GSD, (N - 1) * GSD]}), encoding="utf-8")
        with rasterio.open(d / "building_labels.tif", "w", driver="GTiff", width=N, height=N, count=1, dtype="int32", crs=CRS, transform=tr, nodata=0) as ds:
            ds.write(labels, 1)
        res["artifacts"]["buildings_json"] = "buildings.json"
    return res


def test_screen_change_finds_collapses_and_new_structure(tmp_path):
    lab, h, bld = _scene()
    rng = np.random.default_rng(7)
    rgb = _texture(1)
    pre = h + rng.normal(0, 0.4, h.shape)
    post_true = h.copy()
    post_true[lab == 3] = 0.8  # collapsed
    post_true[lab == 11] = 1.2  # collapsed
    post_true[200:225, 200:225] = 9.0  # new structure
    post = post_true + rng.normal(0, 0.4, h.shape)
    # the "after" acquisition is misregistered by (+3, -2) px: shift both its image and its heights
    shift = (3.0, -2.0)
    post = ndimage.shift(post, shift, order=1, mode="nearest")
    rgb_post = np.dstack([ndimage.shift(rgb[..., i].astype(float), shift, order=1, mode="nearest") for i in range(3)]).clip(0, 255).astype(np.uint8)
    pre_res = _job(tmp_path / "pre", pre, rgb, buildings=bld, labels=lab)
    post_res = _job(tmp_path / "post", post, rgb_post)
    assert overlap_fraction(pre_res, post_res) == 1.0
    out = screen_change(tmp_path / "pre", pre_res, tmp_path / "post", post_res, prefix="change_x", labels={"before": "t0", "after": "t1"})
    s = out["summary"]
    reg = s["registration"]
    assert reg["applied"] and abs(reg["dy_px"] + 3.0) < 0.3 and abs(reg["dx_px"] - 2.0) < 0.3
    cls = {b["id"]: b["class"] for b in out["buildings"]}
    assert cls[3] == "MAJOR_HEIGHT_LOSS" and cls[11] == "MAJOR_HEIGHT_LOSS"
    assert sum(c == "NO_SIGNIFICANT_CHANGE" for i, c in cls.items() if i not in (3, 11)) >= 12  # edges may lose cover
    assert s["buildings"]["counts"]["HEIGHT_GAIN"] == 0
    assert s["pixels"]["gain_area_m2"] > 100 and s["pixels"]["loss_area_m2"] > 150
    assert s["noise"]["nmad_pixels_m"] < 1.5 and s["noise"]["threshold_pixels_m"] >= 2.5
    for k in ("dh_tif", "class_tif", "overlay_png", "after_png", "json", "buildings_geojson", "buildings_csv"):
        assert (tmp_path / "pre" / s["artifacts"][k]).exists(), k
    with rasterio.open(tmp_path / "pre" / s["artifacts"]["class_tif"]) as ds:
        c = ds.read(1)
        assert ds.crs.to_string() == CRS and (c == -1).any() and (c == 1).any()
    gj = json.loads((tmp_path / "pre" / s["artifacts"]["buildings_geojson"]).read_text(encoding="utf-8"))
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 37.5 < lon < 38.5 and 27 < lat < 29  # UTM 37N, easting 400 km (100 km west of 39 E), northing 3100 km


def test_screen_change_refuses_disjoint_areas(tmp_path):
    lab, h, bld = _scene()
    pre_res = _job(tmp_path / "pre", h, _texture(1), buildings=bld, labels=lab)
    post_res = _job(tmp_path / "post", h, _texture(2))
    post_res["grid"]["transform"][0] += 10000.0  # 10 km away
    import pytest

    with pytest.raises(ValueError, match="barely overlap"):
        screen_change(tmp_path / "pre", pre_res, tmp_path / "post", post_res, prefix="c")


def test_nmad_is_robust_to_outliers():
    x = np.r_[np.random.default_rng(0).normal(0, 1.0, 5000), np.full(500, 30.0)]
    assert 0.9 < nmad(x) < 1.2
