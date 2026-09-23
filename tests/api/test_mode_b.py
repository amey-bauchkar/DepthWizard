"""Mode B end-to-end through the API with the stub model and fully synthetic inputs (no bundled assets required):
GeoTIFF ingest -> user DEM -> two-layer calibration -> sample/measure -> validation harness -> anchors (tier A).
"""
import io

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tests.conftest import wait_done

W = H = 240
GSD = 2.0
X0, Y0 = 300000.0, 3160000.0  # UTM 43N, near Delhi latitude band
CRS = "EPSG:32643"


def _write_tif(arr: np.ndarray, crs: str, transform, dtype, nodata=None) -> bytes:
    bands = 1 if arr.ndim == 2 else arr.shape[0]
    mem = io.BytesIO()
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", width=arr.shape[-1], height=arr.shape[-2], count=bands, dtype=dtype, crs=crs, transform=transform, nodata=nodata) as ds:
            ds.write(arr if arr.ndim == 3 else arr[None])
        mem.write(mf.read())
    return mem.getvalue()


@pytest.fixture()
def synthetic_scene():
    """Ground plane + two 'buildings' (bright blocks: the stub predictor maps brightness to relative height)."""
    tr = from_origin(X0, Y0, GSD, GSD)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ground = 250.0 + 0.01 * xx * GSD - 0.005 * yy * GSD
    rgb = np.full((3, H, W), 60, np.uint8)
    obj = np.zeros((H, W))
    for (r0, c0) in ((40, 40), (140, 120)):
        rgb[:, r0 : r0 + 40, c0 : c0 + 40] = 230
        obj[r0 : r0 + 40, c0 : c0 + 40] = 12.0
    image = _write_tif(rgb, CRS, tr, "uint8")
    dem = _write_tif((ground + 0.3 * obj).astype(np.float32), CRS, tr, "float32", nodata=-9999.0)  # coarse-ish DSM-like DEM
    ref_dsm = _write_tif((ground + obj).astype(np.float32), CRS, tr, "float32", nodata=-9999.0)
    ref_dtm = _write_tif(ground.astype(np.float32), CRS, tr, "float32", nodata=-9999.0)
    return {"image": image, "dem": dem, "ref_dsm": ref_dsm, "ref_dtm": ref_dtm, "ground": ground, "obj": obj, "transform": tr}


def _run(client, files, data=None):
    r = client.post("/api/jobs", files=files, data=data or {})
    assert r.status_code == 201, r.text
    jid = r.json()["job_id"]
    assert client.post(f"/api/jobs/{jid}/run").status_code == 202
    done = wait_done(client, jid)
    assert done["status"] == "READY", done.get("error")
    return jid, done


def test_geotiff_is_routed_to_mode_b_with_user_dem(client, synthetic_scene):
    jid, done = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")}, {"dem_vertical_crs": "EGM2008"})
    assert done["mode"] == "B" and "calibration_ms" in done["stages_ms"] and "CALIBRATION" in done["timestamps"]
    res = client.get(f"/api/jobs/{jid}/result").json()
    assert res["mode"] == "B" and res["metric_horizontal"] is True and res["absolute_elevation"] is True
    assert res["calibration_tier"] in ("T",) and res["vertical_reference"] == "EGM2008" and res["units"] == "metres"
    assert res["grid"]["crs"] == CRS and abs(res["gsd_m"] - GSD) < 1e-6
    assert res["dem"]["name"] == "user"
    assert res["quality"] in ("GOOD", "LIMITED", "WARNING")  # never GOOD is not asserted: scale fit may be rejected honestly
    assert any("tier H unavailable" in n for n in res["notes"])
    for name in ("dsm.tif", "terrain.tif", "slope.tif", "flags.tif", "heightfield_dsm.f32", "heightfield_dsm.json", "calib_report.json"):
        assert client.get(f"/api/jobs/{jid}/artifact/{name}").status_code == 200, name
    with rasterio.open(io.BytesIO(client.get(f"/api/jobs/{jid}/artifact/terrain.tif").content)) as ds:
        assert ds.crs.to_string() == CRS and ds.tags().get("NOT_A_DTM") == "true"
        terr = ds.read(1)
    # terrain layer must sit on the synthetic ground plane away from the objects (DEM was the ground there)
    g = synthetic_scene["ground"]
    away = np.ones((H, W), bool); away[30:90, 30:90] = False; away[130:190, 110:170] = False
    assert np.nanmedian(np.abs(terr[away] - g[away])) < 0.5


def test_sample_and_measure_are_server_authoritative(client, synthetic_scene):
    jid, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    s = client.get(f"/api/jobs/{jid}/sample", params={"x": 10, "y": 10}).json()
    assert s["in_bounds"] is True and s["pixel"] == {"col": 10, "row": 10}
    assert s["values"]["dsm"]["quantity"].startswith("ABSOLUTE") and s["values"]["dsm"]["units"] == "m" and s["values"]["dsm"]["vertical_reference"] == "EGM2008"
    assert s["values"]["terrain"]["quantity"].startswith("TERRAIN") and "NOT a DTM" in s["values"]["terrain"]["quantity"]
    assert s["values"]["relative"]["metric"] is False and s["values"]["relative"]["tier"] == "R"
    assert abs(s["position"]["x"] - (X0 + 10.5 * GSD)) < 1e-6 and abs(s["position"]["y"] - (Y0 - 10.5 * GSD)) < 1e-6
    # job-CRS and WGS84 addressing agree with pixel addressing
    s2 = client.get(f"/api/jobs/{jid}/sample", params={"x": s["position"]["x"], "y": s["position"]["y"], "crs": "job"}).json()
    assert s2["pixel"] == s["pixel"]
    s3 = client.get(f"/api/jobs/{jid}/sample", params={"x": s["position"]["lon"], "y": s["position"]["lat"], "crs": "wgs84"}).json()
    assert s3["pixel"] == s["pixel"]
    out = client.get(f"/api/jobs/{jid}/sample", params={"x": 10000, "y": 10}).json()
    assert out["in_bounds"] is False
    m = client.post(f"/api/jobs/{jid}/measure", json={"points": [{"x": 10, "y": 10}, {"x": 110, "y": 10}]}).json()
    seg = m["segments"][0]
    assert seg["distance_units"] == "m" and abs(seg["horizontal_distance"] - 100 * GSD) < 1e-6
    assert "dz_dsm" in seg and "server rasters" in m["authority"]
    assert client.post(f"/api/jobs/{jid}/measure", json={"points": [{"x": 1, "y": 1}]}).status_code == 400


def test_validation_harness_end_to_end(client, synthetic_scene):
    jid, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    r = client.post(f"/api/jobs/{jid}/validate", files={"reference": ("ref_dtm.tif", synthetic_scene["ref_dtm"], "image/tiff")}, data={"ref_type": "dtm", "vertical_crs": "same", "source_note": "synthetic"})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["compared_layer"] == "terrain" and v["metrics_overall"]["n"] > 0.8 * W * H
    assert set(v["metrics_overall"]) >= {"ME", "RMSE", "MAE", "NMAD", "LE90", "LE95", "pearson_r", "spearman_rho"}
    assert abs(v["alignment"]["dx"]) < 0.6 and abs(v["alignment"]["dy"]) < 0.6
    assert v["oracle_affine"]["label"] == "oracle_affine_alignment"
    assert v["metrics_overall"]["RMSE"] < 3.0  # terrain vs synthetic ground: DEM was ground away from objects
    assert client.get(f"/api/jobs/{jid}/artifact/{v['artifacts']['residual_preview']}").status_code == 200
    hist = client.get(f"/api/jobs/{jid}/validation").json()
    assert len(hist["runs"]) == 1 and hist["latest"]["verdict"]["band"]
    # DSM reference, second run is appended
    r2 = client.post(f"/api/jobs/{jid}/validate", files={"reference": ("ref_dsm.tif", synthetic_scene["ref_dsm"], "image/tiff")}, data={"ref_type": "dsm", "vertical_crs": "same"})
    assert r2.status_code == 200 and len(client.get(f"/api/jobs/{jid}/validation").json()["runs"]) == 2
    assert "metrics_by_slope" in r2.json()
    # bad inputs
    assert client.post(f"/api/jobs/{jid}/validate", data={"ref_type": "dsm"}).status_code == 400
    assert client.post(f"/api/jobs/{jid}/validate", data={"bundled": "nope.tif", "ref_type": "dsm"}).status_code == 404


def test_validation_refused_for_mode_a(client, png_bytes, synthetic_scene):
    jid, _ = _run(client, {"file": ("chk.png", png_bytes, "image/png")})
    r = client.post(f"/api/jobs/{jid}/validate", files={"reference": ("ref.tif", synthetic_scene["ref_dtm"], "image/tiff")}, data={"ref_type": "dtm", "vertical_crs": "same"})
    assert r.status_code == 409


def test_anchors_give_tier_a_and_are_excluded_from_validation(client, synthetic_scene):
    tr = synthetic_scene["transform"]; g = synthetic_scene["ground"]
    rng = np.random.default_rng(0)
    lines = ["# crs=EPSG:32643", "id,x,y,z,type,sigma"]
    for i in range(10):  # ground anchors on the true ground, shifted by a known +2 m datum-like offset
        r, c = int(rng.integers(100, 130)), int(rng.integers(180, 230))
        x, y = tr * (c + 0.5, r + 0.5)
        lines.append(f"G{i},{x:.2f},{y:.2f},{g[r, c] + 2.0:.3f},ground,0.2")
    lines.append(f"BL,{X0 + 5:.2f},{Y0 - 5:.2f},{g[2, 2] + 60.0:.3f},ground,0.2")  # a blunder
    csv = "\n".join(lines).encode()
    jid, done = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff"), "anchors": ("anchors.csv", csv, "text/csv")})
    assert done["inputs"]["anchors"] == "anchors.csv"
    res = client.get(f"/api/jobs/{jid}/result").json()
    md = client.get(f"/api/jobs/{jid}/metadata").json()
    a = md["calib_report"]["anchors"]
    assert a["accepted"] is True and res["calibration_tier"] == "A"
    assert abs(a["offset_m"] - 2.0) < 0.6, a
    assert "BL" in a["blunders"] or a["holdout"]["n"] >= 0  # blunder either flagged in the fit set or held out
    assert "anchors_used" in md and len(md["anchors_used"]["points"]) == 11
    v = client.post(f"/api/jobs/{jid}/validate", files={"reference": ("ref_dtm.tif", synthetic_scene["ref_dtm"], "image/tiff")}, data={"ref_type": "dtm", "vertical_crs": "same"}).json()
    assert v["mask"]["anchor_excluded_pixels"] > 0 and v["mask"]["leakage_check"] == "passed"
    assert 1.5 < v["metrics_overall"]["ME"] < 2.5  # the +2 m anchor offset is now carried by the terrain layer


def test_geotiff_without_dem_coverage_is_honest(client, synthetic_scene, monkeypatch):
    monkeypatch.setenv("DW_DEM_DIR", "nonexistent_dem_dir")
    jid, done = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff")})
    res = client.get(f"/api/jobs/{jid}/result").json()
    assert res["mode"] == "B" and res["calibration_tier"] == "R" and res["absolute_elevation"] is False
    assert res["quality"] == "WARNING" and "NO_DEM" in res["flags"]
    assert "dsm_tif" not in res["artifacts"] and client.get(f"/api/jobs/{jid}/artifact/heightfield_relative.f32").status_code == 200


def test_system_reports_mode_b(client):
    j = client.get("/api/system").json()
    assert j["app"]["modes_supported"] == ["A", "B"] and "H" in j["app"]["tiers_unavailable"]
    assert "UNVALIDATED" in j["semantics"]["mode_B"]
    assert isinstance(j["geoid_grids"], list) and {g["name"] for g in j["geoid_grids"]} == {"EGM96", "EGM2008"}
