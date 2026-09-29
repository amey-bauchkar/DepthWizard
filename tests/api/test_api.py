import numpy as np
import pytest

from tests.conftest import wait_done


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["model"]["available"] is True and j["model"]["name"] == "stub"


def test_system_semantics(client):
    j = client.get("/api/system").json()
    assert j["app"]["modes_supported"] == ["A", "B"] and "metric=false" in j["semantics"]["mode_A"]


def test_create_job_and_run_success(client, png_bytes):
    r = client.post("/api/jobs", files={"file": ("chk.png", png_bytes, "image/png")})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "UPLOADED" and job["input_filename"] == "chk.png"
    r = client.post(f"/api/jobs/{job['job_id']}/run")
    assert r.status_code == 202
    done = wait_done(client, job["job_id"])
    assert done["status"] == "READY", done.get("error")
    for k in ("preprocessing_ms", "inference_ms", "raster_ms", "total_ms"):
        assert k in done["stages_ms"] and done["stages_ms"][k] >= 0
    for st in ("CREATED", "UPLOADED", "PREPROCESSING", "INFERENCE", "RASTERIZING", "READY"):
        assert st in done["timestamps"]
    res = client.get(f"/api/jobs/{job['job_id']}/result").json()
    assert res["mode"] == "A" and res["metric"] is False and res["calibration_tier"] == "R" and res["units"] == "relative" and res["vertical_reference"] is None
    assert res["grid"]["crs"] is None and res["grid"]["width"] == 320 and res["grid"]["height"] == 240
    for name in res["artifacts"].values():
        assert client.get(f"/api/jobs/{job['job_id']}/artifact/{name}").status_code == 200
    md = client.get(f"/api/jobs/{job['job_id']}/metadata").json()
    assert md["meta"]["has_georeferencing"] is False and md["prep"]["inference_width"] % 14 == 0


def test_invalid_upload_rejected(client):
    r = client.post("/api/jobs", files={"file": ("x.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    r = client.post("/api/jobs", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_FILE"


def test_failed_job_is_honest(client):
    r = client.post("/api/jobs", files={"file": ("bad.png", b"not really a png", "image/png")})
    jid = r.json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    done = wait_done(client, jid)
    assert done["status"] == "FAILED" and done["error"]["code"] == "INVALID_FILE"
    assert done["error"]["message"] == "Unsupported or corrupted image."
    assert client.get(f"/api/jobs/{jid}/result").status_code == 409
    assert client.get(f"/api/jobs/{jid}/artifact/rdsm.tif").status_code == 404


def test_unknown_job(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.post("/api/jobs/doesnotexist/run").status_code == 404


def test_run_twice_rejected(client, png_bytes):
    jid = client.post("/api/jobs", files={"file": ("chk.png", png_bytes, "image/png")}).json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    wait_done(client, jid)
    assert client.post(f"/api/jobs/{jid}/run").status_code == 409


def test_repeatability_two_jobs_same_input_identical(client, png_bytes):
    ids = []
    for _ in range(2):
        jid = client.post("/api/jobs", files={"file": ("chk.png", png_bytes, "image/png")}).json()["job_id"]
        client.post(f"/api/jobs/{jid}/run")
        assert wait_done(client, jid)["status"] == "READY"
        ids.append(jid)
    a = client.get(f"/api/jobs/{ids[0]}/artifact/heightfield.f32").content
    b = client.get(f"/api/jobs/{ids[1]}/artifact/heightfield.f32").content
    assert a == b


def test_viewer_data_contract(client, rgba_bytes):
    jid = client.post("/api/jobs", files={"file": ("chk.png", rgba_bytes, "image/png")}).json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    assert wait_done(client, jid)["status"] == "READY"
    res = client.get(f"/api/jobs/{jid}/result").json()
    hf = res["heightfield"]
    raw = client.get(f"/api/jobs/{jid}/artifact/heightfield.f32").content
    arr = np.frombuffer(raw, "<f4")
    assert arr.size == hf["width"] * hf["height"]
    finite = arr[np.isfinite(arr)]
    assert finite.size > 0 and finite.min() >= 0.0 and finite.max() <= 1.0
    # transparent strip -> NaN nodata in heightfield
    grid = arr.reshape(hf["height"], hf["width"])
    assert np.isnan(grid[:30]).all() and np.isfinite(grid[30:]).all()
    # texture footprint matches heightfield footprint (same aspect ratio; full-res here)
    assert (hf["texture_width"], hf["texture_height"]) == (320, 240) and (hf["width"], hf["height"]) == (320, 240)
    assert abs(hf["texture_width"] / hf["texture_height"] - hf["width"] / hf["height"]) < 1e-9
    assert hf["metric"] is False and hf["calibration_tier"] == "R" and hf["units"] == "relative"


@pytest.mark.slow
def test_real_model_on_demo_tile(tmp_path, monkeypatch):
    """Runs the actual Depth Anything V2 Small weights (skipped if not installed)."""
    from pathlib import Path

    from backend.config.settings import load_settings

    monkeypatch.setenv("DW_MODEL_NAME", "da-v2-small-baseline")
    monkeypatch.setenv("DW_DATA_DIR", str(tmp_path / "data"))
    s = load_settings()
    if not (s.models_dir / "da-v2-small-baseline" / "1.0.0" / "depth_anything_v2_vits.pth").exists():
        pytest.skip("weights not installed")
    from fastapi.testclient import TestClient

    from backend.main import create_app

    c = TestClient(create_app(s))
    data = Path("assets/demo/sample_urban.jpg").read_bytes()
    jid = c.post("/api/jobs", files={"file": ("sample_urban.jpg", data, "image/jpeg")}).json()["job_id"]
    c.post(f"/api/jobs/{jid}/run")
    done = wait_done(c, jid, timeout=300)
    assert done["status"] == "READY", done.get("error")
    assert done["model"]["name"] == "da-v2-small-baseline" and done["model"]["sha256"].startswith("715fade1")
    res = c.get(f"/api/jobs/{jid}/result").json()
    assert res["metric"] is False and res["calibration_tier"] == "R"
    arr = np.frombuffer(c.get(f"/api/jobs/{jid}/artifact/heightfield.f32").content, "<f4")
    assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1 and arr.std() > 0.05


@pytest.mark.slow
def test_real_model_mode_b_demo_geotiff_beats_nothing_it_should_not(tmp_path, monkeypatch):
    """Real weights, bundled Zürich GeoTIFF + Copernicus DEM: tiled inference runs, tier T with calibrated detail,
    the fused DSM keeps the DEM's 30 m cell means (the fusion never shifts the DEM at its own resolution)."""
    from pathlib import Path

    import rasterio

    from backend.config.settings import load_settings

    monkeypatch.setenv("DW_MODEL_NAME", "da-v2-small-baseline")
    monkeypatch.setenv("DW_DATA_DIR", str(tmp_path / "data"))
    s = load_settings()
    tif = Path("assets/demo/swissimage_2019_2682-1247_2m.tif")
    if not (s.models_dir / "da-v2-small-baseline" / "1.0.0" / "depth_anything_v2_vits.pth").exists() or not tif.exists():
        pytest.skip("weights or demo assets not installed")
    from fastapi.testclient import TestClient

    from backend.main import create_app

    c = TestClient(create_app(s))
    jid = c.post("/api/jobs", files={"file": (tif.name, tif.read_bytes(), "image/tiff")}).json()["job_id"]
    c.post(f"/api/jobs/{jid}/run")
    done = wait_done(c, jid, timeout=600)
    assert done["status"] == "READY", done.get("error")
    assert done["mode"] == "B" and done["stages_ms"]["tiled_inference_ms"] > 0
    res = c.get(f"/api/jobs/{jid}/result").json()
    metric = bool(c.get("/health").json()["model"]["metric_model"]["available"])  # fine-tuned model installed?
    assert res["calibration_tier"] == "T" and res["vertical_reference"] == "EGM2008"
    assert ("METRIC_NDSM_MODEL" if metric else "OBJECT_SCALE_DEM_FIT") in res["flags"]
    rep = c.get(f"/api/jobs/{jid}/metadata").json()["calib_report"]
    assert rep["fusion"]["accepted"] and rep["tiled_inference"]["n_tiles"] >= 4
    assert rep["tiled_inference"]["quantity"] == ("metric_ndsm_metres" if metric else "relative_inverse_depth")
    d = Path(s.jobs_dir) / jid
    with rasterio.open(d / "dsm.tif") as a, rasterio.open(d / "terrain.tif") as t, rasterio.open(d / "ndsm.tif") as n:
        dsm, terr, ndsm = a.read(1, masked=True).filled(np.nan), t.read(1, masked=True).filled(np.nan), n.read(1, masked=True).filled(np.nan)
    ok = np.isfinite(dsm) & np.isfinite(terr)
    assert (terr[ok] <= dsm[ok] + 1e-3).all() and np.allclose((dsm - terr)[ok], ndsm[ok], atol=1e-3)


def test_non_georeferenced_tiff_runs_in_mode_a(client):
    """PS: PNG, JPG or TIFF input. A TIFF without CRS (16-bit here) is Mode A (relative), not rejected."""
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    rng = np.random.default_rng(0)
    arr = (rng.random((3, 64, 80)) * 4000).astype("uint16")
    mem = rasterio.MemoryFile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with mem.open(driver="GTiff", width=80, height=64, count=3, dtype="uint16") as ds:
            ds.write(arr)
    r = client.post("/api/jobs", files={"file": ("plain.tif", mem.read(), "image/tiff")})
    assert r.status_code == 201
    jid = r.json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    done = wait_done(client, jid)
    assert done["status"] == "READY", done.get("error")
    res = client.get(f"/api/jobs/{jid}/result").json()
    assert res["mode"] == "A" and res["calibration_tier"] == "R" and res["grid"]["width"] == 80


def test_delete_job(client, png_bytes):
    jid = client.post("/api/jobs", files={"file": ("chk.png", png_bytes, "image/png")}).json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    wait_done(client, jid)
    r = client.delete(f"/api/jobs/{jid}")
    assert r.status_code == 204 and r.content == b""
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_upload_size_cap_and_job_id_format(client, png_bytes):
    s = client.app.state.settings
    old = s.ingest.max_upload_mb
    s.ingest.max_upload_mb = 0.00001  # ~10 bytes
    try:
        r = client.post("/api/jobs", files={"file": ("big.png", png_bytes, "image/png")})
        assert r.status_code == 413 and r.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    finally:
        s.ingest.max_upload_mb = old
    for bad in ("..", "abc", "0123456789ab0", "..%5C..%5Cx"):
        assert client.get(f"/api/jobs/{bad}/report").status_code == 404


def test_zip_bomb_is_refused(tmp_path):
    import zipfile

    import pytest

    from core.ingest.isro import convert

    z = tmp_path / "bomb.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BAND1.tif", b"\0" * (3 * 1024 * 1024))
    with pytest.raises(ValueError, match="once extracted"):
        convert(z, tmp_path / "o.tif", max_uncompressed_mb=1)
