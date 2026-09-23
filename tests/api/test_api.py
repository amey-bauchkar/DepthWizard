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
def test_real_metric_model_on_demo_tile(tmp_path, monkeypatch):
    """Runs the fine-tuned Depth Anything V2 Small metric head weights."""
    from pathlib import Path

    from backend.config.settings import load_settings

    monkeypatch.setenv("DW_MODEL_NAME", "da-v2-small-metric")
    monkeypatch.setenv("DW_DATA_DIR", str(tmp_path / "data"))
    s = load_settings()
    if not (s.models_dir / "da-v2-small-metric" / "1.0.0" / "depth_anything_v2_metric_s.pth").exists():
        pytest.skip("metric weights not installed")
    from fastapi.testclient import TestClient

    from backend.main import create_app

    c = TestClient(create_app(s))
    data = Path("assets/demo/sample_urban.jpg").read_bytes()
    jid = c.post("/api/jobs", files={"file": ("sample_urban.jpg", data, "image/jpeg")}).json()["job_id"]
    c.post(f"/api/jobs/{jid}/run")
    done = wait_done(c, jid, timeout=300)
    assert done["status"] == "READY", done.get("error")
    assert done["model"]["name"] == "da-v2-small-metric" and done["model"]["sha256"].startswith("fd6d317c")
    res = c.get(f"/api/jobs/{jid}/result").json()
    assert res["metric"] is False and res["calibration_tier"] == "R"
    arr = np.frombuffer(c.get(f"/api/jobs/{jid}/artifact/heightfield.f32").content, "<f4")
    assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1 and arr.std() > 0.05

