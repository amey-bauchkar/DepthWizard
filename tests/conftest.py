import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def checkerboard() -> np.ndarray:
    """240x320 RGB checkerboard with a white top-left marker and a black bottom-right marker."""
    img = np.zeros((240, 320, 3), np.uint8)
    yy, xx = np.mgrid[0:240, 0:320]
    chk = ((xx // 40 + yy // 40) % 2) == 0
    img[chk] = 200
    img[~chk] = 50
    img[:20, :20] = 255
    img[-20:, -20:] = 0
    return img


@pytest.fixture()
def png_bytes(checkerboard) -> bytes:
    b = io.BytesIO()
    Image.fromarray(checkerboard).save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def jpg_bytes(checkerboard) -> bytes:
    b = io.BytesIO()
    Image.fromarray(checkerboard).save(b, "JPEG", quality=95)
    return b.getvalue()


@pytest.fixture()
def rgba_bytes(checkerboard) -> bytes:
    rgba = np.dstack([checkerboard, np.full(checkerboard.shape[:2], 255, np.uint8)])
    rgba[:30, :, 3] = 0  # transparent strip -> nodata
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def stub_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_MODEL_NAME", "stub")
    monkeypatch.setenv("DW_DATA_DIR", str(tmp_path / "data"))
    from backend.config.settings import load_settings

    return load_settings()


@pytest.fixture()
def client(stub_settings):
    from fastapi.testclient import TestClient

    from backend.main import create_app

    return TestClient(create_app(stub_settings))


def wait_done(client, job_id: str, timeout: float = 60.0):
    import time

    t0 = time.time()
    while True:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("READY", "FAILED") or time.time() - t0 > timeout:
            return j
        time.sleep(0.05)
