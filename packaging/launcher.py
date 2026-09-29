"""DepthWizard standalone launcher (the entry point of the packaged app; also runs from source).

    DepthWizard.exe                 start the local server on a free port and open the browser
    DepthWizard.exe --no-browser    server only
    DepthWizard.exe --selftest      offline check: geoid grids, models, then one real demo job end to end; exit 0 / 1

Job data are written next to the executable (DepthWizard_data/), never inside the bundle, so the app also works when
the bundle folder itself is read-only. Everything runs on 127.0.0.1; nothing needs the internet except the optional
online extras (live rainfall, Sentinel-2 scar check, Bhuvan layers), which report when they are offline.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
HOME = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[1]
if not FROZEN:
    sys.path.insert(0, str(HOME))
os.environ.setdefault("DW_DATA_DIR", str(HOME / "DepthWizard_data"))


def free_port(preferred: int) -> int:
    for port in [preferred, *range(8001, 8050)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port between 8000 and 8049")


def selftest() -> int:
    """Offline end-to-end check. Prints one line per step; exit code 0 only if every step passed."""
    from fastapi.testclient import TestClient

    from backend.config.settings import REPO_ROOT, load_settings
    from backend.main import create_app

    ok = True

    def step(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}", flush=True)

    c = TestClient(create_app(load_settings()))  # registers the bundled geoid grids (offline PROJ)
    from core.geo.vertical import transform_heights_xy

    try:  # a point in Sikkim: EGM2008 undulation ~ -35 m; a 0 m answer means the geoid grid is missing (PROJ ballpark)
        import numpy as np

        z, _ = transform_heights_xy(np.array([88.64]), np.array([27.6]), np.array([0.0]), "EPSG:4326", "ellipsoidal", "EGM2008")
        d = float(z[0])
        step("geoid grids (ellipsoid -> EGM2008)", 20.0 < abs(d) < 60.0, f"{d:+.1f} m")
    except Exception as e:  # noqa: BLE001
        step("geoid grids (ellipsoid -> EGM2008)", False, repr(e))
    h = c.get("/health").json()
    step("model loaded", bool(h.get("model", {}).get("available", True)), str(h.get("model", {}).get("name", h)))
    demo = REPO_ROOT / "assets" / "demo" / "india" / "chungthang_rgb_0.5m.tif"
    t0 = time.perf_counter()
    jid = c.post("/api/jobs", files={"file": (demo.name, demo.read_bytes(), "image/tiff")}).json()["job_id"]
    c.post(f"/api/jobs/{jid}/run")
    while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
        time.sleep(1)
    step("demo job (Chungthang, Mode B)", j["status"] == "READY", f"{time.perf_counter() - t0:.0f} s")
    if j["status"] == "READY":
        r = c.get(f"/api/jobs/{jid}/result").json()
        step("metric DSM with a datum", bool(r.get("metric")) and r.get("vertical_reference") == "EGM2008", f"tier {r.get('calibration_tier')}")
        f = c.post(f"/api/jobs/{jid}/disaster/flood", json={"waterLevel_m": 10, "model": "river"})
        step("flood screening", f.status_code == 200)
        p = c.get(f"/api/jobs/{jid}/report.pdf")
        step("damage report PDF", p.status_code == 200 and p.content[:4] == b"%PDF", f"{len(p.content) // 1024} KB")
        s = c.get(f"/api/jobs/{jid}/export/scene.html")
        step("offline 3D scene export", s.status_code == 200 and len(s.content) > 1_000_000, f"{len(s.content) // 1_000_000} MB")
        c.delete(f"/api/jobs/{jid}")
    print("SELFTEST", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="DepthWizard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import uvicorn

    from backend.main import app

    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"DepthWizard is running at {url}\nJob data: {os.environ['DW_DATA_DIR']}\nClose this window to stop.", flush=True)
    if not a.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
