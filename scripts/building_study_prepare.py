"""Process the demo scenes used by the building-detection study (scripts/validate_buildings.py) with the current pipeline.

Writes data/building_study/<scene>/ (the job directory of each scene) and data/building_study/index.json.
The study measures the RAW image detector, so bundled footprints and the object filter are switched off here
(DW_FOOTPRINTS_DIR -> an empty folder, DW_LOD1_OBJECT_FILTER=0); the filter is trained and evaluated by validate_buildings.py.
Usage: python scripts/building_study_prepare.py [scene ...]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "building_study"
os.environ["DW_DATA_DIR"] = str(OUT / "_data")
os.environ["DW_FOOTPRINTS_DIR"] = str(OUT / "_no_footprints")
os.environ["DW_LOD1_OBJECT_FILTER"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from backend.config.settings import load_settings  # noqa: E402
from backend.main import create_app  # noqa: E402

SCENES = {
    "namchi": "india/namchi_rgb_0.5m.tif", "chungthang": "india/chungthang_rgb_0.5m.tif", "teesta_east": "india/teesta_east_rgb_0.5m.tif",
    "teesta_west": "india/teesta_west_rgb_0.5m.tif", "chungthang_west": "india/chungthang_west_rgb_0.5m.tif", "north_sikkim_alpine": "india/north_sikkim_alpine_rgb_0.5m.tif",
    "zurich": "swissimage_2019_2682-1247_0.5m.tif", "islahiye": "change/islahiye_before_2022-12-27_0.5m.tif",
}


def main() -> int:
    names = sys.argv[1:] or list(SCENES)
    c = TestClient(create_app(load_settings()))
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    for n in names:
        f = ROOT / "assets" / "demo" / SCENES[n]
        jid = c.post("/api/jobs", files={"file": (f.name, f.read_bytes(), "image/tiff")}).json()["job_id"]
        c.post(f"/api/jobs/{jid}/run")
        t0 = time.time()
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(1)
        src = OUT / "_data" / "jobs" / jid
        dst = OUT / n
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        index[n] = {"job": jid, "status": j["status"], "seconds": round(time.time() - t0), "input": SCENES[n]}
        print(n, j["status"], round(time.time() - t0), "s", flush=True)
        idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
