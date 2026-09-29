"""Process the scenes used by the hazard-screening study (scripts/validate_hazards.py) with the CURRENT pipeline
(bundled footprints on), into data/hazard_study/<scene>/ + index.json.

Usage: python scripts/hazard_study_prepare.py [scene ...]
       python scripts/hazard_study_prepare.py chungthang --dem my_cartosat_10m.tif [--dem-vcrs EGM2008] [--tag cartosat]
         -> data/hazard_study/chungthang__cartosat/ processed with that DEM; then
            python scripts/validate_flood.py chungthang__cartosat   (scores it against the same observed flood)"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "hazard_study"
os.environ["DW_DATA_DIR"] = str(OUT / "_data")

from fastapi.testclient import TestClient  # noqa: E402

from backend.config.settings import load_settings  # noqa: E402
from backend.main import create_app  # noqa: E402

SCENES = {"teesta_west": "india/teesta_west_rgb_0.5m.tif", "namchi": "india/namchi_rgb_0.5m.tif", "chungthang": "india/chungthang_rgb_0.5m.tif",
          "zurich": "swissimage_2019_2682-1247_0.5m.tif", "islahiye": "change/islahiye_before_2022-12-27_0.5m.tif",
          "nepal_sunkoshi": "nepal/sunkoshi_rgb_0.5m.tif"}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--dem", type=Path, help="a DEM GeoTIFF to use instead of the bundled ones (e.g. Cartosat-1 10 m)")
    ap.add_argument("--dem-vcrs", default="EGM2008", help="vertical datum of --dem (EGM2008, EGM96 or ellipsoidal)")
    ap.add_argument("--tag", help="output suffix for --dem runs (default: the DEM file name)")
    a = ap.parse_args()
    c = TestClient(create_app(load_settings()))
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    for n in a.scenes or list(SCENES):
        f = ROOT / "assets" / "demo" / SCENES[n]
        files = {"file": (f.name, f.read_bytes(), "image/tiff")}
        data = {}
        if a.dem:
            files["dem"] = (a.dem.name, a.dem.read_bytes(), "image/tiff")
            data["dem_vertical_crs"] = a.dem_vcrs
        jid = c.post("/api/jobs", files=files, data=data).json()["job_id"]
        c.post(f"/api/jobs/{jid}/run")
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(1)
        if a.dem:
            n = f"{n}__{a.tag or a.dem.stem}"
        dst = OUT / n
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(OUT / "_data" / "jobs" / jid, dst)
        index[n] = {"job": jid, "status": j["status"], "input": str(f.relative_to(ROOT / "assets" / "demo")).replace("\\", "/"), "dem": str(a.dem) if a.dem else "bundled"}
        idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(n, j["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
