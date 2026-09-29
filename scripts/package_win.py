"""Build the standalone Windows app and prove it works offline.

    python scripts/package_win.py            build dist/DepthWizard/ (one folder) + zip, then run its self-test
    python scripts/package_win.py --no-zip   skip the zip

Steps: 1. frontend build check  2. PyInstaller (packaging/depthwizard.spec)  3. dist/DepthWizard/DepthWizard.exe
--selftest with networking env vars pointing nowhere (proves the core runs offline)  4. zip for distribution.
Needs: pip install pyinstaller (dev only). The result runs on a Windows 10/11 x64 machine with no Python or Node.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    if not (ROOT / "frontend" / "dist" / "standalone" / "standalone.js").exists():
        raise SystemExit("frontend not built: cd frontend && npm ci && npm run build")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(ROOT / "packaging" / "depthwizard.spec")], check=True, cwd=ROOT)
    app = DIST / "DepthWizard"
    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file()) / 1e9
    print(f"built {app} ({size:.2f} GB) in {(time.time() - t0) / 60:.1f} min", flush=True)
    env = {**os.environ, "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
           "DW_DATA_DIR": str(DIST / "selftest_data"), "PYTHONPATH": ""}
    r = subprocess.run([str(app / "DepthWizard.exe"), "--selftest"], env=env, cwd=DIST)
    shutil.rmtree(DIST / "selftest_data", ignore_errors=True)
    if r.returncode != 0:
        raise SystemExit("self-test of the packaged app FAILED")
    if not a.no_zip:
        z = shutil.make_archive(str(DIST / "DepthWizard-win64"), "zip", DIST, "DepthWizard")
        print("zip:", z, f"{Path(z).stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
