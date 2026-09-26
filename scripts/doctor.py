"""Environment doctor: reports versions and flags missing components. Exit code 1 if a REQUIRED item is missing."""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_PY = (3, 12)
REQUIRED = ["torch", "torchvision", "safetensors", "numpy", "scipy", "skimage", "rasterio", "pyproj", "PIL", "fastapi", "uvicorn", "pydantic", "yaml", "multipart"]


def ver(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "present")
    except Exception:
        return None


def cmd(*args: str) -> str | None:
    exe = shutil.which(args[0])
    if not exe:
        return None
    try:
        return subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main() -> int:
    rep: dict = {"os": platform.platform(), "python": platform.python_version(), "python_exe": sys.executable}
    problems: list[str] = []
    if sys.version_info[:2] < MIN_PY:
        problems.append(f"Python {platform.python_version()} < required {MIN_PY[0]}.{MIN_PY[1]}")
    rep["node"] = cmd("node", "--version")
    rep["npm"] = cmd("npm", "--version")
    if rep["node"] is None:
        problems.append("node not found (frontend build needs Node 20+; not needed if frontend/dist is already built)")
    for m in REQUIRED:
        rep[m] = ver(m)
        if rep[m] is None:
            problems.append(f"missing python package: {m}")
    try:
        import torch

        rep["cuda_available"] = torch.cuda.is_available()
        rep["torch_cuda_build"] = torch.version.cuda
        rep["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import rasterio, pyproj  # noqa: E401

        rep["gdal"] = rasterio.__gdal_version__
        rep["proj"] = pyproj.proj_version_str
        rep["proj_data_dir"] = pyproj.datadir.get_data_dir()
    except Exception:
        pass
    try:
        from core.geo.vertical import grid_status

        rep["geoid_grids"] = [s.__dict__ for s in grid_status()]
        if not all(s.found for s in grid_status()):
            rep["geoid_grids_note"] = "Geoid grids absent: Mode B absolute elevation will be REFUSED (Phase 8 C-1). Mode A is unaffected."
    except Exception as e:  # noqa: BLE001
        rep["geoid_grids"] = f"check failed: {e}"
    idx = ROOT / "models" / "INDEX.json"
    rep["model_index"] = idx.exists()
    if idx.exists():
        data = json.loads(idx.read_text(encoding="utf-8"))
        for name, versions in data["models"].items():
            for v, entry in versions.items():
                p = ROOT / "models" / entry["file"]
                rep[f"weights:{name}@{v}"] = {"present": p.exists(), "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None}
                if not p.exists():
                    problems.append(f"weights missing for {name}@{v}: run scripts/fetch_model.py")
    else:
        problems.append("models/INDEX.json missing")
    rep["frontend_dist"] = (ROOT / "frontend" / "dist" / "index.html").exists()
    if not rep["frontend_dist"]:
        problems.append("frontend not built: cd frontend && npm ci && npm run build")
    rep["dem_tiles"] = sorted(p.name for p in (ROOT / "assets" / "dem").glob("*.tif"))
    rep["problems"] = problems
    print(json.dumps(rep, indent=2, default=str))
    print()
    if problems:
        print("RESULT: problems found:")
        for pr in problems:
            print("  -", pr)
    else:
        print("RESULT: OK - ready to run (python run_server.py)")
    return 1 if any("missing python package" in p or "weights missing" in p or "INDEX.json" in p or "< required" in p for p in problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
