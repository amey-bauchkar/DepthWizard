# PyInstaller spec: one-folder Windows bundle of DepthWizard (build with: python scripts/package_win.py).
# The repository layout is recreated inside the bundle's _internal/ folder, so every Path(__file__).parents[2]
# lookup (REPO_ROOT, assets/proj, ml/registry/vendor) resolves exactly as it does from source.
# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
# DEM tiles for the bundled demo scenes only (CartoDEM is NRSC-licensed and never bundled; study-only tiles excluded)
DEMS = ["Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif", "Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif",
        "Copernicus_DSM_COG_10_N37_00_E036_00_DEM.tif", "Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif",
        "Copernicus_DSM_COG_10_N47_00_E008_00_DEM.tif"]
MODELS = ["da-v2-small-baseline", "da-v2-small-ndsm"]  # da-v2-small-metric is not used by the app

datas = [
    (str(ROOT / "frontend" / "dist"), "frontend/dist"),
    (str(ROOT / "configs"), "configs"),
    (str(ROOT / "ml" / "registry" / "vendor"), "ml/registry/vendor"),
    (str(ROOT / "core" / "terrain" / "building_filter_model.json"), "core/terrain"),
    (str(ROOT / "core" / "inference" / "confidence_calibration.json"), "core/inference"),
    (str(ROOT / "models" / "INDEX.json"), "models"),
    (str(ROOT / "assets" / "dem" / "cartodem" / "README.md"), "assets/dem/cartodem"),
]
for sub in ("demo", "proj", "footprints", "waterways", "roads", "reference"):
    if (ROOT / "assets" / sub).exists():
        datas.append((str(ROOT / "assets" / sub), f"assets/{sub}"))
for d in DEMS:
    if (ROOT / "assets" / "dem" / d).exists():
        datas.append((str(ROOT / "assets" / "dem" / d), "assets/dem"))
for m in MODELS:
    datas.append((str(ROOT / "models" / m), f"models/{m}"))
for pkg in ("rasterio", "pyproj", "matplotlib", "certifi"):
    datas += collect_data_files(pkg)

hiddenimports = (collect_submodules("backend") + collect_submodules("core") + collect_submodules("uvicorn")
                 + collect_submodules("rasterio") + ["matplotlib.backends.backend_agg", "matplotlib.backends.backend_pdf", "networkx"])

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PySide6", "IPython", "notebook", "jupyter", "pytest", "ruff", "pyright", "torch.utils.tensorboard",
              # installed on the build machine but never imported by the app (checked with grep); keeps the bundle lean
              "transformers", "duckdb", "opentelemetry", "pandas", "sklearn", "pyarrow", "polars", "numba", "llvmlite",
              "onnxruntime", "jax", "tensorflow", "timm", "sliderule", "shapely", "xformers"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DepthWizard", console=True, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="DepthWizard")
