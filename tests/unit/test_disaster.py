import json

import numpy as np
import pytest
import rasterio

from core.disaster.flood import run_flood_screening, classify_exposure
from core.disaster.accessibility import run_accessibility_screening
from core.geo.grid import Grid
from core.geo.raster_io import write_raster


@pytest.fixture
def mock_job_dir(tmp_path):
    job_dir = tmp_path / "job_test"
    job_dir.mkdir()

    # Create dummy terrain raster
    grid = Grid(width=10, height=10, transform=rasterio.Affine.translation(0, 0), dtype="float32", nodata=-9999.0, metric=True)
    
    # Terrain from 10 to 19
    terrain = np.linspace(10, 19, 100).reshape(10, 10).astype(np.float32)
    terrain[0, 0] = -9999.0  # nodata test
    write_raster(job_dir / "terrain.tif", terrain, grid)

    # Create dummy slope raster
    slope = np.full((10, 10), 10.0, dtype=np.float32)
    slope[5:, :] = 20.0
    write_raster(job_dir / "slope.tif", slope, grid)

    # Create dummy buildings
    buildings = {
        "buildings": [
            {"id": 1, "base_elev_m": 12.0, "height_m": 5.0, "area_m2": 100.0},
            {"id": 2, "base_elev_m": 16.0, "height_m": 10.0, "area_m2": 200.0},
        ]
    }
    (job_dir / "buildings.json").write_text(json.dumps(buildings))

    return job_dir


def test_classify_exposure():
    assert classify_exposure(-1.0) == "NONE"
    assert classify_exposure(0.0) == "NONE"
    assert classify_exposure(0.1) == "LOW"
    assert classify_exposure(1.0) == "MODERATE"
    assert classify_exposure(2.0) == "HIGH"
    assert classify_exposure(4.0) == "VERY HIGH"


def test_run_flood_screening(mock_job_dir):
    result = {"vertical_reference": "EGM2008", "calibration_tier": "T"}
    water_level = 15.0
    summary = run_flood_screening(mock_job_dir, water_level, result)

    assert summary["scenario"] == "flood_screening"
    assert summary["waterLevel_m"] == 15.0
    assert summary["affectedBuildingsCount"] == 1

    # Verify building
    b1 = summary["buildings"][0]
    assert b1["id"] == 1
    assert b1["flood_depth_m"] == 3.0
    assert b1["exposure"] == "HIGH"

    # Verify output rasters
    assert (mock_job_dir / "flood_depth.tif").exists()
    assert (mock_job_dir / "flood_preview.png").exists()

    with rasterio.open(mock_job_dir / "flood_depth.tif") as ds:
        depth = ds.read(1)
        # 0,0 is nodata, should be -9999
        assert depth[0, 0] == -9999.0
        # Value at 11 should be 15 - 11 = 4
        # Wait, the terrain is 10 to 19. It crosses 15 halfway.
        # Depth > 0 means flooded.


def test_run_flood_screening_below_min(mock_job_dir):
    result = {}
    summary = run_flood_screening(mock_job_dir, 5.0, result)
    assert summary["affectedAreaM2"] == 0.0
    assert summary["affectedBuildingsCount"] == 0


def test_run_accessibility_screening_legacy_slope_fallback(mock_job_dir):
    # jobs without a terrain layer fall back to slope.tif (surface slope) and say so
    (mock_job_dir / "terrain.tif").unlink()
    result = {}
    max_slope = 15.0
    summary = run_accessibility_screening(mock_job_dir, max_slope, result)
    
    assert summary["scenario"] == "accessibility_screening"
    assert summary["maxSlopeDeg"] == 15.0
    # Half the raster is slope 10, half is 20
    assert summary["accessibleAreaM2"] == 50.0  # 50 pixels * 1m^2
    assert (mock_job_dir / "accessibility.tif").exists()
    assert "surface slope" in summary["elevationSource"]
