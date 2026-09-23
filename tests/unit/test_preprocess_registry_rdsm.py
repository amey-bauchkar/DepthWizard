import json

import numpy as np
import pytest

from backend.errors import ModelUnavailableError
from core.dsm.rdsm import make_rdsm, write_raster
from core.inference.predictor import StubPredictor, select_device
from core.preprocess.preprocess import prepare, target_size
from ml.registry.registry import ModelRegistry


# ---------------- preprocessing ----------------
def test_target_size_multiple_of_14_and_lower_bound():
    w, h = target_size(500, 500, 518, 14)
    assert w % 14 == 0 and h % 14 == 0 and min(w, h) >= 518 - 14
    w, h = target_size(1000, 500, 518, 14)
    assert h == 518 and w == 1036


def test_prepare_deterministic_and_manifest(checkerboard):
    a, m1 = prepare(checkerboard, input_size=518, size_multiple=14)
    b, m2 = prepare(checkerboard, input_size=518, size_multiple=14)
    assert a.shape == (3, m1.inference_height, m1.inference_width)
    assert np.array_equal(a, b) and m1.to_dict() == m2.to_dict()
    assert (m1.original_width, m1.original_height) == (320, 240)
    assert m1.inference_width % 14 == 0 and m1.inference_height % 14 == 0
    assert abs(m1.resize_factor_x - m1.inference_width / 320) < 1e-12
    # normalisation: a pure-white pixel maps to (1-mean)/std
    white = np.full((64, 64, 3), 255, np.uint8)
    t, m = prepare(white, input_size=56, size_multiple=14)
    expect = (1.0 - np.array(m.normalize_mean)) / np.array(m.normalize_std)
    assert np.allclose(t[:, 5, 5], expect, atol=1e-5)


# ---------------- registry ----------------
def test_registry_resolves_stub(tmp_path):
    card = ModelRegistry(tmp_path).resolve("stub", "0")
    assert card.name == "stub" and card.output_quantity == "relative_brightness"


def test_registry_missing_index(tmp_path):
    with pytest.raises(ModelUnavailableError):
        ModelRegistry(tmp_path).resolve("da-v2-small-baseline", "1.0.0")


def test_registry_missing_weights_and_bad_hash(tmp_path):
    idx = {"models": {"m": {"1": {"file": "m/1/w.pth", "sha256": "00" * 32}}}}
    (tmp_path / "INDEX.json").write_text(json.dumps(idx))
    with pytest.raises(ModelUnavailableError):
        ModelRegistry(tmp_path).resolve("m", "1")
    (tmp_path / "m" / "1").mkdir(parents=True)
    (tmp_path / "m" / "1" / "w.pth").write_bytes(b"weights")
    with pytest.raises(ModelUnavailableError):  # checksum mismatch
        ModelRegistry(tmp_path).resolve("m", "1", verify_hash=True)
    card = ModelRegistry(tmp_path).resolve("m", "1", verify_hash=False)
    assert card.sha256_actual and card.sha256_actual != card.sha256_expected


def test_device_selection():
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in ("cpu", "cuda")
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(ModelUnavailableError):
            select_device("cuda")


def test_stub_predictor_shapes(stub_settings, checkerboard):
    p = StubPredictor(stub_settings).predict(checkerboard)
    assert p.relative_depth.shape == (240, 320) and p.output_quantity == "relative_brightness"
    assert np.isfinite(p.relative_depth).all()


# ---------------- rDSM ----------------
def test_rdsm_normalisation_semantics(tmp_path):
    rel_depth = np.linspace(2.0, 5.0, 100 * 80, dtype=np.float32).reshape(80, 100)
    rel, grid, stats = make_rdsm(rel_depth, method="minmax")
    assert rel.min() == 0.0 and rel.max() == 1.0 and rel.dtype == np.float32
    assert stats.metric is False and stats.calibration_tier == "R" and stats.units == "relative"
    assert grid.crs is None and grid.transform is None and grid.tier == "R" and grid.metric is False and grid.vertical_reference is None
    assert grid.to_dict()["has_georeferencing"] is False


def test_rdsm_nodata_and_percentiles(tmp_path):
    z = np.random.default_rng(0).normal(size=(50, 60)).astype(np.float32)
    mask = np.ones_like(z, bool)
    mask[:10] = False
    rel, grid, stats = make_rdsm(z, mask, method="percentile", percentiles=(1, 99), nodata=-9999.0)
    assert (rel[:10] == -9999.0).all() and ((rel[10:] >= 0) & (rel[10:] <= 1)).all()
    assert abs(stats.valid_fraction - 0.8) < 1e-6
    p = write_raster(tmp_path / "r.tif", rel, grid, {"MODEL": "stub@0"})
    import rasterio

    with rasterio.open(p) as ds:
        assert ds.nodata == -9999.0 and ds.crs is None and ds.width == 60 and ds.height == 50
        tags = ds.tags()
        assert tags["METRIC"] == "false" and tags["CALIBRATION_TIER"] == "R" and tags["UNITS"] == "relative" and tags["VERTICAL_CRS"] == "none" and tags["MODEL"] == "stub@0"
        back = ds.read(1)
    assert np.array_equal(back, rel)
