import numpy as np
import pytest

from backend.errors import EmptyInputError, InvalidFileError, UnsupportedFormatError
from core.ingest.ingest import ingest_image


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_valid_png(tmp_path, png_bytes):
    res = ingest_image(_write(tmp_path, "a.png", png_bytes))
    assert res.meta.mode == "A" and res.meta.has_georeferencing is False and res.meta.crs is None and res.meta.gsd_m is None
    assert res.meta.mode_label == "MODE_A / NON_GEOREFERENCED / RELATIVE_OUTPUT_ONLY"
    assert res.rgb.shape == (240, 320, 3) and res.rgb.dtype == np.uint8 and res.valid_mask is None
    assert res.meta.format == "PNG" and len(res.meta.sha256) == 64


def test_valid_jpg(tmp_path, jpg_bytes):
    res = ingest_image(_write(tmp_path, "a.jpg", jpg_bytes))
    assert res.meta.format == "JPEG" and res.rgb.shape == (240, 320, 3)


def test_rgba_gives_valid_mask(tmp_path, rgba_bytes):
    res = ingest_image(_write(tmp_path, "a.png", rgba_bytes))
    assert res.meta.has_alpha and res.valid_mask is not None
    assert res.valid_mask[:30].sum() == 0 and res.valid_mask[30:].all()
    assert res.rgb.shape[2] == 3


def test_corrupted_file(tmp_path):
    with pytest.raises(InvalidFileError):
        ingest_image(_write(tmp_path, "bad.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 100))


def test_signature_mismatch(tmp_path, jpg_bytes):
    with pytest.raises(InvalidFileError):
        ingest_image(_write(tmp_path, "renamed.png", jpg_bytes))


def test_unsupported_extension(tmp_path, png_bytes):
    with pytest.raises(UnsupportedFormatError):
        ingest_image(_write(tmp_path, "a.gif", png_bytes))


def test_empty_file(tmp_path):
    with pytest.raises(EmptyInputError):
        ingest_image(_write(tmp_path, "e.png", b""))


def test_oversized_image_is_resampled_not_refused(tmp_path, png_bytes):
    res = ingest_image(_write(tmp_path, "a.png", png_bytes), max_dim=100)  # 320 x 240 checkerboard
    assert max(res.rgb.shape[:2]) == 100 and res.rgb.shape[:2] == (75, 100)


def test_oversized_geotiff_keeps_its_extent(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from core.ingest.geotiff import ingest_geotiff

    src = tmp_path / "big.tif"
    with rasterio.open(src, "w", driver="GTiff", width=400, height=300, count=3, dtype="uint8", crs="EPSG:32645", transform=from_origin(600000, 3000000, 0.6, 0.6)) as ds:
        ds.write(np.random.default_rng(0).integers(0, 255, (3, 300, 400), dtype=np.uint8))
    res = ingest_geotiff(src, tmp_path, max_dim=200)
    assert (res.grid.width, res.grid.height) == (200, 150)
    assert abs(res.meta.gsd_m - 1.2) < 1e-6 and res.meta.resampled["from_px"] == [400, 300]
    assert np.allclose(res.grid.bounds, (600000, 3000000 - 180, 600000 + 240, 3000000))


def test_missing_file(tmp_path):
    with pytest.raises(InvalidFileError):
        ingest_image(tmp_path / "nope.png")
