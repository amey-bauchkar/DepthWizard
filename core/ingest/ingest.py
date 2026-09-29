"""Image ingestion for Mode A (PNG / JPEG / non-georeferenced TIFF). Georeferenced GeoTIFFs go to Mode B
(core.ingest.geotiff).

Validates existence, extension, magic bytes, decodability, dimensions, channels; returns an RGB uint8 array,
an optional validity mask (from alpha / nodata), and an InputMeta that explicitly declares MODE_A /
non-georeferenced / relative-output-only. Nothing here ever claims a CRS or metric scale.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 1_000_000_000  # large satellite PNG / JPG are legitimate; uploads are size-capped (ingest.max_upload_mb)

from backend.errors import EmptyInputError, ImageTooLargeError, InvalidFileError, UnsupportedFormatError

MAGIC = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
    b"II+\x00": "TIFF",  # BigTIFF
    b"MM\x00+": "TIFF",
}
EXT_TO_FORMAT = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}


def _read_tiff_rgb(p: Path, max_dim: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Read any TIFF (8/16-bit, float, 1-4+ bands) as RGB uint8 via rasterio, percentile-stretching non-uint8 data."""
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    from core.geo.raster_io import _to_uint8

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(p) as ds:
            if ds.width < 8 or ds.height < 8:
                raise InvalidFileError(f"image too small: {ds.width}x{ds.height}")
            f = max(ds.width, ds.height) / max_dim
            shape = (int(round(ds.height / f)), int(round(ds.width / f))) if f > 1 else (ds.height, ds.width)
            from rasterio.enums import Resampling

            rd = (lambda b: ds.read(b, out_shape=shape, resampling=Resampling.average)) if f > 1 else (lambda b: ds.read(b))
            bands = [rd(1)] * 3 if ds.count < 3 else [rd(b) for b in (1, 2, 3)]
            u8, masks = zip(*[_to_uint8(b, ds.nodata) for b in bands])
            valid = masks[0] & masks[1] & masks[2]
            if ds.count >= 4 and any(str(ci).lower().endswith("alpha") for ci in ds.colorinterp):
                valid &= rd(ds.count) > 0
            return np.dstack(u8), (None if valid.all() else valid), int(ds.count)


@dataclass
class InputMeta:
    mode: str  # "A"
    mode_label: str  # "MODE_A / NON_GEOREFERENCED / RELATIVE_OUTPUT_ONLY"
    filename: str
    format: str
    width: int
    height: int
    channels: int
    dtype: str
    has_alpha: bool
    has_georeferencing: bool
    crs: str | None
    gsd_m: float | None
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestResult:
    rgb: np.ndarray  # HxWx3 uint8
    valid_mask: np.ndarray | None  # HxW bool (alpha > 0) or None
    meta: InputMeta


def sniff_format(head: bytes) -> str | None:
    for magic, fmt in MAGIC.items():
        if head.startswith(magic):
            return fmt
    return None


def ingest_image(path: str | Path, *, max_dim: int = 4096, allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> IngestResult:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise InvalidFileError(f"file not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise EmptyInputError(f"empty file: {p}")
    ext = p.suffix.lower()
    if ext not in allowed_extensions:
        raise UnsupportedFormatError(f"extension {ext!r} not in {allowed_extensions}")
    with p.open("rb") as f:
        head = f.read(16)
    fmt = sniff_format(head)
    if fmt is None:
        raise InvalidFileError(f"file signature does not match PNG/JPEG/TIFF: {head[:8]!r}")
    if EXT_TO_FORMAT[ext] != fmt:
        raise InvalidFileError(f"extension {ext} does not match signature {fmt}")
    if fmt == "TIFF":
        try:
            rgb, valid_mask, channels_in = _read_tiff_rgb(p, max_dim)
        except (ImageTooLargeError, InvalidFileError):
            raise
        except Exception as e:  # noqa: BLE001
            raise InvalidFileError(f"undecodable TIFF: {e}") from e
        return IngestResult(rgb=rgb, valid_mask=valid_mask, meta=InputMeta(
            mode="A", mode_label="MODE_A / NON_GEOREFERENCED / RELATIVE_OUTPUT_ONLY", filename=p.name, format=fmt,
            width=int(rgb.shape[1]), height=int(rgb.shape[0]), channels=channels_in, dtype="uint8", has_alpha=valid_mask is not None,
            has_georeferencing=False, crs=None, gsd_m=None, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size_bytes=int(size)))
    try:
        with Image.open(p) as im:
            im.load()  # forces full decode -> raises on truncated/corrupt data
            width, height = im.size
            if width < 8 or height < 8:
                raise InvalidFileError(f"image too small: {width}x{height}")
            if max(width, height) > max_dim:  # larger than the RAM budget: area-average down, never refused
                f = max(width, height) / max_dim
                im = im.resize((max(8, int(round(width / f))), max(8, int(round(height / f)))), Image.Resampling.BOX)
                width, height = im.size
            has_alpha = im.mode in ("RGBA", "LA", "PA") or ("transparency" in im.info)
            valid_mask = None
            if has_alpha:
                alpha = np.asarray(im.convert("RGBA"))[..., 3]
                valid_mask = alpha > 0
            channels_in = len(im.getbands())
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except ImageTooLargeError:
        raise
    except InvalidFileError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidFileError(f"undecodable image: {e}") from e
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    meta = InputMeta(
        mode="A",
        mode_label="MODE_A / NON_GEOREFERENCED / RELATIVE_OUTPUT_ONLY",
        filename=p.name,
        format=fmt,
        width=int(width),
        height=int(height),
        channels=int(channels_in),
        dtype="uint8",
        has_alpha=bool(has_alpha),
        has_georeferencing=False,
        crs=None,
        gsd_m=None,
        sha256=sha,
        size_bytes=int(size),
    )
    return IngestResult(rgb=rgb, valid_mask=valid_mask, meta=meta)
