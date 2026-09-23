"""Image ingestion for Mode A (PNG/JPEG). GeoTIFF ingestion arrives in Sprint 2.

Validates existence, extension, magic bytes, decodability, dimensions, channels; returns an RGB uint8 array,
an optional validity mask (from alpha), and an InputMeta that explicitly declares MODE_A / non-georeferenced /
relative-output-only. Nothing here ever claims a CRS or metric scale for PNG/JPEG.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.errors import EmptyInputError, ImageTooLargeError, InvalidFileError, UnsupportedFormatError

MAGIC = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
}
EXT_TO_FORMAT = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


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


def ingest_image(path: str | Path, *, max_dim: int = 4096, allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> IngestResult:
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
    try:
        with Image.open(p) as im:
            im.load()  # forces full decode -> raises on truncated/corrupt data
            width, height = im.size
            if width < 8 or height < 8:
                raise InvalidFileError(f"image too small: {width}x{height}")
            if max(width, height) > max_dim:
                raise ImageTooLargeError(f"{width}x{height} exceeds max_image_dim={max_dim}")
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
