"""Deterministic preprocessing for the depth backbone (Sprint 1, Mode A).

Resizes the RGB image so that the SHORTER side equals `input_size` (DA-V2 'lower_bound' convention), keeps the
aspect ratio, rounds both sides to a multiple of `size_multiple` (14 for ViT-S/14), then applies ImageNet
mean/std normalisation. The original image is never modified; the manifest records every parameter so the
inverse mapping back to the original grid is exact.

The canonical-GSD branch (metric path) is deliberately NOT here — it belongs to Mode B (Sprint 2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


@dataclass
class PrepManifest:
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    resize_factor_x: float
    resize_factor_y: float
    resample: str
    input_size: int
    size_multiple: int
    normalize_mean: list[float]
    normalize_std: list[float]
    layout: str = "CHW float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_size(width: int, height: int, input_size: int, multiple: int) -> tuple[int, int]:
    """Shorter side -> input_size, aspect preserved, both sides rounded to nearest multiple (min = multiple)."""
    scale = input_size / min(width, height)
    w = max(multiple, int(round(width * scale / multiple)) * multiple)
    h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return w, h


def prepare(rgb: np.ndarray, *, input_size: int = 518, size_multiple: int = 14, mean: list[float] | None = None, std: list[float] | None = None, resample: str = "bicubic") -> tuple[np.ndarray, PrepManifest]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    h0, w0 = rgb.shape[:2]
    w1, h1 = target_size(w0, h0, input_size, size_multiple)
    im = Image.fromarray(rgb, mode="RGB").resize((w1, h1), RESAMPLE[resample])
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    manifest = PrepManifest(w0, h0, w1, h1, w1 / w0, h1 / h0, resample, input_size, size_multiple, list(mean), list(std))
    return chw, manifest
