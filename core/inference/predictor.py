"""Depth backbone inference (Sprint 1). Output is RELATIVE (inverse-depth-like), never metres.

`Predictor.predict(rgb)` -> Prediction with `relative_depth` resampled to the ORIGINAL image grid,
timings, device and provenance. `StubPredictor` (brightness-based, deterministic) is used by tests and
selected with model.name == "stub".
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.config.settings import Settings
from backend.errors import InferenceFailedError, ModelUnavailableError
from core.preprocess.preprocess import PrepManifest, prepare
from ml.registry.registry import ModelCard, ModelRegistry

VENDOR = Path(__file__).resolve().parents[2] / "ml" / "registry" / "vendor"

DA_V2_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


@dataclass
class Prediction:
    relative_depth: np.ndarray  # HxW float32 on the ORIGINAL grid; unitless, higher = closer to camera
    model_name: str
    model_version: str
    model_sha256: str | None
    output_quantity: str
    device: str
    input_shape: tuple[int, int, int]  # C,H,W fed to the network
    output_shape: tuple[int, int]  # H,W of raw network output
    timings_ms: dict[str, float]
    prep: PrepManifest
    stats: dict[str, float] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("relative_depth", "prep")}
        d["prep"] = self.prep.to_dict()
        return d


def select_device(pref: str = "auto") -> str:
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise ModelUnavailableError("device=cuda requested but CUDA is not available")
        return "cuda"
    if pref == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BasePredictor:
    card: ModelCard
    device: str

    def _resample_to_original(self, raw: np.ndarray, h0: int, w0: int) -> np.ndarray:
        t = torch.from_numpy(raw)[None, None]
        out = F.interpolate(t, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
        return out.numpy().astype(np.float32)


class Predictor(BasePredictor):
    """Depth Anything V2 wrapper around the vendored model code."""

    def __init__(self, settings: Settings, registry: ModelRegistry | None = None):
        self.settings = settings
        reg = registry or ModelRegistry(settings.models_dir)
        self.card = reg.resolve(settings.model.name, settings.model.version, verify_hash=settings.model.verify_hash)
        self.device = select_device(settings.model.device)
        t0 = time.perf_counter()
        self.model = self._load(self.card)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def _load(self, card: ModelCard):
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from depth_anything_v2.dpt import DepthAnythingV2  # vendored, Apache-2.0
            enc = card.extra.get("encoder", "vits") if card.extra else "vits"
            model = DepthAnythingV2(**DA_V2_CONFIGS[enc])
            sd = torch.load(card.weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)
            model.eval().to(self.device)
            return model
        except ModelUnavailableError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailableError(f"failed to load model {card.name}@{card.version}: {e}") from e

    def predict(self, rgb: np.ndarray) -> Prediction:
        t_all = time.perf_counter()
        s = self.settings
        t0 = time.perf_counter()
        chw, prep = prepare(rgb, input_size=s.model.input_size, size_multiple=s.model.size_multiple, mean=s.preprocess.normalize_mean, std=s.preprocess.normalize_std, resample=s.preprocess.resample)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        try:
            x = torch.from_numpy(chw)[None].to(self.device)
            t1 = time.perf_counter()
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = self.model(x)
                else:
                    y = self.model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t1) * 1000.0
            raw = y[0].float().cpu().numpy().astype(np.float32)
        except Exception as e:  # noqa: BLE001
            raise InferenceFailedError(f"forward pass failed: {e}") from e
        if not np.all(np.isfinite(raw)):
            raise InferenceFailedError("model produced non-finite values")
        t2 = time.perf_counter()
        rel = self._resample_to_original(raw, prep.original_height, prep.original_width)
        post_ms = (time.perf_counter() - t2) * 1000.0
        return Prediction(
            relative_depth=rel,
            model_name=self.card.name,
            model_version=self.card.version,
            model_sha256=self.card.sha256_actual,
            output_quantity=self.card.output_quantity,
            device=self.device,
            input_shape=tuple(chw.shape),  # type: ignore[arg-type]
            output_shape=tuple(raw.shape),  # type: ignore[arg-type]
            timings_ms={"preprocess_ms": prep_ms, "inference_ms": infer_ms, "postprocess_ms": post_ms, "total_ms": (time.perf_counter() - t_all) * 1000.0, "model_load_ms": self.load_ms},
            prep=prep,
            stats={"raw_min": float(raw.min()), "raw_max": float(raw.max()), "raw_mean": float(raw.mean())},
        )


class StubPredictor(BasePredictor):
    """Deterministic test double: relative 'depth' = image brightness (0..1). Preserves geometry exactly,
    so texture/heightfield alignment tests can use a checkerboard and know the answer."""

    def __init__(self, settings: Settings, registry: ModelRegistry | None = None):
        self.settings = settings
        self.card = (registry or ModelRegistry(settings.models_dir)).resolve("stub", "0")
        self.device = "cpu"
        self.load_ms = 0.0

    def predict(self, rgb: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        s = self.settings
        chw, prep = prepare(rgb, input_size=s.model.input_size, size_multiple=s.model.size_multiple)
        gray = rgb.astype(np.float32).mean(axis=2) / 255.0
        return Prediction(gray.astype(np.float32), "stub", "0", None, "relative_brightness", "cpu", tuple(chw.shape), gray.shape, {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": (time.perf_counter() - t0) * 1000.0, "model_load_ms": 0.0}, prep, {"raw_min": float(gray.min()), "raw_max": float(gray.max()), "raw_mean": float(gray.mean())})  # type: ignore[arg-type]


def build_predictor(settings: Settings) -> BasePredictor:
    if settings.model.name == "stub":
        return StubPredictor(settings)
    return Predictor(settings)
