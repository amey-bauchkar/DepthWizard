"""Layered configuration: configs/default.yaml -> optional DW_CONFIG file -> environment (DW_*) -> per-job options.

Scientific parameters are never hard-coded elsewhere; modules receive a `Settings` object.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


class ModelCfg(BaseModel):
    name: str = "da-v2-small-baseline"
    version: str = "1.0.0"
    verify_hash: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    input_size: int = 518
    size_multiple: int = 14


class IngestCfg(BaseModel):
    max_image_dim: int = 4096
    allowed_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".tif", ".tiff"])


class CalibCfg(BaseModel):
    output_vertical_crs: str = "EGM2008"
    dem_dir: str = "assets/dem"
    dem_posting_m: float = 30.0
    sigma_cells: float = 1.5
    w_min: float = 0.1
    ground_window_m: float = 60.0
    datum_sanity_m: float = 15.0
    min_object_fraction: float = 0.3
    # Object scale plausibility gates (applied to the DEM residual scale fit)
    # min_implied_p99_m: minimum expected p99 object height (m). 0.5m allows modest buildings at 2m GSD.
    min_implied_p99_m: float = 0.5
    max_implied_p99_m: float = 80.0
    min_r: float = 0.05
    anchor_min_n: int = 5
    anchor_holdout_fraction: float = 0.3
    anchor_accept_k: float = 3.0
    anchor_accept_nmad_m: float = 5.0  # offset accepted when fit NMAD <= max(k*sigma, this): terrain-layer scatter, not anchor noise, dominates
    anchor_scale_min_inlier_fraction: float = 0.6


class ValidateCfg(BaseModel):
    border_px: int = 4
    max_shift_px: int = 4
    anchor_exclusion_radius_m: float = 15.0


class PreprocessCfg(BaseModel):
    normalize_mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    resample: Literal["bicubic", "bilinear"] = "bicubic"


class RdsmCfg(BaseModel):
    method: Literal["percentile", "minmax"] = "percentile"
    percentiles: list[float] = Field(default_factory=lambda: [1.0, 99.0])
    nodata: float = -9999.0
    orientation: str = "higher_value_means_higher_surface"


class TerrainCfg(BaseModel):
    max_mesh_dim: int = 768
    max_texture_dim: int = 2048


class StorageCfg(BaseModel):
    data_dir: str = "data"
    jobs_subdir: str = "jobs"


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    frontend_dist: str = "frontend/dist"


class Settings(BaseModel):
    model: ModelCfg = Field(default_factory=ModelCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
    rdsm: RdsmCfg = Field(default_factory=RdsmCfg)
    calib: CalibCfg = Field(default_factory=CalibCfg)
    validation: ValidateCfg = Field(default_factory=ValidateCfg)
    terrain: TerrainCfg = Field(default_factory=TerrainCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)

    # ---- derived paths -------------------------------------------------
    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def models_dir(self) -> Path:
        return Path(os.environ.get("DW_MODELS_DIR", REPO_ROOT / "models"))

    @property
    def dem_dir(self) -> Path:
        return Path(os.environ.get("DW_DEM_DIR", REPO_ROOT / self.calib.dem_dir))

    @property
    def proj_grids_dir(self) -> Path:
        return Path(os.environ.get("DW_PROJ_GRIDS", REPO_ROOT / "assets" / "proj"))

    @property
    def jobs_dir(self) -> Path:
        base = Path(os.environ.get("DW_DATA_DIR", REPO_ROOT / self.storage.data_dir))
        return base / self.storage.jobs_subdir

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _env_overrides() -> dict[str, Any]:
    """DW_MODEL_NAME=stub -> {"model": {"name": "stub"}}; DW_INGEST_MAX_IMAGE_DIM=1024 etc."""
    out: dict[str, Any] = {}
    sections = {"model", "ingest", "preprocess", "rdsm", "calib", "validation", "terrain", "storage", "server"}
    for key, val in os.environ.items():
        if not key.startswith("DW_"):
            continue
        parts = key[3:].lower().split("_", 1)
        if len(parts) != 2 or parts[0] not in sections:
            continue
        section, field = parts
        # coerce simple scalars
        v: Any = val
        if val.lower() in {"true", "false"}:
            v = val.lower() == "true"
        else:
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    pass
        out.setdefault(section, {})[field] = v
    return out


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    data: dict[str, Any] = {}
    default_path = REPO_ROOT / "configs" / "default.yaml"
    if default_path.exists():
        data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    extra = config_path or os.environ.get("DW_CONFIG")
    if extra:
        p = Path(extra)
        if not p.exists():
            raise FileNotFoundError(f"DW_CONFIG file not found: {p}")
        data = _deep_update(data, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    data = _deep_update(data, _env_overrides())
    return Settings.model_validate(data)
