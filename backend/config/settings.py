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


class MetricModelCfg(BaseModel):
    """Optional fine-tuned metric nDSM model (tier H), used for Mode B tiled inference when installed.
    Install with scripts/install_finetuned_model.py; if missing, Mode B falls back to the zero-shot model."""
    enabled: bool = True
    name: str = "da-v2-small-ndsm"
    version: str = "latest"  # newest installed version, or pin e.g. "1.0.0"
    verify_hash: bool = True


class IngestCfg(BaseModel):
    max_image_dim: int = 6400  # working-grid limit (px); larger inputs are area-averaged to it (6000 px peaked at 6.6 GB RAM)
    max_upload_mb: float = 600.0  # any single uploaded file (image, DEM, zip product); larger -> 413
    max_zip_uncompressed_mb: float = 3000.0  # zipped ISRO product, total size once extracted (zip-bomb guard)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".tif", ".tiff"])


class CalibCfg(BaseModel):
    output_vertical_crs: str = "EGM2008"
    dem_dir: str = "assets/dem"
    dem_posting_m: float = 30.0
    dem_priority: list[str] = Field(default_factory=lambda: ["cartodem", "copernicus"])  # CartoDEM (assets/dem/cartodem/) preferred where it covers the scene
    cartodem_vertical_crs: str = "auto"  # auto = decided by comparison with Copernicus (EGM96 vs ellipsoidal); or EGM96 / ellipsoidal / EGM2008
    sigma_cells: float = 1.5
    w_min: float = 0.1
    ground_window_m: float = 60.0
    datum_sanity_m: float = 15.0
    anchor_min_n: int = 5
    anchor_holdout_fraction: float = 0.3
    anchor_accept_k: float = 3.0
    anchor_accept_nmad_m: float = 5.0  # offset accepted when fit NMAD <= max(k*sigma, this): terrain-layer scatter, not anchor noise, dominates


class FusionCfg(BaseModel):
    """Tiled inference + DEM-preserving detail fusion (core.calib.fusion)."""
    enabled: bool = True
    inference_gsd_m: float = 0.5  # Mode B: imagery is resampled so the model sees ~this GSD per tile pixel
    min_upsample: float = 0.25
    max_upsample: float = 4.0
    tile_px: int = 518  # DA-V2 native input size (multiple of 14)
    overlap: float = 0.25
    max_tiles: int = 400  # model tiles per job: full 0.5 m detail up to ~3.5 x 3.5 km at 0.6 m (CPU ~2 s / tile)
    max_tile_gain: float | None = None  # optional cap on the per-tile DEM-band gain (m per relative unit)
    anchor_gain_max: float = 4.0  # plausibility bound on the anchor-fitted detail gain
    mode_a_tiling: bool = True  # Mode A: refine large images with native-resolution tiles
    mode_a_metric_model: bool = True  # Mode A: use the fine-tuned nDSM model (when installed) instead of zero-shot depth (docs/validation_mode_a.md)
    metric_composition: Literal["highpass", "terrain_plus_ndsm"] = "highpass"  # how a metric nDSM model is combined with the DEM
    metric_ground_max_m: float = 1.0  # model nDSM below this = ground (terrain-layer support)
    metric_detail_gain: float = 1.0  # g: DSM = DEM + g * hp(nDSM)            (chosen on validation regions: scripts/select_dem_trust.py)
    metric_dem_object_fraction: float = 0.75  # f: terrain = DEM - f * lp(nDSM)  (share of the smoothed object height the DEM holds)


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
    model_metric: MetricModelCfg = Field(default_factory=MetricModelCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
    rdsm: RdsmCfg = Field(default_factory=RdsmCfg)
    calib: CalibCfg = Field(default_factory=CalibCfg)
    fusion: FusionCfg = Field(default_factory=FusionCfg)
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
    def footprints_dir(self) -> Path:
        """Bundled open building footprints (assets/footprints/index.json, scripts/fetch_building_footprints.py)."""
        return Path(os.environ.get("DW_FOOTPRINTS_DIR", REPO_ROOT / "assets" / "footprints"))

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
    sections = {"model", "ingest", "preprocess", "rdsm", "calib", "fusion", "validation", "terrain", "storage", "server"}
    for key, val in os.environ.items():
        if not key.startswith("DW_"):
            continue
        low = key[3:].lower()
        if low.startswith("model_metric_"):  # two-word section: DW_MODEL_METRIC_ENABLED=false
            section, field = "model_metric", low[len("model_metric_"):]
        else:
            parts = low.split("_", 1)
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
