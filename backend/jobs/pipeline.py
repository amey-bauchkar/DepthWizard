"""Mode A pipeline stages (Sprint 1). Each stage is a pure function over the job directory and returns the
artefacts it wrote plus timings. Stages: PREPROCESSING -> INFERENCE -> RASTERIZING (rDSM + heightfield)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.config.settings import Settings
from backend.logging_setup import JobLogger
from core.dsm.rdsm import make_rdsm, write_preview, write_raster
from core.ingest.ingest import IngestResult, ingest_image
from core.inference.predictor import BasePredictor
from core.terrain.heightfield import build_heightfield, write_heightfield, write_texture


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def stage_ingest(job_dir: Path, input_path: Path, settings: Settings, log: JobLogger) -> IngestResult:
    t0 = time.perf_counter()
    res = ingest_image(input_path, max_dim=settings.ingest.max_image_dim, allowed_extensions=tuple(settings.ingest.allowed_extensions))
    _dump(job_dir / "meta.json", res.meta.to_dict())
    # display copy of the input (PNG for the browser; original is untouched)
    Image.fromarray(res.rgb, "RGB").save(job_dir / "input_preview.png")
    log.event("UPLOADED", "ingested", width=res.meta.width, height=res.meta.height, sha256=res.meta.sha256, format=res.meta.format, ms=round((time.perf_counter() - t0) * 1000, 1))
    return res


def stage_inference(job_dir: Path, ing: IngestResult, predictor: BasePredictor, log: JobLogger) -> tuple[np.ndarray, dict[str, Any]]:
    pred = predictor.predict(ing.rgb)
    prov = pred.provenance()
    _dump(job_dir / "prep.json", {**pred.prep.to_dict(), "device": pred.device, "model_input_size": pred.prep.input_size})
    _dump(job_dir / "prediction.json", prov)
    np.save(job_dir / "relative_depth.npy", pred.relative_depth)
    log.event("INFERENCE", "prediction complete", model=pred.model_name, version=pred.model_version, device=pred.device, input_shape=list(pred.input_shape), output_shape=list(pred.output_shape), **{k: round(v, 1) for k, v in pred.timings_ms.items()})
    return pred.relative_depth, prov


def stage_rasterize(job_dir: Path, ing: IngestResult, rel_depth: np.ndarray, prov: dict[str, Any], settings: Settings, log: JobLogger) -> dict[str, Any]:
    t0 = time.perf_counter()
    rc = settings.rdsm
    rel, grid, stats = make_rdsm(rel_depth, ing.valid_mask, method=rc.method, percentiles=tuple(rc.percentiles), nodata=rc.nodata, orientation=rc.orientation)  # type: ignore[arg-type]
    tags = {"MODEL": f"{prov['model_name']}@{prov['model_version']}", "MODEL_SHA256": prov.get("model_sha256") or "n/a", "INPUT_SHA256": ing.meta.sha256, "OUTPUT_QUANTITY": "relative_height_normalised", "SOURCE_QUANTITY": prov.get("output_quantity", "")}
    write_raster(job_dir / "rdsm.tif", rel, grid, tags)
    write_preview(job_dir / "rdsm_preview.png", rel, rc.nodata, mode="ramp")
    write_preview(job_dir / "depth_preview.png", rel, rc.nodata, mode="gray")
    raster_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    tex_path, tex_size = write_texture(job_dir / "texture.jpg", ing.rgb, max_dim=settings.terrain.max_texture_dim)
    heights, valid, hmeta = build_heightfield(rel, rc.nodata, max_mesh_dim=settings.terrain.max_mesh_dim, texture_size=tex_size)
    write_heightfield(job_dir / "heightfield.f32", heights)
    _dump(job_dir / "heightfield.json", hmeta.to_dict())
    hf_ms = (time.perf_counter() - t1) * 1000.0
    result = {
        "mode": "A",
        "mode_label": ing.meta.mode_label,
        "metric": False,
        "calibration_tier": "R",
        "units": "relative",
        "vertical_reference": None,
        "grid": grid.to_dict(),
        "rdsm_stats": stats.to_dict(),
        "heightfield": hmeta.to_dict(),
        "artifacts": {
            "input_preview": "input_preview.png",
            "depth_preview": "depth_preview.png",
            "rdsm_preview": "rdsm_preview.png",
            "rdsm_tif": "rdsm.tif",
            "relative_depth_npy": "relative_depth.npy",
            "heightfield": "heightfield.f32",
            "heightfield_meta": "heightfield.json",
            "texture": "texture.jpg",
            "prep": "prep.json",
            "prediction": "prediction.json",
            "meta": "meta.json",
        },
        "timings_ms": {"raster_ms": raster_ms, "heightfield_ms": hf_ms},
    }
    _dump(job_dir / "result.json", result)
    log.event("RASTERIZING", "rDSM + heightfield written", raster_ms=round(raster_ms, 1), heightfield_ms=round(hf_ms, 1), mesh_dim=[hmeta.width, hmeta.height], downsample_factor=hmeta.downsample_factor)
    return result
