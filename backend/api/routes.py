"""HTTP API. Mode A (PNG/JPEG -> relative) and Mode B (GeoTIFF -> DEM-anchored absolute DSM).
All responses state mode/metric/tier/vertical reference explicitly; measurements are sampled server-side."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, Body, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.config.settings import REPO_ROOT
from backend.errors import DepthWizardError, InvalidFileError, JobNotFoundError, UnsupportedFormatError
from backend.jobs import query
from backend.jobs.manager import JobManager
from core.geo.vertical import grid_status

REFERENCE_DIR = REPO_ROOT / "assets" / "reference"
DEMO_DIR = REPO_ROOT / "assets" / "demo"

router = APIRouter()


def _mgr(request: Request) -> JobManager:
    return request.app.state.jobs


@router.get("/health")
def health(request: Request):
    mgr = _mgr(request)
    return {"status": "ok", "version": request.app.version, "model": mgr.model_status(), "device_available": {"cuda": torch.cuda.is_available(), "cpu": True}, "python": platform.python_version(), "torch": torch.__version__}


@router.get("/api/system")
def system(request: Request):
    import numpy, rasterio, pyproj  # noqa: E401

    mgr = _mgr(request)
    s = mgr.settings
    return {
        "app": {"name": "DepthWizard", "version": request.app.version, "modes_supported": ["A", "B"], "tiers_available": ["R", "T", "A"], "tiers_unavailable": {"H": "no fine-tuned metric head in this build (no GPU / no training data)"}},
        "runtime": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "numpy": numpy.__version__, "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str},
        "model": mgr.model_status(),
        "config": {"hash": s.config_hash(), "model": s.model.model_dump(), "ingest": s.ingest.model_dump(), "rdsm": s.rdsm.model_dump(), "calib": s.calib.model_dump(), "validation": s.validation.model_dump(), "terrain": s.terrain.model_dump()},
        "geoid_grids": [g.__dict__ for g in grid_status()],
        "dem_tiles": sorted(p.name for p in s.dem_dir.glob("*.tif")) if s.dem_dir.exists() else [],
        "semantics": {
            "mode_A": "PNG/JPEG -> relative surface structure (rDSM). Unitless. metric=false. calibration_tier=R. No vertical reference.",
            "mode_B": "GeoTIFF -> terrain layer (DEM, datum-transformed) + object layer (zero-shot relative structure x calibrated scale). metric=true. calibration_tier=T (DEM) or A (anchors). Object scale from DEM residual is UNVALIDATED; quality <= LIMITED unless anchors.",
        },
    }


@router.post("/api/jobs", status_code=201)
async def create_job(request: Request, file: UploadFile = File(...), dem: UploadFile | None = File(None), anchors: UploadFile | None = File(None), dem_vertical_crs: str = Form("EGM2008")):
    """Create a job from an image (PNG/JPEG -> Mode A; GeoTIFF -> Mode B). Optional: user DEM GeoTIFF (+ its vertical CRS) and an anchors CSV (id,x,y,z,type[,sigma])."""
    mgr = _mgr(request)
    if not file.filename:
        raise InvalidFileError("missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in mgr.settings.ingest.allowed_extensions:
        raise UnsupportedFormatError(f"extension {ext!r}")
    data = await file.read()
    if not data:
        raise InvalidFileError("empty upload")
    job = mgr.create()
    mgr.attach_upload(job.job_id, file.filename, data)
    if dem is not None and dem.filename:
        ddata = await dem.read()
        if ddata:
            mgr.attach_extra(job.job_id, "dem", dem.filename, ddata)
            mgr.set_input_option(job.job_id, "dem_vcrs", dem_vertical_crs)
    if anchors is not None and anchors.filename:
        adata = await anchors.read()
        if adata:
            mgr.attach_extra(job.job_id, "anchors", anchors.filename, adata)
    return mgr.get(job.job_id).to_dict()


@router.get("/api/demo")
def demo_items():
    """Bundled demo inputs and LiDAR references (swisstopo OGD, see assets/demo/README.md)."""
    items: list[dict[str, Any]] = []
    manifest = DEMO_DIR / "manifest.json"
    if manifest.exists():
        items = json.loads(manifest.read_text(encoding="utf-8")).get("items", [])
    refs = sorted(p.name for p in REFERENCE_DIR.glob("*.tif")) if REFERENCE_DIR.exists() else []
    return {"items": items, "references": refs}


@router.get("/api/jobs/{job_id}/sample")
def job_sample(request: Request, job_id: str, x: float = Query(...), y: float = Query(...), crs: str = Query("pixel")):
    mgr = _mgr(request)
    return query.sample(mgr._job_dir(job_id), mgr.result(job_id), x, y, crs)


@router.post("/api/jobs/{job_id}/measure")
def job_measure(request: Request, job_id: str, body: dict[str, Any] = Body(...)):
    mgr = _mgr(request)
    return query.measure(mgr._job_dir(job_id), mgr.result(job_id), body.get("points", []), body.get("crs", "pixel"))


@router.post("/api/jobs/{job_id}/validate")
async def job_validate(request: Request, job_id: str, reference: UploadFile | None = File(None), bundled: str | None = Form(None), ref_type: str = Form("dsm"), vertical_crs: str = Form("same"), source_note: str = Form(""), acquisition_date: str | None = Form(None)):
    """Validate a Mode B job against a reference raster (upload) or a bundled reference name. Synchronous."""
    mgr = _mgr(request)
    job_dir = mgr._job_dir(job_id)
    result = mgr.result(job_id)
    if reference is not None and reference.filename:
        data = await reference.read()
        if not data:
            raise InvalidFileError("empty reference upload")
        mgr.attach_extra(job_id, "reference", reference.filename, data)
        ref_path = job_dir / mgr.get(job_id).inputs["reference"]
        note = source_note or reference.filename
    elif bundled:
        ref_path = REFERENCE_DIR / Path(bundled).name
        if not ref_path.exists():
            raise JobNotFoundError(f"bundled reference {bundled!r} not found")
        note = source_note or f"bundled: {ref_path.name}"
    else:
        raise InvalidFileError("provide a reference upload or a bundled reference name")
    return query.validate_job(job_dir, result, ref_path, ref_type, vertical_crs, note, mgr.settings, acquisition_date)


@router.get("/api/jobs/{job_id}/validation")
def job_validation(request: Request, job_id: str):
    mgr = _mgr(request)
    mgr.get(job_id)
    p = mgr._job_dir(job_id) / "validation.json"
    if not p.exists():
        return {"runs": [], "latest": None}
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/api/jobs")
def list_jobs(request: Request):
    return [j.to_dict() for j in _mgr(request).list()]


@router.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str):
    return _mgr(request).get(job_id).to_dict()


@router.post("/api/jobs/{job_id}/run", status_code=202)
def run_job(request: Request, job_id: str):
    return _mgr(request).run(job_id).to_dict()


@router.get("/api/jobs/{job_id}/result")
def job_result(request: Request, job_id: str):
    return _mgr(request).result(job_id)


@router.get("/api/jobs/{job_id}/metadata")
def job_metadata(request: Request, job_id: str):
    return _mgr(request).metadata(job_id)


@router.get("/api/jobs/{job_id}/artifact/{name}")
def job_artifact(request: Request, job_id: str, name: str):
    p = _mgr(request).artifact_path(job_id, name)
    media = {".png": "image/png", ".jpg": "image/jpeg", ".tif": "image/tiff", ".json": "application/json", ".f32": "application/octet-stream", ".npy": "application/octet-stream", ".jsonl": "application/x-ndjson"}.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(p, media_type=media, filename=p.name)


@router.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(request: Request, job_id: str):
    _mgr(request).delete(job_id)
    return JSONResponse(status_code=204, content=None)


def error_response(exc: DepthWizardError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})
