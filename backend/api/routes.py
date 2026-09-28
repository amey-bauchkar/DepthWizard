"""HTTP API. Mode A (PNG/JPEG -> relative) and Mode B (GeoTIFF -> DEM-anchored absolute DSM).
All responses state mode/metric/tier/vertical reference explicitly; measurements are sampled server-side."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import rasterio
import torch
from fastapi import APIRouter, Body, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.config.settings import REPO_ROOT
from backend.errors import DepthWizardError, InvalidFileError, JobNotFoundError, UnsupportedFormatError
from backend.jobs import query
from backend.jobs.manager import JobManager
from core.geo.vertical import grid_status
from core.disaster.flood import run_flood_screening
from core.disaster.accessibility import run_accessibility_screening
from core.disaster.landing_zones import run_landing_zone_screening
from core.screening_params import DEFAULT_MAX_SLOPE_DEG, HLZ_DEFAULT_SIZE

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
    has_h = mgr.metric_predictor() is not None
    return {
        "app": {"name": "DepthWizard", "version": request.app.version, "modes_supported": ["A", "B"], "tiers_available": ["R", "H", "T", "A"] if has_h else ["R", "T", "A"], "tiers_unavailable": {} if has_h else {"H": "no fine-tuned metric nDSM model installed (train with notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb, install with scripts/install_finetuned_model.py)"}},
        "runtime": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "numpy": numpy.__version__, "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str},
        "model": mgr.model_status(),
        "config": {"hash": s.config_hash(), "model": s.model.model_dump(), "ingest": s.ingest.model_dump(), "rdsm": s.rdsm.model_dump(), "calib": s.calib.model_dump(), "fusion": s.fusion.model_dump(), "validation": s.validation.model_dump(), "terrain": s.terrain.model_dump()},
        "geoid_grids": [g.__dict__ for g in grid_status()],
        "dem_tiles": sorted(p.name for p in s.dem_dir.glob("*.tif")) if s.dem_dir.exists() else [],
        "semantics": {
            "mode_A": "PNG/JPEG/non-georeferenced TIFF -> relative surface structure (rDSM). Unitless. metric=false. calibration_tier=R. No vertical reference.",
            "mode_B": "GeoTIFF -> DSM = DEM (datum-transformed) + zero-shot model detail below one DEM posting, scaled per tile against the DEM band (tiled inference). metric=true. calibration_tier=T (DEM) or A (anchors). DEM-band detail scale is UNVALIDATED without anchors/reference; quality <= LIMITED unless anchors.",
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


@router.post("/api/inspect")
async def inspect_input(request: Request, file: UploadFile = File(...), has_dem: bool = Form(False), has_anchors: bool = Form(False)):
    """Input check before a job: header + overview only; reports mode, resolution fit, DEM coverage, expected tier and accuracy."""
    from backend.jobs.inspect import inspect_upload

    mgr = _mgr(request)
    data = await file.read()
    if not data:
        raise InvalidFileError("empty upload")
    mp = mgr.metric_predictor()
    try:
        return inspect_upload(file.filename or "upload", data, mgr.settings, metric_card=mp.card if mp is not None else None, has_user_dem=has_dem, has_anchors=has_anchors)
    except rasterio.errors.RasterioIOError as e:
        raise InvalidFileError(f"unreadable GeoTIFF: {e}") from e


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


@router.post("/api/jobs/{job_id}/validate_points")
async def job_validate_points(request: Request, job_id: str, points: UploadFile | None = File(None), bundled: str | None = Form(None), source_note: str = Form("")):
    """Validate a Mode B job against sparse checkpoints (CSV: id,lon,lat,h_ground[,h_canopy,...], '# vcrs=...').
    `bundled` = a demo checkpoint file listed in assets/demo/manifest.json (e.g. ICESat-2 over Sikkim)."""
    mgr = _mgr(request)
    job_dir = mgr._job_dir(job_id)
    result = mgr.result(job_id)
    if points is not None and points.filename:
        data = await points.read()
        if not data:
            raise InvalidFileError("empty checkpoint upload")
        mgr.attach_extra(job_id, "checkpoints", points.filename, data)
        path = job_dir / mgr.get(job_id).inputs["checkpoints"]
        note = source_note or points.filename
    elif bundled:
        allowed = {i.get("reference_points") for i in json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8")).get("items", [])} - {None}
        if bundled not in allowed:
            raise JobNotFoundError(f"bundled checkpoints {bundled!r} not found")
        path = DEMO_DIR / bundled
        note = source_note or f"bundled: {bundled}"
    else:
        raise InvalidFileError("provide a checkpoint CSV upload or a bundled checkpoint name")
    return query.validate_points_job(job_dir, result, path, mgr.settings, note)


def _buildings(request: Request, job_id: str) -> dict[str, Any]:
    mgr = _mgr(request)
    result = mgr.result(job_id)
    name = (result.get("artifacts") or {}).get("buildings_json")
    if not name:
        raise JobNotFoundError("this job has no LoD-1 buildings (Mode B with model detail only)")
    from core.terrain import buildings as B

    return B.load(mgr._job_dir(job_id) / name)


@router.get("/api/jobs/{job_id}/buildings")
def job_buildings(request: Request, job_id: str, min_height: float = Query(0.0, ge=0), min_area: float = Query(0.0, ge=0), limit: int = Query(500, ge=1, le=20000)):
    """Building table (tallest first) with robust heights, measured typical error, ground/roof elevation, area, volume, floors range."""
    from core.terrain import buildings as B

    data = _buildings(request, job_id)
    recs = B.records(data, min_height_m=min_height, min_area_m2=min_area)
    return {"summary": B.summary(data, recs), "buildings": recs[:limit]}


@router.get("/api/jobs/{job_id}/buildings.geojson")
def job_buildings_geojson(request: Request, job_id: str, min_height: float = Query(0.0, ge=0), min_area: float = Query(0.0, ge=0)):
    from core.terrain import buildings as B

    data = _buildings(request, job_id)
    body = json.dumps(B.to_geojson(data, B.records(data, min_height_m=min_height, min_area_m2=min_area)))
    return Response(body, media_type="application/geo+json", headers={"content-disposition": f'attachment; filename="buildings_{job_id}.geojson"'})


@router.get("/api/jobs/{job_id}/buildings.csv")
def job_buildings_csv(request: Request, job_id: str, min_height: float = Query(0.0, ge=0), min_area: float = Query(0.0, ge=0)):
    from core.terrain import buildings as B

    data = _buildings(request, job_id)
    return Response(B.to_csv(B.records(data, min_height_m=min_height, min_area_m2=min_area)), media_type="text/csv", headers={"content-disposition": f'attachment; filename="buildings_{job_id}.csv"'})


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
    return Response(status_code=204)


@router.post("/api/jobs/{job_id}/disaster/flood")
def job_disaster_flood(request: Request, job_id: str, body: dict[str, Any] = Body(...)):
    mgr = _mgr(request)
    job_dir = mgr._job_dir(job_id)
    result = mgr.result(job_id)
    try:
        water_level = float(body.get("waterLevel_m", 0.0))
        return run_flood_screening(job_dir, water_level, result, connected_only=bool(body.get("connectedOnly", False)))
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_PARAMETER", "message": str(e)}})


@router.post("/api/jobs/{job_id}/disaster/accessibility")
def job_disaster_accessibility(request: Request, job_id: str, body: dict[str, Any] = Body(...)):
    mgr = _mgr(request)
    job_dir = mgr._job_dir(job_id)
    result = mgr.result(job_id)
    try:
        max_slope = float(body.get("maxSlopeDeg", DEFAULT_MAX_SLOPE_DEG))
        return run_accessibility_screening(job_dir, max_slope, result)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_PARAMETER", "message": str(e)}})



@router.post("/api/jobs/{job_id}/disaster/landing_zones")
def job_disaster_landing_zones(request: Request, job_id: str, body: dict[str, Any] = Body(...)):
    mgr = _mgr(request)
    job_dir = mgr._job_dir(job_id)
    result = mgr.result(job_id)
    try:
        size = int(body.get("size", HLZ_DEFAULT_SIZE))
        max_slope = body.get("maxSlopeDeg")
        return run_landing_zone_screening(job_dir, result, size=size, max_slope_deg=None if max_slope is None else float(max_slope), exclude_flooded=bool(body.get("excludeFlooded", False)))
    except (ValueError, TypeError) as e:
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_PARAMETER", "message": str(e)}})


def error_response(exc: DepthWizardError) -> JSONResponse:
    return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})
