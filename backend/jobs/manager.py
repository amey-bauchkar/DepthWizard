"""Job manager: filesystem-backed jobs (data/jobs/<id>/job.json) + single in-process worker thread.

States: CREATED -> UPLOADED -> PREPROCESSING -> INFERENCE -> (Mode A: RASTERIZING | Mode B: CALIBRATION) -> READY | FAILED.
Timestamps and stage timings are recorded from actual clocks; nothing is estimated.
On startup, jobs found mid-flight are marked FAILED (interrupted) — never resumed silently.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.config.settings import Settings
from backend.errors import DepthWizardError, JobNotFoundError, JobStateError
from backend.jobs import pipeline, pipeline_b
from core.ingest.geotiff import is_georeferenced_tiff
from backend.logging_setup import JobLogger
from core.calib.fusion import needs_tiling
from core.inference.predictor import BasePredictor, build_metric_predictor, build_predictor

STATES = ["CREATED", "UPLOADED", "PREPROCESSING", "INFERENCE", "CALIBRATION", "RASTERIZING", "READY", "FAILED"]
IN_FLIGHT = {"PREPROCESSING", "INFERENCE", "CALIBRATION", "RASTERIZING"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Job:
    job_id: str
    status: str = "CREATED"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    input_filename: str | None = None
    input_sha256: str | None = None
    mode: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    stages_ms: dict[str, float] = field(default_factory=dict)
    timestamps: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    config_hash: str | None = None
    warnings: list[str] = field(default_factory=list)
    progress: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs_dir = settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dw-worker")
        self._predictor: BasePredictor | None = None
        self._predictor_error: DepthWizardError | None = None
        self._predictor_lock = threading.Lock()
        self._metric: BasePredictor | None = None
        self._metric_error: DepthWizardError | None = None
        self._metric_tried = False
        self._load_existing()

    # ---- persistence -------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _save(self, job: Job) -> None:
        job.updated_at = _now()
        (self._job_dir(job.job_id) / "job.json").write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _load_existing(self) -> None:
        for d in sorted(self.jobs_dir.glob("*")):
            jf = d / "job.json"
            if not jf.exists():
                continue
            try:
                job = Job(**json.loads(jf.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
            if job.status in IN_FLIGHT:
                job.status = "FAILED"
                job.error = {"code": "INTERRUPTED", "message": "Processing was interrupted (server restart).", "detail": "", "recoverable": True}
                self._save(job)
            self._jobs[job.job_id] = job

    # ---- model -------------------------------------------------------------
    def predictor(self) -> BasePredictor:
        with self._predictor_lock:
            if self._predictor is None:
                self._predictor = build_predictor(self.settings)  # raises ModelUnavailableError with a clear message
            return self._predictor

    def metric_predictor(self) -> BasePredictor | None:
        """Fine-tuned metric nDSM model (tier H) if installed and enabled; None otherwise (reason in model_status)."""
        with self._predictor_lock:
            if not self._metric_tried:
                self._metric_tried = True
                try:
                    self._metric = build_metric_predictor(self.settings)
                except DepthWizardError as e:
                    self._metric_error = e
            return self._metric

    def model_status(self) -> dict[str, Any]:
        mp = self.metric_predictor()
        metric = {"available": True, "name": mp.card.name, "version": mp.card.version, "sha256": mp.card.sha256_actual, "output_quantity": mp.card.output_quantity, "validation": (mp.card.extra or {}).get("validation")} if mp else {"available": False, "reason": self._metric_error.detail if self._metric_error else "not loaded"}
        try:
            p = self.predictor()
            return {"available": True, "name": p.card.name, "version": p.card.version, "device": p.device, "sha256": p.card.sha256_actual, "output_quantity": p.card.output_quantity, "licence": p.card.licence, "load_ms": getattr(p, "load_ms", None), "metric_model": metric}
        except DepthWizardError as e:
            return {"available": False, "error": e.to_dict(), "name": self.settings.model.name, "version": self.settings.model.version, "metric_model": metric}

    # ---- lifecycle ---------------------------------------------------------
    def create(self) -> Job:
        job_id = uuid.uuid4().hex[:12]
        d = self._job_dir(job_id)
        d.mkdir(parents=True, exist_ok=False)
        job = Job(job_id=job_id, config_hash=self.settings.config_hash())
        job.timestamps["CREATED"] = job.created_at
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def attach_upload(self, job_id: str, filename: str, data: bytes) -> Job:
        job = self.get(job_id)
        if job.status not in ("CREATED", "UPLOADED"):
            raise JobStateError(f"job {job_id} is {job.status}")
        safe = Path(filename).name or "upload.bin"
        ext = Path(safe).suffix.lower()
        target = self._job_dir(job_id) / f"input{ext}"
        target.write_bytes(data)
        job.input_filename = safe
        job.status = "UPLOADED"
        job.timestamps["UPLOADED"] = _now()
        self._save(job)
        return job

    def attach_extra(self, job_id: str, kind: str, filename: str, data: bytes) -> Job:
        """Attach optional calibration/validation inputs: kind in {dem, anchors, reference}."""
        job = self.get(job_id)
        if job.status in IN_FLIGHT:
            raise JobStateError("job is running")
        ext = Path(filename).suffix.lower() or ".bin"
        target = self._job_dir(job_id) / f"{kind}{ext}"
        target.write_bytes(data)
        job.inputs[kind] = target.name
        self._save(job)
        return job

    def set_input_option(self, job_id: str, key: str, value: str) -> None:
        job = self.get(job_id)
        job.inputs[key] = value
        self._save(job)

    def run(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status != "UPLOADED":
            raise JobStateError(f"job {job_id} is {job.status}; expected UPLOADED")
        self._set(job, "PREPROCESSING")
        self._executor.submit(self._execute, job_id)
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.status in IN_FLIGHT:
            raise JobStateError("cannot delete a running job")
        from backend.jobs.query import CACHE

        CACHE.drop(self._job_dir(job_id))
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)
        with self._lock:
            self._jobs.pop(job_id, None)

    def _set(self, job: Job, status: str, progress: dict[str, Any] | None = None) -> None:
        job.status = status
        job.timestamps[status] = _now()
        if progress is not None:
            job.progress = progress
        self._save(job)

    # ---- worker ------------------------------------------------------------
    def _execute(self, job_id: str) -> None:
        job = self.get(job_id)
        d = self._job_dir(job_id)
        log = JobLogger(d, job_id)
        inputs = sorted(d.glob("input.*"))
        try:
            if not inputs:
                raise DepthWizardError("no input file attached")
            src = inputs[0]
            t0 = time.perf_counter()
            self._set(job, "PREPROCESSING", {
                "percent": 5,
                "stage": "PREPROCESSING",
                "message": "Ingesting and validating input…",
                "eta_s": 50.0,
            })
            mode_b = src.suffix.lower() in (".tif", ".tiff") and is_georeferenced_tiff(src)
            job.mode = "B" if mode_b else "A"
            if mode_b:
                ing = pipeline_b.stage_ingest_geotiff(d, src, self.settings, log)
                rgb, valid_mask, meta_sha = ing.rgb, ing.valid_mask, ing.meta.sha256
            else:
                ing_a = pipeline.stage_ingest(d, src, self.settings, log)
                rgb, valid_mask, meta_sha = ing_a.rgb, ing_a.valid_mask, ing_a.meta.sha256
            job.input_sha256 = meta_sha
            job.stages_ms["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._set(job, "INFERENCE", {
                "percent": 15,
                "stage": "INFERENCE",
                "message": "Depth Anything V2 — whole image pass…",
                "eta_s": 45.0,
            })
            predictor = self.predictor()
            job.model = {"name": predictor.card.name, "version": predictor.card.version, "sha256": predictor.card.sha256_actual, "device": predictor.device}
            t1 = time.perf_counter()
            # stage_inference only needs .rgb; both ingest results expose it
            rel_depth, prov = pipeline.stage_inference(d, ing if mode_b else ing_a, predictor, log)  # type: ignore[arg-type]
            job.stages_ms["model_forward_ms"] = round(prov["timings_ms"]["inference_ms"], 1)
            # overlapping-tile inference: sub-DEM-posting detail (Mode B) / native-resolution detail (large Mode A images)
            tiled = None
            fz = self.settings.fusion
            if fz.enabled and (mode_b or (fz.mode_a_tiling and needs_tiling(rgb.shape[0], rgb.shape[1], self.settings.model.input_size))):
                t_tiles = time.perf_counter()
                # Mode B: the fine-tuned metric nDSM model when installed (tier H heights), else the zero-shot model
                tile_pred = (self.metric_predictor() or predictor) if mode_b else predictor
                if tile_pred is not predictor:
                    job.model["tiles"] = {"name": tile_pred.card.name, "version": tile_pred.card.version, "sha256": tile_pred.card.sha256_actual, "output_quantity": tile_pred.card.output_quantity}

                def on_tile_progress(done_n: int, total_n: int, elapsed_s: float) -> None:
                    # Scale tile progress linearly from 20% to 80%
                    pct = int(20 + round((done_n / max(total_n, 1)) * 60))
                    avg_per_tile = elapsed_s / max(done_n, 1)
                    rem_tiles = total_n - done_n
                    calib_est_s = 5.0 if mode_b else 1.0
                    eta_s = max(1.0, round(rem_tiles * avg_per_tile + calib_est_s, 1))
                    job.progress = {
                        "percent": pct,
                        "current": done_n,
                        "total": total_n,
                        "eta_s": eta_s,
                        "elapsed_s": round(elapsed_s, 1),
                        "stage": "INFERENCE",
                        "message": f"Tile {done_n}/{total_n} · Depth Anything V2",
                    }
                    self._save(job)

                tiled = pipeline_b.stage_tiled_inference(d, rgb, ing.meta.gsd_m if mode_b else None, tile_pred, self.settings, log, progress_cb=on_tile_progress)
                job.stages_ms["tiled_inference_ms"] = round((time.perf_counter() - t_tiles) * 1000, 1)
            job.stages_ms["inference_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            t2 = time.perf_counter()
            if mode_b:
                self._set(job, "CALIBRATION", {
                    "percent": 85,
                    "stage": "CALIBRATION",
                    "eta_s": 5.0,
                    "message": "DEM fusion, ground separation & LoD-1 buildings…",
                })
                dem_file = d / job.inputs["dem"] if "dem" in job.inputs else None
                anchors_file = d / job.inputs["anchors"] if "anchors" in job.inputs else None
                pipeline_b.stage_calibrate_and_compose(d, ing, rel_depth, prov, self.settings, log, tiled=tiled, user_dem=dem_file, user_dem_vcrs=job.inputs.get("dem_vcrs", "EGM2008"), anchors_path=anchors_file)
                job.stages_ms["calibration_ms"] = round((time.perf_counter() - t2) * 1000, 1)
            else:
                self._set(job, "RASTERIZING", {
                    "percent": 85,
                    "stage": "RASTERIZING",
                    "eta_s": 2.0,
                    "message": "Generating relative surface & heightfield…",
                })
                pipeline.stage_rasterize(d, ing_a, rel_depth, prov, self.settings, log, tiled=tiled)
                job.stages_ms["raster_ms"] = round((time.perf_counter() - t2) * 1000, 1)
            job.stages_ms["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._set(job, "READY", {
                "percent": 100,
                "stage": "READY",
                "eta_s": 0.0,
                "message": "Complete",
            })
            log.event("READY", "job complete", mode=job.mode, **job.stages_ms)
        except DepthWizardError as e:
            job.error = e.to_dict()
            log.exception(job.status, f"{e.code}: {e.detail}", code=e.code)
            self._set(job, "FAILED")
        except Exception as e:  # noqa: BLE001
            job.error = {"code": "INTERNAL_ERROR", "message": "Unable to process image.", "detail": f"{type(e).__name__}: {e}", "recoverable": False}
            log.event(job.status, "unexpected failure", level=40, traceback=traceback.format_exc(), code="INTERNAL_ERROR")
            self._set(job, "FAILED")

    def result(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job.status != "READY":
            raise JobStateError(f"job {job_id} is {job.status}")
        return json.loads((self._job_dir(job_id) / "result.json").read_text(encoding="utf-8"))

    def artifact_path(self, job_id: str, name: str) -> Path:
        self.get(job_id)
        safe = Path(name).name
        p = self._job_dir(job_id) / safe
        if not p.exists():
            raise JobNotFoundError(f"artifact {safe} not found for job {job_id}")
        return p

    def metadata(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        d = self._job_dir(job_id)
        out: dict[str, Any] = {"job": job.to_dict()}
        for name in ("meta.json", "prep.json", "prediction.json", "heightfield.json", "calib_report.json", "validation.json", "anchors_used.json"):
            p = d / name
            if p.exists():
                out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
        return out
