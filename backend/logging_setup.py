"""Structured JSON logging. Application log to stderr; per-job log.jsonl written by the job manager."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("job_id", "stage", "code"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and getattr(h, "_dw", False) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    h._dw = True  # type: ignore[attr-defined]
    root.addHandler(h)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import warnings
    from rasterio.errors import NotGeoreferencedWarning

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)  # expected for Mode A pixel-space rasters


class JobLogger:
    """Appends structured events to <job_dir>/log.jsonl and mirrors to the app logger."""

    def __init__(self, job_dir: Path, job_id: str):
        self.path = job_dir / "log.jsonl"
        self.job_id = job_id
        self.log = logging.getLogger("depthwizard.job")

    def event(self, stage: str, msg: str, level: int = logging.INFO, **fields: Any) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": self.job_id, "stage": stage, "msg": msg, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.log.log(level, msg, extra={"job_id": self.job_id, "stage": stage, **({"code": fields["code"]} if "code" in fields else {})})

    def exception(self, stage: str, msg: str, **fields: Any) -> None:
        import traceback

        self.event(stage, msg, level=logging.ERROR, traceback=traceback.format_exc(), **fields)
