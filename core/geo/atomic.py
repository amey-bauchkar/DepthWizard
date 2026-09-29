"""Atomic file replacement that survives Windows file locks.

Every writer uses a UNIQUE temporary name (two overlapping requests never share one), then os.replace with short
retries: on Windows a file that a browser or another request is reading is briefly locked (WinError 32).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Callable


def unique_tmp(path: Path) -> Path:
    return path.with_name(f".{path.stem}_{uuid.uuid4().hex[:8]}.tmp{path.suffix}")


def replace_with_retry(tmp: Path, path: Path, attempts: int = 8) -> None:
    for k in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.05 * (k + 1))


def write_atomic(path: Path, write: Callable[[Path], None]) -> Path:
    """write(tmp_path) produces the file; it then atomically replaces `path` (the temp file never survives)."""
    path = Path(path)
    tmp = unique_tmp(path)
    try:
        write(tmp)
        replace_with_retry(tmp, path)
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path
