"""In-memory cache for the per-job rasters that hazard screenings read.

A screening is re-run many times on one job (the water-level and slope sliders), and re-reading the same GeoTIFFs
dominated its latency. Entries are keyed by the source files' (mtime, size), so a re-processed job never gets stale
arrays, and the cache holds at most MAX_BYTES of arrays (least recently used entries are dropped first). Cached arrays
are read-only: a caller that needs to modify one must copy it.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

MAX_BYTES = 1_200_000_000

_lock = threading.Lock()
_store: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
_bytes = 0


def _stamp(paths: list[Path]) -> tuple:
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def _freeze(v: Any) -> int:
    """Mark every array in v read-only; return their total size in bytes."""
    if isinstance(v, np.ndarray):
        v.flags.writeable = False
        return int(v.nbytes)
    if isinstance(v, (tuple, list)):
        return sum(_freeze(x) for x in v)
    if isinstance(v, dict):
        return sum(_freeze(x) for x in v.values())
    return 0


def cached(name: str, paths: list[Path], build: Callable[[], Any], extra: tuple = ()) -> Any:
    """build() once per (name, extra, state of paths); later calls with unchanged files return the same objects."""
    global _bytes
    key = (name, extra, _stamp(paths))
    with _lock:
        hit = _store.get(key)
        if hit is not None:
            _store.move_to_end(key)
            return hit[0]
    val = build()
    size = _freeze(val)
    with _lock:
        if key not in _store and size <= MAX_BYTES:
            _store[key] = (val, size)
            _bytes += size
            while _bytes > MAX_BYTES and _store:
                _k, (_v, s) = _store.popitem(last=False)
                _bytes -= s
    return val


def clear() -> None:
    global _bytes
    with _lock:
        _store.clear()
        _bytes = 0
