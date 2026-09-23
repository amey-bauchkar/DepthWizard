"""Vertical-reference handling with the Phase 8 C-1 safety guard.

Phase 8 (L0-01b) showed that with geoid grid files absent, PROJ silently selects a
"ballpark vertical transformation" that applies a 0 m correction -- even with only_best=True.
Over India that would label ellipsoidal heights as EGM2008 with a 24-99 m error.

Rules implemented here:
  1. Network access is disabled for PROJ (offline determinism).
  2. Required grid files must be present in a PROJ data directory (bundled or user dir).
  3. The selected pipeline description is inspected; anything containing "ballpark" is rejected.
  4. A known-point self-test asserts a non-zero undulation before a transformer is trusted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyproj
from pyproj import Transformer
from pyproj.exceptions import ProjError

from backend.errors import GridsMissingError, VerticalTransformUnsafeError

# Vertical CRS registry (EPSG codes) and the PROJ grid each requires.
VERTICAL_CRS = {
    "ellipsoidal": None,  # WGS84 ellipsoidal height (EPSG:4979 3D)
    "EGM96": "EPSG:5773",
    "EGM2008": "EPSG:3855",
}
COMPOUND_3D = {"EGM96": "EPSG:9707", "EGM2008": "EPSG:9518"}  # WGS84 + geoid height compound CRSs
REQUIRED_GRIDS = {"EGM96": "us_nga_egm96_15.tif", "EGM2008": "us_nga_egm08_25.tif"}

# Known point for self-test (Delhi). Phase 8 L0-01 values (online, PROJ 9.8.1): EGM96 -52.602, EGM2008 -52.555 m.
SELFTEST_POINT = (77.21, 28.61)
SELFTEST_EXPECTED_N = {"EGM96": -52.602, "EGM2008": -52.555}
SELFTEST_TOL_M = 0.05


@dataclass(frozen=True)
class GridStatus:
    name: str
    filename: str
    found: bool
    path: str | None


def proj_data_dirs() -> list[Path]:
    dirs: list[Path] = []
    extra = os.environ.get("DW_PROJ_GRIDS")
    if extra:
        dirs.append(Path(extra))
    bundled = Path(__file__).resolve().parents[2] / "assets" / "proj"
    if bundled.exists():
        dirs.append(bundled)
    try:
        dirs.append(Path(pyproj.datadir.get_user_data_dir()))
    except Exception:  # pragma: no cover
        pass
    dirs.append(Path(pyproj.datadir.get_data_dir()))
    return dirs


def grid_status(names: Iterable[str] = ("EGM96", "EGM2008")) -> list[GridStatus]:
    out = []
    for n in names:
        fn = REQUIRED_GRIDS[n]
        found = None
        for d in proj_data_dirs():
            p = d / fn
            if p.exists():
                found = p
                break
        out.append(GridStatus(n, fn, found is not None, str(found) if found else None))
    return out


def ensure_grids(names: Iterable[str] = ("EGM96", "EGM2008")) -> list[GridStatus]:
    """Raise GridsMissingError unless every required grid file is present locally."""
    st = grid_status(names)
    missing = [s for s in st if not s.found]
    if missing:
        raise GridsMissingError(
            "missing PROJ geoid grids: " + ", ".join(f"{m.name} ({m.filename})" for m in missing),
        )
    # make the user dir containing DW_PROJ_GRIDS visible to PROJ
    extra = os.environ.get("DW_PROJ_GRIDS")
    if extra:
        pyproj.datadir.append_data_dir(extra)
    for s in st:
        if s.found and s.path:
            try:
                pyproj.datadir.append_data_dir(str(Path(s.path).parent))
            except Exception:
                pass
    return st


def _crs_for(name_or_code: str) -> str:
    if name_or_code in COMPOUND_3D:
        return COMPOUND_3D[name_or_code]
    if name_or_code == "ellipsoidal":
        return "EPSG:4979"
    return name_or_code


def safe_vertical_transformer(src: str, dst: str) -> Transformer:
    """Build an offline transformer between vertical references and REJECT ballpark pipelines.

    src/dst: "ellipsoidal" | "EGM96" | "EGM2008" | explicit compound CRS code.
    """
    register_bundled_grids()
    pyproj.network.set_network_enabled(False)
    for name in (src, dst):
        if name in REQUIRED_GRIDS:
            ensure_grids([name])
    try:
        t = Transformer.from_crs(_crs_for(src), _crs_for(dst), always_xy=True, only_best=True)
    except ProjError as e:  # pragma: no cover - PROJ may raise when only_best cannot be honoured
        raise VerticalTransformUnsafeError(f"PROJ refused transformation {src}->{dst}: {e}") from e
    desc = (t.description or "").lower()
    # description may be lazy ("unavailable until proj_trans is called") -> force a call, then inspect
    lon, lat = SELFTEST_POINT
    _x, _y, h = t.transform(lon, lat, 0.0)
    desc = used_operation_description(t)
    if "ballpark" in desc.lower():
        raise VerticalTransformUnsafeError(f"PROJ selected a ballpark vertical transformation for {src}->{dst}: {desc}")
    if not np.isfinite(h):
        raise VerticalTransformUnsafeError(f"non-finite result from {src}->{dst} self-test")
    # self-test against known undulation when one side is ellipsoidal
    for geoid, expected in SELFTEST_EXPECTED_N.items():
        if {src, dst} == {"ellipsoidal", geoid}:
            n_est = -h if src == "ellipsoidal" else h  # h_geoid = h_ell - N  => N = -h when input h_ell=0
            if abs(n_est - expected) > SELFTEST_TOL_M:
                raise VerticalTransformUnsafeError(
                    f"self-test failed for {src}->{dst}: undulation {n_est:.3f} m vs expected {expected:.3f} m (grids wrong or ballpark)"
                )
    return t


def register_bundled_grids(path: str | os.PathLike[str] | None = None) -> None:
    """Make the bundled grid directory (assets/proj by default, or DW_PROJ_GRIDS) visible to PROJ, offline."""
    d = Path(path) if path else Path(os.environ.get("DW_PROJ_GRIDS", Path(__file__).resolve().parents[2] / "assets" / "proj"))
    if d.exists():
        os.environ.setdefault("DW_PROJ_GRIDS", str(d))
        pyproj.datadir.append_data_dir(str(d))
    pyproj.network.set_network_enabled(False)


def _vertical_code(name: str) -> str:
    return VERTICAL_CRS.get(name, name) or "ellipsoidal"


def safe_compound_transformer(horizontal_crs: str, src_v: str, dst_v: str) -> Transformer:
    """Transformer between two vertical references over the SAME projected horizontal CRS, offline,
    rejecting ballpark pipelines. src_v/dst_v: 'ellipsoidal' | 'EGM96' | 'EGM2008' | any vertical EPSG code (e.g. 'EPSG:5728' = LN02)."""
    register_bundled_grids()
    def compound(v: str) -> str:
        code = _vertical_code(v)
        return f"{horizontal_crs}" if code in (None, "ellipsoidal") else f"{horizontal_crs}+{code.split(':')[-1]}"
    src_c, dst_c = compound(src_v), compound(dst_v)
    if src_v == "ellipsoidal":
        src_c = CRS_3D_OF.get(horizontal_crs, horizontal_crs)
    if dst_v == "ellipsoidal":
        dst_c = CRS_3D_OF.get(horizontal_crs, horizontal_crs)
    try:
        t = Transformer.from_crs(src_c, dst_c, always_xy=True, only_best=True)
    except ProjError as e:
        raise VerticalTransformUnsafeError(f"PROJ refused {src_v}->{dst_v} over {horizontal_crs}: {e}") from e
    return t


CRS_3D_OF: dict[str, str] = {}


def used_operation_description(t: Transformer) -> str:
    """Description of the operation PROJ actually used for the last transform call (multi-candidate transformers
    report 'unavailable until proj_trans is called' in .description) — this is what the C-1 guard must inspect."""
    parts = [t.description or ""]
    try:
        op = t.get_last_used_operation()
        parts.append(op.description or "")
        parts.append(op.name or "")
    except Exception:  # noqa: BLE001 - no operation used yet
        pass
    return " | ".join(p for p in parts if p)


def check_transformer_not_ballpark(t: Transformer, x: float, y: float, label: str) -> str:
    _ = t.transform(x, y, 0.0)
    desc = used_operation_description(t)
    if "ballpark" in desc.lower():
        raise VerticalTransformUnsafeError(f"PROJ selected a ballpark vertical transformation for {label}: {desc}")
    return desc


def transform_heights_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray, horizontal_crs: str, src_v: str, dst_v: str) -> tuple[np.ndarray, dict]:
    """Convert heights between vertical references at projected positions (x, y in `horizontal_crs`)."""
    if src_v == dst_v:
        return np.asarray(z, float), {"transformed": False}
    t = safe_compound_transformer(horizontal_crs, src_v, dst_v)
    xs = np.asarray(x, float); ys = np.asarray(y, float)
    check_transformer_not_ballpark(t, float(xs.flat[0]), float(ys.flat[0]), f"{src_v}->{dst_v}")
    _x, _y, zz = t.transform(xs, ys, np.asarray(z, float))
    if not np.all(np.isfinite(zz)):
        raise VerticalTransformUnsafeError(f"non-finite heights from {src_v}->{dst_v}")
    desc = used_operation_description(t)  # populated only after a transform call
    if "ballpark" in desc.lower():
        raise VerticalTransformUnsafeError(f"PROJ selected a ballpark vertical transformation for {src_v}->{dst_v}: {desc}")
    return np.asarray(zz), {"transformed": True, "pipeline": desc}


def transform_heights(lon: np.ndarray, lat: np.ndarray, z: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert heights between vertical references at given geographic positions (safe path only)."""
    if src == dst:
        return np.asarray(z, dtype=np.float64)
    t = safe_vertical_transformer(src, dst)
    _x, _y, zz = t.transform(np.asarray(lon, float), np.asarray(lat, float), np.asarray(z, float))
    return np.asarray(zz)
