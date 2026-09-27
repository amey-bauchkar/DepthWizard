"""Standalone offline 3D scene: everything the 3D explorer needs, embedded in ONE html file that opens by double-click
(no server, no Python, no internet). Heightfields are the job's viewer heightfields quantized to 16 bit (step <= 1 cm
for scenes with < 650 m of relief, reported in the file), the texture is the job's texture.jpg, buildings are the
LoD-1 blocks. Every number shown by the offline viewer is sampled from these embedded arrays and says so.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import Transformer

LAYER_LABELS = {
    "dsm": "DSM · surface elevation (ground + buildings + trees)",
    "terrain": "Terrain · bare-ground elevation (DEM-derived)",
    "ndsm": "nDSM · height above ground",
    "relative": "Relative structure (no units)",
}
BUILDING_KEYS = ("id", "height_m", "height_median_m", "height_p10_m", "height_p90_m", "base_elev_m", "ground_min_m", "roof_elev_m", "area_m2", "volume_m3", "floors_range", "quality_flags")


def quantize(z: np.ndarray) -> dict[str, Any]:
    """float heightfield -> uint16 + offset/scale (65535 = nodata). Max rounding error = scale / 2."""
    finite = np.isfinite(z)
    lo = float(np.min(z[finite])) if finite.any() else 0.0
    hi = float(np.max(z[finite])) if finite.any() else 0.0
    scale = (hi - lo) / 65000.0 if hi > lo else 1e-3
    q = np.full(z.shape, 65535, np.uint16)
    q[finite] = np.clip(np.round((z[finite] - lo) / scale), 0, 65000).astype(np.uint16)
    return {"b64": base64.b64encode(q.astype("<u2").tobytes()).decode("ascii"), "offset": lo, "scale": scale, "nodata": 65535, "max_rounding_error": scale / 2.0}


def dequantize(q: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(q["b64"]), "<u2").reshape(shape).astype(np.float64)
    out = q["offset"] + a * q["scale"]
    out[a == q["nodata"]] = np.nan
    return out


def _measurement_grid(tif: Path, f: int, shape: tuple[int, int]) -> np.ndarray | None:
    """nodata-aware f x f block mean of a job raster on the heightfield grid (ceil blocks, as the heightfield)."""
    if not tif.exists():
        return None
    import rasterio

    from core.terrain.heightfield import _block_sum_padded

    with rasterio.open(tif) as ds:
        a = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
    if f > 1:
        valid = np.isfinite(a)
        s = _block_sum_padded(np.where(valid, a, 0.0), f)
        n = _block_sum_padded(valid.astype(np.float64), f)
        a = np.where(n > 0, s / np.maximum(n, 1e-9), np.nan)
    return a if a.shape == tuple(shape) else None


def _lonlat_grid(result: dict[str, Any], n: int = 5) -> dict[str, Any] | None:
    g = result.get("grid") or {}
    if not g.get("crs") or not g.get("transform"):
        return None
    tr = Affine.from_gdal(*g["transform"])
    to_ll = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True)
    cols = np.linspace(0, g["width"], n)
    rows = np.linspace(0, g["height"], n)
    lon, lat = [], []
    for r in rows:
        for c in cols:
            x, y = tr * (c, r)
            lo, la = to_ll.transform(x, y)
            lon.append(round(float(lo), 8))
            lat.append(round(float(la), 8))
    return {"n": n, "width": g["width"], "height": g["height"], "lon": lon, "lat": lat}


def _validation_summary(job_dir: Path) -> dict[str, Any] | None:
    p = job_dir / "validation.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    lt = h.get("latest")
    if lt and (lt.get("metrics_overall") or {}).get("RMSE") is not None:
        m, b = lt["metrics_overall"], lt.get("metrics_baseline") or {}
        out["raster"] = {"layer": lt.get("compared_layer"), "reference": (lt.get("reference") or {}).get("source_note") if isinstance(lt.get("reference"), dict) else lt.get("source_note"),
                         "rmse_m": m.get("RMSE"), "me_m": m.get("ME"), "nmad_m": m.get("NMAD"), "n": m.get("n"), "baseline_rmse_m": b.get("RMSE"), "text": (lt.get("verdict") or {}).get("text"), "ran_at": lt.get("ran_at")}
    lp = h.get("latest_points")
    if lp:
        out["points"] = {"source": lp.get("source"), "n_in_grid": lp.get("n_in_grid"), "ran_at": lp.get("ran_at"),
                         "metrics": {k: {"rmse_m": v.get("RMSE"), "me_m": v.get("ME"), "n": v.get("n")} for k, v in (lp.get("metrics") or {}).items() if isinstance(v, dict) and v.get("RMSE") is not None}}
    return out or None


def demo_source(input_filename: str | None, manifest_path: Path) -> dict[str, Any] | None:
    """Attribution for bundled demo inputs (matched by file name); None for user uploads."""
    if not input_filename or not manifest_path.exists():
        return None
    for it in json.loads(manifest_path.read_text(encoding="utf-8")).get("items", []):
        if Path(it.get("file", "")).name == Path(input_filename).name:
            return {"label": it.get("label"), "attribution": it.get("source"), "country": it.get("country")}
    return None


def accuracy_lines(result: dict[str, Any]) -> list[str]:
    u = result.get("uncertainty") or {}
    lines = []
    if "ndsm" in u:
        n = u["ndsm"]
        lines.append(f"Height above ground: typical error ±{n['object_m']:.1f} m on buildings/trees, ±{(n.get('ground_m') or 0):.1f} m on open ground ({n['source']}).")
    if "dsm" in u:
        lines.append(f"Surface elevation (DSM): typical error ±{u['dsm']['value_m']:.1f} m; terrain ±{u['terrain']['value_m']:.1f} m ({u['dsm']['source']}).")
    if not lines:
        lines.append("No held-out measurement backs the heights of this scene: treat them as relative / uncalibrated.")
    return lines


def scene_payload(job_dir: Path, result: dict[str, Any], job: dict[str, Any], *, app_version: str, manifest_path: Path) -> dict[str, Any]:
    job_dir = Path(job_dir)
    mode = result.get("mode", "A")
    layers: dict[str, Any] = {}
    names = ("dsm", "terrain", "ndsm", "relative") if mode == "B" else ("relative",)
    for name in names:
        li = (result.get("layers") or {}).get(name) or {}
        hf, hm = li.get("heightfield"), li.get("heightfield_meta")
        if mode == "A" and name == "relative":
            hf, hm = (result.get("artifacts") or {}).get("heightfield", "heightfield.f32"), None
        if not hf or not (job_dir / hf).exists():
            continue
        meta = json.loads((job_dir / hm).read_text(encoding="utf-8")) if hm and (job_dir / hm).exists() else dict(result.get("heightfield") or {})
        z = np.fromfile(job_dir / hf, dtype="<f4").reshape(meta["height"], meta["width"])
        q = quantize(z)
        metric = mode == "B" and name != "relative"
        layers[name] = {"label": LAYER_LABELS[name], "metric": metric, "units": "m" if metric else "relative",
                        "vertical_reference": (result.get("vertical_reference") if name in ("dsm", "terrain") else "height above ground" if name == "ndsm" else None),
                        "meta": meta, "q": q}
        # readings never come from the display mesh (edge-sharpened for looks): embed the plain block mean of the
        # full-resolution raster on the same grid, like the app samples its rasters server-side
        m = _measurement_grid(job_dir / f"{name}.tif", int(meta.get("downsample_factor") or 1), (meta["height"], meta["width"])) if metric else None
        if m is not None:
            layers[name]["m"] = quantize(m)
    if not layers:
        raise ValueError("job has no heightfields to export")
    tex = (result.get("artifacts") or {}).get("texture")
    texture = None
    if tex and (job_dir / tex).exists():
        mime = "image/jpeg" if tex.lower().endswith((".jpg", ".jpeg")) else "image/png"
        texture = {"mime": mime, "b64": base64.b64encode((job_dir / tex).read_bytes()).decode("ascii")}
    buildings = None
    bj = (result.get("artifacts") or {}).get("buildings_json")
    if bj and (job_dir / bj).exists():
        d = json.loads((job_dir / bj).read_text(encoding="utf-8"))
        buildings = {k: d.get(k) for k in ("count", "min_height_m", "gsd_m", "extent", "height_error", "vertical_reference", "segmentation_method")}
        buildings["buildings"] = [{**{k: b.get(k) for k in BUILDING_KEYS if k in b}, "coords": [[round(x, 2), round(y, 2)] for x, y in b["coords"]]} for b in d.get("buildings", [])]
    dem = result.get("dem") or {}
    return {
        "format": "depthwizard-scene", "version": 1,
        "title": (demo_source(job.get("input_filename"), manifest_path) or {}).get("label") or job.get("input_filename") or "DepthWizard scene",
        "exported_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "app_version": app_version,
        "job": {"id": job.get("job_id"), "input_filename": job.get("input_filename"), "input_sha256": job.get("input_sha256"), "processed_at": (job.get("timestamps") or {}).get("READY"),
                "model": job.get("model"), "config_hash": job.get("config_hash"), "method_version": result.get("method_version")},
        "source": demo_source(job.get("input_filename"), manifest_path),
        "mode": mode, "tier": result.get("calibration_tier"), "quality": result.get("quality"), "quality_triggers": result.get("quality_triggers") or [],
        "flags": result.get("flags") or [], "notes": result.get("notes") or [], "vertical_reference": result.get("vertical_reference"),
        "units": result.get("units"), "gsd_m": result.get("gsd_m"), "crs": (result.get("grid") or {}).get("crs"), "transform": (result.get("grid") or {}).get("transform"),
        "size": [(result.get("grid") or {}).get("width"), (result.get("grid") or {}).get("height")], "lonlat_grid": _lonlat_grid(result),
        "dem": {"product": dem.get("product"), "posting_m": dem.get("posting_m"), "vertical_crs": dem.get("vertical_crs")} if dem else None,
        "uncertainty": result.get("uncertainty") or {}, "accuracy": accuracy_lines(result), "validation": _validation_summary(job_dir),
        "layers": layers, "default_layer": "city" if (buildings and "terrain" in layers) else ("dsm" if "dsm" in layers else next(iter(layers))),
        "texture": texture, "buildings": buildings,
    }


def _json_script_safe(s: str) -> str:
    """JSON inside <script type="application/json">: "<" only occurs inside JSON strings, where \\u003c is valid."""
    return s.replace("<", "\\u003c")


def _js_script_safe(s: str) -> str:
    """Inline JS: only "</script" and "<!--" can end / confuse the HTML script element; both can only occur inside JS
    strings, regex literals or comments, where "<\\/script" and "<\\!--" mean the same thing."""
    return re.sub(r"</(script)", r"<\\/\1", s, flags=re.IGNORECASE).replace("<!--", "<\\!--")


def standalone_html(payload: dict[str, Any], bundle_dir: Path) -> str:
    js_p, css_p = Path(bundle_dir) / "standalone.js", Path(bundle_dir) / "standalone.css"
    if not js_p.exists():
        raise FileNotFoundError(f"standalone viewer bundle missing ({js_p}): run `npm run build` in frontend/")
    js = js_p.read_text(encoding="utf-8")
    css = css_p.read_text(encoding="utf-8") if css_p.exists() else ""
    title = str(payload.get("title") or "DepthWizard scene").replace("<", "&lt;").replace(">", "&gt;")
    data = _json_script_safe(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<meta name=\"generator\" content=\"DepthWizard {payload.get('app_version', '')}\">\n"
        f"<title>{title} · DepthWizard offline 3D scene</title>\n<style>{css}</style>\n</head>\n<body>\n"
        "<div id=\"dw-app\"><noscript>This offline 3D scene needs JavaScript and WebGL (any current Chrome, Edge, Firefox or Safari).</noscript></div>\n"
        f"<script id=\"dw-scene\" type=\"application/json\">{data}</script>\n"
        f"<script>{_js_script_safe(js)}</script>\n</body>\n</html>\n"
    )
