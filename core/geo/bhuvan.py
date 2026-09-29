"""Bhuvan (NRSC / ISRO) map layers as overlays aligned to a job's grid.

Bhuvan publishes its thematic layers as OGC WMS (images). Its WFS (vector download) is disabled, so these layers are
for display and the report only; analyses that need vectors use OpenStreetMap (roads, rivers) or the user's upload.
The WMS image is requested in WGS84 for the scene's bounds and reprojected onto the job grid, then cached as
bhuvan_<layer>.png next to the job's other previews. Terms: Bhuvan data are provided by NRSC for non-commercial use,
with attribution "Bhuvan, NRSC/ISRO".
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

VEC = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
RAS_LULC = "https://bhuvan-ras2.nrsc.gov.in/cgi-bin/LULC250K.exe"
ATTRIBUTION = "Bhuvan, NRSC/ISRO"
# Checked 2026-09: the national layers mmi:india_roads (WMS error) and LULC250K (empty at scene scale) do not render
# for a ~1 km scene, so state layers are listed. Other states follow the same names: mmi:<ST>_ROAD_NETWORK_Q4_2022,
# basemap:<ST>_LULC (see the GetCapabilities of VEC).
# id: (label, service, layer, states) - states: None = all of India, else a (W, S, E, N) box it applies to
LAYERS: dict[str, tuple[str, str, str, tuple[float, float, float, float] | None]] = {
    "drainage": ("Drainage network (India)", VEC, "hydrology:bdrain_qgs", None),
    "sk_roads": ("Sikkim road network 2022", VEC, "mmi:SK_ROAD_NETWORK_Q4_2022", (88.0, 27.05, 88.95, 28.15)),
    "sk_lulc": ("Sikkim land use / land cover", VEC, "basemap:SK_LULC", (88.0, 27.05, 88.95, 28.15)),
    "sk_slope": ("Sikkim slope", VEC, "sdv:sk_slope", (88.0, 27.05, 88.95, 28.15)),
}
INDIA = (68.0, 6.5, 97.5, 37.5)
MAX_PX = 2048


def available(bounds_wgs84: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    w, s, e, n = bounds_wgs84

    def inside(b):
        return w >= b[0] and e <= b[2] and s >= b[1] and n <= b[3]

    if not inside(INDIA):
        return []
    return [{"id": k, "label": v[0], "attribution": ATTRIBUTION} for k, v in LAYERS.items() if v[3] is None or inside(v[3])]


def overlay(job_dir: Path, layer_id: str, *, refresh: bool = False) -> Path:
    """bhuvan_<layer_id>.png on the job grid (RGBA, transparent where the layer is empty)."""
    if layer_id not in LAYERS:
        raise ValueError(f"unknown Bhuvan layer {layer_id!r}; choose one of {sorted(LAYERS)}")
    out = job_dir / f"bhuvan_{layer_id}.png"
    if out.exists() and not refresh:
        return out
    ref = job_dir / "terrain.tif" if (job_dir / "terrain.tif").exists() else job_dir / "input.tif"
    with rasterio.open(ref) as ds:
        crs, tr, H, W = ds.crs, ds.transform, ds.height, ds.width
        b = ds.bounds
    if crs is None:
        raise ValueError("Bhuvan overlays need a georeferenced job")
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)
    k = min(1.0, MAX_PX / max(W, H))
    ow, oh = max(64, int(W * k)), max(64, int(H * k))
    _, service, layer, _ = LAYERS[layer_id]
    q = urlencode({"service": "WMS", "version": "1.1.1", "request": "GetMap", "layers": layer, "styles": "", "srs": "EPSG:4326",
                   "bbox": f"{w},{s},{e},{n}", "width": ow, "height": oh, "format": "image/png", "transparent": "true"})
    with urllib.request.urlopen(urllib.request.Request(f"{service}?{q}", headers={"User-Agent": "DepthWizard/1.0"}), timeout=60) as r:
        ctype, data = r.headers.get("Content-Type", ""), r.read()
    if "image" not in ctype:
        raise RuntimeError(f"Bhuvan returned {ctype or 'no image'} for {layer}: {data[:200]!r}")
    src = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")).transpose(2, 0, 1)
    dst = np.zeros((4, oh, ow), np.uint8)
    dst_tr = tr * tr.scale(W / ow, H / oh)
    for i in range(4):
        reproject(src[i], dst[i], src_transform=from_bounds(w, s, e, n, src.shape[2], src.shape[1]), src_crs="EPSG:4326", dst_transform=dst_tr, dst_crs=crs, resampling=Resampling.nearest)
    from core.geo.atomic import write_atomic

    img = Image.fromarray(dst.transpose(1, 2, 0), "RGBA").resize((W, H), Image.NEAREST)
    write_atomic(out, lambda t: img.save(t, format="PNG"))
    return out
