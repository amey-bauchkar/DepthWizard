"""Observed footprint of a real flood at a demo scene, from Sentinel-2:
  chungthang       4 Oct 2023 Teesta flood (South Lhonak GLOF), Sikkim
  nepal_sunkoshi   27-28 Sep 2024 monsoon flood, Sunkoshi / Roshi Khola confluence, Nepal

Same-season pair (one year apart, before and after the event) from Sentinel-2 L2A on Microsoft Planetary Computer
(Copernicus data, free). A 10 m pixel is "flood-affected" when, after the flood, it is bare sediment or water where
before it was vegetated or built (NDVI drop), or it is newly bright bare deposit:
    affected = valid & (NDVI_after < 0.25) & ((NDVI_before - NDVI_after) > 0.15 | new water (MNDWI_after > 0.1 > MNDWI_before))
and cloud / shadow pixels (scene classification SCL 3, 8, 9, 10) in either image are excluded.
The GLOF left kilometres of fresh sediment along the Teesta, so this "scar" is its observable extent.

Writes data/glof/<scene>_affected.tif (0/1, 10 m, scene CRS) and <scene>_s2.json. Usage:
  python scripts/glof_observed_extent.py chungthang | nepal_sunkoshi
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PC = "https://planetarycomputer.microsoft.com/api"
SCENES = {  # scene image, before window, after window (same season)
    "chungthang": ("india/chungthang_rgb_0.5m.tif", "2022-10-15/2022-12-31", "2023-10-20/2023-12-31"),
    "nepal_sunkoshi": ("nepal/sunkoshi_rgb_0.5m.tif", "2023-10-15/2023-12-31", "2024-10-10/2024-12-31"),
}


from core.geo.sentinel2 import indices, search as _pick, token as _token  # noqa: E402  (shared with core.disaster.landslide)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "chungthang"
    image, before, after = SCENES[name]
    with rasterio.open(ROOT / "assets" / "demo" / image) as ds:
        crs, b = ds.crs, ds.bounds
    bbox = list(transform_bounds(crs, "EPSG:4326", *b))
    shape = (int((b.top - b.bottom) / 10), int((b.right - b.left) / 10))
    tr = rasterio.transform.from_origin(b.left, b.top, 10.0, 10.0)
    tok = _token()
    best = {}
    for tag, rng in (("before", before), ("after", after)):
        for it in _pick(bbox, rng)[:6]:
            nd, mw, ok = indices(it, tok, crs, tr, shape)
            if ok.mean() > 0.97:
                best[tag] = (it, nd, mw, ok)
                break
        if tag not in best:
            raise SystemExit(f"no cloud-free {tag} image")
    (_, nb, mb, okb), (_, na, ma, oka) = best["before"], best["after"]
    valid = okb & oka
    affected = valid & (na < 0.25) & (((nb - na) > 0.15) | ((ma > 0.1) & (mb <= 0.1)))
    out = ROOT / "data" / "glof"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out / f"{name}_affected.tif", "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=crs, transform=tr, nodata=255) as ds:
        ds.write(np.where(valid, affected, 255).astype(np.uint8), 1)
    with rasterio.open(out / f"{name}_river_before.tif", "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=crs, transform=tr, nodata=255) as ds:
        ds.write(np.where(valid, mb > 0.1, 255).astype(np.uint8), 1)  # the river channel that was water before the flood
    meta = {"before": best["before"][0]["id"], "after": best["after"][0]["id"], "valid_fraction": round(float(valid.mean()), 3),
            "affected_ha": round(float(affected.sum()) * 0.01, 2), "rule": [ln.strip() for ln in __doc__.split("\n") if ln.strip().startswith(("affected =", "and cloud"))]}
    (out / f"{name}_s2.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
