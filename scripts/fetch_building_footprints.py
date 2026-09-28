"""Bundle open building footprints for LoD-1 (core/terrain/footprints.py): Microsoft Global ML Building Footprints.

Licence: ODbL 1.0 (Open Data Commons Open Database License), (c) Microsoft. Attribution and share-alike apply to the
footprint database; DepthWizard stores the subset for each area unchanged in assets/footprints/<name>.geojson.

Usage:
  python scripts/fetch_building_footprints.py                  # every georeferenced demo scene in assets/demo/manifest.json
  python scripts/fetch_building_footprints.py --bbox W S E N --name my_area   # any area (lon/lat)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "footprints"
CACHE = ROOT / "data" / "ref_buildings"
LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
SOURCE = "Microsoft Global ML Building Footprints"
LICENCE = "ODbL 1.0 (c) Microsoft"
MARGIN_DEG = 0.002  # ~200 m around the scene


def _tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    s = math.sin(math.radians(lat))
    return int((lon + 180) / 360 * 2 ** z), int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 2 ** z)


def _quadkey(tx: int, ty: int, z: int) -> str:
    q = ""
    for i in range(z, 0, -1):
        m = 1 << (i - 1)
        q += str((1 if tx & m else 0) + (2 if ty & m else 0))
    return q


def quadkeys(w: float, s: float, e: float, n: float, z: int = 9) -> set[str]:
    x0, y0 = _tile(w, n, z)
    x1, y1 = _tile(e, s, z)
    return {_quadkey(x, y, z) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def fetch(name: str, bbox: tuple[float, float, float, float], rows: list[dict]) -> dict:
    w, s, e, n = bbox[0] - MARGIN_DEG, bbox[1] - MARGIN_DEG, bbox[2] + MARGIN_DEG, bbox[3] + MARGIN_DEG
    CACHE.mkdir(parents=True, exist_ok=True)
    feats = []
    for q in sorted(quadkeys(w, s, e, n)):
        for r in (r for r in rows if r["QuadKey"] == q):
            f = CACHE / f"ms_{r['Location']}_{q}.geojsonl.gz"
            if not f.exists():
                print(f"  downloading {r['Location']} {q} ({r['Size']})")
                urllib.request.urlretrieve(r["Url"], f)
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    ft = json.loads(line)
                    ring = ft["geometry"]["coordinates"][0]
                    cx, cy = sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)
                    if w <= cx <= e and s <= cy <= n:
                        feats.append({"type": "Feature", "properties": {"source": SOURCE}, "geometry": ft["geometry"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": name, "source": SOURCE, "licence": LICENCE, "features": feats}, separators=(",", ":")), encoding="utf-8")
    return {"file": f"{name}.geojson", "bounds_wgs84": [round(v, 6) for v in (w, s, e, n)], "source": SOURCE, "licence": LICENCE, "count": len(feats)}


def scene_bbox(tif: Path) -> tuple[float, float, float, float]:
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(tif) as ds:
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--name")
    a = ap.parse_args()
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(LINKS, timeout=120).read().decode())))
    jobs: dict[str, tuple[float, float, float, float]] = {}
    if a.bbox:
        jobs[a.name or "area"] = tuple(a.bbox)  # type: ignore[assignment]
    else:
        for it in json.loads((ROOT / "assets" / "demo" / "manifest.json").read_text(encoding="utf-8"))["items"]:
            if it.get("mode") == "B" and it["file"].endswith(".tif"):
                jobs[Path(it["file"]).stem] = scene_bbox(ROOT / "assets" / "demo" / it["file"])
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"files": []}
    for name, bb in jobs.items():
        entry = fetch(name, bb, rows)
        index["files"] = [f for f in index["files"] if f["file"] != entry["file"]] + [entry]
        print(f"{name}: {entry['count']} footprints")
    idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
    (OUT / "README.md").write_text("# Building footprints for LoD-1\n\nSubsets of **Microsoft Global ML Building Footprints** "
                                   "(https://github.com/microsoft/GlobalMLBuildingFootprints), licensed under the Open Data Commons "
                                   "Open Database License (ODbL 1.0), (c) Microsoft. Stored unchanged, one GeoJSON per demo area; "
                                   "regenerate with `python scripts/fetch_building_footprints.py`.\n\nDepthWizard uses a file only "
                                   "when it covers the scene; building heights always come from the DepthWizard nDSM.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
