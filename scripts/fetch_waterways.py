"""Bundle mapped rivers and streams for river-flood screening (stream burning in core/disaster/hand.py).

Source: OpenStreetMap (waterway = river / stream / canal / drain) via the Overpass API. Licence: ODbL 1.0,
(c) OpenStreetMap contributors. Stored per demo area in assets/waterways/<name>.geojson with an index.json.

Usage:
  python scripts/fetch_waterways.py                         # every georeferenced demo scene in assets/demo/manifest.json
  python scripts/fetch_waterways.py --bbox W S E N --name my_area
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "waterways"
OVERPASS = "https://overpass-api.de/api/interpreter"
SOURCE, LICENCE = "OpenStreetMap waterways", "ODbL 1.0 (c) OpenStreetMap contributors"
MARGIN_DEG = 0.003


def query(bbox: tuple[float, float, float, float]) -> list[dict]:
    w, s, e, n = bbox
    q = f'[out:json][timeout:90];way["waterway"~"^(river|stream|canal|drain)$"]({s},{w},{n},{e});out tags geom;'
    for attempt in range(6):
        try:
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "DepthWizard/1.0 (flood screening)"})
            return json.load(urllib.request.urlopen(req, timeout=180))["elements"]
        except urllib.error.HTTPError as ex:
            if ex.code not in (429, 504):
                raise
            wait = 20 * (attempt + 1)
            print(f"  Overpass busy (HTTP {ex.code}), retrying in {wait} s")
            time.sleep(wait)
    raise SystemExit("Overpass API unavailable; try again later")


def fetch(name: str, bbox: tuple[float, float, float, float]) -> dict:
    b = (bbox[0] - MARGIN_DEG, bbox[1] - MARGIN_DEG, bbox[2] + MARGIN_DEG, bbox[3] + MARGIN_DEG)
    feats = []
    for el in query(b):
        coords = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        if len(coords) >= 2:
            t = el.get("tags", {})
            feats.append({"type": "Feature", "properties": {"waterway": t.get("waterway"), "name": t.get("name"), "osm_id": el["id"]}, "geometry": {"type": "LineString", "coordinates": coords}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": name, "source": SOURCE, "licence": LICENCE, "features": feats}, separators=(",", ":")), encoding="utf-8")
    return {"file": f"{name}.geojson", "bounds_wgs84": [round(v, 6) for v in b], "source": SOURCE, "licence": LICENCE, "count": len(feats)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--name")
    a = ap.parse_args()
    from scripts.fetch_building_footprints import scene_bbox

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
        entry = fetch(name, bb)
        index["files"] = [f for f in index["files"] if f["file"] != entry["file"]] + [entry]
        idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"{name}: {entry['count']} waterways")
        time.sleep(5)  # be polite to the public Overpass server
    (OUT / "README.md").write_text("# Mapped waterways for river-flood screening\n\nRivers, streams, canals and drains from **OpenStreetMap** "
                                   "(ODbL 1.0, (c) OpenStreetMap contributors), one GeoJSON per demo area; regenerate with "
                                   "`python scripts/fetch_waterways.py`. They are burned into the terrain for the height-above-nearest-"
                                   "drainage computation so that channels follow the mapped rivers instead of the coarse DEM.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
