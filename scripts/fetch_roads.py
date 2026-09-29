"""Bundle mapped roads, tracks, footpaths and named settlements for road-access screening (core/disaster/roads.py).

Source: OpenStreetMap via the Overpass API (ODbL 1.0, (c) OpenStreetMap contributors):
  * highway = motorway ... residential, service, unclassified, track (motorable) and path / footway / steps (foot);
  * place = city / town / village / hamlet / isolated_dwelling / locality / suburb / neighbourhood (settlement names).
Bhuvan's road layers (NRSC) are shown as a map overlay in the app, but its vector download (WFS) is disabled, so
the network analysis uses OSM. Stored per demo area in assets/roads/<name>.geojson with an index.json.

Usage:
  python scripts/fetch_roads.py                            # every georeferenced demo scene in assets/demo/manifest.json
  python scripts/fetch_roads.py --bbox W S E N --name my_area
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
OUT = ROOT / "assets" / "roads"
OVERPASS = "https://overpass-api.de/api/interpreter"
SOURCE, LICENCE = "OpenStreetMap roads, paths and places", "ODbL 1.0 (c) OpenStreetMap contributors"
MARGIN_DEG = 0.004  # ~400 m beyond the scene: roads leaving the scene must be seen to leave it
MOTORABLE = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|living_street|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
FOOT = "path|footway|steps|pedestrian|bridleway"
PLACES = "city|town|village|hamlet|isolated_dwelling|locality|suburb|neighbourhood"
EXTRA = {"nepal_sunkoshi": "nepal/sunkoshi_rgb_0.5m.tif"}  # study scenes not in the demo manifest


def query(bbox: tuple[float, float, float, float]) -> list[dict]:
    w, s, e, n = bbox
    q = (f'[out:json][timeout:120];(way["highway"~"^({MOTORABLE}|{FOOT})$"]({s},{w},{n},{e});'
         f'node["place"~"^({PLACES})$"]({s},{w},{n},{e}););out tags geom;')
    for attempt in range(6):
        try:
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "DepthWizard/1.0 (road access screening)"})
            return json.load(urllib.request.urlopen(req, timeout=240))["elements"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as ex:
            if isinstance(ex, urllib.error.HTTPError) and ex.code not in (429, 502, 503, 504):
                raise
            wait = 20 * (attempt + 1)
            print(f"  Overpass busy ({ex}), retrying in {wait} s")
            time.sleep(wait)
    raise SystemExit("Overpass API unavailable; try again later")


def fetch(name: str, bbox: tuple[float, float, float, float]) -> dict:
    b = (bbox[0] - MARGIN_DEG, bbox[1] - MARGIN_DEG, bbox[2] + MARGIN_DEG, bbox[3] + MARGIN_DEG)
    feats = []
    n_road = n_place = 0
    for el in query(b):
        t = el.get("tags", {})
        if el["type"] == "node":
            feats.append({"type": "Feature", "properties": {"kind": "place", "place": t.get("place"), "name": t.get("name") or t.get("name:en"), "population": t.get("population"), "osm_id": el["id"]},
                          "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]}})
            n_place += 1
            continue
        coords = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        if len(coords) >= 2:
            hw = t.get("highway")
            feats.append({"type": "Feature", "properties": {"kind": "road", "highway": hw, "motorable": hw not in FOOT.split("|"), "name": t.get("name") or t.get("ref"), "ref": t.get("ref"),
                                                            "bridge": t.get("bridge") == "yes", "osm_id": el["id"]},
                          "geometry": {"type": "LineString", "coordinates": coords}})
            n_road += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": name, "source": SOURCE, "licence": LICENCE, "features": feats}, separators=(",", ":")), encoding="utf-8")
    return {"file": f"{name}.geojson", "bounds_wgs84": [round(v, 6) for v in b], "source": SOURCE, "licence": LICENCE, "roads": n_road, "places": n_place}


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
        files = [it["file"] for it in json.loads((ROOT / "assets" / "demo" / "manifest.json").read_text(encoding="utf-8"))["items"] if it.get("mode") == "B" and it["file"].endswith(".tif")]
        for f in files:
            jobs[Path(f).stem] = scene_bbox(ROOT / "assets" / "demo" / f)
        for n, f in EXTRA.items():
            if (ROOT / "assets" / "demo" / f).exists():
                jobs[n] = scene_bbox(ROOT / "assets" / "demo" / f)
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"files": []}
    for name, bb in jobs.items():
        entry = fetch(name, bb)
        index["files"] = [f for f in index["files"] if f["file"] != entry["file"]] + [entry]
        idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"{name}: {entry['roads']} roads / paths, {entry['places']} named places", flush=True)
        time.sleep(5)  # be polite to the public Overpass server
    (OUT / "README.md").write_text("# Mapped roads and settlements for road-access screening\n\nRoads, tracks, footpaths and named places from "
                                   "**OpenStreetMap** (ODbL 1.0, (c) OpenStreetMap contributors), one GeoJSON per demo area; regenerate with "
                                   "`python scripts/fetch_roads.py`. Used by `core/disaster/roads.py` to find road links cut by a hazard and the "
                                   "settlements that lose every route out of the scene.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
