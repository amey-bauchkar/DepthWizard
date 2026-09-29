"""Road access screening (Disaster Management): which road links a hazard cuts, and which settlements lose every
route out of the scene. Network screening on mapped roads, NOT traffic routing.

Network   OSM roads / paths (assets/roads, scripts/fetch_roads.py) as a graph: nodes = way vertices (shared vertices
          join ways), edges = consecutive vertices. Ways are fetched ~400 m beyond the scene, so every node outside the
          scene is an EXIT (the outside world, assumed open: nothing is known there).
Cut       an edge is cut when any point sampled along it (<= 1 pixel apart) inside the scene is hazardous:
            flood       flood_depth.tif > ROAD_FLOOD_IMPASSABLE_M (the last flood run: its model and water height)
            landslide   landslide_hazard.tif class >= the chosen class (the last landslide run), optional
          Bridges (bridge=yes) are cut by flood only where the water is deeper than ROAD_BRIDGE_CLEARANCE_M.
Access    a settlement (building cluster, core.disaster.settlements, or an OSM place) joins the network at its nearest
          node within ROAD_SETTLEMENT_LINK_M. CUT OFF = it reached an exit on the intact network and reaches none once
          the cut edges are removed. Computed for vehicles (motorable ways) and on foot (all ways).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer

from core.disaster.settlements import POPULATION_NOTE, clusters
from core.screening_params import ROAD_BRIDGE_CLEARANCE_M, ROAD_FLOOD_IMPASSABLE_M, ROAD_SETTLEMENT_LINK_M

METHOD = "road-access-1 (OSM network, hazard cuts, connectivity to scene exits)"
WARNINGS = [
    "Road access screening on mapped roads only: unmapped tracks, road condition, debris and bridge damage are not known.",
    "Roads leaving the scene are assumed open beyond it.",
    f"Bridges are assumed to clear the water up to {ROAD_BRIDGE_CLEARANCE_M:g} m (deck heights are not known).",
]
COLOURS = {"open": (34, 197, 94, 235), "cut": (239, 68, 68, 255), "stranded": (245, 158, 11, 235), "foot": (148, 163, 184, 200)}


def _roads_file(bounds_wgs84: tuple[float, float, float, float]) -> tuple[Path, dict[str, Any]] | None:
    from backend.config.settings import REPO_ROOT
    from core.terrain.footprints import discover

    return discover(bounds_wgs84, Path(os.environ.get("DW_ROADS_DIR", REPO_ROOT / "assets" / "roads")))


def _read_mask(path: Path, fn) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        a = ds.read(1, masked=True)
    return fn(a.filled(0))


def build_graph(feats: list[dict[str, Any]], to_px, shape: tuple[int, int], cut_mask: np.ndarray | None, bridge_cut_mask: np.ndarray | None) -> nx.Graph:
    """Graph over road vertices in pixel coordinates; edge attrs: length_px, cut, motorable, name, inside."""
    H, W = shape
    G = nx.Graph()
    for f in feats:
        p = f["properties"]
        pts = [to_px(x, y) for x, y in f["geometry"]["coordinates"]]
        mask = bridge_cut_mask if p.get("bridge") else cut_mask
        for (c0, r0), (c1, r1) in zip(pts[:-1], pts[1:]):
            a, b = (round(c0, 1), round(r0, 1)), (round(c1, 1), round(r1, 1))
            L = math.hypot(c1 - c0, r1 - r0)
            n = max(2, int(math.ceil(L)) + 1)
            cs, rs = np.linspace(c0, c1, n), np.linspace(r0, r1, n)
            ins = (cs >= 0) & (cs < W) & (rs >= 0) & (rs < H)
            cut = bool(mask is not None and ins.any() and mask[rs[ins].astype(int), cs[ins].astype(int)].any())
            for node, (cc, rr) in ((a, (c0, r0)), (b, (c1, r1))):
                if node not in G:
                    G.add_node(node, exit=not (0 <= cc < W and 0 <= rr < H))
            if G.has_edge(a, b):  # duplicated way: keep the worse case
                G.edges[a, b]["cut"] |= cut
                continue
            G.add_edge(a, b, length_px=L, cut=cut, motorable=bool(p.get("motorable", True)), name=p.get("name"), inside=bool(ins.any()), bridge=bool(p.get("bridge")))
    return G


def _reach_exit(G: nx.Graph, use_cut: bool, motorable_only: bool) -> set:
    """Nodes that can reach an exit node."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((a, b) for a, b, d in G.edges(data=True) if (not use_cut or not d["cut"]) and (d["motorable"] or not motorable_only))
    out = set()
    for comp in nx.connected_components(H):
        if any(G.nodes[n]["exit"] for n in comp):
            out |= comp
    return out


def _nearest(nodes: np.ndarray, keys: list, c: float, r: float, max_px: float):
    if not len(nodes):
        return None, None
    d = np.hypot(nodes[:, 0] - c, nodes[:, 1] - r)
    j = int(np.argmin(d))
    return (keys[j], float(d[j])) if d[j] <= max_px else (None, float(d[j]))


def run_road_access(job_dir: Path, result: dict[str, Any], *, include_landslide: bool = False, landslide_min_class: int = 4) -> dict[str, Any]:
    ref = job_dir / "terrain.tif" if (job_dir / "terrain.tif").exists() else job_dir / "dsm.tif"
    with rasterio.open(ref) as ds:
        crs, tr, shape = ds.crs, ds.transform, ds.shape
    if crs is None:
        raise ValueError("Road access screening needs a georeferenced job")
    px = math.sqrt(abs(tr.a * tr.e - tr.b * tr.d))
    from core.geo.raster_io import grid_bounds_wgs84
    from core.geo.grid import Grid

    hit = _roads_file(grid_bounds_wgs84(Grid(width=shape[1], height=shape[0], transform=tr, crs=crs.to_string())))
    if hit is None:
        raise ValueError("No mapped roads cover this scene. Run: python scripts/fetch_roads.py --bbox W S E N --name <area>")
    gj = json.loads(hit[0].read_text(encoding="utf-8"))
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = ~tr

    def to_px(lon: float, lat: float) -> tuple[float, float]:
        return inv * to.transform(lon, lat)

    hazards, cut, bridge_cut = [], np.zeros(shape, bool), np.zeros(shape, bool)
    flood = json.loads((job_dir / "disaster_flood.json").read_text(encoding="utf-8")) if (job_dir / "disaster_flood.json").exists() else None
    fd = _read_mask(job_dir / "flood_depth.tif", lambda a: a) if flood else None
    if fd is not None and fd.shape == shape:
        cut |= fd > ROAD_FLOOD_IMPASSABLE_M
        bridge_cut |= fd > ROAD_BRIDGE_CLEARANCE_M
        hazards.append({"hazard": "flood", "model": flood.get("model"), "waterLevel_m": flood.get("waterLevel_m"), "rule": f"water depth > {ROAD_FLOOD_IMPASSABLE_M:g} m"})
    ls = None
    if include_landslide:
        lp = job_dir / "disaster_landslide.json"
        if not lp.exists():
            raise ValueError("Run landslide screening first to include it")
        ls = _read_mask(job_dir / "landslide_hazard.tif", lambda a: (a >= landslide_min_class) & (a < 255))
        if ls is not None and ls.shape != shape:  # hazard is on a coarser working grid: nearest-neighbour onto this one
            ri = np.minimum((np.arange(shape[0]) * ls.shape[0]) // shape[0], ls.shape[0] - 1)
            ci = np.minimum((np.arange(shape[1]) * ls.shape[1]) // shape[1], ls.shape[1] - 1)
            ls = ls[np.ix_(ri, ci)]
        if ls is not None:
            cut |= ls
            bridge_cut |= ls
            hazards.append({"hazard": "landslide", "rule": f"hazard class >= {landslide_min_class}"})
    if not hazards:
        raise ValueError("Run a flood (or landslide) screening first: road access is checked against the last hazard result")

    feats = [f for f in gj["features"] if f["properties"].get("kind", "road") == "road" and f["geometry"]["type"] == "LineString"]
    G = build_graph(feats, to_px, shape, cut, bridge_cut)
    places = []
    for f in gj["features"]:
        if f["properties"].get("kind") == "place" and f["properties"].get("name"):
            c, r = to_px(*f["geometry"]["coordinates"])
            places.append({"name": f["properties"]["name"], "place": f["properties"].get("place"), "col": c, "row": r, "inside": 0 <= c < shape[1] and 0 <= r < shape[0]})
    setts = clusters(job_dir, shape, px, [p for p in places if -800 / px <= p["col"] <= shape[1] + 800 / px])
    seen: dict[str, int] = {}
    for s in setts:  # several clusters near one place name: "Chungthang", "Chungthang (2)", ...
        seen[s["name"]] = seen.get(s["name"], 0) + 1
        if seen[s["name"]] > 1:
            s["name"] = f"{s['name']} ({seen[s['name']]})"
    named = {s["name"] for s in setts}
    for p in places:  # named places with no building cluster of their own (e.g. footprints missing)
        if p["inside"] and p["name"] not in named:
            setts.append({"id": len(setts) + 1, "name": p["name"], "namedFrom": "OpenStreetMap place (no building cluster)", "col": round(p["col"], 1), "row": round(p["row"], 1), "buildings": [], "nBuildings": 0, "population": None})

    reach = {(m, u): _reach_exit(G, u, m) for m in (True, False) for u in (False, True)}
    keys = {m: [n for n in G.nodes if any(G.edges[n, v]["motorable"] or not m for v in G.neighbors(n))] for m in (True, False)}
    arr = {m: np.array(keys[m], dtype=np.float64).reshape(-1, 2) for m in (True, False)}
    link = ROAD_SETTLEMENT_LINK_M / px
    for s in setts:
        for m, tag in ((True, "vehicle"), (False, "foot")):
            node, d = _nearest(arr[m], keys[m], s["col"], s["row"], link)
            if node is None:
                s[tag] = "NO_MAPPED_ROAD"
            elif node not in reach[(m, False)]:
                s[tag] = "NO_EXIT_BEFORE"  # the mapped network never left the scene from here
            else:
                s[tag] = "OPEN" if node in reach[(m, True)] else "CUT_OFF"
            s[f"{tag}LinkM"] = round(d * px, 0) if d is not None else None
        s["status"] = ("CUT_OFF" if s["vehicle"] == "CUT_OFF" and s["foot"] in ("CUT_OFF", "NO_MAPPED_ROAD", "NO_EXIT_BEFORE")
                       else "VEHICLE_CUT_FOOT_OPEN" if s["vehicle"] == "CUT_OFF" else "OPEN" if s["vehicle"] == "OPEN" else s["vehicle"])

    # roads: cut edges, and open edges stranded (no exit any more for vehicles)
    lens = {"cut": 0.0, "stranded": 0.0, "open": 0.0}
    cut_names: dict[str, float] = {}
    for a, b, d in G.edges(data=True):
        if not d["inside"] or not d["motorable"]:
            continue
        L = d["length_px"] * px
        st = "cut" if d["cut"] else ("stranded" if a not in reach[(True, True)] and a in reach[(True, False)] else "open")
        d["status"] = st
        lens[st] += L
        if st == "cut":
            cut_names[d["name"] or "unnamed road"] = cut_names.get(d["name"] or "unnamed road", 0.0) + L
    _preview(job_dir / "roads_preview.png", G, setts, shape)
    _geojson(job_dir / "roads_status.geojson", G, setts, tr, crs)

    cut_off = [s for s in setts if s["status"] == "CUT_OFF"]
    foot_only = [s for s in setts if s["status"] == "VEHICLE_CUT_FOOT_OPEN"]
    summary = {
        "scenario": "road_access", "method": METHOD, "hazards": hazards,
        "roads": {"source": hit[1]["source"], "licence": hit[1]["licence"], "motorableKm": round(sum(lens.values()) / 1000, 2), "cutKm": round(lens["cut"] / 1000, 2),
                  "strandedKm": round(lens["stranded"] / 1000, 2), "cutRoads": [{"name": k, "cutM": round(v)} for k, v in sorted(cut_names.items(), key=lambda kv: -kv[1])]},
        "settlements": [{k: v for k, v in s.items() if k != "buildings"} for s in setts],
        "nSettlements": len(setts), "nCutOff": len(cut_off), "nFootOnly": len(foot_only),
        "populationCutOff": sum(s["population"] or 0 for s in cut_off), "populationFootOnly": sum(s["population"] or 0 for s in foot_only),
        "rules": {"floodImpassableM": ROAD_FLOOD_IMPASSABLE_M, "bridgeClearanceM": ROAD_BRIDGE_CLEARANCE_M, "settlementLinkM": ROAD_SETTLEMENT_LINK_M},
        "previewResult": "roads_preview.png", "vectorResult": "roads_status.geojson",
        "warnings": WARNINGS + [POPULATION_NOTE] + ([f"The flood result used is the last run: {hazards[0]['model']} model, height {hazards[0]['waterLevel_m']:g} m."] if flood else []),
    }
    (job_dir / "disaster_roads.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _preview(path: Path, G: nx.Graph, setts: list[dict[str, Any]], shape: tuple[int, int]) -> None:
    H, W = shape
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    w = max(3, int(round(min(W, H) / 400)))
    for order in ("foot", "open", "stranded", "cut"):
        for a, b, d in G.edges(data=True):
            st = d.get("status") if d["motorable"] else ("cut" if d["cut"] else "foot")
            if st == order and d["inside"]:
                dr.line([a, b], fill=COLOURS[order], width=w * (2 if order == "cut" else 1))
    for s in setts:
        c = {"CUT_OFF": COLOURS["cut"], "VEHICLE_CUT_FOOT_OPEN": COLOURS["stranded"], "OPEN": COLOURS["open"]}.get(s["status"], (148, 163, 184, 255))
        r = 4 * w
        dr.ellipse([s["col"] - r, s["row"] - r, s["col"] + r, s["row"] + r], fill=c, outline=(255, 255, 255, 255), width=max(1, w // 2))
    from core.geo.atomic import write_atomic

    write_atomic(path, lambda t: im.save(t, format="PNG"))


def _geojson(path: Path, G: nx.Graph, setts: list[dict[str, Any]], tr, crs) -> None:
    to = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def ll(c: float, r: float) -> list[float]:
        return [round(v, 7) for v in to.transform(*(tr * (c, r)))]

    feats = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [ll(*a), ll(*b)]},
              "properties": {"kind": "road", "status": d.get("status", "cut" if d["cut"] else "open"), "motorable": d["motorable"], "name": d["name"], "bridge": d["bridge"]}}
             for a, b, d in G.edges(data=True) if d["inside"]]
    feats += [{"type": "Feature", "geometry": {"type": "Point", "coordinates": ll(s["col"], s["row"])},
               "properties": {"kind": "settlement", **{k: v for k, v in s.items() if k not in ("buildings", "col", "row")}}} for s in setts]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
