"""One-click damage-assessment report (PDF) for a district disaster officer, built from the job's LATEST hazard results
(disaster_flood.json, disaster_landslide.json, disaster_landing_zones.json, disaster_roads.json). A section whose
screening has not been run is listed as "not run", never filled with defaults the user did not choose.

Page 1  situation summary: key numbers, parameters, data sources
Page 2  map: image + flood depth + landslide hazard + roads (cut / open) + settlements + landing sites
Page 3  action lists: cut-off settlements, exposed buildings, landing sites, rainfall trigger
Page 4  limits and method notes (every warning of every result)
Uses matplotlib only (no extra dependency).
"""
from __future__ import annotations

import datetime as _dt
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.disaster.settlements import POPULATION_NOTE, population

MAP_PX = 1400
TITLE = "DepthWizard — rapid damage and access assessment"


def _load(job_dir: Path, name: str) -> dict[str, Any] | None:
    p = job_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _area(m2: float | None) -> str:
    if m2 is None:
        return "–"
    return f"{m2 / 1e4:,.1f} ha" if m2 >= 1e4 else f"{m2:,.0f} m²"


def collect(job_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Every number shown in the report (also returned by the API as JSON)."""
    flood, ls, hlz, roads = (_load(job_dir, f"disaster_{n}.json") for n in ("flood", "landslide", "landing_zones", "roads"))
    by_id = {int(b["id"]): b for b in (_load(job_dir, "buildings.json") or {}).get("buildings", [])}
    k: dict[str, Any] = {"generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "job": job_dir.name}
    job = _load(job_dir, "job.json") or {}
    k["input"] = job.get("input_filename")
    k["crs"] = (result.get("grid") or {}).get("crs") if isinstance(result.get("grid"), dict) else None
    k["totalBuildings"] = len(by_id)
    k["totalPopulation"] = round(population(list(by_id.values())))
    if flood:
        aff = flood.get("buildings", [])
        exposed = [by_id[int(b["id"])] for b in aff if int(b["id"]) in by_id]
        u = flood.get("uncertainty") or {}
        k["flood"] = {"model": flood["model"], "height": flood["waterLevel_m"], "heightRef": flood.get("waterLevelVerticalReference"), "area": flood["affectedAreaM2"],
                      "areaLikely": u.get("likelyAreaM2"), "areaPossible": u.get("possibleAreaM2"), "buildings": flood["affectedBuildingsCount"],
                      "buildingsLikely": u.get("buildingsLikely"), "buildingsPossible": u.get("buildingsPossible"), "byClass": flood.get("exposureCounts"),
                      "population": round(population(exposed)), "reliability": u.get("note"),
                      "worst": sorted(aff, key=lambda b: -(b.get("flood_depth_m") or 0))[:15]}
    if ls:
        k["landslide"] = {"classes": ls["classAreaPct"], "best": ls.get("classAreaPctIfGeologyBest"), "worst": ls.get("classAreaPctIfGeologyWorst"),
                          "highArea": ls["highOrWorseAreaM2"], "buildings": ls["buildingsHighOrWorse"], "unknown": ls["factors"]["unknown"], "rain": ls.get("rainfall"), "scars": ls.get("scars")}
    if hlz:
        wet = None
        if flood and not hlz.get("excludeFlooded") and (job_dir / "flood_depth.tif").exists():
            import rasterio

            with rasterio.open(job_dir / "flood_depth.tif") as ds:
                wet = np.nan_to_num(ds.read(1, masked=True).filled(0)) > 0
        for s in hlz["sites"]:
            s["_flooded"] = bool(wet is not None and wet[min(wet.shape[0] - 1, s["row"]), min(wet.shape[1] - 1, s["col"])])
        dropped = sum(s["_flooded"] for s in hlz["sites"])
        hlz = {**hlz, "sites": [s for s in hlz["sites"] if not s["_flooded"]]}
        k["hlzFloodedDropped"] = dropped
        k["hlz"] = {"size": hlz["size"], "diameter": hlz["padDiameterM"], "n": hlz["nSites"], "confidence": hlz.get("confidenceCounts"), "advice": hlz.get("advice"),
                    "sites": [{"id": s["id"], "lat": s.get("lat"), "lon": s.get("lon"), "slope": s["slopeDeg"], "clear": s["clearBearings"], "class": s["class"],
                               "confidence": (s.get("confidence") or {}).get("label"), "col": s["col"], "row": s["row"]} for s in hlz["sites"][:12]]}
    if roads:
        k["roads"] = {"cutKm": roads["roads"]["cutKm"], "strandedKm": roads["roads"]["strandedKm"], "totalKm": roads["roads"]["motorableKm"], "cutRoads": roads["roads"]["cutRoads"][:8],
                      "nCutOff": roads["nCutOff"], "nFootOnly": roads["nFootOnly"], "popCutOff": roads["populationCutOff"], "popFootOnly": roads["populationFootOnly"],
                      "settlements": [s for s in roads["settlements"] if s["status"] in ("CUT_OFF", "VEHICLE_CUT_FOOT_OPEN")][:15], "hazards": roads["hazards"]}
    k["warnings"] = {n: r.get("warnings", []) for n, r in (("Flood", flood), ("Landslide", ls), ("Landing zones", hlz), ("Road access", roads)) if r}
    k["notRun"] = [n for n, r in (("flood screening", flood), ("landslide screening", ls), ("helicopter landing zones", hlz), ("road access", roads)) if not r]
    return k


def _map(job_dir: Path, k: dict[str, Any]) -> np.ndarray:
    base = Image.open(job_dir / "input_preview.png").convert("RGBA")
    W, H = base.size
    s = MAP_PX / max(W, H)
    size = (max(1, int(W * s)), max(1, int(H * s)))
    img = base.resize(size, Image.BILINEAR)
    img = Image.blend(img, Image.new("RGBA", size, (255, 255, 255, 255)), 0.25)  # lighten so overlays read
    for name in ("landslide_preview.png", "flood_preview.png", "roads_preview.png"):
        p = job_dir / name
        if p.exists():
            ov = Image.open(p).convert("RGBA").resize(size, Image.NEAREST if name != "flood_preview.png" else Image.BILINEAR)
            img = Image.alpha_composite(img, ov)
    return np.asarray(img)


def build_pdf(job_dir: Path, result: dict[str, Any], out: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    k = collect(job_dir, result)
    out = out or job_dir / "damage_report.pdf"
    from core.geo.atomic import replace_with_retry, unique_tmp

    tmp = unique_tmp(out)  # unique: two quick clicks must not share a temp file
    A4 = (8.27, 11.69)
    wrap = lambda t, w=95: "\n".join(textwrap.wrap(str(t), w))  # noqa: E731

    def page(title: str):
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.965, TITLE, fontsize=9, color="#555")
        fig.text(0.06, 0.94, title, fontsize=15, weight="bold")
        fig.text(0.94, 0.965, f"generated {k['generated']}", fontsize=8, color="#555", ha="right")
        fig.text(0.06, 0.02, "Screening results for planning and reconnaissance only - not a survey, forecast or clearance. Verify on the ground.", fontsize=7, color="#a00")
        return fig

    def table(fig, y: float, rows: list[list[str]], widths: list[float], head: bool = True, fs: float = 8.5) -> float:
        for i, row in enumerate(rows):
            x = 0.06
            for cell, w in zip(row, widths):
                fig.text(x, y, cell, fontsize=fs, weight="bold" if (head and i == 0) else "normal", va="top")
                x += w
            y -= 0.02 * (1 + max(str(c).count("\n") for c in row))
        return y - 0.01

    with PdfPages(tmp) as pdf:
        # 1. summary
        fig = page("Situation summary")
        y = 0.9
        fig.text(0.06, y, f"Scene: {k.get('input') or k['job']}   ·   job {k['job']}   ·   {k['totalBuildings']} buildings, ~{k['totalPopulation']:,} people (estimate)", fontsize=9)
        y -= 0.04
        rows = [["Hazard / need", "Result", "Detail"]]
        f = k.get("flood")
        if f:
            h = f"{f['height']:g} m above the river" if f["model"] == "river" else f"{f['height']:g} m water level"
            rows.append(["Flood", f"{_area(f['area'])} flooded", f"{h}; range {_area(f['areaLikely'])} likely - {_area(f['areaPossible'])} possible" if f["areaLikely"] is not None else h])
            rows.append(["  buildings exposed", f"{f['buildings']}", f"~{f['population']:,} people; by depth class: " + ", ".join(f"{c} {n}" for c, n in (f["byClass"] or {}).items() if n)])
        r = k.get("roads")
        if r:
            rows.append(["Roads", f"{r['cutKm']:g} km cut", f"of {r['totalKm']:g} km motorable; {r['strandedKm']:g} km open but cut off from the outside"])
            rows.append(["  settlements cut off", f"{r['nCutOff']} (+{r['nFootOnly']} foot only)", f"~{r['popCutOff']:,} people with no route out; ~{r['popFootOnly']:,} reachable on foot only"])
        ls = k.get("landslide")
        if ls:
            hi = ls["classes"].get("HIGH", 0) + ls["classes"].get("VERY HIGH", 0)
            rng = ""
            if ls["best"]:
                rng = f"; {ls['best'].get('HIGH', 0) + ls['best'].get('VERY HIGH', 0):.0f}-{ls['worst'].get('HIGH', 0) + ls['worst'].get('VERY HIGH', 0):.0f} % depending on geology ({', '.join(ls['unknown'])} unknown)"
            rows.append(["Landslide hazard", f"{hi:.0f} % high or very high", f"{_area(ls['highArea'])}, {ls['buildings']} buildings{rng}"])
            rain = ls.get("rain") or {}
            if rain.get("worst"):
                w = rain["worst"]
                rows.append(["  rainfall trigger", "EXCEEDED" if rain["exceeded"] else f"{100 * w['ratio']:.0f} % of threshold", f"{w['rainMm']:g} mm in {w['durationH']} h ({rain.get('source') or 'user'})"])
            if ls.get("scars") and ls["scars"].get("count") is not None:
                rows.append(["  new slope scars", f"{ls['scars']['count']}", f"{_area(ls['scars']['areaM2'])} of new bare ground on steep slopes (Sentinel-2)"])
        hz = k.get("hlz")
        if hz:
            c = hz["confidence"] or {}
            if k.get("hlzFloodedDropped"):
                rows.append(["Helicopter landing", f"{len(hz['sites'])} usable sites", f"{k['hlzFloodedDropped']} of {hz['n']} candidate sites lie in the current flood and are left out; re-run landing zones with 'exclude flooded ground'"])
            else:
                rows.append(["Helicopter landing", f"{hz['n']} candidate sites", f"size {hz['size']} ({hz['diameter']:g} m pad); confidence high {c.get('HIGH', 0)}, medium {c.get('MEDIUM', 0)}, low {c.get('LOW', 0)}"])
        y = table(fig, y, [[a, b, wrap(c, 60)] for a, b, c in rows], [0.2, 0.2, 0.5])
        if k["notRun"]:
            fig.text(0.06, y, wrap("Not run (no result in this report): " + ", ".join(k["notRun"]) + "."), fontsize=8.5, color="#a60", va="top")
            y -= 0.04
        fig.text(0.06, y, "Data sources", fontsize=11, weight="bold", va="top")
        y -= 0.025
        src = ["Image: the uploaded scene (see its licence); heights: DepthWizard image model calibrated to the DEM.",
               "Terrain: CartoDEM (NRSC/ISRO) or Copernicus GLO-30. Rivers, roads, places: OpenStreetMap (ODbL).",
               "Buildings: bundled footprints (Microsoft, ODbL) or DepthWizard detection. Rain: Open-Meteo (CC BY 4.0).",
               "Landslide factors: BIS IS 14496 (Part 2). Landing zones: US Army FM 3-21.38. " + POPULATION_NOTE]
        fig.text(0.06, y, "\n".join(wrap(s) for s in src), fontsize=8, va="top", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # 2. map
        fig = page("Map")
        ax = fig.add_axes([0.06, 0.2, 0.88, 0.7])
        im = _map(job_dir, k)
        ax.imshow(im)
        sc = im.shape[1] / Image.open(job_dir / "input_preview.png").size[0]
        for s in (hz or {}).get("sites", []):
            ax.plot(s["col"] * sc, s["row"] * sc, marker="H", ms=11, mfc="#16a34a" if s["confidence"] == "HIGH" else "#65a30d", mec="white", mew=1.2)
            ax.text(s["col"] * sc + 12, s["row"] * sc, str(s["id"]), fontsize=7, color="white", weight="bold")
        for st in (r or {}).get("settlements", []):
            ax.text(st["col"] * sc + 14, st["row"] * sc - 10, st["name"], fontsize=7, color="#7f1d1d", weight="bold", bbox={"fc": "white", "alpha": 0.7, "pad": 1, "lw": 0})
        ax.set_xticks([])
        ax.set_yticks([])
        leg = ["Blue: flood depth (light = shallow)" if f else None, "Red / green lines: roads cut / open; amber: open but cut off" if r else None,
               "Dots: settlements (red cut off, amber foot only, green open)" if r else None, "Yellow-red shading: landslide hazard (moderate to very high)" if ls else None,
               "H: candidate helicopter landing site (number = rank)" if hz else None]
        fig.text(0.06, 0.18, "\n".join(x for x in leg if x), fontsize=8.5, va="top", linespacing=1.5)
        pdf.savefig(fig)
        plt.close(fig)

        # 3. action lists
        fig = page("Action lists")
        y = 0.9
        if r and r["settlements"]:
            fig.text(0.06, y, "Settlements that lost their road link", fontsize=11, weight="bold", va="top")
            y = table(fig, y - 0.03, [["Settlement", "Status", "Buildings", "People (est.)"]] + [[s["name"], s["status"].replace("_", " ").lower(), str(s["nBuildings"]), f"{s['population'] or '–'}"] for s in r["settlements"]], [0.35, 0.25, 0.15, 0.15])
        if r and r["cutRoads"]:
            fig.text(0.06, y, "Roads cut", fontsize=11, weight="bold", va="top")
            y = table(fig, y - 0.03, [["Road", "Length cut"]] + [[c["name"], f"{c['cutM']:,} m"] for c in r["cutRoads"]], [0.5, 0.2])
        if hz and hz["sites"]:
            fig.text(0.06, y, f"Candidate landing sites (size {hz['size']}, {hz['diameter']:g} m) - reconnoitre before use", fontsize=11, weight="bold", va="top")
            y = table(fig, y - 0.03, [["#", "Lat, lon", "Slope", "Clear approach bearings", "Confidence"]] + [[str(s["id"]), f"{s['lat']:.5f}, {s['lon']:.5f}" if s["lat"] else "–", f"{s['slope']:.1f}°",
                                                                                                      ", ".join(f"{b:g}°" for b in s["clear"][:6]) or "none", s["confidence"] or "–"] for s in hz["sites"]], [0.05, 0.28, 0.1, 0.35, 0.15])
        elif hz and hz.get("advice"):
            fig.text(0.06, y, wrap("Landing zones: " + hz["advice"]), fontsize=8.5, va="top")
            y -= 0.05
        if f and f["worst"] and y > 0.2:
            fig.text(0.06, y, "Most exposed buildings (flood)", fontsize=11, weight="bold", va="top")
            table(fig, y - 0.03, [["Building", "Depth at low side", "Class", "Wet share"]] + [[f"#{b['id']}", f"{b['flood_depth_m']:g} m", b["exposure"], f"{round(100 * b['wet_fraction'])} %" if b.get("wet_fraction") is not None else "–"] for b in f["worst"] if y > 0.1][:12], [0.2, 0.2, 0.2, 0.2])
        pdf.savefig(fig)
        plt.close(fig)

        # 4. limits
        fig = page("Limits of these results")
        y = 0.9
        if f and f.get("reliability"):
            fig.text(0.06, y, wrap("Flood range reliability: " + f["reliability"]), fontsize=8.5, va="top")
            y -= 0.05
        for name, ws in k["warnings"].items():
            fig.text(0.06, y, name, fontsize=10, weight="bold", va="top")
            y -= 0.022
            txt = "\n".join("• " + "\n  ".join(textwrap.wrap(w, 100)) for w in ws)
            fig.text(0.06, y, txt, fontsize=7.5, va="top", linespacing=1.35)
            y -= 0.0145 * (txt.count("\n") + 1) + 0.015
        pdf.savefig(fig)
        plt.close(fig)
        d = pdf.infodict()
        d["Title"], d["Subject"], d["Creator"] = TITLE, "Rapid damage and access assessment", "DepthWizard"
    replace_with_retry(tmp, out)
    return out
