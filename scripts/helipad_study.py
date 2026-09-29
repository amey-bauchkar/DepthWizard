"""Landing-zone screening vs REAL helipads: does DepthWizard find the helipads that exist?

Truth: helipads mapped in OpenStreetMap (aeroway=helipad|heliport, ODbL) that lie inside free Maxar Open Data imagery
(CC BY-NC 4.0) with a 1.2 km chip around them:
  * India-Floods-Oct-2023 (Sikkim),
  * Nepal-Earthquake-Nov-2023 (Jajarkot / Rukum West, Karnali hills).
Rooftop pads (location=roof or on a building) are excluded: the screening looks for ground pads.

For every helipad: the 1.2 km chip centred on it is processed by the CURRENT pipeline (Mode B, Copernicus GLO-30),
then landing zones are screened for sizes 1-3 (no user settings). Per helipad and size:
  * hit        a reported site centre lies within HIT_M of the mapped helipad centre;
  * rank       the site's rank in the list (1 = best);
  * why        if missed: the rejection reasons among pad centres within HIT_M (landing_feasible.tif bitmask);
  * chance     the share of the scene within HIT_M of ANY reported site: the hit rate a screening that places the same
               number of sites at random would get. Hit rate / mean chance = how much better than random.
The helipad positions are OSM's (not surveyed); a pad mapped 20-30 m off, or a helipad built after / removed before
the image date, counts as a miss here. Rows are reported individually so each can be checked on the image.

Usage:
  python scripts/helipad_study.py            (discover -> fetch -> process -> screen; each step cached)
Writes data/helipad_study/..., docs/helipad_validation.{md,json}
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "helipad_study"
os.environ["DW_DATA_DIR"] = str(OUT / "_data")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import reproject  # noqa: E402

EVENTS = ["India-Floods-Oct-2023", "Nepal-Earthquake-Nov-2023"]
MAXAR = "https://maxar-opendata.s3.amazonaws.com/events/{}/collection.json"
OVERPASS = "https://overpass-api.de/api/interpreter"
COP = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
SIZE_M, GSD, HIT_M = 1200, 0.5, 30.0
UA = {"User-Agent": "DepthWizard/1.0 (helipad validation study)"}


def _get(u: str) -> dict:
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if k == 3:
                raise
            time.sleep(3 * (k + 1))
    raise RuntimeError


def _overpass(q: str) -> dict:
    for k in range(5):
        try:
            req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("  overpass retry:", e, flush=True)
            time.sleep(15 * (k + 1))
    raise SystemExit("Overpass unavailable")


def _items(ev: str) -> list[dict]:
    EV = MAXAR.format(ev)
    urls = []
    for link in _get(EV)["links"]:
        if link["rel"] == "child":
            cu = urljoin(EV, link["href"])
            c = _get(cu)
            urls += [(c["id"], urljoin(cu, x["href"])) for x in c["links"] if x["rel"] == "item"]

    def one(a):
        acq, iu = a
        it = _get(iu)
        p = it["properties"]
        return {"acq": acq, "geom": it["geometry"], "date": p.get("datetime", "")[:10], "offnadir": p.get("view:off_nadir"), "href": urljoin(iu, it["assets"]["visual"]["href"])}

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, urls))


def discover() -> list[dict]:
    """Helipads (OSM) whose whole 1.2 km chip is inside one Maxar acquisition (the least off-nadir one is used)."""
    p = OUT / "sites.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    sites = []
    for ev in EVENTS:
        items = _items(ev)
        acq_geom: dict[str, dict] = {}
        for i in items:
            a = acq_geom.setdefault(i["acq"], {"geoms": [], "hrefs": [], "date": i["date"], "offnadir": i["offnadir"]})
            a["geoms"].append(shape(i["geom"]))
            a["hrefs"].append(i["href"])
        for a in acq_geom.values():
            a["union"] = unary_union(a["geoms"])
        x0, y0, x1, y1 = unary_union([a["union"] for a in acq_geom.values()]).bounds
        q = f'[out:json][timeout:240];(nwr["aeroway"~"^(helipad|heliport)$"]({y0},{x0},{y1},{x1}););out center tags;'
        els = _overpass(q)["elements"]
        print(ev, len(items), "image tiles,", len(els), "OSM helipads in the event bbox", flush=True)
        for e in els:
            t = e.get("tags", {})
            if t.get("location") == "roof" or "building" in t or t.get("surface") == "roof":
                continue
            lat, lon = e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))
            dlat, dlon = SIZE_M / 2 / 111_320, SIZE_M / 2 / (111_320 * np.cos(np.radians(lat)))
            chip = box(lon - dlon * 1.05, lat - dlat * 1.05, lon + dlon * 1.05, lat + dlat * 1.05)
            cov = sorted(((acq, a) for acq, a in acq_geom.items() if a["union"].contains(chip)), key=lambda c: c[1]["offnadir"] or 99)
            if not cov:
                continue
            acq, a = cov[0]
            sites.append({"id": f"{e['type']}_{e['id']}", "osm": f"https://www.openstreetmap.org/{e['type']}/{e['id']}", "name": t.get("name"), "lat": lat, "lon": lon,
                          "event": ev, "acq": acq, "date": a["date"], "offnadir": a["offnadir"], "hrefs": a["hrefs"]})
    # the same helipad mapped twice (node + way) or neighbouring pads within one chip: keep the first per 150 m
    kept: list[dict] = []
    for s in sites:
        if all(abs(s["lat"] - k["lat"]) * 111_320 > 150 or abs(s["lon"] - k["lon"]) * 111_320 * np.cos(np.radians(s["lat"])) > 150 for k in kept):
            kept.append(s)
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    return kept


def _utm(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def fetch_chip(s: dict) -> Path:
    out = OUT / "chips" / f"{s['id']}_rgb_0.5m.tif"
    if out.exists():
        return out
    epsg = _utm(s["lon"], s["lat"])
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(s["lon"], s["lat"])
    n = int(SIZE_M / GSD)
    left, top = round(x - SIZE_M / 2), round(y + SIZE_M / 2)
    tr = from_origin(left, top, GSD, GSD)
    dst, got = np.zeros((3, n, n), np.uint8), np.zeros((n, n), bool)
    for href in s["hrefs"]:
        with rasterio.open(href) as src:
            from rasterio.warp import transform_bounds

            b = transform_bounds(src.crs, f"EPSG:{epsg}", *src.bounds)
            if b[2] <= left or b[0] >= left + SIZE_M or b[3] <= top - SIZE_M or b[1] >= top:
                continue
            tmp = np.zeros((3, n, n), np.uint8)
            for i in range(3):
                reproject(rasterio.band(src, i + 1), tmp[i], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=f"EPSG:{epsg}", resampling=Resampling.average, src_nodata=0, dst_nodata=0)
            m = (tmp.max(axis=0) > 0) & ~got
            dst[:, m] = tmp[:, m]
            got |= m
    if got.mean() < 0.95:
        raise RuntimeError(f"only {got.mean():.0%} covered")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=f"EPSG:{epsg}", transform=tr, compress="jpeg", jpeg_quality=92, photometric="ycbcr", tiled=True) as ds:
        ds.write(dst)
        ds.update_tags(SOURCE=f"Maxar Open Data Program, event {s['event']}, acquisition {s['acq']}", LICENSE="CC BY-NC 4.0", ATTRIBUTION="Maxar Technologies, Maxar Open Data Program")
    return out


def ensure_dem(lat: float, lon: float) -> None:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    name = f"Copernicus_DSM_COG_10_{ns}{abs(int(np.floor(lat))):02d}_00_{ew}{abs(int(np.floor(lon))):03d}_00_DEM"
    p = ROOT / "assets" / "dem" / f"{name}.tif"
    if not p.exists():
        print("  downloading", name, flush=True)
        urllib.request.urlretrieve(COP.format(n=name), p)


def process(s: dict, chip: Path, client) -> Path:
    dst = OUT / s["id"]
    if (dst / "result.json").exists():
        return dst
    jid = client.post("/api/jobs", files={"file": (chip.name, chip.read_bytes(), "image/tiff")}).json()["job_id"]
    client.post(f"/api/jobs/{jid}/run")
    while (j := client.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
        time.sleep(1)
    if j["status"] != "READY":
        raise RuntimeError(f"job {jid} {j['status']}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(OUT / "_data" / "jobs" / jid, dst)
    return dst


def screen_site(s: dict, job: Path) -> dict:
    from core.disaster.landing_zones import R_FEASIBLE, R_INVALID, R_OBJECT, R_ROUGH, R_SLOPE, run_landing_zone_screening

    res = json.loads((job / "result.json").read_text(encoding="utf-8"))
    with rasterio.open(job / "dsm.tif") as ds:
        crs, tr, shp = ds.crs, ds.transform, ds.shape
    hx, hy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(s["lon"], s["lat"])
    col, row = ~tr * (hx, hy)
    yy, xx = np.mgrid[0:shp[0], 0:shp[1]]
    px = abs(tr.a)
    near = np.hypot((xx + 0.5 - col) * px, (yy + 0.5 - row) * px) <= HIT_M
    rows = []
    for size in (1, 2, 3):
        o = run_landing_zone_screening(job, res, size=size)
        with rasterio.open(job / "landing_feasible.tif") as ds:
            r = ds.read(1)
        d = [(float(np.hypot(q["x"] - hx, q["y"] - hy)), q) for q in o["sites"]]
        hit = sorted((x for x in d if x[0] <= HIT_M), key=lambda x: x[1]["id"])
        # chance: scene share within HIT_M of any reported site centre (random placement of the same sites)
        cov = np.zeros(shp, bool)
        for q in o["sites"]:
            cov |= np.hypot((xx + 0.5 - q["col"] - 0.5) * px, (yy + 0.5 - q["row"] - 0.5) * px) <= HIT_M
        rn = r[near]
        why = {k: round(100 * float(((rn & bit) > 0).mean()), 0) for k, bit in (("object", R_OBJECT), ("slope", R_SLOPE), ("rough", R_ROUGH), ("noData", R_INVALID))}
        rows.append({"size": size, "sites": o["nSites"], "hit": bool(hit), "rank": hit[0][1]["id"] if hit else None,
                     "confidence": hit[0][1]["confidence"]["label"] if hit else None, "nearestSiteM": round(min(x[0] for x in d), 1) if d else None,
                     "feasibleNearPct": round(100 * float((rn == R_FEASIBLE).mean()), 0), "whyNotNearPct": why, "chance": round(float(cov.mean()), 4)})
    return {k: s[k] for k in ("id", "osm", "name", "lat", "lon", "event", "acq", "date", "offnadir")} | {"quality": res.get("quality"), "sizes": rows}


def report(rows: list[dict]) -> str:
    L = ["# Landing-zone screening vs real helipads", "",
         "Generated by `python scripts/helipad_study.py` (raw: `docs/helipad_validation.json`).",
         f"Truth: OpenStreetMap helipads (ground pads) inside free Maxar Open Data imagery ({', '.join(EVENTS)}). Each helipad's 1.2 km chip is processed by the current pipeline; landing zones are screened with default settings. **Hit** = a reported site centre within {HIT_M:g} m of the mapped helipad. **Chance** = the hit rate a screening placing the same number of sites at random would get.", "",
         "## Summary", "", "| Size | Helipads | Hits | Hit rate | Chance (mean) | Better than random | Sites per 1.44 km² (median) |", "|---|---|---|---|---|---|---|"]
    for size in (1, 2, 3):
        z = [s for r in rows for s in r["sizes"] if s["size"] == size]
        h = sum(s["hit"] for s in z)
        ch = float(np.mean([s["chance"] for s in z]))
        L.append(f"| {size} | {len(z)} | {h} | {100 * h / len(z):.0f} % | {100 * ch:.1f} % | {(h / len(z)) / ch:.0f}× |" if ch > 0 else f"| {size} | {len(z)} | {h} | {100 * h / len(z):.0f} % | 0 % | – |")
        L[-1] += f" {int(np.median([s['sites'] for s in z]))} |"
    anyhit = sum(any(s["hit"] for s in r["sizes"]) for r in rows)
    L += ["", f"**Helipads found at any size: {anyhit} of {len(rows)}.**", "",
          "## Per helipad", "", "| Helipad | Image (off-nadir) | Size 1 | Size 2 | Size 3 | If missed: pad centres within 30 m rejected for (size 1) |", "|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for s in r["sizes"]:
            cells.append(f"hit, rank {s['rank']} ({s['confidence']})" if s["hit"] else (f"miss (nearest {s['nearestSiteM']:g} m)" if s["nearestSiteM"] is not None else "miss (no sites)"))
        s1 = r["sizes"][0]
        why = "" if s1["hit"] else ", ".join(f"{k} {v:g} %" for k, v in s1["whyNotNearPct"].items() if v) or f"feasible {s1['feasibleNearPct']:g} % (not chosen: spacing / corridor)"
        L.append(f"| [{r['name'] or r['id']}]({r['osm']}) | {r['event'].split('-')[0]} {r['date']} ({r['offnadir']:.0f}°) | " + " | ".join(cells) + f" | {why} |")
    L += ["", "## Reading", "",
          "* The chance column is the fair baseline: a screening that reports many sites would hit helipads by luck. The ratio says how much better than random placement the screening is.",
          "* Misses are explained by the rejection reasons at the helipad itself. \"object\" usually means a tree, vehicle or building within the pad plus the 10 m buffer (or trees the model reads over the pad), \"slope\" a pad the 30 m DEM and image model read as too steep.",
          "* Limits: OSM positions are volunteered, not surveyed; a helipad may have been built after, or overgrown before, the image date. The Nepal images are 28–41° off-nadir, which the model was not trained for. Each row links to the OSM object so it can be checked.",
          ""]
    return "\n".join(L)


def main() -> int:
    sites = discover()
    print(len(sites), "helipads with full image coverage", flush=True)
    from fastapi.testclient import TestClient

    from backend.config.settings import load_settings
    from backend.main import create_app

    client = TestClient(create_app(load_settings()))
    rows = []
    for s in sites:
        try:
            ensure_dem(s["lat"], s["lon"])
            chip = fetch_chip(s)
            job = process(s, chip, client)
            row = screen_site(s, job)
        except Exception as e:  # noqa: BLE001 - report and continue with the others
            print("  SKIP", s["id"], e, flush=True)
            continue
        rows.append(row)
        print(s["id"], s["name"], [(x["size"], x["hit"], x["sites"], x["nearestSiteM"]) for x in row["sizes"]], flush=True)
    (ROOT / "docs" / "helipad_validation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (ROOT / "docs" / "helipad_validation.md").write_text(report(rows), encoding="utf-8")
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
