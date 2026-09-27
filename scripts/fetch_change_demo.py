"""Fetch a before / after demo pair for DepthWizard's change screening — no account or login needed.

Islahiye (Gaziantep province, Türkiye) was heavily damaged by the Mw 7.8 Kahramanmaras earthquake of 6 February 2023.
Maxar's Open Data Program published a pre-event and a post-event WorldView acquisition of the town, both close to
nadir (14.8 deg and 5.7 deg off-nadir), which keeps building lean small between the two dates.

  * RGB: Maxar Open Data Program, event "Kahramanmaras-turkey-earthquake-23", WorldView visual COGs read over HTTP and
    resampled to 0.5 m GeoTIFFs on ONE shared UTM grid (EPSG:32637). Licence CC BY-NC 4.0 (non-commercial;
    attribution "Maxar Technologies, Maxar Open Data Program").
  * DEM: Copernicus GLO-30 tile N37E036 (AWS open data), EGM2008.

Usage:  python scripts/fetch_change_demo.py
Writes assets/demo/change/<pair>_{before,after}_<date>_0.5m.tif, the DEM tile to assets/dem/, and two Mode B demo
entries (role before / after, same "pair") to assets/demo/manifest.json. Process both in the app, then compare them
in "Before / after change screening".
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import reproject  # noqa: E402
from pyproj import Transformer  # noqa: E402

OUT = ROOT / "assets" / "demo" / "change"
DEM_DIR = ROOT / "assets" / "dem"
COP = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
GSD = 0.5
PAIRS = {
    "islahiye": dict(
        event="Kahramanmaras-turkey-earthquake-23", lon=36.6310, lat=37.0260, size_m=1200, epsg=32637, country="TR",
        dem="Copernicus_DSM_COG_10_N37_00_E036_00_DEM",
        before=dict(acq="10300100E0287700", date="2022-12-27", off_nadir=14.8),
        after=dict(acq="1040010082698700", date="2023-02-07", off_nadir=5.7),
        place="Islahiye, Türkiye",
        desc="Town centre of Islahiye (Gaziantep), heavily damaged by the Mw 7.8 Kahramanmaras earthquake of 6 Feb 2023.",
    ),
}


def _get(u: str) -> dict:
    with urllib.request.urlopen(u, timeout=90) as r:
        return json.load(r)


def visual_hrefs(event: str, acq: str) -> list[str]:
    ev = f"https://maxar-opendata.s3.amazonaws.com/events/{event}/collection.json"
    for l in _get(ev)["links"]:
        if l["rel"] != "child":
            continue
        cu = urljoin(ev, l["href"])
        c = _get(cu)
        if c["id"] != acq:
            continue
        out = []
        for x in c["links"]:
            if x["rel"] == "item":
                iu = urljoin(cu, x["href"])
                out.append(urljoin(iu, _get(iu)["assets"]["visual"]["href"]))
        return out
    raise SystemExit(f"acquisition {acq} not found in {ev}")


def fetch(name: str, p: dict, role: str) -> Path:
    a = p[role]
    out = OUT / f"{name}_{role}_{a['date']}_0.5m.tif"
    if out.exists():
        print("cached", out.name)
        return out
    crs = f"EPSG:{p['epsg']}"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(p["lon"], p["lat"])
    n = int(p["size_m"] / GSD)
    left, top = round(x - p["size_m"] / 2), round(y + p["size_m"] / 2)  # both dates on exactly this grid
    tr = from_origin(left, top, GSD, GSD)
    dst = np.zeros((3, n, n), np.uint8)
    got = np.zeros((n, n), bool)
    for href in visual_hrefs(p["event"], a["acq"]):
        with rasterio.open(href) as src:
            from rasterio.warp import transform_bounds

            b = transform_bounds(src.crs, crs, *src.bounds)
            if b[2] <= left or b[0] >= left + p["size_m"] or b[3] <= top - p["size_m"] or b[1] >= top:
                continue
            tmp = np.zeros((3, n, n), np.uint8)
            for i in range(3):
                reproject(rasterio.band(src, i + 1), tmp[i], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.average, src_nodata=0, dst_nodata=0)
            m = (tmp.max(axis=0) > 0) & ~got
            dst[:, m] = tmp[:, m]
            got |= m
            print(f"  {name} {role}: +{int(m.sum())} px from {href.split('/')[-1]}")
    if got.mean() < 0.95:
        raise SystemExit(f"{name} {role}: only {got.mean():.0%} of the AOI covered by {a['acq']}")
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=crs, transform=tr, compress="jpeg", jpeg_quality=92, photometric="ycbcr", tiled=True) as ds:
        ds.write(dst)
        ds.update_tags(SOURCE=f"Maxar Open Data Program, event {p['event']}, acquisition {a['acq']} ({a['date']})", LICENSE="CC BY-NC 4.0", ATTRIBUTION="Maxar Technologies, Maxar Open Data Program", ACQUISITION_DATE=a["date"])
    print("wrote", out.name, f"{n}x{n}")
    return out


def fetch_dem(name: str) -> Path:
    out = DEM_DIR / f"{name}.tif"
    if not out.exists():
        print("downloading", out.name)
        urllib.request.urlretrieve(COP.format(n=name), out)
    return out


def update_manifest(name: str, p: dict, files: dict[str, Path]) -> None:
    mp = ROOT / "assets" / "demo" / "manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    ids = {f"change_{name}_{r}" for r in files}
    items = [i for i in man["items"] if i["id"] not in ids]
    for role, f in files.items():
        a = p[role]
        items.append({
            "id": f"change_{name}_{role}", "mode": "B", "country": p["country"], "pair": name, "role": role, "date": a["date"],
            "label": f"{p['place']} · {role.upper()} {a['date']} · Maxar WorldView · 0.5 m · Mode B (change demo)",
            "file": f"change/{f.name}",
            "source": f"Maxar Open Data Program (CC BY-NC 4.0), acquisition {a['acq']} ({a['date']}, {a['off_nadir']} deg off-nadir); {p['desc']}",
        })
    man["items"] = items
    mp.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for name, p in PAIRS.items():
        fetch_dem(p["dem"])
        files = {role: fetch(name, p, role) for role in ("before", "after")}
        update_manifest(name, p, files)
    print("done: process both images in DepthWizard, then open 'Before / after change screening' on the BEFORE result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
