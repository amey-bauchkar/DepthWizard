"""Fetch the Indian demo / validation package (Sikkim, Teesta basin) — no account or login needed.

  * RGB: Maxar Open Data Program, event "India-Floods-Oct-2023" (South Lhonak GLOF, Sikkim), WorldView ARD
    visual COGs, read over HTTP and resampled to 0.5 m GeoTIFFs (EPSG:32645). Licence CC BY-NC 4.0 (non-commercial;
    attribution "Maxar Technologies, Maxar Open Data Program").
  * DEM: Copernicus GLO-30 tile N27E088 (AWS open data), EGM2008.
  * Independent checkpoints: NASA ICESat-2 laser altimetry (ATL08-style ground + canopy heights, 20 m segments,
    2018-2025) via the public SlideRule service (https://slideruleearth.io). Requires `pip install sliderule`
    (see requirements/india-data.txt). Heights are ellipsoidal (WGS84); DepthWizard converts them to EGM2008 with its
    datum guard before comparing.

Usage:  python scripts/fetch_india_demo.py [--aoi namchi chungthang] [--no-icesat2]
Writes assets/demo/india/<aoi>_rgb_0.5m.tif, assets/demo/india/<aoi>_icesat2.csv, assets/dem/Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif,
and adds Mode B demo entries to assets/demo/manifest.json.
"""
from __future__ import annotations

import argparse
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

EVENT = "https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/collection.json"
OUT = ROOT / "assets" / "demo" / "india"
DEM_DIR = ROOT / "assets" / "dem"
COP = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"

# AOIs: centre (lon, lat), side length (m), Maxar acquisition id, description
AOIS = {
    "namchi": dict(lon=88.3615, lat=27.1655, size_m=1200, acq="1040010073381800", label="Namchi, South Sikkim (hill town) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B",
                   desc="Dense hill town on steep slopes; district HQ of Namchi (South Sikkim)."),
    "chungthang": dict(lon=88.6455, lat=27.6030, size_m=1200, acq="10300100CE8D0400", label="Chungthang, North Sikkim (valley town, forest) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B",
                       desc="Steep forested valley and town at the Lachen-Lachung confluence; Teesta-III dam area hit by the Oct-2023 GLOF."),
    # additional test sites, chosen for ICESat-2 track density (>= 2.8 km from the sites above)
    "teesta_east": dict(lon=88.4057, lat=27.1799, size_m=1200, acq="1040010073381800", label="Teesta valley east, South Sikkim (hill villages) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B",
                        desc="Terraced slopes and scattered villages east of Namchi."),
    "teesta_west": dict(lon=88.3185, lat=27.1519, size_m=1200, acq="1040010073381800", label="Teesta valley west, South Sikkim (rural slopes) · Maxar WorldView 2022-03-14 · 0.5 m · Mode B",
                        desc="Rural terraced hillsides and forest patches west of Namchi."),
    "chungthang_west": dict(lon=88.6185, lat=27.6313, size_m=1200, acq="10300100CE8D0400", label="Chungthang west, North Sikkim (forested slopes) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B",
                            desc="Steep forested mountainside above the Lachen valley."),
    "north_sikkim_alpine": dict(lon=88.2955, lat=27.9083, size_m=1200, acq="10300100CF621C00", label="North Sikkim alpine (barren / glacial) · Maxar WorldView 2022-03-07 · 0.5 m · Mode B",
                                desc="High-altitude barren and glacial terrain in the South Lhonak region (near-nadir image, off-nadir 2 deg)."),
}
GSD = 0.5
EPSG = 32645


def _get(u: str) -> dict:
    with urllib.request.urlopen(u, timeout=90) as r:
        return json.load(r)


def maxar_visual_hrefs(acq: str) -> list[str]:
    cat = _get(EVENT)
    for l in cat["links"]:
        if l["rel"] != "child":
            continue
        cu = urljoin(EVENT, l["href"])
        c = _get(cu)
        if c["id"] != acq:
            continue
        hrefs = []
        for x in c["links"]:
            if x["rel"] == "item":
                iu = urljoin(cu, x["href"])
                it = _get(iu)
                hrefs.append(urljoin(iu, it["assets"]["visual"]["href"]))
        return hrefs
    raise SystemExit(f"acquisition {acq} not found in {EVENT}")


def fetch_rgb(name: str, a: dict) -> Path:
    out = OUT / f"{name}_rgb_0.5m.tif"
    if out.exists():
        print("cached", out.name)
        return out
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(a["lon"], a["lat"])
    n = int(a["size_m"] / GSD)
    left, top = round(x - a["size_m"] / 2), round(y + a["size_m"] / 2)
    tr = from_origin(left, top, GSD, GSD)
    dst = np.zeros((3, n, n), np.uint8)
    got = np.zeros((n, n), bool)
    for href in maxar_visual_hrefs(a["acq"]):
        with rasterio.open(href) as src:
            b = src.bounds
            if b.right <= left or b.left >= left + a["size_m"] or b.top <= top - a["size_m"] or b.bottom >= top:
                continue
            tmp = np.zeros((3, n, n), np.uint8)
            for i in range(3):
                reproject(rasterio.band(src, i + 1), tmp[i], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=f"EPSG:{EPSG}", resampling=Resampling.average, src_nodata=0, dst_nodata=0)
            m = (tmp.max(axis=0) > 0) & ~got
            dst[:, m] = tmp[:, m]
            got |= m
            print(f"  {name}: +{int(m.sum())} px from {href.split('/')[-1]}")
    if got.mean() < 0.95:
        raise SystemExit(f"{name}: only {got.mean():.0%} of the AOI covered by {a['acq']}")
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=f"EPSG:{EPSG}", transform=tr, compress="jpeg", jpeg_quality=92, photometric="ycbcr", tiled=True, nodata=None) as ds:
        ds.write(dst)
        ds.update_tags(SOURCE=f"Maxar Open Data Program, event India-Floods-Oct-2023, acquisition {a['acq']}", LICENSE="CC BY-NC 4.0", ATTRIBUTION="Maxar Technologies, Maxar Open Data Program")
    print("wrote", out.name, f"{n}x{n}")
    return out


def fetch_dem() -> Path:
    name = "Copernicus_DSM_COG_10_N27_00_E088_00_DEM"
    out = DEM_DIR / f"{name}.tif"
    if not out.exists():
        print("downloading", out.name)
        urllib.request.urlretrieve(COP.format(n=name), out)
    return out


def fetch_icesat2(name: str, a: dict) -> Path | None:
    out = OUT / f"{name}_icesat2.csv"
    if out.exists():
        print("cached", out.name)
        return out
    try:
        from sliderule import icesat2, sliderule
    except ImportError:
        print("  ICESat-2 skipped: `pip install -r requirements/india-data.txt` to fetch checkpoints")
        return None
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(a["lon"], a["lat"])
    back = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    h = a["size_m"] / 2
    ring = [back.transform(x + dx, y + dy) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h), (-h, -h))]
    sliderule.init("slideruleearth.io", verbose=False)
    parms = {"poly": [{"lon": lo, "lat": la} for lo, la in ring], "t0": "2018-10-01T00:00:00Z", "t1": "2025-12-31T00:00:00Z", "srt": icesat2.SRT_LAND, "len": 20.0, "res": 20.0, "cnf": 0, "pass_invalid": True,
             "atl08_class": ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"], "phoreal": {"binsize": 1.0, "geoloc": "center", "use_abs_h": False, "send_waveform": False}}
    gdf = icesat2.atl08p(parms)
    if gdf is None or len(gdf) == 0:
        print("  no ICESat-2 data")
        return None
    rows = ["# ICESat-2 (NASA) ground/canopy heights via SlideRule PhoREAL, 20 m segments. h_ground: ellipsoidal WGS84 (m). h_canopy: RH98-like canopy height above ground (m).",
            "# vcrs=ellipsoidal", "id,lon,lat,h_ground,h_canopy,gnd_photons,veg_photons,night,date,rgt,cycle,spot"]
    for i, (t, r) in enumerate(gdf.iterrows()):
        rows.append(f"S{i:05d},{r.geometry.x:.7f},{r.geometry.y:.7f},{r.h_te_median:.3f},{r.h_canopy:.3f},{int(r.gnd_ph_count)},{int(r.veg_ph_count)},{int(r.solar_elevation < 0)},{str(t)[:10]},{int(r.rgt)},{int(r.cycle)},{int(r.spot)}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("wrote", out.name, len(gdf), "segments")
    return out


def update_manifest(name: str, a: dict, rgb: Path, pts: Path | None) -> None:
    mp = ROOT / "assets" / "demo" / "manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    items = [i for i in man["items"] if i["id"] != f"india_{name}"]
    item = {"id": f"india_{name}", "label": a["label"], "file": f"india/{rgb.name}", "mode": "B", "country": "IN", "source": f"Maxar Open Data Program (CC BY-NC 4.0), acquisition {a['acq']}; {a['desc']}"}
    if pts:
        item["reference_points"] = f"india/{pts.name}"
        item["reference_points_vertical_crs"] = "ellipsoidal"
    items.append(item)
    man["items"] = items
    mp.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", nargs="*", default=list(AOIS))
    ap.add_argument("--no-icesat2", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_dem()
    for name in a.aoi:
        cfg = AOIS[name]
        rgb = fetch_rgb(name, cfg)
        pts = None if a.no_icesat2 else fetch_icesat2(name, cfg)
        update_manifest(name, cfg, rgb, pts)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
