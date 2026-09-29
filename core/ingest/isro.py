"""ISRO optical products -> the natural-colour GeoTIFF DepthWizard ingests.

Products from NRSC (Bhoonidhi) arrive as a zip or folder with one GeoTIFF per band (BAND1.tif ... or *_B2.tif ...) and
a metadata text file (BAND_META.txt / *_META.txt), or as one multi-band GeoTIFF. DepthWizard's image model expects
red, green, blue in bands 1-3, so bands are re-ordered by sensor:

    Cartosat-2 / 2E / 3 MX: B1 blue, B2 green, B3 red, B4 NIR   -> RGB = (B3, B2, B1)
    Resourcesat-2 / 2A LISS-IV MX: B2 green, B3 red, B4 NIR (no blue band) -> simulated natural colour
                                   R = B3, G = (3 B2 + B4) / 4, B = B2   (the usual SPOT-style simulation; flagged)
    PAN (Cartosat-1/2/3, single band) -> grey; with an MX product in the same package: Brovey pan-sharpening
                                   RGB_sharp = RGB_mx(resampled to PAN) * PAN / mean(RGB_mx)
All bands share one percentile stretch (P1-P99.5 over the three bands together), so the colour balance is kept.
The output is uint8, 3 bands, on the finest grid, with the product's CRS; NoData (0 in every band) stays 0.
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

SENSORS = {  # regex on metadata / file names -> (sensor, mx band order)
    "LISS4": (r"LISS[-_ ]?(IV|4)|\bL4(MX|FMX)?\b|RESOURCESAT|RS2A?\b|IRS[-_ ]?R2", "liss4"),
    "CARTOSAT": (r"CARTOSAT|\bC2[EF]?\b|\bC3\b|CARTO", "cartosat"),
}


def _band_no(name: str) -> int | None:
    m = re.search(r"(?:BAND|_B)(\d)(?!\d)", name.upper())
    return int(m.group(1)) if m else None


def _sensor(text: str) -> str | None:
    t = text.upper()
    for key, (rx, kind) in SENSORS.items():
        if re.search(rx, t):
            return kind
    return None


def _stretch(bands: list[np.ndarray], nodata: np.ndarray) -> list[np.ndarray]:
    vals = np.concatenate([b[~nodata].ravel() for b in bands]) if (~nodata).any() else np.array([0.0, 1.0])
    lo, hi = np.percentile(vals, [1.0, 99.5])
    out = []
    for b in bands:
        u = np.clip((b - lo) / max(hi - lo, 1e-9) * 254 + 1, 1, 255).astype(np.uint8)
        u[nodata] = 0
        out.append(u)
    return out


def _read(p: Path) -> tuple[np.ndarray, Any]:
    with rasterio.open(p) as ds:
        return ds.read().astype(np.float64), ds.profile


def convert(src: Path, out: Path, *, max_uncompressed_mb: float = 3000.0) -> dict[str, Any]:
    """src: a zip, a folder, or a multi-band GeoTIFF. Writes out (GeoTIFF RGB uint8). Returns what was done."""
    src = Path(src)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        if src.suffix.lower() == ".zip":
            with zipfile.ZipFile(src) as z:
                infos = z.infolist()
                total = sum(i.file_size for i in infos)
                if len(infos) > 500 or total > max_uncompressed_mb * 1024 * 1024:
                    raise ValueError(f"the zip holds {len(infos)} files / {total / 1e6:.0f} MB once extracted (limit 500 files / {max_uncompressed_mb:g} MB)")
                z.extractall(root)  # zipfile drops '..' and absolute parts of member names (no path traversal)
        elif src.is_dir():
            root = src
        else:
            root = src.parent
        files = [src] if src.suffix.lower() in (".tif", ".tiff") else sorted(p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
        meta_txt = " ".join(p.read_text(errors="ignore")[:20000] for p in root.rglob("*") if p.suffix.lower() == ".txt" and "META" in p.name.upper()) if src.suffix.lower() != ".tif" else ""
        kind = _sensor(meta_txt + " " + " ".join(f.name for f in files) + " " + src.name)
        if not files:
            raise ValueError("no GeoTIFF band files found in the product")
        def inside(f: Path) -> str:  # path within the product only (never the user's own folder names)
            try:
                return str(f.relative_to(root)).upper()
            except ValueError:
                return f.name.upper()

        pan = [f for f in files if "PAN" in inside(f)]
        mx = [f for f in files if f not in pan]
        by_band = {n: f for f in mx if (n := _band_no(f.name)) is not None}
        notes: list[str] = []
        if len(mx) == 1 and _band_no(mx[0].name) is None:  # one multi-band GeoTIFF
            arr, prof = _read(mx[0])
            bands = {i + 1: arr[i] for i in range(arr.shape[0])}
        elif by_band:
            prof = None
            bands = {}
            for n, f in sorted(by_band.items()):
                a, pr = _read(f)
                bands[n] = a[0]
                prof = prof or pr
        elif pan:
            bands, prof = {}, None
        else:
            raise ValueError("could not identify the band files (expected BAND1.tif ... or *_B2.tif ...)")

        if bands:
            if kind == "liss4" or (set(bands) >= {2, 3, 4} and 1 not in bands):
                R, G_, N = bands[3], bands[2], bands[4]
                rgb = [R, (3 * G_ + N) / 4.0, G_]
                notes.append("LISS-IV has no blue band: natural colour simulated as R=B3, G=(3 B2 + B4)/4, B=B2")
                kind = kind or "liss4"
            elif set(bands) >= {1, 2, 3}:
                rgb = [bands[3], bands[2], bands[1]]
                notes.append("MX bands re-ordered: R=B3, G=B2, B=B1")
                kind = kind or "cartosat"
            else:
                raise ValueError(f"unsupported band set {sorted(bands)}")
            nod = np.all([b == 0 for b in rgb], axis=0)
        if pan:
            P, pprof = _read(pan[0])
            P = P[0]
            if bands:
                up = []
                for b in rgb:
                    d = np.zeros(P.shape)
                    reproject(b, d, src_transform=prof["transform"], src_crs=prof["crs"], dst_transform=pprof["transform"], dst_crs=pprof["crs"], resampling=Resampling.bilinear)
                    up.append(d)
                m = np.mean(up, axis=0)
                rgb = [u * P / np.maximum(m, 1e-6) for u in up]
                notes.append("Brovey pan-sharpened: MX colour on the PAN grid")
            else:
                rgb = [P, P, P]
                notes.append("PAN only: grey image")
            prof, nod = pprof, P == 0
        u8 = _stretch(rgb, nod)
        prof.update(driver="GTiff", count=3, dtype="uint8", nodata=0, compress="deflate", photometric="RGB", tiled=True)
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
        with rasterio.open(out, "w", **prof) as ds:
            ds.write(np.stack(u8))
            ds.update_tags(SOURCE=f"ISRO product {src.name}", SENSOR=kind or "unknown", PROCESSING="; ".join(notes))
    if prof.get("crs") is None:
        notes.append("the product has no CRS: the result is Mode A (relative heights)")
    return {"sensor": kind, "notes": notes, "width": prof["width"], "height": prof["height"], "crs": str(prof.get("crs"))}


def convert_bytes(filename: str, data: bytes, *, max_uncompressed_mb: float = 3000.0) -> tuple[str, bytes, dict[str, Any]]:
    """Upload helper: a zipped ISRO product -> ('<name>_rgb.tif', GeoTIFF bytes, info)."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / Path(filename).name
        src.write_bytes(data)
        out = Path(td) / (Path(filename).stem + "_rgb.tif")
        info = convert(src, out, max_uncompressed_mb=max_uncompressed_mb)
        return out.name, out.read_bytes(), info

