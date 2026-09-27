"""GIS data package: one zip an analyst can open directly in QGIS / ArcGIS / Blender / any STAC-aware catalogue.

    <name>/README.txt                    what every file is, datum, tier, measured accuracy, licences
    <name>/scene_3d_offline.html         the standalone 3D explorer (double-click, no install)
    <name>/rasters/*.tif + *.qml         Cloud-Optimized GeoTIFFs (DSM, terrain, nDSM, slope, flags, orthophoto,
                                         input DEM when redistributable) with QGIS default styles
    <name>/vector/buildings.gpkg         LoD-1 footprints in the job CRS with heights, error band, volume, floors
                                         range (QGIS style embedded); buildings.geojson (WGS84) and buildings.csv
    <name>/3d/scene.glb                  textured terrain + LoD-1 blocks (glTF 2.0), origin in asset.extras
    <name>/metadata/stac_item.json       STAC 1.0 item (projection + raster extensions)
    <name>/metadata/provenance.json      job, model, calibration report, validation history
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.shutil import copy as rio_copy

from core.dsm.derive import FLAG_BITS
from core.export import gpkg
from core.export.glb import write_glb
from core.export.scene import accuracy_lines, demo_source, scene_payload, standalone_html
from core.terrain import buildings as bld

RASTERS = {  # job file -> (package name, title, overview resampling)
    "dsm.tif": ("dsm.tif", "Digital Surface Model (ground + objects), metres", "AVERAGE"),
    "terrain.tif": ("terrain.tif", "Terrain layer (bare ground, DEM-derived), metres", "AVERAGE"),
    "ndsm.tif": ("ndsm.tif", "nDSM: height above ground, metres", "AVERAGE"),
    "slope.tif": ("slope.tif", "Slope, degrees", "AVERAGE"),
    "flags.tif": ("flags.tif", "Quality flags (bit field, see README)", "NEAREST"),
    "dem.tif": ("dem_input.tif", "Input DEM resampled to the job grid, metres", "AVERAGE"),
    "input.tif": ("orthophoto.tif", "Input image (orthophoto)", "AVERAGE"),
}
RAMPS = {
    "elevation": [(0.0, "#2c7bb6"), (0.25, "#abd9e9"), (0.5, "#ffffbf"), (0.75, "#fdae61"), (1.0, "#d7191c")],
    "height": [(0.0, "#f7fcf5"), (0.15, "#c7e9c0"), (0.35, "#74c476"), (0.6, "#238b45"), (1.0, "#00441b")],
    "slope": [(0.0, "#ffffcc"), (0.33, "#fed976"), (0.66, "#fd8d3c"), (1.0, "#bd0026")],
}
NO_REDISTRIBUTION_DEMS = {"cartodem"}  # NRSC/Bhoonidhi terms: keep the raw DEM local unless redistribution is confirmed


def _hex_rgb(c: str) -> str:
    return ",".join(str(int(c[i:i + 2], 16)) for i in (1, 3, 5))


def raster_qml(lo: float, hi: float, ramp: str, unit: str) -> str:
    items = "\n".join(f'          <item alpha="255" value="{lo + t * (hi - lo):.4f}" label="{lo + t * (hi - lo):.1f} {unit}" color="{c}"/>' for t, c in RAMPS[ramp])
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1" classificationMin="{lo:.4f}" classificationMax="{hi:.4f}" nodataColor="">
      <rasterTransparency/>
      <minMaxOrigin><limits>None</limits><extent>WholeRaster</extent><statAccuracy>Estimated</statAccuracy><cumulativeCutLower>0.02</cumulativeCutLower><cumulativeCutUpper>0.98</cumulativeCutUpper><stdDevFactor>2</stdDevFactor></minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" minimumValue="{lo:.4f}" maximumValue="{hi:.4f}" labelPrecision="1">
{items}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


BUILDING_CLASSES = [(0.0, 5.0, "below 5 m", "#fff5eb"), (5.0, 10.0, "5 - 10 m", "#fdd0a2"), (10.0, 15.0, "10 - 15 m", "#fd8d3c"), (15.0, 25.0, "15 - 25 m", "#d94801"), (25.0, 1000.0, "25 m and above", "#7f2704")]


def buildings_qml() -> str:
    def props(color: str) -> str:
        kv = {"color": f"{_hex_rgb(color)},255", "outline_color": "70,70,70,255", "outline_style": "solid", "outline_width": "0.2", "outline_width_unit": "MM", "style": "solid"}
        opt = "".join(f'<Option type="QString" name="{k}" value="{v}"/>' for k, v in kv.items())
        prop = "".join(f'<prop k="{k}" v="{v}"/>' for k, v in kv.items())
        return f'<Option type="Map">{opt}</Option>{prop}'
    ranges = "\n".join(f'      <range lower="{lo:.6f}" upper="{hi:.6f}" symbol="{i}" label="{lab}" render="true"/>' for i, (lo, hi, lab, _c) in enumerate(BUILDING_CLASSES))
    symbols = "\n".join(f'      <symbol type="fill" name="{i}" alpha="0.9" clip_to_extent="1" force_rhr="0"><layer class="SimpleFill" enabled="1" locked="0" pass="0">{props(c)}</layer></symbol>' for i, (_lo, _hi, _lab, c) in enumerate(BUILDING_CLASSES))
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" styleCategories="AllStyleCategories">
  <renderer-v2 type="graduatedSymbol" attr="height_m" graduatedMethod="GraduatedColor" symbollevels="0" enableorderby="0" forceraster="0">
    <ranges>
{ranges}
    </ranges>
    <symbols>
{symbols}
    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>2</layerGeometryType>
</qgis>
"""


def to_cog(src: Path, dst: Path, resampling: str) -> None:
    with rasterio.open(src) as ds:
        rgb8 = ds.count == 3 and ds.dtypes[0] == "uint8"
        is_float = ds.dtypes[0].startswith("float")
    opts: dict[str, Any] = {"BLOCKSIZE": 512, "OVERVIEW_RESAMPLING": resampling, "BIGTIFF": "IF_SAFER"}
    if rgb8:
        opts.update(COMPRESS="JPEG", QUALITY=90)
    else:
        opts.update(COMPRESS="DEFLATE", PREDICTOR="YES" if is_float else "NO")
    rio_copy(str(src), str(dst), driver="COG", **opts)


def _reprojected(job_dir: Path) -> bool:
    m = job_dir / "meta.json"
    return bool(json.loads(m.read_text(encoding="utf-8")).get("reprojected")) if m.exists() else True


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(s).stem)[:48].strip("_")
    return s or "scene"


def _building_features(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recs = bld.records(data)
    by_id = {b["id"]: b for b in data.get("buildings", [])}
    feats = []
    for r in recs:
        ring = bld._to_crs(data, by_id[r["id"]]["coords"])  # noqa: SLF001 - the one CRS mapping used by every export
        props = {k: v for k, v in r.items() if k not in ("scene_xy",)}
        feats.append({"rings": [ring], "properties": props})
    return recs, feats


def _stac_item(job: dict[str, Any], result: dict[str, Any], assets: dict[str, Any], app_version: str) -> dict[str, Any]:
    g = result.get("grid") or {}
    tr = Affine.from_gdal(*g["transform"])
    from pyproj import CRS, Transformer

    to_ll = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True)
    corners = [tr * (0, 0), tr * (g["width"], 0), tr * (g["width"], g["height"]), tr * (0, g["height"])]
    ring = [list(map(lambda v: round(v, 8), to_ll.transform(x, y))) for x, y in corners]
    ring.append(ring[0])
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    epsg = CRS.from_user_input(g["crs"]).to_epsg()
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "type": "Feature", "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/projection/v1.1.0/schema.json", "https://stac-extensions.github.io/raster/v1.1.0/schema.json", "https://stac-extensions.github.io/processing/v1.1.0/schema.json"],
        "id": f"depthwizard-{job.get('job_id')}",
        "geometry": {"type": "Polygon", "coordinates": [ring]}, "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "properties": {
            "datetime": now, "created": now, "title": f"DepthWizard height products for {job.get('input_filename')}",
            "description": "Single-view height estimation: DSM, terrain layer and nDSM from one optical image, calibrated with a national DEM. The acquisition date of the input image is not known to DepthWizard; datetime is the processing time.",
            "proj:epsg": epsg, "proj:shape": [g["height"], g["width"]], "proj:transform": list(tr)[:6],
            "gsd": result.get("gsd_m"), "processing:software": {"DepthWizard": app_version}, "processing:level": "L3",
            "depthwizard:calibration_tier": result.get("calibration_tier"), "depthwizard:quality": result.get("quality"),
            "depthwizard:vertical_reference": result.get("vertical_reference"), "depthwizard:uncertainty": result.get("uncertainty"),
        },
        "links": [], "assets": assets,
    }


def build_package(job_dir: Path, result: dict[str, Any], job: dict[str, Any], *, out_zip: Path, app_version: str, manifest_path: Path, bundle_dir: Path | None) -> dict[str, Any]:
    job_dir = Path(job_dir)
    if result.get("mode") != "B":
        raise ValueError("the GIS package needs a georeferenced (Mode B) job; use the offline 3D scene for Mode A")
    name = f"DepthWizard_{_safe_name(job.get('input_filename') or 'scene')}_{job.get('job_id')}"
    work = out_zip.parent / f"_pkg_{job.get('job_id')}"
    if work.exists():
        shutil.rmtree(work)
    root = work / name
    for sub in ("rasters", "vector", "3d", "metadata"):
        (root / sub).mkdir(parents=True)
    contents: dict[str, Any] = {"files": [], "skipped": []}
    dem_name = (result.get("dem") or {}).get("name")
    legends = {k: (v or {}).get("legend") or {} for k, v in (result.get("layers") or {}).items()}
    stac_assets: dict[str, Any] = {}
    # --- rasters (COG) + QGIS default styles
    for src_name, (dst_name, title, rs) in RASTERS.items():
        src = job_dir / src_name
        if not src.exists():
            continue
        if src_name == "dem.tif" and dem_name in NO_REDISTRIBUTION_DEMS:
            contents["skipped"].append(f"rasters/{dst_name}: {dem_name} redistribution terms not confirmed (the DSM / terrain derived from it are included)")
            continue
        if src_name == "input.tif" and _reprojected(job_dir):
            contents["skipped"].append(f"rasters/{dst_name}: the input was reprojected at ingest, so the original file is not on the job grid")
            continue
        to_cog(src, root / "rasters" / dst_name, rs)
        contents["files"].append(f"rasters/{dst_name}")
        key = dst_name.removesuffix(".tif")
        stac_assets[key] = {"href": f"../rasters/{dst_name}", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "title": title, "roles": ["data"] if key != "orthophoto" else ["visual"]}
        with rasterio.open(root / "rasters" / dst_name) as ds:
            if ds.count == 1 and ds.dtypes[0].startswith("float"):
                stac_assets[key]["raster:bands"] = [{"data_type": ds.dtypes[0], "nodata": ds.nodata, "unit": "degree" if key == "slope" else "metre"}]
            if key in ("dsm", "terrain", "dem_input", "ndsm", "slope") and ds.count == 1:
                a = ds.read(1, out_shape=(max(1, ds.height // 8), max(1, ds.width // 8)), masked=True).astype("float64").filled(np.nan)
                lg = legends.get(key if key != "dem_input" else "dsm") or {}
                lo = lg.get("lo") if isinstance(lg.get("lo"), (int, float)) else float(np.nanpercentile(a, 2))
                hi = lg.get("hi") if isinstance(lg.get("hi"), (int, float)) else float(np.nanpercentile(a, 98))
                if key == "ndsm":
                    lo, hi = 0.0, max(5.0, float(np.nanpercentile(a, 99)))
                if key == "slope":
                    lo, hi = 0.0, 60.0
                (root / "rasters" / f"{key}.qml").write_text(raster_qml(float(lo), float(hi), "height" if key == "ndsm" else "slope" if key == "slope" else "elevation", "°" if key == "slope" else "m"), encoding="utf-8")
                contents["files"].append(f"rasters/{key}.qml")
    # --- buildings
    bj = (result.get("artifacts") or {}).get("buildings_json")
    data = json.loads((job_dir / bj).read_text(encoding="utf-8")) if bj and (job_dir / bj).exists() else None
    if data and data.get("buildings"):
        recs, feats = _building_features(data)
        gpkg.write_polygon_layer(root / "vector" / "buildings.gpkg", "buildings", data["grid"]["crs"], feats, description="DepthWizard LoD-1 building footprints with predicted heights", qml_style=buildings_qml())
        (root / "vector" / "buildings.geojson").write_text(json.dumps(bld.to_geojson(data, recs)), encoding="utf-8")
        (root / "vector" / "buildings.csv").write_text(bld.to_csv(recs), encoding="utf-8")
        contents["files"] += ["vector/buildings.gpkg", "vector/buildings.geojson", "vector/buildings.csv"]
        contents["buildings"] = len(recs)
        stac_assets["buildings"] = {"href": "../vector/buildings.gpkg", "type": "application/geopackage+sqlite3", "title": "LoD-1 building footprints with heights", "roles": ["data"]}
    # --- 3D model (terrain heightfield + LoD-1 blocks, exact georeferencing)
    g = result["grid"]
    tr = Affine.from_gdal(*g["transform"])
    W, H = g["width"], g["height"]
    cx, cy = tr * (W / 2.0, H / 2.0)
    gsd = float(result.get("gsd_m") or abs(tr.a))
    lay = (result.get("layers") or {}).get("terrain") or (result.get("layers") or {}).get("dsm") or {}
    if lay.get("heightfield") and (job_dir / lay["heightfield"]).exists():
        meta = json.loads((job_dir / lay["heightfield_meta"]).read_text(encoding="utf-8"))
        z = np.fromfile(job_dir / lay["heightfield"], dtype="<f4").reshape(meta["height"], meta["width"]).astype(np.float64)
        f = int(meta.get("downsample_factor") or 1)
        cols = (np.arange(meta["width"]) * f + f / 2.0)  # heightfield sample i = centre of source block i
        rows = (np.arange(meta["height"]) * f + f / 2.0)
        xs = np.array([(tr * (c, 0))[0] for c in cols]) - cx
        ns = np.array([(tr * (0, r))[1] for r in rows]) - cy
        us, vs = cols / W, rows / H
        blds = None
        if data and data.get("buildings"):
            blds = []
            for b in data["buildings"]:
                ring = bld._to_crs(data, b["coords"])  # noqa: SLF001
                blds.append({**b, "coords": [(x - cx, y - cy) for x, y in ring]})
        x0, y0 = tr * (0, 0)
        uv_of = lambda e, n: ((cx + e - x0) / (W * gsd), (y0 - (cy + n)) / (H * gsd))  # noqa: E731
        tex = (result.get("artifacts") or {}).get("texture")
        extras = {"crs": g["crs"], "origin_easting": cx, "origin_northing": cy, "vertical_reference": result.get("vertical_reference"), "axes": "glTF: +X east, +Y up, -Z north; metres",
                  "note": "Y = elevation - origin_elevation_m. Terrain from the DepthWizard terrain layer; blocks = LoD-1 buildings (height = volume / area of the predicted nDSM)."}
        zref = float(np.nanmin(z)) if np.isfinite(z).any() else 0.0
        extras["origin_elevation_m"] = zref
        st = write_glb(root / "3d" / "scene.glb", heights=z, axes=(xs, ns, us, vs), buildings=blds, uv_of=uv_of,
                       texture_jpeg=(job_dir / tex).read_bytes() if tex and tex.lower().endswith((".jpg", ".jpeg")) and (job_dir / tex).exists() else None,
                       extras=extras, zref=zref, generator=f"DepthWizard {app_version}")
        contents["files"].append("3d/scene.glb")
        contents["glb"] = st
        stac_assets["model3d"] = {"href": "../3d/scene.glb", "type": "model/gltf-binary", "title": "Textured terrain + LoD-1 buildings (glTF 2.0)", "roles": ["visual"]}
    # --- offline 3D scene
    if bundle_dir is not None:
        try:
            html = standalone_html(scene_payload(job_dir, result, job, app_version=app_version, manifest_path=manifest_path), bundle_dir)
            (root / "scene_3d_offline.html").write_text(html, encoding="utf-8")
            contents["files"].append("scene_3d_offline.html")
        except FileNotFoundError as e:
            contents["skipped"].append(f"scene_3d_offline.html: {e}")
    # --- metadata
    (root / "metadata" / "stac_item.json").write_text(json.dumps(_stac_item(job, result, stac_assets, app_version), indent=1), encoding="utf-8")
    prov = {"job": job, "result": result, "flag_bits": FLAG_BITS, "calib_report": json.loads((job_dir / "calib_report.json").read_text(encoding="utf-8")) if (job_dir / "calib_report.json").exists() else None,
            "validation": json.loads((job_dir / "validation.json").read_text(encoding="utf-8")) if (job_dir / "validation.json").exists() else None, "exported_by": f"DepthWizard {app_version}"}
    (root / "metadata" / "provenance.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    contents["files"] += ["metadata/stac_item.json", "metadata/provenance.json"]
    (root / "README.txt").write_text(readme(job, result, contents, app_version, demo_source(job.get("input_filename"), manifest_path)), encoding="utf-8")
    # --- zip (COG / GLB are already compressed: stored; text: deflated)
    tmp = out_zip.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as z_:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                comp = zipfile.ZIP_STORED if p.suffix in (".tif", ".glb") else zipfile.ZIP_DEFLATED
                z_.write(p, p.relative_to(work).as_posix(), compress_type=comp)
    tmp.replace(out_zip)
    shutil.rmtree(work, ignore_errors=True)
    contents["zip_bytes"] = out_zip.stat().st_size
    contents["folder"] = name
    return contents


def readme(job: dict[str, Any], result: dict[str, Any], contents: dict[str, Any], app_version: str, source: dict[str, Any] | None) -> str:
    dem = result.get("dem") or {}
    tier_txt = {"T": "T: absolute elevations from the DEM (datum-checked) + object heights from the model",
                "A": "A: like T, plus a vertical offset fitted to your ground control points",
                "H": "H: heights above ground from the fine-tuned model; no absolute elevation (no DEM)",
                "R": "R: relative structure only, no metres"}.get(str(result.get("calibration_tier")), str(result.get("calibration_tier")))
    L = [
        "DepthWizard height products",
        "=" * 28, "",
        f"Input image      : {job.get('input_filename')}  (sha256 {str(job.get('input_sha256'))[:16]}...)",
        f"Processed        : {(job.get('timestamps') or {}).get('READY')} with DepthWizard {app_version} (method {result.get('method_version')})",
        f"Grid             : {result['grid']['width']} x {result['grid']['height']} px at {result.get('gsd_m')} m, {result['grid']['crs']}",
        f"Vertical datum   : {result.get('vertical_reference') or 'none (relative)'} (heights in metres)",
        f"Calibration tier : {tier_txt}",
        f"Quality          : {result.get('quality')}" + (f" ({'; '.join(result.get('quality_triggers') or [])})" if result.get("quality_triggers") else ""),
        f"DEM used         : {dem.get('product') or 'none'}",
        "", "Measured accuracy (typical error, 1 sigma)", "-" * 42,
        *[f"* {x}" for x in accuracy_lines(result)],
        "", "Contents", "-" * 8,
        "scene_3d_offline.html   3D explorer: double-click to open in Chrome / Edge / Firefox. Works offline, no installation.",
        "rasters/dsm.tif         surface elevation (ground + buildings + trees)",
        "rasters/terrain.tif     bare-ground elevation (DEM-derived terrain layer; NOT a surveyed DTM)",
        "rasters/ndsm.tif        height above ground (dsm - terrain)",
        "rasters/slope.tif       slope of the DSM in degrees",
        "rasters/flags.tif       quality flags per pixel (bit field: " + ", ".join(f"{v}={k}" for k, v in FLAG_BITS.items()) + ")",
        "rasters/orthophoto.tif  the input image",
        "rasters/*.qml           QGIS styles, applied automatically when a raster is added",
        "vector/buildings.gpkg   LoD-1 building footprints (job CRS) with height, height interval, ground / roof",
        "                        elevation, area, volume and a floors RANGE; QGIS style embedded",
        "vector/buildings.geojson / buildings.csv   the same table in WGS84",
        "3d/scene.glb            textured terrain + building blocks (Blender: File > Import > glTF 2.0)",
        "metadata/stac_item.json STAC 1.0 item for catalogues; provenance.json = full processing record",
        "",
        "All rasters are Cloud-Optimized GeoTIFFs with internal overviews; drag the folder into QGIS.",
    ]
    if contents.get("skipped"):
        L += ["", "Not included", "-" * 12, *[f"* {s}" for s in contents["skipped"]]]
    L += ["", "How to read the numbers", "-" * 23,
          "* Heights come from a single image. They are a fast first look, not a replacement for stereo or LiDAR surveys.",
          "* Building height = median predicted nDSM over the footprint; the block height is volume / area.",
          "* Floors are a range assuming 3.0-3.5 m per storey, not a count.",
          "", "Licences and attribution", "-" * 24]
    if source and source.get("attribution"):
        L.append(f"* Input imagery: {source['attribution']}")
    else:
        L.append("* Input imagery: supplied by the user; its licence applies to rasters/orthophoto.tif and the textured 3D files.")
    if dem.get("name") == "cartodem":
        L.append("* DEM: CartoDEM v3 R1, (c) NRSC / ISRO (Bhuvan / Bhoonidhi). Derived products; the raw DEM is not redistributed.")
    elif dem.get("name") == "bundled":
        L.append("* DEM: Copernicus DEM GLO-30, (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.")
    L += ["* Height model: Depth Anything V2 Small (Apache-2.0), fine-tuned by the DepthWizard team on swisstopo (OGD) and USGS 3DEP / NAIP (public domain) data.",
          "* DepthWizard software: see the project licence."]
    return "\n".join(L) + "\n"
