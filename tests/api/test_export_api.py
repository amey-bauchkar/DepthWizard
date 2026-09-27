"""Offline 3D scene + GIS package exports through the API on a synthetic Mode B job (stub model, no bundled assets)."""
import io
import json
import re
import sqlite3
import zipfile

import rasterio

from core.export.glb import read_glb
from tests.api.test_mode_b import _run, synthetic_scene  # noqa: F401 - fixture re-export


def _fake_bundle(client, tmp_path):
    d = tmp_path / "dist" / "standalone"
    d.mkdir(parents=True)
    (d / "standalone.js").write_text("window.__dw_ok = 1;", encoding="utf-8")
    (d / "standalone.css").write_text("body{}", encoding="utf-8")
    client.app.state.jobs.settings.server.frontend_dist = str(tmp_path / "dist")


def test_offline_scene_export(client, synthetic_scene, tmp_path):
    _fake_bundle(client, tmp_path)
    jid, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    r = client.get(f"/api/jobs/{jid}/export/scene.html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["content-disposition"].startswith("attachment") and "_3D_offline.html" in r.headers["content-disposition"]
    assert client.get(f"/api/jobs/{jid}/export/scene.html", params={"inline": 1}).headers["content-disposition"].startswith("inline")
    sc = json.loads(re.search(r'<script id="dw-scene" type="application/json">(.*?)</script>', r.text, re.S).group(1))
    assert sc["format"] == "depthwizard-scene" and sc["mode"] == "B" and sc["crs"] == "EPSG:32643"
    assert {"dsm", "terrain", "relative"} <= set(sc["layers"]) and sc["texture"]["mime"] == "image/jpeg"
    dsm = sc["layers"]["dsm"]
    assert dsm["metric"] and dsm["q"]["max_rounding_error"] < 0.01 and "m" in dsm  # readings from the full-res rasters
    assert len(sc["lonlat_grid"]["lon"]) == sc["lonlat_grid"]["n"] ** 2 and sc["accuracy"]
    assert "window.__dw_ok" in r.text


def test_offline_scene_export_without_bundle_is_a_clear_error(client, synthetic_scene, tmp_path):
    client.app.state.jobs.settings.server.frontend_dist = str(tmp_path / "missing")
    jid, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    r = client.get(f"/api/jobs/{jid}/export/scene.html")
    assert r.status_code == 409 and "npm run build" in r.json()["error"]["message"]


def test_gis_package_export(client, synthetic_scene, tmp_path):
    _fake_bundle(client, tmp_path)
    jid, _ = _run(client, {"file": ("scene.tif", synthetic_scene["image"], "image/tiff"), "dem": ("dem.tif", synthetic_scene["dem"], "image/tiff")})
    r = client.get(f"/api/jobs/{jid}/export/package.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = {n.split("/", 1)[1] for n in z.namelist()}
    assert {"README.txt", "scene_3d_offline.html", "rasters/dsm.tif", "rasters/terrain.tif", "rasters/dsm.qml", "rasters/orthophoto.tif", "rasters/dem_input.tif", "3d/scene.glb", "metadata/stac_item.json", "metadata/provenance.json"} <= names
    root = z.namelist()[0].split("/")[0]
    with rasterio.open(io.BytesIO(z.read(f"{root}/rasters/dsm.tif"))) as ds:
        assert ds.crs.to_string() == "EPSG:32643" and ds.overviews(1) is not None and ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
    stac = json.loads(z.read(f"{root}/metadata/stac_item.json"))
    assert stac["stac_version"] == "1.0.0" and stac["properties"]["proj:epsg"] == 32643 and "dsm" in stac["assets"]
    (tmp_path / "s.glb").write_bytes(z.read(f"{root}/3d/scene.glb"))
    js, _bin = read_glb(tmp_path / "s.glb")
    assert js["asset"]["extras"]["crs"] == "EPSG:32643" and any(m["name"] == "terrain" for m in js["meshes"])
    readme = z.read(f"{root}/README.txt").decode("utf-8")
    assert "Vertical datum   : EGM2008" in readme and "Calibration tier" in readme
    if "vector/buildings.gpkg" in names:
        (tmp_path / "b.gpkg").write_bytes(z.read(f"{root}/vector/buildings.gpkg"))
        assert sqlite3.connect(tmp_path / "b.gpkg").execute("SELECT srs_id FROM gpkg_geometry_columns").fetchone()[0] == 32643
    # cached: a second request serves the same file
    assert client.get(f"/api/jobs/{jid}/export/package.zip").content == r.content


def test_gis_package_refused_for_mode_a(client, png_bytes, tmp_path):
    _fake_bundle(client, tmp_path)
    jid, _ = _run(client, {"file": ("chk.png", png_bytes, "image/png")})
    assert client.get(f"/api/jobs/{jid}/export/package.zip").status_code == 409
    r = client.get(f"/api/jobs/{jid}/export/scene.html")  # the offline 3D scene works for relative results too
    assert r.status_code == 200 and '"mode":"A"' in r.text
