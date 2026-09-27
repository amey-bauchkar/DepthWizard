"""Exports: GeoPackage writer, glTF (GLB) writer, ear-clipping, heightfield quantization, standalone HTML embedding."""
import io
import json
import re
import sqlite3
import struct

import numpy as np
from PIL import Image

from core.export import gpkg
from core.export.glb import read_glb, triangulate, write_glb
from core.export.scene import dequantize, quantize, standalone_html


def test_gpkg_roundtrip_and_metadata(tmp_path):
    feats = [
        {"rings": [[(500000.0, 3000000.0), (500010.0, 3000000.0), (500010.0, 3000008.0), (500000.0, 3000008.0)]], "properties": {"id": 1, "height_m": 12.5, "flags": ["EDGE"], "name": "a"}},
        {"rings": [[(500020.0, 3000000.0), (500030.0, 3000000.0), (500025.0, 3000006.0)]], "properties": {"id": 2, "height_m": 3.0, "flags": [], "name": None}},
    ]
    p = gpkg.write_polygon_layer(tmp_path / "b.gpkg", "buildings", "EPSG:32643", feats, qml_style="<!DOCTYPE qgis>\n<qgis/>")
    con = sqlite3.connect(p)
    assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
    assert con.execute("PRAGMA user_version").fetchone()[0] == 10300
    row = con.execute("SELECT data_type, srs_id, min_x, min_y, max_x, max_y FROM gpkg_contents WHERE table_name='buildings'").fetchone()
    assert row == ("features", 32643, 500000.0, 3000000.0, 500030.0, 3000008.0)
    assert con.execute("SELECT geometry_type_name, srs_id FROM gpkg_geometry_columns").fetchone() == ("POLYGON", 32643)
    assert con.execute("SELECT count(*) FROM gpkg_spatial_ref_sys WHERE srs_id IN (-1, 0, 4326, 32643)").fetchone()[0] == 4
    assert con.execute("SELECT useAsDefault, f_table_name FROM layer_styles").fetchone() == (1, "buildings")
    con.close()
    back = gpkg.read_polygons(p, "buildings")
    assert [r["id"] for r in back] == [1, 2] and back[0]["height_m"] == 12.5 and json.loads(back[0]["flags"]) == ["EDGE"]
    ring = back[1]["rings"][0]
    assert ring[0] == ring[-1] and len(ring) == 4  # closed on write
    assert ring[:3] == [(500020.0, 3000000.0), (500030.0, 3000000.0), (500025.0, 3000006.0)]
    blob = sqlite3.connect(p).execute("SELECT geom FROM buildings WHERE fid=1").fetchone()[0]
    assert blob[:4] == b"GP\x00\x03" and struct.unpack_from("<i", blob, 4)[0] == 32643
    assert struct.unpack_from("<4d", blob, 8) == (500000.0, 500010.0, 3000000.0, 3000008.0)  # envelope minx, maxx, miny, maxy


def _area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def test_triangulate_concave_polygon_covers_its_area():
    L = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3), (0, 2)]  # L-shape with a collinear vertex (0, 2)
    tris = triangulate(L)
    assert all(_area([L[a], L[b], L[c]]) > 0 for a, b, c in tris)  # all counter-clockwise, none degenerate
    assert abs(sum(_area([L[a], L[b], L[c]]) for a, b, c in tris) - _area(L)) < 1e-9


def test_glb_is_valid_and_skips_nodata_cells(tmp_path):
    z = np.array([[10, 11, 12, 13, 14], [10, 11, np.nan, 13, 14], [10, 11, 12, 13, 14], [10, 10, 10, 10, 10]], np.float64)
    xs = np.arange(5) * 2.0 - 4.0
    ns = 3.0 - np.arange(4) * 2.0
    us, vs = np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 4)
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(buf, "JPEG")
    blds = [{"coords": [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0), (0.0, 0.0)], "base_elev_m": 11.0, "ground_min_m": 10.5, "height_m": 6.0}]
    st = write_glb(tmp_path / "s.glb", heights=z, axes=(xs, ns, us, vs), buildings=blds, uv_of=lambda e, n: (0.5, 0.5), texture_jpeg=buf.getvalue(), extras={"crs": "EPSG:32643"}, zref=10.0)
    js, binary = read_glb(tmp_path / "s.glb")
    assert js["asset"]["version"] == "2.0" and js["asset"]["extras"]["crs"] == "EPSG:32643"
    assert len(binary) % 4 == 0 and js["buffers"][0]["byteLength"] == len(binary)
    assert st["terrain_triangles"] == 2 * 4 * 3 - 2 * 4  # 12 cells, the 4 around the NaN sample dropped
    for acc in js["accessors"]:
        bv = js["bufferViews"][acc["bufferView"]]
        assert bv["byteOffset"] % 4 == 0 and bv["byteOffset"] + bv["byteLength"] <= len(binary)
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            pos = js["accessors"][prim["attributes"]["POSITION"]]
            idx = js["accessors"][prim["indices"]]
            ib = js["bufferViews"][idx["bufferView"]]
            ind = np.frombuffer(binary, np.uint32, idx["count"], ib["byteOffset"])
            assert ind.max() < pos["count"] and "min" in pos and "max" in pos
            pb = js["bufferViews"][pos["bufferView"]]
            P = np.frombuffer(binary, np.float32, pos["count"] * 3, pb["byteOffset"]).reshape(-1, 3)
            assert np.isfinite(P).all()
    roofs = [p for m in js["meshes"] for p in m["primitives"] if js["materials"][p["material"]]["name"] == "building roofs"][0]
    top = js["accessors"][roofs["attributes"]["POSITION"]]
    assert abs(top["max"][1] - (11.0 + 6.0 - 10.0)) < 1e-5  # roof = base + height - zref (glTF Y up)
    assert js["images"][0]["mimeType"] == "image/jpeg"


def test_quantize_roundtrip_error_and_nodata():
    rng = np.random.default_rng(1)
    z = rng.uniform(1500, 1900, (37, 41))
    z[3, 4] = np.nan
    q = quantize(z)
    back = dequantize(q, z.shape)
    assert np.isnan(back[3, 4]) and np.isfinite(back).sum() == z.size - 1
    assert np.nanmax(np.abs(back - z)) <= q["max_rounding_error"] + 1e-9
    assert q["max_rounding_error"] < 0.005  # 400 m of relief -> sub-centimetre steps


def test_standalone_html_embeds_payload_safely(tmp_path):
    (tmp_path / "standalone.js").write_text('var s = "</script><!-- x";console.log(s);', encoding="utf-8")
    (tmp_path / "standalone.css").write_text("body{margin:0}", encoding="utf-8")
    payload = {"title": "Evil </script><script>alert(1)</script> & <b>", "app_version": "t", "notes": ["a</SCRIPT>b"]}
    html = standalone_html(payload, tmp_path)
    assert html.lower().count("</script>") == 2  # exactly the two real closing tags
    m = re.search(r'<script id="dw-scene" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(m.group(1)) == payload
    assert "<title>Evil &lt;/script&gt;" in html
    assert '"<\\/script><\\!-- x"' in html  # JS strings keep their meaning
