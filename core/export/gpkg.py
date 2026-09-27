"""Minimal OGC GeoPackage (1.3) writer for polygon layers, using only sqlite3.

Writes the mandatory metadata tables (gpkg_spatial_ref_sys, gpkg_contents, gpkg_geometry_columns), one feature table
with GeoPackage binary geometries (header + little-endian WKB polygons with an [minx, maxx, miny, maxy] envelope), and
optionally a default QGIS style in the `layer_styles` table (QGIS applies it on load).
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable

from pyproj import CRS

_SQL_TYPE = {int: "INTEGER", float: "DOUBLE", str: "TEXT", bool: "BOOLEAN"}


def _gpb_polygon(rings: list[list[tuple[float, float]]], srs_id: int) -> bytes:
    """GeoPackage binary: 'GP', version 0, flags (little endian, envelope type 1), srs_id, envelope, WKB polygon."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    head = b"GP" + bytes([0, 0b00000011]) + struct.pack("<i", srs_id) + struct.pack("<4d", min(xs), max(xs), min(ys), max(ys))
    wkb = [struct.pack("<BII", 1, 3, len(rings))]
    for r in rings:
        pts = list(r)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        wkb.append(struct.pack("<I", len(pts)))
        wkb.append(b"".join(struct.pack("<2d", x, y) for x, y in pts))
    return head + b"".join(wkb)


def _srs_row(crs: CRS) -> tuple[str, int, str, int, str, str]:
    auth = crs.to_authority() or ("NONE", 0)
    code = int(auth[1]) if str(auth[1]).isdigit() else 99999
    try:
        wkt = crs.to_wkt("WKT1_GDAL")
    except Exception:  # noqa: BLE001 - WKT1 cannot express every CRS
        wkt = crs.to_wkt()
    return (crs.name, code, auth[0], code, wkt, "")


def _sql_type(values: Iterable[Any]) -> str:
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            return "BOOLEAN"
        return _SQL_TYPE.get(type(v), "TEXT")
    return "TEXT"


def write_polygon_layer(path: str | Path, table: str, crs: str, features: list[dict[str, Any]], *, description: str = "", qml_style: str | None = None) -> Path:
    """features: [{"rings": [[(x, y), ...], ...], "properties": {...}}]. Lists/dicts in properties are stored as JSON text."""
    path = Path(path)
    if path.exists():
        path.unlink()
    c = CRS.from_user_input(crs)
    srs = _srs_row(c)
    srs_id = srs[1]
    cols: list[str] = []
    for f in features:
        for k in f["properties"]:
            if k not in cols:
                cols.append(k)

    def norm(v: Any) -> Any:
        return json.dumps(v) if isinstance(v, (list, dict)) else v

    types = {k: _sql_type(norm(f["properties"].get(k)) for f in features) for k in cols}
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA application_id = 1196444487")  # 'GPKG'
        cur.execute("PRAGMA user_version = 10300")
        cur.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY, organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT)")
        rows = [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
            _srs_row(CRS.from_epsg(4326)),
        ]
        if srs_id not in (-1, 0, 4326):
            rows.append(srs)
        cur.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", rows)
        cur.execute("CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER, CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))")
        cur.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL, CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name), CONSTRAINT uk_gc_table_name UNIQUE (table_name), CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name), CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id))")
        col_sql = "".join(f', "{k}" {types[k]}' for k in cols)
        cur.execute(f'CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, geom POLYGON{col_sql})')
        allx = [p[0] for f in features for r in f["rings"] for p in r]
        ally = [p[1] for f in features for r in f["rings"] for p in r]
        bbox = (min(allx), min(ally), max(allx), max(ally)) if allx else (None, None, None, None)
        cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id) VALUES (?,?,?,?,?,?,?,?,?)", (table, "features", table, description, *bbox, srs_id))
        cur.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)", (table, "geom", "POLYGON", srs_id, 0, 0))
        ph = ",".join("?" * (len(cols) + 1))
        names = ", ".join(["geom"] + [f'"{k}"' for k in cols])
        cur.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({ph})', [(_gpb_polygon(f["rings"], srs_id), *[norm(f["properties"].get(k)) for k in cols]) for f in features])
        # no R-tree index: the spec requires ST_* maintenance triggers for it, and a few thousand footprints need none
        if qml_style:
            cur.execute("CREATE TABLE layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT(256), f_table_schema TEXT(256), f_table_name TEXT(256), f_geometry_column TEXT(256), styleName TEXT(30), styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT(30), ui TEXT(30), update_time DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")
            cur.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, styleName, styleQML, styleSLD, useAsDefault, description, owner) VALUES ('', '', ?, 'geom', ?, ?, '', 1, 'DepthWizard default style', '')", (table, f"{table}_style", qml_style))
            cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description) VALUES ('layer_styles', 'attributes', 'layer_styles', 'QGIS layer styles')")
        con.commit()
    finally:
        con.close()
    return path


def read_polygons(path: str | Path, table: str) -> list[dict[str, Any]]:
    """Read back a layer written by write_polygon_layer (tests / round-trip checks)."""
    con = sqlite3.connect(Path(path))
    try:
        cur = con.execute(f'SELECT * FROM "{table}"')
        names = [d[0] for d in cur.description]
        out = []
        for row in cur.fetchall():
            rec = dict(zip(names, row))
            g = rec.pop("geom")
            assert g[:2] == b"GP"
            off = 8 + 32
            _bo, _typ, n_rings = struct.unpack_from("<BII", g, off)
            off += 9
            rings = []
            for _ in range(n_rings):
                (n,) = struct.unpack_from("<I", g, off)
                off += 4
                rings.append([struct.unpack_from("<2d", g, off + 16 * i) for i in range(n)])
                off += 16 * n
            rec["rings"] = rings
            out.append(rec)
        return out
    finally:
        con.close()
