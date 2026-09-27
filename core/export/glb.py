"""Binary glTF 2.0 (.glb) writer: textured terrain mesh + LoD-1 building blocks, no dependencies beyond numpy.

Axes follow glTF (right-handed, Y up): X = east, Y = up (metres above `origin_elevation_m`), Z = south (-north).
The scene origin (grid centre in the job CRS and the reference elevation) is written to `asset.extras` and to the root
node's extras, so the model can be placed back on the map (e.g. in Blender with BlenderGIS, or Cesium).
Opens in Blender, Windows 3D Viewer, three.js / Babylon viewers and most 3D-tiles converters.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_FLOAT, _UINT32 = 5126, 5125


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_tri(p, a, b, c) -> bool:  # noqa: ANN001 - closed triangle test (CCW a, b, c)
    return _cross(a, b, p) >= -1e-12 and _cross(b, c, p) >= -1e-12 and _cross(c, a, p) >= -1e-12


def triangulate(poly: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple polygon given counter-clockwise without the closing vertex. Collinear vertices are
    dropped; if no ear is found (self-intersecting input) the rest is fan-triangulated so a roof is always produced."""
    idx = list(range(len(poly)))
    tris: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        n = len(idx)
        for i in range(n):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % n]
            cr = _cross(poly[a], poly[b], poly[c])
            if abs(cr) < 1e-12:  # collinear / duplicate: remove without a triangle
                idx.pop(i)
                break
            if cr < 0:
                continue
            if any(_in_tri(poly[p], poly[a], poly[b], poly[c]) for p in idx if p not in (a, b, c)):
                continue
            tris.append((a, b, c))
            idx.pop(i)
            break
        else:
            tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
            return tris
    if len(idx) == 3 and abs(_cross(poly[idx[0]], poly[idx[1]], poly[idx[2]])) >= 1e-12:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


class _Bin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def view(self, raw: bytes, target: int | None = None) -> int:
        while len(self.data) % 4:
            self.data += b"\0"
        v: dict[str, Any] = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.data += raw
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, kind: str, target: int, *, minmax: bool = False) -> int:
        comp = _UINT32 if arr.dtype == np.uint32 else _FLOAT
        a: dict[str, Any] = {"bufferView": self.view(np.ascontiguousarray(arr).tobytes(), target), "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            a["min"] = [float(v) for v in arr.min(axis=0)]
            a["max"] = [float(v) for v in arr.max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1


def _primitive(b: _Bin, pos: np.ndarray, nrm: np.ndarray, idx: np.ndarray, material: int, uv: np.ndarray | None = None) -> dict[str, Any]:
    attrs = {"POSITION": b.accessor(pos.astype(np.float32), "VEC3", _ARRAY_BUFFER, minmax=True), "NORMAL": b.accessor(nrm.astype(np.float32), "VEC3", _ARRAY_BUFFER)}
    if uv is not None:
        attrs["TEXCOORD_0"] = b.accessor(uv.astype(np.float32), "VEC2", _ARRAY_BUFFER)
    return {"attributes": attrs, "indices": b.accessor(idx.astype(np.uint32).ravel(), "SCALAR", _ELEMENT_ARRAY_BUFFER), "material": material, "mode": 4}


def terrain_arrays(z: np.ndarray, xs: np.ndarray, ns: np.ndarray, us: np.ndarray, vs: np.ndarray, zref: float, max_dim: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid mesh from a heightfield z[row, col] (row 0 = north). xs[col] / ns[row]: east / north offsets (m) of the
    samples from the scene origin; us[col] / vs[row]: texture coordinates. Returns positions, normals, uvs, indices."""
    step = max(1, int(np.ceil(max(z.shape) / max_dim)))
    zs = z[::step, ::step].astype(np.float64)
    x, n, u, v = xs[::step], ns[::step], us[::step], vs[::step]
    H, W = zs.shape
    valid = np.isfinite(zs)
    fill = np.where(valid, zs, np.nanmedian(zs) if valid.any() else zref)
    X, N = np.meshgrid(x, n)
    pos = np.stack([X, fill - zref, -N], axis=-1).reshape(-1, 3)
    dzdn, dzdx = np.gradient(fill, n, x) if H > 1 and W > 1 else (np.zeros_like(fill), np.zeros_like(fill))
    nrm = np.stack([-dzdx, np.ones_like(fill), dzdn], axis=-1)  # up-normal of y = h(x, -n) in glTF axes
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    U, V = np.meshgrid(u, v)
    uv = np.stack([U, V], axis=-1).reshape(-1, 2)
    rr, cc = np.mgrid[0:H - 1, 0:W - 1]
    a = (rr * W + cc).astype(np.int64).ravel()
    b, c_, d = a + 1, a + W, a + W + 1
    ok = (valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]).ravel()
    tris = np.concatenate([np.stack([a, c_, b], 1)[ok], np.stack([b, c_, d], 1)[ok]])
    return pos, nrm.reshape(-1, 3), np.clip(uv, 0.0, 1.0), tris


def building_arrays(buildings: list[dict[str, Any]], zref: float, uv_of: Any = None) -> dict[str, np.ndarray]:
    """LoD-1 prisms from footprints given as (east, north) offsets from the scene origin in `coords`: walls (flat
    normals) and roofs (ear-clipped, optional UVs from uv_of(east, north))."""
    wp, wn, wi, rp, rn, ruv, ri = [], [], [], [], [], [], []
    for bld in buildings:
        ring = [tuple(map(float, p)) for p in bld["coords"]]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        area2 = sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))
        if abs(area2) < 1e-9:
            continue
        if area2 < 0:
            ring = ring[::-1]
        base = float(bld["base_elev_m"])
        bottom = min(base, float(bld.get("ground_min_m", base))) - zref
        top = base + max(0.5, float(bld["height_m"])) - zref
        for i in range(len(ring)):
            (x0, n0), (x1, n1) = ring[i], ring[(i + 1) % len(ring)]
            dx, dn = x1 - x0, n1 - n0
            ln = (dx * dx + dn * dn) ** 0.5
            if ln < 1e-9:
                continue
            k = len(wp)
            wp += [(x0, bottom, -n0), (x1, bottom, -n1), (x1, top, -n1), (x0, top, -n0)]
            wn += [(dn / ln, 0.0, dx / ln)] * 4
            wi += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
        k = len(rp)
        for x, n in ring:
            rp.append((x, top, -n))
            rn.append((0.0, 1.0, 0.0))
            ruv.append(uv_of(x, n) if uv_of else (0.0, 0.0))
        ri += [(k + a, k + b, k + c) for a, b, c in triangulate(ring)]
    f = lambda v, w: np.array(v, np.float64).reshape(-1, w)  # noqa: E731
    return {"wall_pos": f(wp, 3), "wall_nrm": f(wn, 3), "wall_idx": np.array(wi, np.int64).reshape(-1, 3), "roof_pos": f(rp, 3), "roof_nrm": f(rn, 3), "roof_uv": f(ruv, 2), "roof_idx": np.array(ri, np.int64).reshape(-1, 3)}


def write_glb(path: str | Path, *, heights: np.ndarray | None = None, axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None, buildings: list[dict[str, Any]] | None = None, uv_of: Any = None, texture_jpeg: bytes | None = None, extras: dict[str, Any] | None = None, zref: float | None = None, generator: str = "DepthWizard") -> dict[str, Any]:
    """Write the GLB. heights: terrain (or DSM) heightfield in metres, row 0 = north, with axes = (xs, ns, us, vs) as in
    terrain_arrays. buildings: footprints in (east, north) offsets from the same origin; uv_of(east, north) -> (u, v)
    textures the roofs with the same image as the terrain."""
    b = _Bin()
    materials: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    tex_index = None
    if texture_jpeg:
        images.append({"bufferView": b.view(texture_jpeg), "mimeType": "image/jpeg"})
        textures.append({"source": 0, "sampler": 0})
        tex_index = 0
    if zref is None:
        zref = float(np.nanmin(heights)) if heights is not None and np.isfinite(heights).any() else 0.0
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = [{"name": "DepthWizard scene", "children": [], "extras": extras or {}}]
    stats: dict[str, Any] = {"zref": zref}
    if heights is not None and axes is not None:
        pos, nrm, uv, tris = terrain_arrays(heights, *axes, zref)
        mat = {"name": "terrain", "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}, "doubleSided": True}
        if tex_index is not None:
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_index}
        materials.append(mat)
        meshes.append({"name": "terrain", "primitives": [_primitive(b, pos, nrm, tris, len(materials) - 1, uv if tex_index is not None else None)]})
        nodes.append({"name": "terrain", "mesh": len(meshes) - 1})
        stats.update(terrain_vertices=int(pos.shape[0]), terrain_triangles=int(tris.shape[0]))
    if buildings:
        uv_of = uv_of if tex_index is not None else None
        ba = building_arrays(buildings, zref, uv_of)
        prims = []
        if len(ba["wall_idx"]):
            materials.append({"name": "building walls", "pbrMetallicRoughness": {"baseColorFactor": [0.93, 0.92, 0.89, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}})
            prims.append(_primitive(b, ba["wall_pos"], ba["wall_nrm"], ba["wall_idx"], len(materials) - 1))
        if len(ba["roof_idx"]):
            roof = {"name": "building roofs", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0] if uv_of else [0.72, 0.74, 0.76, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.85}, "doubleSided": True}
            if uv_of:
                roof["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_index}
            materials.append(roof)
            prims.append(_primitive(b, ba["roof_pos"], ba["roof_nrm"], ba["roof_idx"], len(materials) - 1, ba["roof_uv"] if uv_of else None))
        if prims:
            meshes.append({"name": "LoD-1 buildings", "primitives": prims})
            nodes.append({"name": "LoD-1 buildings", "mesh": len(meshes) - 1})
        stats.update(buildings=len(buildings), wall_triangles=int(len(ba["wall_idx"])), roof_triangles=int(len(ba["roof_idx"])))
    nodes[0]["children"] = list(range(1, len(nodes)))
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator, "extras": extras or {}},
        "scene": 0, "scenes": [{"name": "DepthWizard", "nodes": [0]}], "nodes": nodes, "meshes": meshes,
        "materials": materials, "accessors": b.accessors, "bufferViews": b.views, "buffers": [{"byteLength": 0}],
    }
    if images:
        gltf.update(images=images, textures=textures, samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
    while len(b.data) % 4:
        b.data += b"\0"
    gltf["buffers"][0]["byteLength"] = len(b.data)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(b.data)
    out = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(b.data), 0x004E4942) + bytes(b.data)
    Path(path).write_bytes(out)
    stats["bytes"] = total
    return stats


def read_glb(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """Parse a GLB into (json, binary chunk) — used by tests and the export self-check."""
    raw = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(raw), "not a valid GLB 2.0 file"
    jl, jt = struct.unpack_from("<II", raw, 12)
    assert jt == 0x4E4F534A
    js = json.loads(raw[20:20 + jl])
    bl, bt = struct.unpack_from("<II", raw, 20 + jl)
    assert bt == 0x004E4942
    return js, raw[28 + jl:28 + jl + bl]
