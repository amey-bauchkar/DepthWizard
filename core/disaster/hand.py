"""HAND: Height Above Nearest Drainage (Nobre et al. 2011, J. Hydrology 404) for river-flood screening in valleys.

A single flat water level ("bathtub") is meaningless in hilly terrain: the Teesta valley drops tens of metres across one
scene. HAND measures every cell's height above the channel cell its water drains to, so a river STAGE h (metres above
the channel) floods {HAND <= h}: the water surface follows the valley gradient. This is the standard first-order method
behind HAND-based flood-inundation mapping (e.g. NOAA National Water Model HAND-FIM). It is still a screening: no
discharge, no hydraulics, no timing.

Steps on the terrain layer (DEM-derived, so a few metres of grid are enough; default HAND_CELL_M):
  1. priority-flood depression filling with an epsilon gradient (Barnes, Lehman & Mulla 2014): every cell drains
     to the raster edge, and flat or filled areas keep a defined flow direction;
  2. D8 flow directions: steepest descent on the conditioned surface;
  3. flow accumulation -> drainage network: contributing area >= drainage_area_m2;
  4. HAND = z(cell) - z(first drainage cell downstream), with z the ORIGINAL terrain.
Cells whose water leaves the scene before meeting a drainage cell have no defined HAND (NaN): the channel they drain
to is outside the scene.

Stream burning: where rivers are mapped (OpenStreetMap waterways bundled in assets/waterways, or supplied), the flow
routing surface gets a smooth trench toward them (AGREE, Hellweger 1997): BURN_DEPTH_M at the mapped line, fading to 0
at BURN_BUFFER_M. Mapped cells are drainage cells by definition. Flow within the buffer therefore joins the mapped
river instead of a parallel thalweg of the 30 m DEM. The trench is used for routing only; HAND uses the true terrain.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
from scipy import ndimage

HAND_CELL_M = 3.0
"""ALGORITHMIC. Working grid for the hydrology. The terrain layer carries DEM-scale relief (30 m posting), so a 3 m
grid resolves every drainage line the DEM contains; HAND is interpolated back to the job grid."""

DRAINAGE_AREA_M2 = 20_000.0
"""POLICY. Contributing area that starts a channel (2 ha). Smaller values add gullies (HAND then measures height above
small streams); larger values keep only the main valleys. Configurable per request; returned with every result."""

BURN_DEPTH_M = 20.0
"""ALGORITHMIC. Trench depth at the mapped river line for flow routing (not for HAND)."""

BURN_BUFFER_M = 90.0
"""ALGORITHMIC. Trench half-width: three 30 m DEM postings, the scale at which the DEM's valley line can sit beside the
real river. Flow generated within this distance of a mapped river is routed into it."""

_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _resample_mean(z: np.ndarray, f: int) -> np.ndarray:
    h, w = z.shape
    hh, ww = -(-h // f), -(-w // f)
    pad = np.full((hh * f, ww * f), np.nan)
    pad[:h, :w] = z
    with np.errstate(invalid="ignore"):
        return np.nanmean(pad.reshape(hh, f, ww, f), axis=(1, 3))


def priority_flood(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Depression filling with epsilon: returns a surface where every valid cell has a strictly descending path to the
    edge (or to a NoData cell). NaN cells stay NaN and act as outlets."""
    h, w = z.shape
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nan).astype(np.float64)
    done = ~valid
    heap: list[tuple[float, int, int]] = []
    edge = np.zeros_like(valid)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= ndimage.binary_dilation(~valid, structure=np.ones((3, 3), bool))
    for r, c in zip(*np.nonzero(edge & valid)):
        heap.append((float(filled[r, c]), int(r), int(c)))
        done[r, c] = True
    heapq.heapify(heap)
    while heap:
        zc, r, c = heapq.heappop(heap)
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not done[rr, cc]:
                done[rr, cc] = True
                nz = filled[rr, cc]
                if nz <= zc:
                    nz = zc + eps
                    filled[rr, cc] = nz
                heapq.heappush(heap, (nz, rr, cc))
    return filled


def d8(filled: np.ndarray, cell: float) -> np.ndarray:
    """Flat index of the steepest-descent neighbour of every cell (-1: no lower neighbour = outlet)."""
    h, w = filled.shape
    best = np.zeros((h, w))
    down = np.full((h, w), -1, np.int64)
    idx = np.arange(h * w).reshape(h, w)
    zp = np.pad(filled, 1, constant_values=np.inf)
    ip = np.pad(idx, 1, constant_values=-1)
    for dr, dc in _D8:
        nb = zp[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (filled - nb) / (cell * math.hypot(dr, dc))
        better = np.isfinite(slope) & (slope > best)
        best = np.where(better, slope, best)
        down = np.where(better, ip[1 + dr:1 + dr + h, 1 + dc:1 + dc + w], down)
    down[~np.isfinite(filled)] = -1
    return down.ravel()


def accumulation(filled: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Number of cells draining through every cell (itself included), by processing cells from high to low."""
    z = np.where(np.isfinite(filled.ravel()), filled.ravel(), -np.inf)
    order = np.argsort(-z, kind="stable")
    acc = np.ones(z.size)
    dl, al = down.tolist(), acc  # list indexing is faster in the loop
    for i in order.tolist():
        j = dl[i]
        if j >= 0:
            al[j] += al[i]
    return acc


def hand(terrain: np.ndarray, cell_m: float, drainage_area_m2: float = DRAINAGE_AREA_M2, burn: np.ndarray | None = None) -> dict[str, Any]:
    """HAND on a grid of `cell_m` metres. burn: mapped channel cells (bool). Returns {"hand", "drainage", "acc_m2", "stats"}."""
    burn = burn & np.isfinite(terrain) if burn is not None else None
    routed = terrain
    if burn is not None and burn.any():
        dist = ndimage.distance_transform_edt(~burn) * cell_m
        routed = terrain - BURN_DEPTH_M * np.clip(1.0 - dist / BURN_BUFFER_M, 0.0, 1.0)
    filled = priority_flood(routed)
    down = d8(filled, cell_m)
    acc_m2 = accumulation(filled, down) * cell_m * cell_m
    drain = (acc_m2 >= drainage_area_m2) & np.isfinite(terrain.ravel())
    if burn is not None:
        drain |= burn.ravel()
    n = terrain.size
    # pointer jumping: every cell -> the first drainage cell (or outlet) downstream
    nxt = np.where(drain | (down < 0), np.arange(n), down)
    for _ in range(int(math.ceil(math.log2(max(2, n)))) + 1):
        nn = nxt[nxt]
        if np.array_equal(nn, nxt):
            break
        nxt = nn
    zt = terrain.ravel()
    reach = drain[nxt]
    hd = np.where(reach, zt - zt[nxt], np.nan).reshape(terrain.shape)
    hd = np.where(np.isfinite(hd), np.maximum(hd, 0.0), np.nan)
    return {"hand": hd, "drainage": drain.reshape(terrain.shape), "acc_m2": acc_m2.reshape(terrain.shape),
            "stats": {"cell_m": cell_m, "drainage_area_m2": drainage_area_m2, "drainage_fraction": round(float(drain.mean()), 4), "burned_cells": int(burn.sum()) if burn is not None else 0,
                      "undefined_fraction": round(float(np.isnan(hd[np.isfinite(terrain)]).mean()) if np.isfinite(terrain).any() else 1.0, 4)}}


def hand_on_grid(terrain: np.ndarray, gsd_m: float, *, cell_m: float = HAND_CELL_M, drainage_area_m2: float = DRAINAGE_AREA_M2, burn: np.ndarray | None = None) -> dict[str, Any]:
    """HAND computed on a ~cell_m working grid and brought back to the terrain grid (nearest block). burn: mapped
    channel mask on the terrain grid."""
    f = max(1, int(round(cell_m / gsd_m)))
    zc = _resample_mean(terrain, f) if f > 1 else terrain.astype(np.float64)
    bc = None
    if burn is not None:
        H0, W0 = burn.shape
        hh, ww = -(-H0 // f), -(-W0 // f)
        pad = np.zeros((hh * f, ww * f), bool)
        pad[:H0, :W0] = burn
        bc = pad.reshape(hh, f, ww, f).any(axis=(1, 3))
    out = hand(zc, gsd_m * f, drainage_area_m2, bc)
    H, W = terrain.shape
    up = lambda a: np.repeat(np.repeat(a, f, axis=0), f, axis=1)[:H, :W]  # noqa: E731
    return {"hand": np.where(np.isfinite(terrain), up(out["hand"]), np.nan), "drainage": up(out["drainage"]), "stats": {**out["stats"], "factor": f}}
