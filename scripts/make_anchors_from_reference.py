"""Create a demo anchors CSV by sampling a reference DSM/DTM pair (simulated surveyed control points).

The points are labelled as SIMULATED in the CSV header: they stand in for surveyed GCPs / spot heights so that
tier A (anchor-refined) can be exercised on the bundled demo tiles. Validation excludes pixels within
`validation.anchor_exclusion_radius_m` of every anchor, so anchors never leak into the metrics.

Usage:
  python scripts/make_anchors_from_reference.py --dsm assets/reference/swisssurface3d_urban_2682-1247_dsm_0.5m.tif \
      --dtm assets/reference/swissalti3d_urban_2682-1247_dtm_0.5m.tif --n-ground 12 --n-object 12 \
      --out assets/demo/anchors_urban_simulated.csv --seed 0
Heights are written in the reference's own vertical CRS (LN02 = EPSG:5728 for swisstopo); the CSV carries `# vcrs=`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsm", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--n-ground", type=int, default=12)
    ap.add_argument("--n-object", type=int, default=12)
    ap.add_argument("--vcrs", default="EPSG:5728")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin-px", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    with rasterio.open(a.dsm) as d, rasterio.open(a.dtm) as t:
        dsm = d.read(1).astype(float)
        dtm = t.read(1).astype(float)
        tr, crs = d.transform, d.crs.to_string()
        dsm[dsm == d.nodata] = np.nan
        dtm[dtm == t.nodata] = np.nan
    ndsm = dsm - dtm
    h, w = dsm.shape
    m = a.margin_px
    rows = []
    # ground: flat-ish ground pixels (nDSM < 0.3 m) with low local relief
    gmask = np.zeros_like(dsm, bool)
    gmask[m:-m, m:-m] = np.isfinite(ndsm[m:-m, m:-m]) & (ndsm[m:-m, m:-m] < 0.3)
    gidx = np.flatnonzero(gmask)
    for i, k in enumerate(rng.choice(gidx, a.n_ground, replace=False)):
        r, c = divmod(int(k), w)
        x, y = tr * (c + 0.5, r + 0.5)
        rows.append((f"G{i+1:02d}", x, y, dtm[r, c], "ground", a.sigma))
    omask = np.zeros_like(dsm, bool)
    omask[m:-m, m:-m] = np.isfinite(ndsm[m:-m, m:-m]) & (ndsm[m:-m, m:-m] > 4.0)
    oidx = np.flatnonzero(omask)
    for i, k in enumerate(rng.choice(oidx, a.n_object, replace=False)):
        r, c = divmod(int(k), w)
        x, y = tr * (c + 0.5, r + 0.5)
        rows.append((f"O{i+1:02d}", x, y, dsm[r, c], "object", a.sigma))
    out = Path(a.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# SIMULATED control points sampled from {Path(a.dsm).name} / {Path(a.dtm).name} (seed={a.seed}); not surveyed.\n")
        f.write(f"# crs={crs}\n# vcrs={a.vcrs}\n")
        f.write("id,x,y,z,type,sigma\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.3f},{r[4]},{r[5]}\n")
    print(f"wrote {len(rows)} anchors -> {out}")


if __name__ == "__main__":
    main()
