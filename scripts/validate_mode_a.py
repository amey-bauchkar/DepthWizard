"""Mode A (PNG/JPG, relative DSM) accuracy against airborne LiDAR.

Mode A outputs a unitless relative surface (rdsm.tif, 0..1), so it can only be scored in ways that ignore unknown
scale and offset (the problem statement asks for structure, not metres, in this mode):
  * Pearson r and Spearman rho between rdsm and the LiDAR surface;
  * RMSE / MAE after the best affine fit  z_ref ~ a * rdsm + b  (least squares on all valid pixels) - the "oracle
    scaled" error, i.e. the error that would remain with a perfect scale calibration. Reported next to the
    no-skill baseline: a flat surface at the mean (RMSE = standard deviation of the reference);
  * the same against the LiDAR height above ground (nDSM = DSM - DTM), which removes the regional terrain slope and
    tests whether buildings and trees are seen.

Inputs: the bundled Mode A JPEGs are the Swiss GeoTIFF tiles with georeferencing stripped (same pixels, checked:
mean absolute RGB difference ~2 from JPEG compression), so the tile's grid is used to align the LiDAR.
    python scripts/validate_mode_a.py   -> docs/validation_mode_a.{md,json}
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "mode_a_study"
os.environ["DW_DATA_DIR"] = str(OUT / "_data")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

DEMO, REF = ROOT / "assets" / "demo", ROOT / "assets" / "reference"
TILES = [  # Mode A input, georeferenced twin (grid), LiDAR DSM, LiDAR DTM, label
    ("sample_urban_hd.jpg", "swissimage_2019_2682-1247_0.5m.tif", "swisssurface3d_urban_2682-1247_dsm_0.5m.tif", "swissalti3d_urban_2682-1247_dtm_0.5m.tif", "Zürich urban, 2000 px (0.5 m)"),
    ("sample_urban.jpg", "swissimage_2019_2682-1247_2m.tif", "swisssurface3d_urban_2682-1247_dsm_0.5m.tif", "swissalti3d_urban_2682-1247_dtm_0.5m.tif", "Zürich urban, 500 px (2 m)"),
    ("sample_rural.jpg", "swissimage_2021_2621-1202_2m.tif", "swisssurface3d_rural_2621-1202_dsm_0.5m.tif", "swissalti3d_rural_2621-1202_dtm_0.5m.tif", "Emmental rural / forest / hilly, 500 px (2 m)"),
]


def on_grid(ref: Path, twin: Path) -> np.ndarray:
    with rasterio.open(twin) as t, rasterio.open(ref) as r:
        out = np.full((t.height, t.width), np.nan, np.float64)
        reproject(rasterio.band(r, 1), out, src_transform=r.transform, src_crs=r.crs, dst_transform=t.transform, dst_crs=t.crs,
                  resampling=Resampling.average, src_nodata=r.nodata, dst_nodata=np.nan)
    return out


def score(pred: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, z = pred[ok], ref[ok]
    A = np.c_[p, np.ones_like(p)]
    (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - (a * p + b)
    rho = spearmanr(p[:: max(1, p.size // 200000)], z[:: max(1, p.size // 200000)]).statistic
    return {"n": int(ok.sum()), "pearson_r": round(float(np.corrcoef(p, z)[0, 1]), 3), "spearman_rho": round(float(rho), 3),
            "rmse_after_affine_m": round(float(np.sqrt(np.mean(res ** 2))), 2), "mae_after_affine_m": round(float(np.mean(np.abs(res))), 2),
            "baseline_flat_rmse_m": round(float(np.std(z)), 2), "explained_variance": round(float(1 - np.var(res) / np.var(z)), 3), "affine_a_m_per_unit": round(float(a), 2)}


def main() -> int:
    from fastapi.testclient import TestClient

    from backend.config.settings import load_settings
    from backend.main import create_app

    rows = []
    runs = [(engine, flag, tile) for engine, flag in (("zero-shot (old Mode A)", "false"), ("fine-tuned nDSM model (current Mode A)", "true")) for tile in TILES]
    clients: dict[str, TestClient] = {}
    for engine, flag, (jpg, twin, dsm, dtm, label) in runs:
        if flag not in clients:
            os.environ["DW_FUSION_MODE_A_METRIC_MODEL"] = flag
            clients[flag] = TestClient(create_app(load_settings()))
        c = clients[flag]
        f = DEMO / jpg
        t0 = time.perf_counter()
        jid = c.post("/api/jobs", files={"file": (f.name, f.read_bytes(), "image/jpeg")}).json()["job_id"]
        c.post(f"/api/jobs/{jid}/run")
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(0.5)
        if j["status"] != "READY":
            print(jpg, "FAILED")
            continue
        jd = OUT / "_data" / "jobs" / jid
        with rasterio.open(jd / "rdsm.tif") as ds:
            pred = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        zs, zt = on_grid(REF / dsm, DEMO / twin), on_grid(REF / dtm, DEMO / twin)
        if pred.shape != zs.shape:
            print(jpg, "grid mismatch", pred.shape, zs.shape)
            continue
        row = {"input": jpg, "engine": engine, "label": label, "seconds": round(time.perf_counter() - t0, 1), "vs_dsm": score(pred, zs), "vs_ndsm": score(pred, zs - zt)}
        rows.append(row)
        dst = OUT / f"{Path(jpg).stem}_{flag}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(jd, dst)
        print(engine, jpg, row["vs_dsm"]["pearson_r"], row["vs_ndsm"]["pearson_r"], flush=True)
    (ROOT / "docs" / "validation_mode_a.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    L = ["# Mode A (PNG / JPG, relative DSM) accuracy vs airborne LiDAR", "",
         "Generated by `python scripts/validate_mode_a.py`. Mode A output is unitless, so it is scored scale-free: correlation, and the error left after the best possible scale + offset (an *oracle* calibration, so an upper bound on what any relative map could give). "
         "Reference: swissSURFACE3D (DSM) and swissSURFACE3D − swissALTI3D (height above ground, nDSM), 0.5 m LiDAR averaged to the image grid. **No-skill baseline** = a flat surface (RMSE = the reference's standard deviation).", "",
         "| Tile | Engine | Reference | Pearson r | Spearman ρ | RMSE after best affine (m) | No-skill RMSE (m) | Variance explained |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for k, name in (("vs_dsm", "DSM"), ("vs_ndsm", "height above ground")):
            m = r[k]
            L.append(f"| {r['label']} | {r['engine']} | {name} | {m['pearson_r']} | {m['spearman_rho']} | {m['rmse_after_affine_m']} | {m['baseline_flat_rmse_m']} | {100 * m['explained_variance']:.0f} % |")
    L += ["", "Reading: correlation shows how much of the real surface structure the relative map captures; the affine RMSE vs the no-skill RMSE shows how much error even a perfect calibration would leave. "
          "The tiles are the held-out Swiss test tiles (never used for training or tuning). The switch to the fine-tuned model was decided from its held-out-region results; no setting was tuned on these tiles. "
          "The fine-tuned engine reads heights above ground, so on a hillside it cannot rank the terrain itself (Emmental DSM): without coordinates and a DEM, terrain relief is not recoverable from the image.", ""]
    (ROOT / "docs" / "validation_mode_a.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(L) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
