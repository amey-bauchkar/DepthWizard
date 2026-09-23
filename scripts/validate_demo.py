"""Reproducible validation of the bundled demo tiles against swisstopo LiDAR references.

Runs every Mode B demo tile through the real pipeline (in-process, real model) twice — without anchors (tier T)
and with the simulated anchors (tier A) — validates DSM vs swissSURFACE3D and terrain vs swissALTI3D, and also
reports the raw Copernicus DEM against the same references as the no-model baseline.

Writes docs/validation_results.json and docs/validation_results.md with the ACTUAL numbers of this run.
Usage:  .venv/Scripts/python scripts/validate_demo.py   (needs assets fetched by scripts/fetch_assets.sh + model weights)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DW_DATA_DIR", str(Path(tempfile.mkdtemp(prefix="dw_validate_")) / "data"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from backend.main import create_app  # noqa: E402
from core.calib.dem import discover_dem, load_dem_on_grid  # noqa: E402
from core.geo.grid import Grid  # noqa: E402
from core.geo.raster_io import grid_bounds_wgs84, read_raster_on_grid  # noqa: E402
from core.geo.vertical import transform_heights_xy  # noqa: E402
from core.validate.metrics import metric_set  # noqa: E402

OUT_JSON = ROOT / "docs" / "validation_results.json"
OUT_MD = ROOT / "docs" / "validation_results.md"


def raw_dem_metrics(job_dir: Path, ref_path: Path, ref_vcrs: str, out_vcrs: str) -> dict:
    with rasterio.open(job_dir / "dsm.tif") as ds:
        grid = Grid(width=ds.width, height=ds.height, transform=ds.transform, crs=ds.crs.to_string(), dtype="float32", nodata=ds.nodata, units="m", metric=True, vertical_reference=out_vcrs, tier="T")
    src = discover_dem(grid_bounds_wgs84(grid), Path(os.environ.get("DW_DEM_DIR", ROOT / "assets" / "dem")))
    dem, dvalid, _ = load_dem_on_grid(src, grid, out_vcrs)
    ref, rvalid = read_raster_on_grid(str(ref_path), grid, resampling=Resampling.average)
    cc, rr = np.meshgrid(np.arange(0, grid.width, 32), np.arange(0, grid.height, 32))
    xs, ys = grid.transform * (cc + 0.5, rr + 0.5)  # type: ignore[operator]
    z, _ = transform_heights_xy(np.asarray(xs), np.asarray(ys), np.zeros(np.shape(xs)), grid.crs, ref_vcrs, out_vcrs)
    ref = ref + float(np.mean(z))
    m = dvalid & rvalid & np.isfinite(dem) & np.isfinite(ref)
    m[:4] = m[-4:] = False; m[:, :4] = m[:, -4:] = False
    return metric_set(dem[m], ref[m])


def main() -> None:
    c = TestClient(create_app())
    demo = c.get("/api/demo").json()
    results: dict = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "system": {k: c.get("/api/system").json()[k] for k in ("app", "runtime", "model", "geoid_grids", "dem_tiles")}, "tiles": {}}
    for item in [i for i in demo["items"] if i["mode"] == "B"]:
        t = "urban" if "urban" in item["id"] else "rural"
        tile: dict = {"item": item, "runs": {}}
        for variant in ("no_anchors", "simulated_anchors"):
            files = {"file": (item["file"], (ROOT / "assets" / "demo" / item["file"]).read_bytes(), "image/tiff")}
            if variant == "simulated_anchors":
                files["anchors"] = (f"anchors_{t}.csv", (ROOT / "assets" / "demo" / f"anchors_{t}_simulated.csv").read_bytes(), "text/csv")
            r = c.post("/api/jobs", files=files); r.raise_for_status()
            jid = r.json()["job_id"]
            c.post(f"/api/jobs/{jid}/run")
            while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
                time.sleep(0.3)
            if j["status"] != "READY":
                tile["runs"][variant] = {"status": "FAILED", "error": j.get("error")}
                continue
            res = c.get(f"/api/jobs/{jid}/result").json()
            md = c.get(f"/api/jobs/{jid}/metadata").json()
            run: dict = {"status": "READY", "stages_ms": j["stages_ms"], "tier": res["calibration_tier"], "quality": res["quality"], "quality_triggers": res["quality_triggers"], "flags": res["flags"], "object_scale_source": res["object_scale_source"], "object_scale_m_per_unit": res["object_scale_m_per_unit"], "vertical_reference": res["vertical_reference"], "calib_report": {k: md["calib_report"].get(k) for k in ("terrain", "scale_fit", "anchors", "consistency", "dem")}, "validation": {}}
            for rt, key in (("dsm", "reference_dsm"), ("dtm", "reference_dtm")):
                v = c.post(f"/api/jobs/{jid}/validate", data={"bundled": item[key], "ref_type": rt, "vertical_crs": item["reference_vertical_crs"]})
                v.raise_for_status()
                vj = v.json()
                run["validation"][rt] = {k: vj[k] for k in ("compared_layer", "metrics_overall", "metrics_by_slope", "metrics_by_object", "height_bins", "oracle_affine", "alignment", "mask", "reference", "verdict", "caveats")}
            job_dir = Path(os.environ["DW_DATA_DIR"]) / "jobs" / jid
            if variant == "no_anchors":
                tile["raw_copernicus_vs_reference"] = {rt: raw_dem_metrics(job_dir, ROOT / "assets" / "reference" / item[key], item["reference_vertical_crs"], res["vertical_reference"]) for rt, key in (("dsm", "reference_dsm"), ("dtm", "reference_dtm"))}
            tile["runs"][variant] = run
            print(f"{item['id']} [{variant}] tier {res['calibration_tier']} quality {res['quality']} | " + " | ".join(f"{rt}: RMSE {run['validation'][rt]['metrics_overall']['RMSE']:.2f} ME {run['validation'][rt]['metrics_overall']['ME']:+.2f} NMAD {run['validation'][rt]['metrics_overall']['NMAD']:.2f}" for rt in ("dsm", "dtm")))
        results["tiles"][item["id"]] = tile
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    write_markdown(results)
    print("wrote", OUT_JSON, OUT_MD)


def write_markdown(results: dict) -> None:
    L = [f"# Demo-tile validation results (measured)\n", f"Run at {results['ran_at']} · model {results['system']['model'].get('name')}@{results['system']['model'].get('version')} on {results['system']['model'].get('device')} · {results['system']['runtime']['python']} · torch {results['system']['runtime']['torch']}\n",
         "All numbers below are MEASURED by `scripts/validate_demo.py` on the bundled swisstopo tiles (1 km × 1 km, 2 m GSD) against swissSURFACE3D (DSM) and swissALTI3D (DTM) 0.5 m LiDAR references, area-averaged onto the 2 m job grid, LN02 → EGM2008 converted with the bundled grids. Border 4 px masked; anchor pixels excluded within 15 m. Units: metres.\n",
         "Bands are descriptive (A <2 m, B 2–5 m, C 5–10 m, D >10 m RMSE) — they are not a claim of meeting the problem statement.\n"]
    for tid, tile in results["tiles"].items():
        L.append(f"\n## {tile['item']['label']}\n\n_{tile['item']['source']}_\n")
        L.append("| variant | tier | quality | object scale | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        raw = tile.get("raw_copernicus_vs_reference", {})
        for rt, m in raw.items():
            L.append(f"| raw Copernicus GLO-30 (no model, baseline) | — | — | — | {rt.upper()} reference | {m['n']} | {m['ME']:+.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['NMAD']:.2f} | {m['LE90']:.2f} | {m['pearson_r']:.3f} |")
        for variant, run in tile["runs"].items():
            if run.get("status") != "READY":
                L.append(f"| {variant} | FAILED | | | | | | | | | | |"); continue
            for rt, v in run["validation"].items():
                m = v["metrics_overall"]
                L.append(f"| {variant} | {run['tier']} | {run['quality']} | {run['object_scale_source'] or 'none'} | {v['compared_layer']} vs {rt.upper()} | {m['n']} | {m['ME']:+.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['NMAD']:.2f} | {m['LE90']:.2f} | {m['pearson_r']:.3f} |")
        for variant, run in tile["runs"].items():
            if run.get("status") == "READY":
                L.append(f"\n**{variant}** — triggers: " + "; ".join(run["quality_triggers"]) + f". Timings: {run['stages_ms']}")
                obj = run["validation"]["dsm"].get("metrics_by_object", {})
                if obj:
                    L.append("\nDSM error by reference object class: " + "; ".join(f"{k}: ME {m['ME']:+.2f}, RMSE {m['RMSE']:.2f} (n={m['n']})" for k, m in obj.items()))
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
