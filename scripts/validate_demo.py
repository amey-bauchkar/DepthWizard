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
    xs, ys = grid.transform @ (cc + 0.5, rr + 0.5)  # type: ignore[operator]
    z, _ = transform_heights_xy(np.asarray(xs), np.asarray(ys), np.zeros(np.shape(xs)), grid.crs, ref_vcrs, out_vcrs)
    ref = ref + float(np.mean(z))
    m = dvalid & rvalid & np.isfinite(dem) & np.isfinite(ref)
    m[:4] = m[-4:] = False; m[:, :4] = m[:, -4:] = False
    return metric_set(dem[m], ref[m])


def main() -> None:
    from backend.config.settings import load_settings

    c = TestClient(create_app())
    has_metric = bool(c.get("/health").json()["model"].get("metric_model", {}).get("available"))
    s0 = load_settings()
    s0.model_metric.enabled = False
    cz = TestClient(create_app(s0)) if has_metric else None  # same pipeline, zero-shot model only (for comparison)
    demo = c.get("/api/demo").json()
    from backend.jobs.pipeline_b import METHOD_VERSION

    results: dict = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "method_version": METHOD_VERSION, "system": {k: c.get("/api/system").json()[k] for k in ("app", "runtime", "model", "geoid_grids", "dem_tiles")}, "tiles": {}}
    for item in [i for i in demo["items"] if i["mode"] == "B" and i.get("reference_dsm")]:
        t = "urban" if "urban" in item["id"] else "rural"
        tile: dict = {"item": item, "runs": {}}
        variants = ([("zeroshot_no_anchors", cz)] if cz else []) + [("no_anchors", c), ("simulated_anchors", c)]
        for variant, cl in variants:
            files = {"file": (item["file"], (ROOT / "assets" / "demo" / item["file"]).read_bytes(), "image/tiff")}
            if variant == "simulated_anchors":
                files["anchors"] = (f"anchors_{t}.csv", (ROOT / "assets" / "demo" / f"anchors_{t}_simulated.csv").read_bytes(), "text/csv")
            r = cl.post("/api/jobs", files=files); r.raise_for_status()
            jid = r.json()["job_id"]
            cl.post(f"/api/jobs/{jid}/run")
            while (j := cl.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
                time.sleep(0.3)
            if j["status"] != "READY":
                tile["runs"][variant] = {"status": "FAILED", "error": j.get("error")}
                continue
            res = cl.get(f"/api/jobs/{jid}/result").json()
            md = cl.get(f"/api/jobs/{jid}/metadata").json()
            run: dict = {"status": "READY", "stages_ms": j["stages_ms"], "tier": res["calibration_tier"], "quality": res["quality"], "quality_triggers": res["quality_triggers"], "flags": res["flags"], "object_scale_source": res["object_scale_source"], "object_scale_m_per_unit": res["object_scale_m_per_unit"], "vertical_reference": res["vertical_reference"], "calib_report": {k: md["calib_report"].get(k) for k in ("terrain", "fusion", "metric_model", "anchors", "consistency", "dem")}, "tile_model": (j.get("model") or {}).get("tiles", {}).get("name", (j.get("model") or {}).get("name")), "validation": {}}
            for rt, key in (("dsm", "reference_dsm"), ("dtm", "reference_dtm")):
                v = cl.post(f"/api/jobs/{jid}/validate", data={"bundled": item[key], "ref_type": rt, "vertical_crs": item["reference_vertical_crs"]})
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


def _pct(new: float, old: float) -> str:
    return f"{100 * (new - old) / old:+.1f} %"


def write_markdown(results: dict) -> None:
    model = results["system"]["model"]
    rt_ = results["system"]["runtime"]
    L = [
        "# Demo-tile validation results (measured)\n",
        f"Run at {results['ran_at']} · model {model.get('name')}@{model.get('version')} on {model.get('device')} · Python {rt_['python']} · torch {rt_['torch']} · method {results.get('method_version', '')}\n",
        "All numbers are MEASURED by `scripts/validate_demo.py` (reproducible, deterministic): every Mode B demo tile is run through the real pipeline and compared with swisstopo airborne-LiDAR rasters — swissSURFACE3D (DSM) and swissALTI3D (DTM), 0.5 m — area-averaged onto the job grid, LN02 → EGM2008 converted with the bundled geoid grids. Border 4 px (16 px at 0.5 m) masked. Units: metres. *Baseline* = the raw Copernicus GLO-30 DEM bilinearly resampled to the same grid (what you get without DepthWizard).\n",
        "## Summary — DSM and terrain RMSE against LiDAR (lower is better)\n",
    ]
    has_zs = any("zeroshot_no_anchors" in t["runs"] for t in results["tiles"].values())
    cols = (["zeroshot_no_anchors"] if has_zs else []) + ["no_anchors", "simulated_anchors"]
    names = {"zeroshot_no_anchors": "zero-shot model, tier T", "no_anchors": ("fine-tuned model, tier T (no anchors)" if has_zs else "tier T (no anchors)"), "simulated_anchors": ("fine-tuned model + simulated anchors (tier A where the anchor fit was accepted)" if has_zs else "tier A (simulated anchors)")}
    L += ["| tile | layer vs reference | Copernicus GLO-30 alone | " + " | ".join(f"DepthWizard {names[v]}" for v in cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for tile in results["tiles"].values():
        raw = tile.get("raw_copernicus_vs_reference", {})
        for rt, lname in (("dsm", "DSM vs LiDAR DSM"), ("dtm", "terrain vs LiDAR DTM")):
            if rt not in raw:
                continue
            base = raw[rt]["RMSE"]
            cells = []
            for variant in cols:
                run = tile["runs"].get(variant, {})
                if run.get("status") != "READY":
                    cells.append("failed")
                    continue
                v = run["validation"][rt]["metrics_overall"]["RMSE"]
                cells.append(f"**{v:.2f}** ({_pct(v, base)})")
            L.append(f"| {tile['item']['label']} | {lname} | {base:.2f} | " + " | ".join(cells) + " |")
    L += [
        "",
        "How to read this: the DEM already carries every structure larger than one posting (30 m); DepthWizard keeps it and adds the finer detail from Depth Anything V2 run on ~0.5 m tiles, so at 30 m its DSM is unbiased against the DEM (per 30 m cell it can still differ by a few metres, because the high-pass is Gaussian) and the changes come from sub-30 m structure. *Zero-shot* = the untrained model, whose per-tile scale is fitted against the DEM. *Fine-tuned* = the DepthWizard nDSM model (trained on swisstopo LiDAR from other regions; the tiles below and 5 km around them were excluded from training, validation and model selection), which predicts height above ground in metres directly; terrain = DSM − predicted heights. Tier A anchors fit a gain/offset on the DSM detail (accepted only if leave-one-out error improves).",
        "",
        "Caveats (stated, not hidden): only two 1 km² locations (Swiss urban, and rural/forested hilly); no Indian LiDAR reference was available; the tier-A anchors are *simulated* by sampling the same LiDAR tiles (pixels within 15 m of an anchor are excluded from every metric); the fine-tuned model was trained on Swiss data only, so these tiles measure generalisation to unseen Swiss places, not to Indian imagery. Bands (A < 2 m, B 2–5 m, C 5–10 m, D > 10 m RMSE) are descriptive only.",
        "",
    ]
    for tile in results["tiles"].values():
        L += ["", f"## {tile['item']['label']}", "", f"_{tile['item']['source']}_", ""]
        L += ["| variant | tier | quality | model detail | compared | n | ME | RMSE | MAE | NMAD | LE90 | r |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rt, m in tile.get("raw_copernicus_vs_reference", {}).items():
            L.append(f"| raw Copernicus GLO-30 (no model, baseline) | — | — | — | {rt.upper()} reference | {m['n']} | {m['ME']:+.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['NMAD']:.2f} | {m['LE90']:.2f} | {m['pearson_r']:.3f} |")
        for variant, run in tile["runs"].items():
            if run.get("status") != "READY":
                L.append(f"| {variant} | FAILED | | | | | | | | | | |")
                continue
            for rt, v in run["validation"].items():
                m = v["metrics_overall"]
                L.append(f"| {variant} | {run['tier']} | {run['quality']} | {run['object_scale_source'] or 'none'} | {v['compared_layer']} vs {rt.upper()} | {m['n']} | {m['ME']:+.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['NMAD']:.2f} | {m['LE90']:.2f} | {m['pearson_r']:.3f} |")
        for variant, run in tile["runs"].items():
            if run.get("status") != "READY":
                continue
            L += ["", f"**{variant}** — triggers: " + "; ".join(run["quality_triggers"]) + f". Timings (ms): {run['stages_ms']}"]
            obj = run["validation"]["dsm"].get("metrics_by_object", {})
            if obj:
                L += ["", "DSM error by object class (from the predicted nDSM): " + "; ".join(f"{k}: ME {m['ME']:+.2f}, RMSE {m['RMSE']:.2f} (n={m['n']})" for k, m in obj.items())]
            slope = run["validation"]["dsm"].get("metrics_by_slope", {})
            if slope:
                L += ["", "DSM error by reference slope class: " + "; ".join(f"{k}: RMSE {m['RMSE']:.2f} (n={m['n']})" for k, m in slope.items())]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
