"""Indian validation: Sikkim (Maxar WorldView 0.5 m) scenes vs NASA ICESat-2 checkpoints.

Runs every Indian Mode B demo item (assets/demo/manifest.json, country "IN", with reference_points) through the real
pipeline with the zero-shot model and, when installed, the fine-tuned model; validates terrain / nDSM / DSM against the
ICESat-2 ground and canopy heights; the input DEM (Copernicus GLO-30) on the same checkpoints is the baseline.
Writes docs/validation_india.json and docs/validation_india.md.

Prerequisite: python scripts/fetch_india_demo.py
Usage:        python scripts/validate_india.py
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
os.environ.setdefault("DW_DATA_DIR", str(Path(tempfile.mkdtemp(prefix="dw_india_")) / "data"))

from fastapi.testclient import TestClient  # noqa: E402

from backend.config.settings import load_settings  # noqa: E402
from backend.main import create_app  # noqa: E402

OUT_JSON = ROOT / "docs" / "validation_india.json"
OUT_MD = ROOT / "docs" / "validation_india.md"
KEYS = ("terrain_vs_ground", "input_dem_vs_ground", "ndsm_vs_canopy_height", "dsm_vs_top_of_surface", "input_dem_vs_top_of_surface")


def run_item(c: TestClient, item: dict) -> dict:
    f = ROOT / "assets" / "demo" / item["file"]
    jid = c.post("/api/jobs", files={"file": (f.name, f.read_bytes(), "image/tiff")}).json()["job_id"]
    c.post(f"/api/jobs/{jid}/run")
    while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
        time.sleep(0.5)
    if j["status"] != "READY":
        return {"status": "FAILED", "error": j.get("error")}
    res = c.get(f"/api/jobs/{jid}/result").json()
    v = c.post(f"/api/jobs/{jid}/validate_points", data={"bundled": item["reference_points"]})
    v.raise_for_status()
    return {"status": "READY", "tier": res["calibration_tier"], "quality": res["quality"], "flags": res["flags"], "object_scale_source": res["object_scale_source"], "stages_ms": j["stages_ms"], "tile_model": (j.get("model") or {}).get("tiles", {}).get("name", (j.get("model") or {}).get("name")), "points": v.json()}


def main() -> None:
    s_ft = load_settings()
    c_ft = TestClient(create_app(s_ft))
    has_metric = bool(c_ft.get("/health").json()["model"].get("metric_model", {}).get("available"))
    s_zs = load_settings()
    s_zs.model_metric.enabled = False
    c_zs = TestClient(create_app(s_zs))
    items = [i for i in c_ft.get("/api/demo").json()["items"] if i.get("country") == "IN" and i.get("reference_points")]
    if not items:
        raise SystemExit("no Indian demo items: run python scripts/fetch_india_demo.py first")
    out: dict = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "fine_tuned_model_installed": has_metric, "items": {}}
    for item in items:
        runs = {"zero_shot": run_item(c_zs, item)}
        if has_metric:
            runs["fine_tuned"] = run_item(c_ft, item)
        out["items"][item["id"]] = {"item": item, "runs": runs}
        for k, r in runs.items():
            m = r.get("points", {}).get("metrics", {})
            print(item["id"], k, r.get("tier"), {kk: round(m[kk]["RMSE"], 2) for kk in KEYS if kk in m and "RMSE" in m[kk]}, "n", r.get("points", {}).get("n_in_grid"))
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    write_markdown(out)
    print("wrote", OUT_JSON, OUT_MD)


def _row(name: str, m: dict | None) -> str:
    if not m or "RMSE" not in m:
        return f"| {name} | – | – | – | – | – |"
    return f"| {name} | {m['n']} | {m['ME']:+.2f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['NMAD']:.2f} |"


def write_markdown(out: dict) -> None:
    L = [
        "# Indian validation — Sikkim vs NASA ICESat-2 (measured)\n",
        f"Run at {out['ran_at']} · `python scripts/validate_india.py` · imagery: Maxar Open Data Program (WorldView, 0.5 m, CC BY-NC 4.0) · DEM: CartoDEM v3 R1 (ISRO/NRSC, ellipsoidal heights auto-detected and converted) where installed in assets/dem/cartodem, else Copernicus GLO-30 · checkpoints: NASA ICESat-2 ATL08-style 20 m segments (2018–2025) via SlideRule, ellipsoidal → EGM2008 with the bundled geoid grids.\n",
        "No airborne LiDAR is publicly available for these Indian sites, so accuracy is measured against **independent sparse laser checkpoints**: *ground* = ICESat-2 terrain height, *canopy* = ICESat-2 height of the top surface above ground (trees and buildings), *top of surface* = ground + canopy. Neither the model nor the calibration saw these points or any Indian data.\n",
    ]
    L += ["## Summary (RMSE in metres; lower is better)\n",
          "| site | checkpoints | DSM vs top of surface: input DEM → zero-shot → fine-tuned | height above ground (nDSM vs canopy/structure): zero-shot → fine-tuned | terrain vs ground: input DEM → fine-tuned |",
          "|---|---|---|---|---|"]
    for blk in out["items"].values():
        rz, rf = blk["runs"].get("zero_shot", {}), blk["runs"].get("fine_tuned", {})
        mz = rz.get("points", {}).get("metrics", {}) if rz.get("status") == "READY" else {}
        mf = rf.get("points", {}).get("metrics", {}) if rf.get("status") == "READY" else {}
        g = lambda m, k: f"{m[k]['RMSE']:.2f}" if k in m and "RMSE" in m[k] else "–"  # noqa: E731
        n = (rf or rz).get("points", {}).get("n_in_grid", "–")
        L.append(f"| {blk['item']['id'].replace('india_', '')} | {n} | {g(mf or mz, 'input_dem_vs_top_of_surface')} → {g(mz, 'dsm_vs_top_of_surface')} → **{g(mf, 'dsm_vs_top_of_surface')}** | {g(mz, 'ndsm_vs_canopy_height')} → **{g(mf, 'ndsm_vs_canopy_height')}** | {g(mf or mz, 'input_dem_vs_ground')} → {g(mf, 'terrain_vs_ground')} |")
    acc: dict = {}
    for blk in out["items"].values():
        for rk, r in blk["runs"].items():
            for k, m in r.get("points", {}).get("metrics", {}).items():
                if "RMSE" in m:
                    a = acc.setdefault((rk, k), [0, 0.0, 0.0])
                    a[0] += m["n"]; a[1] += m["n"] * m["RMSE"] ** 2; a[2] += m["n"] * m["ME"]
    pooled = lambda rk, k: f"{(acc[(rk, k)][1] / acc[(rk, k)][0]) ** 0.5:.2f} (ME {acc[(rk, k)][2] / acc[(rk, k)][0]:+.2f})" if (rk, k) in acc else "–"  # noqa: E731
    base = "fine_tuned" if any(k[0] == "fine_tuned" for k in acc) else "zero_shot"
    L += ["\n### Pooled over all sites (RMSE = sqrt of checkpoint-weighted mean squared error; metres)\n",
          "| comparison | input DEM alone | zero-shot model | fine-tuned model |", "|---|---|---|---|",
          f"| terrain vs ICESat-2 ground | {pooled(base, 'input_dem_vs_ground')} | {pooled('zero_shot', 'terrain_vs_ground')} | **{pooled('fine_tuned', 'terrain_vs_ground')}** |",
          f"| DSM vs ICESat-2 top of surface | {pooled(base, 'input_dem_vs_top_of_surface')} | {pooled('zero_shot', 'dsm_vs_top_of_surface')} | **{pooled('fine_tuned', 'dsm_vs_top_of_surface')}** |",
          f"| height above ground vs ICESat-2 canopy/structure | – | {pooled('zero_shot', 'ndsm_vs_canopy_height')} | **{pooled('fine_tuned', 'ndsm_vs_canopy_height')}** |",
          f"\nCheckpoints: {acc.get((base, 'terrain_vs_ground'), [0])[0]} ICESat-2 segments over {len(out['items'])} sites."]
    for iid, blk in out["items"].items():
        L += [f"\n## {blk['item']['label']}\n", f"_{blk['item']['source']}_\n"]
        for rk, r in blk["runs"].items():
            if r.get("status") != "READY":
                L.append(f"\n**{rk}**: FAILED {r.get('error')}\n")
                continue
            p = r["points"]
            m = p["metrics"]
            L += [f"\n**{rk.replace('_', ' ')}** (tile model `{r['tile_model']}`, tier {r['tier']}, quality {r['quality']}; {p['n_in_grid']} checkpoints in the scene)\n",
                  "| comparison (metres) | n | ME | RMSE | MAE | NMAD |", "|---|---|---|---|---|---|",
                  _row("input DEM vs ICESat-2 ground — baseline", m.get("input_dem_vs_ground")),
                  _row("DepthWizard terrain vs ICESat-2 ground", m.get("terrain_vs_ground")),
                  _row("input DEM vs ICESat-2 top of surface — baseline", m.get("input_dem_vs_top_of_surface")),
                  _row("DepthWizard DSM vs ICESat-2 top of surface", m.get("dsm_vs_top_of_surface")),
                  _row("DepthWizard nDSM vs ICESat-2 canopy/structure height", m.get("ndsm_vs_canopy_height"))]
            for sname, sm in p.get("strata", {}).items():
                L.append(f"\n_{sname}_: " + "; ".join(f"{k}: RMSE {v['RMSE']:.2f} (n={v['n']})" for k, v in sm.items() if "RMSE" in v))
    L += ["\n## Caveats\n", "* Sparse checkpoints (tens to a few hundred per scene) along a few ground tracks; metrics carry sampling uncertainty.",
          "* ICESat-2 dates (2018–2025) differ from the image dates (2022); new construction or clearing shows up as error.",
          "* Each 20 m segment is compared with a 10 m-radius disk of the raster; on steep Himalayan slopes this adds error of its own.",
          "* The fine-tuned model was trained only on Swiss data: these numbers are its first measurement on Indian imagery and terrain."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
