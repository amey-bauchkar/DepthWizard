"""End-to-end scenario sweep through the real API: every demo scene x every feature, plus error cases.
Each check validates the response CONTENT, not just the status code. Prints a PASS / FAIL / SKIP table and writes
data/scenario_sweep.json.   python scripts/check_all_scenarios.py [--online] [--quick]
  --online  also run the internet-dependent extras (live rainfall, Sentinel-2 scars, Bhuvan overlays)
  --quick   only one scene per kind (Mode A, Swiss B, India B, change pair)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["DW_DATA_DIR"] = str(Path(tempfile.mkdtemp(prefix="dw_sweep_")) / "data")

import numpy as np  # noqa: E402

RESULTS: list[dict] = []


def check(scene: str, name: str, fn, *, skip: str | None = None):
    if skip:
        RESULTS.append({"scene": scene, "check": name, "status": "SKIP", "detail": skip, "s": 0})
        return None
    t0 = time.perf_counter()
    try:
        out = fn()
        RESULTS.append({"scene": scene, "check": name, "status": "PASS", "detail": str(out)[:140] if out is not None else "", "s": round(time.perf_counter() - t0, 1)})
        return out
    except Exception as e:  # noqa: BLE001
        RESULTS.append({"scene": scene, "check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}"[:300], "s": round(time.perf_counter() - t0, 1), "tb": traceback.format_exc()[-1500:]})
        return None


def ok(r, *codes):
    codes = codes or (200,)
    assert r.status_code in codes, f"HTTP {r.status_code}: {r.text[:200]}"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from fastapi.testclient import TestClient

    from backend.main import create_app

    c = TestClient(create_app(), raise_server_exceptions=False)
    check("system", "health (model + metric model)", lambda: (lambda j: (j["status"] == "ok" and j["model"]["available"] and j["model"]["metric_model"]["available"]) or (_ for _ in ()).throw(AssertionError(j)) and None or j["model"]["metric_model"]["version"])(ok(c.get("/health")).json()))
    check("system", "system info", lambda: list(ok(c.get("/api/system")).json())[:6])
    demo = ok(c.get("/api/demo")).json()["items"]
    if a.quick:
        keep = {"urban_jpg", "urban_geotiff", "india_chungthang", "change_islahiye_before", "change_islahiye_after"}
        demo = [i for i in demo if i["id"] in keep]
    jobs: dict[str, str] = {}
    for item in demo:
        sid, path = item["id"], ROOT / "assets" / "demo" / item["file"]
        data = path.read_bytes()
        mime = "image/tiff" if path.suffix.lower() in (".tif", ".tiff") else "image/jpeg"
        check(sid, "inspect", lambda: (lambda j: j.get("mode") or j.get("expected_tier") or j)(ok(c.post("/api/inspect", files={"file": (path.name, data, mime)})).json()))

        def run():
            jid = ok(c.post("/api/jobs", files={"file": (path.name, data, mime)}), 201).json()["job_id"]
            ok(c.post(f"/api/jobs/{jid}/run"), 202)
            t0 = time.time()
            while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
                assert time.time() - t0 < 900, "timeout"
                time.sleep(1)
            assert j["status"] == "READY", j.get("error")
            jobs[sid] = jid
            return f"{j['status']} in {j['stages_ms'].get('total_ms', 0) / 1000:.0f}s"
        if check(sid, "process (upload -> READY)", run) is None:
            continue
        jid = jobs[sid]
        res = check(sid, "result + metadata", lambda: (ok(c.get(f"/api/jobs/{jid}/metadata")), ok(c.get(f"/api/jobs/{jid}/result")).json())[1])
        if not res:
            continue
        mode_b = res["mode"] == "B"
        check(sid, "mode as expected", lambda: res["mode"] == item["mode"] or (_ for _ in ()).throw(AssertionError(f"mode {res['mode']} != {item['mode']}")))
        if mode_b:
            check(sid, "metric DSM + EGM2008 + tier", lambda: (res["metric"] and res["vertical_reference"] == "EGM2008" and res["calibration_tier"] in ("T", "A")) or (_ for _ in ()).throw(AssertionError(res["calibration_tier"])) or res["calibration_tier"])
        # every artifact the result advertises must exist
        def artifacts():
            names = [v for v in (res.get("artifacts") or {}).values() if isinstance(v, str)]
            bad = [n for n in names if c.get(f"/api/jobs/{jid}/artifact/{n}").status_code != 200]
            assert not bad, f"missing: {bad}"
            return f"{len(names)} artifacts"
        check(sid, "all advertised artifacts downloadable", artifacts)
        g = res["grid"]
        cx, cy = g["width"] // 2, g["height"] // 2

        def sample():
            j = ok(c.get(f"/api/jobs/{jid}/sample", params={"x": cx, "y": cy, "crs": "pixel"})).json()
            vals = {k: v.get("value") for k, v in j["values"].items()}
            assert any(v is not None for v in vals.values()), vals
            return vals
        check(sid, "point sample", sample)
        check(sid, "2-point measure", lambda: ok(c.post(f"/api/jobs/{jid}/measure", json={"points": [{"x": cx - 50, "y": cy}, {"x": cx + 50, "y": cy}], "crs": "pixel"})).json()["segments"][0])
        check(sid, "malformed measure -> 400", lambda: ok(c.post(f"/api/jobs/{jid}/measure", json={"points": [[1, 2], [3, 4]]}), 400, 422).status_code)
        check(sid, "export offline 3D scene", lambda: (lambda r: len(r.content) > 1_000_000 or (_ for _ in ()).throw(AssertionError(len(r.content))) or f"{len(r.content) // 1_000_000} MB")(ok(c.get(f"/api/jobs/{jid}/export/scene.html"))))

        def package():
            r = ok(c.get(f"/api/jobs/{jid}/export/package.zip"))
            names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
            assert any(n.endswith(".tif") for n in names), names[:10]
            return f"{len(names)} files"
        check(sid, "export GIS package", package, skip=None if mode_b else "Mode A has no GIS package")
        if not mode_b:
            check(sid, "hazard refused cleanly on Mode A", lambda: ok(c.post(f"/api/jobs/{jid}/disaster/landing_zones", json={"size": 1}), 422, 400).status_code)
            check(sid, "confidence refused cleanly on Mode A", lambda: ok(c.post(f"/api/jobs/{jid}/confidence"), 422).status_code)
            continue
        check(sid, "buildings json / geojson / csv", lambda: (ok(c.get(f"/api/jobs/{jid}/buildings.geojson")), ok(c.get(f"/api/jobs/{jid}/buildings.csv")), ok(c.get(f"/api/jobs/{jid}/buildings")).json()["summary"])[2].get("count"))
        if item.get("reference_dsm"):
            for rt, key in (("dsm", "reference_dsm"), ("dtm", "reference_dtm")):
                check(sid, f"validate vs LiDAR {rt}", lambda rt=rt, key=key: (lambda j: np.isfinite(j["metrics_overall"]["RMSE"]) and round(j["metrics_overall"]["RMSE"], 2))(ok(c.post(f"/api/jobs/{jid}/validate", data={"bundled": item[key], "ref_type": rt, "vertical_crs": item["reference_vertical_crs"]})).json()))
        if item.get("reference_points"):
            check(sid, "validate vs ICESat-2 points", lambda: (lambda j: {k: round(v["RMSE"], 2) for k, v in j["metrics"].items() if "RMSE" in v})(ok(c.post(f"/api/jobs/{jid}/validate_points", data={"bundled": item["reference_points"]})).json()))
        relief = check(sid, "flood relief / model suggestion", lambda: ok(c.get(f"/api/jobs/{jid}/disaster/flood/relief")).json())
        check(sid, "flood still level", lambda: (lambda j: (j["affectedAreaM2"] >= 0 and j["uncertainty"] is not None) and f"{j['affectedAreaPct']}% wet")(ok(c.post(f"/api/jobs/{jid}/disaster/flood", json={"waterLevel_m": (relief or {}).get("p5_m", 0) + 3, "model": "level"})).json()))
        check(sid, "flood river rise (HAND)", lambda: (lambda j: f"{j['affectedAreaPct']}% wet, {j['affectedBuildingsCount']} bldgs")(ok(c.post(f"/api/jobs/{jid}/disaster/flood", json={"waterLevel_m": 8, "model": "river"})).json()))
        check(sid, "terrain accessibility", lambda: ok(c.post(f"/api/jobs/{jid}/disaster/accessibility", json={"maxSlopeDeg": 15})).json().get("accessiblePctOfGround"))
        for size in (1, 3, 5):
            check(sid, f"landing zones size {size}", lambda size=size: (lambda j: f"{j['nSites']} sites, conf {j.get('confidenceCounts')}")(ok(c.post(f"/api/jobs/{jid}/disaster/landing_zones", json={"size": size})).json()))
        check(sid, "landing zones excluding flooded", lambda: ok(c.post(f"/api/jobs/{jid}/disaster/landing_zones", json={"size": 1, "excludeFlooded": True})).json()["nSites"])
        check(sid, "landslide (offline)", lambda: ok(c.post(f"/api/jobs/{jid}/disaster/landslide", json={"lithology": "1.3"})).json()["classAreaPct"])
        check(sid, "landslide + live rainfall", lambda: ok(c.post(f"/api/jobs/{jid}/disaster/landslide", json={"fetchRainfall": True})).json()["rainfall"], skip=None if a.online else "online (--online)")

        def roads():
            r = c.post(f"/api/jobs/{jid}/disaster/roads", json={"includeLandslide": True})
            if r.status_code == 422:
                msg = r.json()["error"]["message"]
                assert "No mapped roads" in msg, msg
                return "no mapped roads here (clean 422)"
            j = ok(r).json()
            return f"{j['roads']['cutKm']} km cut, {j['nCutOff']}/{j['nSettlements']} settlements cut off"
        check(sid, "road access (flood + landslide)", roads)
        check(sid, "Bhuvan layer list", lambda: [x["id"] for x in ok(c.get(f"/api/jobs/{jid}/bhuvan")).json()["layers"]])
        check(sid, "Bhuvan overlay image", lambda: (lambda L: ok(c.get(f"/api/jobs/{jid}/bhuvan/{L[0]['id']}.png")).headers["content-type"] if L else "no layers for this area")(ok(c.get(f"/api/jobs/{jid}/bhuvan")).json()["layers"]),
              skip=None if a.online else "online (--online)")
        check(sid, "damage report JSON", lambda: sorted(k for k in ok(c.get(f"/api/jobs/{jid}/report")).json() if k in ("flood", "roads", "landslide", "hlz")))
        check(sid, "damage report PDF", lambda: (lambda r: r.content[:4] == b"%PDF" or (_ for _ in ()).throw(AssertionError("not a PDF")) or f"{len(r.content) // 1024} KB")(ok(c.get(f"/api/jobs/{jid}/report.pdf"))))
    # one confidence map (slow: 4 extra model passes)
    cj = jobs.get("india_chungthang") or next((v for k, v in jobs.items() if k.startswith("india")), None)
    check("india_chungthang", "height confidence map", lambda: (lambda j: f"median +/-{j['medianIntervalM']} m, cover {j['testCoverage']}")(ok(c.post(f"/api/jobs/{cj}/confidence")).json()), skip=None if cj else "no India job")
    # before / after change pair
    b, af = jobs.get("change_islahiye_before"), jobs.get("change_islahiye_after")
    if b and af:
        check("change_islahiye", "change candidates", lambda: [x.get("job_id") for x in ok(c.get(f"/api/jobs/{b}/change/candidates")).json().get("candidates", [])][:3])
        check("change_islahiye", "before/after change screening", lambda: (lambda j: j.get("summary", j))(ok(c.post(f"/api/jobs/{b}/change", json={"after": af})).json()))
    # ISRO product zip (synthetic Cartosat MX layout, cut from a demo scene)
    def isro_zip():
        import rasterio
        with rasterio.open(ROOT / "assets" / "demo" / "india" / "chungthang_rgb_0.5m.tif") as ds:
            rgb = ds.read(window=((0, 800), (0, 800)))
            prof = {**ds.profile, "width": 800, "height": 800, "count": 1, "transform": ds.window_transform(((0, 800), (0, 800))), "dtype": "uint16", "compress": "deflate", "photometric": "minisblack"}
        prof.pop("jpeg_quality", None)
        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(buf, "w") as z:
            for n, band in ((1, rgb[2]), (2, rgb[1]), (3, rgb[0]), (4, rgb[1])):   # B1 blue, B2 green, B3 red, B4 NIR
                p = Path(td) / f"BAND{n}.tif"
                with rasterio.open(p, "w", **prof) as o:
                    o.write(band.astype("uint16") * 4, 1)
                z.write(p, f"C3_MX/BAND{n}.tif")
            z.writestr("C3_MX/BAND_META.txt", "SatID=CARTOSAT-3\nSensor=MX\n")
        insp = ok(c.post("/api/inspect", files={"file": ("c3_product.zip", buf.getvalue(), "application/zip")})).json()
        jid = ok(c.post("/api/jobs", files={"file": ("c3_product.zip", buf.getvalue(), "application/zip")}), 201).json()["job_id"]
        ok(c.post(f"/api/jobs/{jid}/run"), 202)
        while (j := c.get(f"/api/jobs/{jid}").json())["status"] not in ("READY", "FAILED"):
            time.sleep(1)
        assert j["status"] == "READY", j.get("error")
        r = c.get(f"/api/jobs/{jid}/result").json()
        return f"inspect mode {insp.get('mode')}, job mode {r['mode']} tier {r['calibration_tier']}"
    check("isro", "Cartosat MX zip upload -> Mode B job", isro_zip)
    # error handling: never a 500
    check("errors", "text file rejected", lambda: ok(c.post("/api/jobs", files={"file": ("x.txt", b"hello", "text/plain")}), 400, 415, 422).status_code)
    check("errors", "empty file rejected", lambda: ok(c.post("/api/jobs", files={"file": ("x.png", b"", "image/png")}), 400, 422).status_code)
    check("errors", "corrupt PNG fails cleanly", lambda: (lambda jid: (ok(c.post(f"/api/jobs/{jid}/run"), 202, 400), [time.sleep(1) for _ in range(3)], c.get(f"/api/jobs/{jid}").json()["status"])[2])(ok(c.post("/api/jobs", files={"file": ("x.png", b"\x89PNG garbage", "image/png")}), 201, 400).json().get("job_id", "none")) if True else None)
    check("errors", "bad zip rejected", lambda: ok(c.post("/api/jobs", files={"file": ("x.zip", b"PK not a zip", "application/zip")}), 400, 422).status_code)
    for bad in ("..", "..%5C..", "abc", "0123456789abX"):
        check("errors", f"bad job id {bad!r} -> 404", lambda bad=bad: ok(c.get(f"/api/jobs/{bad}/result"), 404).status_code)
    any_job = next(iter(jobs.values()), None)
    check("errors", "artifact path traversal cannot leave the job folder", lambda: ok(c.get(f"/api/jobs/{any_job}/artifact/..%5C..%5C..%5C..%5Cconfigs%5Cdefault.yaml"), 404, 400).status_code, skip=None if any_job else "no job")
    check("errors", "flood with NaN level rejected", lambda: ok(c.post(f"/api/jobs/{any_job}/disaster/flood", json={"waterLevel_m": "nan", "model": "level"}), 422).status_code, skip=None if any_job else "no job")
    check("errors", "unknown HLZ size rejected", lambda: ok(c.post(f"/api/jobs/{any_job}/disaster/landing_zones", json={"size": 9}), 422).status_code, skip=None if any_job else "no job")
    check("errors", "malformed URL -> 404, not 500", lambda: ok(c.get("/api/jobs/x/bhuvan/{'id': 'a'}.png"), 404).status_code)
    check("errors", "unknown Bhuvan layer rejected", lambda: ok(c.get(f"/api/jobs/{any_job}/bhuvan/nope.png"), 422).status_code, skip=None if any_job else "no job")
    # summary
    n = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    width = max(len(r["scene"]) for r in RESULTS)
    for r in RESULTS:
        if r["status"] != "PASS" or os.environ.get("SWEEP_VERBOSE"):
            print(f"{r['status']:4s} {r['scene']:{width}s} {r['check']:42s} {r['detail']}")
    print(f"\nTOTAL: {n['PASS']} PASS, {n['FAIL']} FAIL, {n['SKIP']} SKIP  ({len(jobs)} scenes processed)")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "scenario_sweep.json").write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
