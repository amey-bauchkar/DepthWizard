"""Phase 8 Level-0 design-verification suite runner. Runs the reference scripts, captures environment, writes experiment_log.json.
These scripts test the MATHEMATICAL DESIGN specified in Phase 6/7 with synthetic data. They do NOT test the team's codebase (none exists yet)."""
import subprocess, sys, json, platform, datetime, hashlib, importlib.metadata as md, os
scripts = ["l0_01_datum.py", "l0_02_geotiff_roundtrip.py", "l0_04_nc_terrain.py", "l0_05_anchors.py", "l0_06_12_misc.py"]
env = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "os": platform.platform(), "python": sys.version.split()[0],
       "packages": {p: (md.version(p) if p in {d.metadata['Name'] for d in md.distributions()} else "NOT INSTALLED") for p in ["numpy","scipy","scikit-learn","rasterio","pyproj","torch","pillow"]},
       "gpu": "none detected (nvidia-smi not found)", "note": "network required for PROJ CDN geoid grids in l0_01 'online' section"}
log = {"suite": "phase8-L0-design-verification", "environment": env, "runs": []}
for s in scripts:
    t0 = datetime.datetime.now(); r = subprocess.run([sys.executable, s], capture_output=True, text=True); dt = (datetime.datetime.now()-t0).total_seconds()
    log["runs"].append({"script": s, "sha256": hashlib.sha256(open(s,'rb').read()).hexdigest()[:16], "returncode": r.returncode, "seconds": round(dt,2), "stderr_tail": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""})
    print(f"{s}: rc={r.returncode} {dt:.1f}s")
log["results_files"] = sorted(os.listdir("results"))
json.dump(log, open("experiment_log.json","w"), indent=2); print(json.dumps(log["environment"], indent=1))
