# DepthWizard — Single-View Height Estimation and 3D Flythrough

SIH26175 · ISRO · **Sprint 1 build (Mode A, relative output only)**

DepthWizard turns a single nadir RGB image into a surface model. In Sprint 1 the pipeline is:

```
PNG / JPG → validation → preprocessing → Depth Anything V2 Small (baseline, CPU/GPU)
          → relative depth → rDSM (relative surface, [0,1]) → heightfield → Three.js 3D view
```

Everything produced by this build is **RELATIVE and NON-METRIC** (calibration tier **R**). No metres, no
elevation, no vertical reference are claimed. Metric height above ground (tier H), DEM-based terrain and
absolute DSMs (tiers T/A) arrive in later sprints, with the Phase 8 safety corrections already in `core/`.

## Quick start (Windows / Linux)

Prerequisites: Python **3.12** (the venv here was created with `uv python install 3.12`), Node **24 LTS**
(25.x works but is a recorded deviation), ~1 GB disk, no GPU required.

```bash
python -m venv .venv                      # use a 3.12 interpreter
.venv/Scripts/activate                    # Linux: source .venv/bin/activate
pip install -r requirements/dev.txt       # torch: use the PyTorch index for your hardware (see requirements/base.txt)
python scripts/fetch_model.py             # Depth Anything V2 Small weights (99 MB, Apache-2.0), SHA-256 verified
python scripts/doctor.py                  # environment report
cd frontend && npm ci && npm run build && cd ..
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000  (demo tiles: assets/demo, © swisstopo OGD)
```

Development: `cd frontend && npm run dev` (Vite on :5173, proxies `/api`, `/health`, `/demo` to :8000).

Tests: `python -m pytest -q` (40 tests: unit, API, Phase 8 regressions incl. the real model on a demo tile).

## API (Sprint 1)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | status, model availability/version/hash, device |
| GET | `/api/system` | runtime versions, config hash, Mode A semantics |
| POST | `/api/jobs` (multipart `file`) | create job + upload PNG/JPG → `UPLOADED` |
| POST | `/api/jobs/{id}/run` | run pipeline (async, in-process worker) |
| GET | `/api/jobs/{id}` | state machine: CREATED → UPLOADED → PREPROCESSING → INFERENCE → RASTERIZING → READY / FAILED, with measured `stages_ms` |
| GET | `/api/jobs/{id}/result` | result manifest (mode, metric=false, tier R, grid, rDSM stats, heightfield meta, artefacts) |
| GET | `/api/jobs/{id}/metadata` | input meta, prep manifest, prediction provenance, heightfield meta |
| GET | `/api/jobs/{id}/artifact/{name}` | `rdsm.tif` (float32, no CRS, tags METRIC=false/TIER=R), previews, `heightfield.f32`, `texture.jpg`, `log.jsonl` … |

Errors are structured: `{"error": {"code", "message", "detail", "recoverable"}}` — e.g. `UNSUPPORTED_FORMAT`,
`INVALID_FILE`, `IMAGE_TOO_LARGE`, `MODEL_UNAVAILABLE`, `INFERENCE_FAILED`, `JOB_NOT_FOUND`, `JOB_STATE`.

## Repository layout

`backend/` FastAPI app, job manager, config · `core/` ingest, geo (Grid, vertical-datum guard, UTM reprojection),
preprocess, inference, calib (robust anchors), dsm (rDSM), validate (co-registration, metrics), terrain (heightfield) ·
`ml/registry/` model registry + vendored Depth Anything V2 code · `models/` INDEX + model cards (weights fetched) ·
`frontend/` Vite + TypeScript + Three.js viewer · `assets/demo/` open demo tiles · `tests/` unit / api / regression ·
`validation/phase8/` design-verification suite · `configs/default.yaml` all tunable parameters · `docs/`.

## Phase 8 corrections already in the code

* **C-1** `core/geo/vertical.py`: offline PROJ; geoid grid presence check; any transformer whose description contains
  "ballpark" is rejected; known-point self-test (Delhi). Absolute elevation is impossible without grids.
* **C-2** `core/validate/coregister.py`: Horn gradients via `correlate` (sign-tested on a plane); iterative estimator
  with reference smoothing + deterministic grid search; regression test recovers a (2, −1) px synthetic shift.
* **C-3** `core/calib/anchors.py`: median offset, RANSAC scale, N ≥ 5, NMAD blunder flags, hold-out split (no Huber).
* **C-4** `core/geo/reproject.py`: geographic CRS → local UTM before metric operations, with provenance record.

Licences: code Apache-2.0-compatible (vendored DA-V2 under Apache-2.0); demo imagery © swisstopo (OGD, attribution).
