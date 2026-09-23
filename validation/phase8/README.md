# Phase 8 validation artifacts (Level 0 — design verification)

**What this is:** self-contained, seeded reference scripts that verify the *mathematical design* specified in Phase 6/7
(datum handling, GeoTIFF round-trip, ground-masked normalized-convolution terrain, robust anchor fitting, Horn slope,
co-registration, metric identities, ground-pixel dominance, composition, mesh decimation residual, tiling).

**What this is NOT:** a test of the DepthWizard codebase. At the time of Phase 8 (2026-09-20/21) the project directory
contained no source code, weights, data or outputs; Levels 1–10 of the Phase 8 protocol are therefore NOT EXECUTED.

Reproduce: `python run_all.py` (Python 3.12+; numpy, scipy, scikit-learn, rasterio, pyproj; network for PROJ geoid grids).
Results: `results/*.json`; provenance: `experiment_log.json`. Scripts `l0_07b/c/d` were run interactively (see report §24)
and their JSON outputs are included; the corrected co-registration logic is in `l0_07d_coregistration_corrected.json`.
