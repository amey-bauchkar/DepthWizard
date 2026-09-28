# Helicopter landing-zone screening

`POST /api/jobs/{id}/disaster/landing_zones` · code `core/disaster/landing_zones.py` · constants `core/screening_params.py`
(HLZ section) · tests `tests/unit/test_landing_zones.py` · measurement `scripts/hlz_study.py` → `docs/landing_zones_results.md`.

**Output = candidate sites for reconnaissance, never a landing clearance.** Every response carries the limits below.

## Rules and their sources

| Rule | Value | Source |
|---|---|---|
| Pad diameter / site spacing | Size 1–5: 25, 35, 50, 80, 100 m | FM 3-21.38 (2006) §4-3.d |
| Slope | ≤ 7° any helicopter; 7–15° large utility/cargo only (sizes ≥ 3), land upslope → MARGINAL; > 15° never | FM 3-21.38 §4-1.d |
| Obstacle clearance | 10:1, measured from the pad centre (touchdown point) | FM 3-21.38 §4-1.i |
| Corridor width / length / bearings | 1 × pad diameter, 300 m, 16 bearings | DepthWizard policy (the manual gives none) |
| Obstacle buffer | detected obstacles ≥ 10 m outside the pad; spread 10 m sideways in corridors | measured (results §1) |
| Slope margin | measured slope must be ≥ 2° below the limit | measured (results §1b) |
| Roughness | plane-fit RMS ≤ 0.70 m | measured, P95 on LiDAR-landable pads (results §2) |
| Object height margin in corridors | + 1 σ object RMSE of the job (5.5 m) | model card |

Earlier plan text said "7 % grade"; the manual states **degrees**, so the limits are 7° / 15°.

## Method

Pad = disk round in CRS metres (exact under any affine grid). For every centre, FFT correlations (one shared padded
grid) give: invalid/outside cells, detected-object cells (nDSM > 2.5 m or building footprint), wet cells, and the
least-squares plane of the DSM (slope, RMS). Elevations are centred on the scene median first (FFT precision).
Candidates on a lattice (0.1 D) are ranked (slope band, slope, roughness) and thinned at spacing D. Each is checked
on 16 corridors: BLOCKED if any cell rises above s/10 over the pad; UNVERIFIED if the corridor leaves the scene or
has no data (never CLEAR); otherwise CLEAR. Heights are max-pooled to 1 m and dilated one cell, so the check is
conservative. CANDIDATE = slope within the ≤ 7° band and ≥ 1 CLEAR bearing; MARGINAL otherwise; sites with all
bearings blocked are dropped and counted. Refused (HTTP 422): tier R/H, no measured uncertainty (reprocess the job),
pixel too coarse for the pad (< 10 px across).

## Measured against Swiss LiDAR (results file has all tables)

| | Urban 0.5 m | Rural 2 m |
|---|---|---|
| Pad false clear (LiDAR object > 2.5 m in pad), Size 1 / 3 | 0.4 % / 0.0 % | 1.3 % / 8.8 % |
| Pad false clear (slope), Size 1 / 3 | 0.3 % / 0.0 % | 0.9 % / 1.4 % |
| CLEAR bearings actually blocked | – (no clear bearings) | 21 % (Size 1), 26 % (Size 3) |
| Sites with a CLEAR bearing where none is truly clear | – | 0 / 46 (Size 1), 3 / 33 (Size 3) |

Before the buffer and margin, pad false clear was 29 % (urban) and slope false clear 10 % (rural). The buffer was
chosen on urban and held out on rural; the slope margin was chosen on rural and has **no independent test scene**.
Two options to cut corridor false clears (lower object threshold; +1/+2 m height allowance) were measured and
rejected: no gain, or most true-clear bearings lost.

## Limits (all shown in the UI)

- Not detectable: wires, poles, antennas, debris, obstacles < ~2.5 m; ~11 % of real obstacle pixels (trees, edges)
  are read as < 1 m by the model.
- 50 m pads in rural terrain: 8.8 % false clear from small isolated objects no buffer can catch.
- Bearings are candidates: about 1 in 4 CLEAR bearings was blocked on LiDAR.
- Terrain beyond 300 m is not checked; bearings are grid bearings.
- 30 m DEM is coarser than 25 m pads (flag `TERRAIN_COARSER_THAN_PAD`).
- No Indian obstacle truth exists; ICESat-2 cannot validate obstacles. Sikkim demo imagery is pre-event (2022).
- Runtime ~9–12 s on a 2400² scene on this laptop (target was 5 s; not met).
