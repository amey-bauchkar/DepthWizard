"""Every numerical constant used by Building Intelligence and Hazard Screening, with its justification.

Categories (docs/math_audit.md §25):  PHYSICAL  = physical meaning;  STATISTICAL = statistical justification;
POLICY = a user-facing screening policy, configurable, NOT a physical truth;  ALGORITHMIC = numerical necessity.
Nothing in core/terrain/building_stats.py, core/disaster/flood.py or core/disaster/accessibility.py may introduce a
threshold that is not defined here.
"""
from __future__ import annotations

# ---------------------------------------------------------------- building intelligence
FLOOR_HEIGHT_RANGE_M = (3.0, 3.5)
"""PHYSICAL. Typical storey height range (floor-to-floor) used only for the approximate floor-count range."""

MIN_FLOORS_HEIGHT_M = 2.5
"""PHYSICAL. Below one minimum storey (~2.5 m clear height) a structure is not assigned any floors."""

MIN_PIXELS_FOR_QUANTILES = 20
"""STATISTICAL. With n samples the 10th/90th percentiles fall between order statistics n*0.1 and n*0.9. For n < 20
they are interpolated from the two most extreme samples, i.e. they become extreme-value statistics, not quantiles.
Below this count P10/P90 are still reported but the building carries the SMALL_SAMPLE flag."""

MIN_VALID_FRACTION = 0.8
"""STATISTICAL/POLICY. Volume integrates valid pixels and imputes invalid ones with the valid mean
(V = A_px * sum(h_valid) * N / N_valid). The imputation is exact only if the missing pixels share the valid mean; with
more than 20 % missing, the worst-case error |dV| <= A_missing * (P90 - P10) is no longer small against V, so the
building is flagged LOW_VALID_FRACTION."""

PLANE_FIT_MIN_PIXELS = 3
"""ALGORITHMIC. A plane z = a x + b y + c has 3 unknowns."""

PLANE_FIT_MAX_COND = 1e8
"""ALGORITHMIC. Condition-number ceiling of the (centred) least-squares design matrix. Above it the pixel centres are
(nearly) collinear, the tilt is not identifiable and ground_slope_deg is reported as null."""

# ---------------------------------------------------------------- flood screening
FLOOD_CONNECTIVITY = 8
"""POLICY. Neighbourhood used by the connectivity diagnostic. 8-connectivity lets water pass through a diagonal
corner contact between two cells; it is the more permissive choice and therefore never under-reports hazard relative
to 4-connectivity (the set of cells connected under 4 is a subset of those connected under 8)."""

EXPOSURE_GROUND_QUANTILE = 10.0
"""STATISTICAL. The building's exposure depth is W minus the 10th percentile of footprint ground elevation, i.e. the
depth on the low side of the footprint where water reaches the building first. P10 instead of the minimum gives a 10 %
breakdown point (a single erroneous low terrain cell cannot set the depth); instead of the median (the pre-audit
rule, W - median ground) it does not ignore a building that is up to half inundated on a slope."""

EXPOSURE_BINS_M = (0.5, 1.5, 3.0)
"""POLICY. Upper bounds (inclusive) of LOW / MODERATE / HIGH exposure depth in metres; deeper = VERY HIGH, depth <= 0
= NONE. These are screening-communication bins, not derived from a damage or hazard curve: DepthWizard has no
velocity, duration or building-vulnerability data, so no physical hazard class can be justified. Change them here
only; the API returns them with every result so the UI never hard-codes a second copy."""

EXPOSURE_LABELS = ("LOW", "MODERATE", "HIGH", "VERY HIGH")

# ---------------------------------------------------------------- accessibility
DEFAULT_MAX_SLOPE_DEG = 15.0
"""POLICY. Default traversable-slope threshold for the UI slider; the user sets the value used."""

# ---------------------------------------------------------------- helicopter landing-zone (HLZ) screening
# Doctrine source: US Army FM 3-21.38 "Pathfinder Operations" (April 2006), ch. 4, verified 2026-09-27 at
# globalsecurity.org/military/library/policy/army/fm/3-21-38/ch4.htm. Its landing-zone content matches the older
# FM 57-38 ch. 4. DGCA CAR Section 4 Series B Part III (ICAO Annex 14 Vol II) governs certified heliports, not
# field landing zones, and is not used here.

HLZ_SIZES = {
    1: {"diameter_m": 25.0, "label": "Size 1 · 25 m", "category": "observation / light"},
    2: {"diameter_m": 35.0, "label": "Size 2 · 35 m", "category": "utility"},
    3: {"diameter_m": 50.0, "label": "Size 3 · 50 m", "category": "large utility"},
    4: {"diameter_m": 80.0, "label": "Size 4 · 80 m", "category": "cargo"},
    5: {"diameter_m": 100.0, "label": "Size 5 · 100 m", "category": "sling load / unknown aircraft"},
}
"""PHYSICAL (doctrine). Landing-point diameters, FM 3-21.38 para 4-3.d ('Size 1 landing point, 25 meters' ... 'Size 5,
100 meters'). The same values are the minimum centre-to-centre spacing between landing points, which is why sites
are separated by at least one diameter. The manual lists sizes, not aircraft; the category column is the common
training mapping, and the operator must match the actual aircraft."""

HLZ_DEFAULT_SIZE = 3
"""POLICY. Default selection in the UI (50 m). Conservative middle choice; the user selects the aircraft size."""

HLZ_SLOPE_ALL_DEG = 7.0
"""PHYSICAL (doctrine). FM 3-21.38 para 4-1.d: 'All helicopters can land where ground slope measures 7 degrees or
less.' Slope <= this value -> eligible for SUITABLE."""

HLZ_SLOPE_ADVISORY_DEG = 15.0
"""PHYSICAL (doctrine). FM 3-21.38 para 4-1.d: above 7 degrees 'observation and utility helicopters must terminate at
a hover'; 'between 7 and 15 degrees, pathfinders advise the pilots of large utility and cargo helicopters' (land
upslope). So 7 < slope <= 15 is MARGINAL (UPSLOPE_ADVISORY) for sizes >= HLZ_ADVISORY_MIN_SIZE and excluded for
smaller sizes. The manual gives no touchdown guidance above 15 degrees; DepthWizard therefore never offers it."""

HLZ_ADVISORY_MIN_SIZE = 3
"""PHYSICAL (doctrine mapping). Sizes 3-5 correspond to large utility / cargo aircraft, the only classes the manual
allows in the 7-15 degree band."""

HLZ_OBSTACLE_RATIO = 10.0
"""PHYSICAL (doctrine). FM 3-21.38 para 4-1.i: 'obstacle ratio of 10 to 1' for approach and departure: an obstacle
of height H above the landing point needs 10 H of horizontal clearance."""

HLZ_APPROACH_LENGTH_M = 300.0
"""POLICY. Corridor length checked beyond the pad edge. At 10:1 it covers obstacles up to 30 m above the pad
(taller trees, most buildings). The manual gives no length. Terrain rising further out is NOT checked, and every
result says so. Where a corridor leaves the scene before this length, the bearing is UNVERIFIED, never CLEAR."""

HLZ_BEARINGS = 16
"""ALGORITHMIC. Candidate approach directions (22.5 degree step). Each bearing's status is exact for a corridor along
that bearing; directions between the bearings are not claimed."""

HLZ_CORRIDOR_WIDTH_FACTOR = 1.0
"""POLICY. The corridor width is this factor times the pad diameter, and constant along its length. The manual
specifies no corridor width or divergence. A width of one pad diameter is the narrowest a helicopter centred on the
pad can use; this is a DepthWizard choice."""

HLZ_OBJECT_MARGIN_SIGMA = 1.0
"""STATISTICAL. In the approach test, every object cell (nDSM > object threshold) is raised by this many times the
job's measured object-height RMSE (result.uncertainty.ndsm.object_m; 5.5 m for model da-v2-small-ndsm@1.0.0 on
held-out Swiss LiDAR). The model is not unbiased per object, so a 1-sigma margin is a minimum, not a guarantee."""

HLZ_ROUGHNESS_MAX_M = 0.70
"""STATISTICAL (dataset-derived, docs/landing_zones_results.md section 2). Largest allowed RMS residual of the DSM
about the least-squares plane over the pad: the P95 of that residual on LiDAR-landable Size-1 pad centres in the
Swiss urban and rural test scenes (n = 27 973), so that 95 % of truly landable pads pass the roughness test."""

HLZ_MIN_PAD_PIXELS_ACROSS = 10
"""ALGORITHMIC. With fewer than 10 pixels across the pad, the disk, slope and obstacle tests are too coarse to mean
anything, and the request is refused."""

HLZ_SITE_LATTICE_FRACTION = 0.1
"""ALGORITHMIC. Candidate pad centres are evaluated on a lattice of spacing (this x diameter), at least one pixel.
Each centre's clear/slope/roughness test is exact; the lattice only limits where the centres may lie
(2.5 m for Size 1)."""

HLZ_MAX_SITES = 50
"""ALGORITHMIC. Maximum number of reported sites; bounds the approach-check cost and the list length."""

HLZ_MAX_EVALUATED = 200
"""ALGORITHMIC. Maximum number of ranked candidates whose corridors are checked while looking for HLZ_MAX_SITES
sites that are not fully blocked."""

HLZ_CORRIDOR_CELL_M = 1.0
"""ALGORITHMIC. Before the corridor check, heights are max-pooled to cells of about this size (never finer than a
pixel) and dilated by one cell. Corridors are then sampled with spacing <= one pooled cell in pixel space, so every
cell that intersects a corridor is seen: the check is conservative, and an obstacle can appear up to ~2 cells
earlier or wider, never later."""

HLZ_OBJECT_BUFFER_M = 10.0
"""STATISTICAL (dataset-derived, docs/landing_zones_results.md section 1). Detected obstacles must lie at least this far
outside the pad edge, and in the corridor check their heights are spread this far sideways. The model misplaces and
under-segments obstacle edges: with no buffer, 29 % of pad centres accepted in the Zurich urban scene had a LiDAR
object > 2.5 m inside the pad. Chosen on the urban scene as the smallest buffer giving <= 2 % false clear; on the
rural scene (not used for the choice) the Size-1 rate is 1.3 %. It cannot fix obstacles the model misses entirely
(Size-3 rural: 8.8 %)."""

HLZ_SLOPE_MARGIN_DEG = 2.0
"""STATISTICAL (dataset-derived, docs/landing_zones_results.md section 1b). Measured pad slopes must be this much
below the doctrine limits (7 / 15 degrees), because the pad slope error has an NMAD of ~1 degree and a P95 of 2-3
degrees against LiDAR. With no margin, 10 % of rural pads accepted as <= 7 degrees were steeper on LiDAR; with 2
degrees it is <= 1.4 % in both scenes. Chosen on the rural scene (the urban scene is nearly flat and cannot constrain
it), so no independent scene has tested this value."""

HLZ_SCENE_BIAS_MIN_N = 20
"""STATISTICAL. The scene-measured under-reading of obstacle tops (mean error of the nDSM P98 against laser canopy /
structure heights, e.g. ICESat-2) is added to the approach-check margin only when at least this many checkpoints
support it; with fewer, the mean error is dominated by sampling noise (standard error ~ RMSE / sqrt(n) ~ 2 m at n = 20)."""

HLZ_SLOPE_SIGMA_DEG = 1.5
"""STATISTICAL (docs/landing_zones_results.md section 3). 1-sigma error of the measured pad slope used for the per-site
confidence: the P95 absolute slope error against LiDAR is 2.2-3.0 degrees across scenes and sizes, i.e. ~1.5 degrees
if Gaussian (the NMAD, 0.3-1.0 degrees, understates the tails). The slope limit is compared with the slope
WITHOUT the margin HLZ_SLOPE_MARGIN_DEG, so the confidence reports how far inside the doctrine limit a site sits."""

HLZ_BEARING_FALSE_CLEAR = 0.25
"""STATISTICAL (docs/landing_zones_results.md section 4). Share of approach bearings shown CLEAR that the LiDAR DSM
blocks (36/175 = 21 % Size 1, 28/107 = 26 % Size 3, rural). Per-site approach confidence = 1 - rate^k for k CLEAR
bearings, which reproduces the measured share of sites whose clear bearings are all blocked (3 of 79, ~4 %) at the
typical k = 2-3."""

HLZ_CONFIDENCE_BANDS = (0.9, 0.6)
"""POLICY. Site confidence >= first value: HIGH; >= second: MEDIUM; else LOW. A scene without laser-checkpoint
validation of obstacle heights is capped at MEDIUM."""

ROAD_FLOOD_IMPASSABLE_M = 0.3
"""POLICY. A road cell is cut when the screened water depth exceeds this. ~30 cm of moving water can float a car
(US NWS 'Turn Around Don't Drown'; also the usual threshold in flood-routing studies)."""

ROAD_BRIDGE_CLEARANCE_M = 5.0
"""ASSUMPTION. Mapped bridges (OSM bridge=yes) stay open until the water under them is deeper than this: the deck
height is not known, and a river channel below a bridge is always 'wet'. Stated with every result."""

ROAD_SETTLEMENT_LINK_M = 250.0
"""POLICY. A settlement is joined to the network at the nearest road node within this distance; farther away it is
reported as having no mapped road."""

SETTLEMENT_CLUSTER_M = 25.0
"""ALGORITHMIC. Buildings closer than this (edge to edge) form one settlement cluster; clusters of at least
SETTLEMENT_MIN_BUILDINGS are reported (named after the nearest OSM place within SETTLEMENT_NAME_M)."""
SETTLEMENT_MIN_BUILDINGS = 3
SETTLEMENT_NAME_M = 400.0

HOUSEHOLD_SIZE = 4.9
"""STATISTICAL (Census of India 2011: 1,210.9 million persons / 246.7 million households). Population estimate =
sum over buildings of floors x HOUSEHOLD_SIZE, i.e. ONE household per floor of every detected building. Rough and
labelled as such: it counts non-residential buildings too; replace with census / survey data where available."""
