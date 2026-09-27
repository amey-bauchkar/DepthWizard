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
