"""Calibration tier decision and quality state (Phase 5 §10, Phase 6 §10).

Tiers: R relative · H metric height above ground (requires a fine-tuned metric head — NOT available in this
build) · T absolute elevation from DEM terrain (+ scene-level object scale) · A anchor-refined.
Quality: GOOD / LIMITED / WARNING / INVALID / UNVALIDATED with the triggering values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TierDecision:
    tier: str
    metric_horizontal: bool
    metric_vertical: bool
    absolute_elevation: bool
    vertical_reference: str | None
    quality: str
    triggers: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def decide(*, georeferenced: bool, dem_found: bool, terrain_stats: dict[str, Any] | None, scale_fit: dict[str, Any] | None, anchor_fit: dict[str, Any] | None, out_vcrs: str, datum_ok: bool, consistency: dict[str, Any] | None, cfg: dict[str, Any], is_metric_head: bool = False) -> TierDecision:
    if not georeferenced:
        return TierDecision("R", False, False, False, None, "UNVALIDATED", ["no georeferencing"], [], ["Relative surface structure only; no scale, no elevation."])
    flags: list[str] = []
    trig: list[str] = []
    if is_metric_head:
        notes: list[str] = ["Object heights come directly from the fine-tuned metric nDSM neural backbone (Tier H active)."]
    else:
        notes: list[str] = ["Object heights come from a zero-shot relative model scaled by calibration; no fine-tuned metric head in this build (tier H unavailable)."]
    if not dem_found:
        if is_metric_head:
            return TierDecision("H", True, True, False, None, "GOOD", ["metric nDSM available from fine-tuned head; no DEM for absolute elevation"], flags, notes)
        return TierDecision("R", True, False, False, None, "WARNING", ["no DEM covers the AOI"], ["NO_DEM"], notes + ["Horizontal scale known (GSD); no absolute elevation possible without a DEM or anchors."])
    if not datum_ok:
        if is_metric_head:
            return TierDecision("H", True, True, False, None, "LIMITED", ["datum transform unsafe; fallback to height-above-ground metric layer"], ["DATUM_UNSAFE"], notes)
        return TierDecision("R", True, False, False, None, "INVALID", ["vertical datum transform unsafe (geoid grids missing or ballpark)"], ["DATUM_UNSAFE"], notes)
    ts = terrain_stats or {}
    quality = "GOOD"
    if ts.get("raw_fallback_fraction", 0) > 0.5:
        quality = "WARNING"; trig.append(f"raw-DEM fallback on {ts['raw_fallback_fraction']:.0%} of cells"); flags.append("TERRAIN_RAW_DEM")
    elif ts.get("support_mean", 1) < 0.5:
        quality = "LIMITED"; trig.append(f"mean ground support {ts['support_mean']:.2f} < 0.5")
    if ts.get("dem_void_fraction", 0) > 0.05:
        flags.append("DEM_VOID"); trig.append(f"DEM voids {ts['dem_void_fraction']:.1%}")
    if consistency and abs(consistency.get("ME", 0)) > cfg.get("datum_sanity_m", 15.0):
        return TierDecision("R", True, False, False, None, "INVALID", trig + [f"terrain-vs-DEM mean offset {consistency['ME']:.1f} m > sanity threshold"], flags + ["DATUM_SUSPECT"], notes)
    tier = "T"
    anchors_ok = bool(anchor_fit and anchor_fit.get("accepted"))
    if anchors_ok:
        tier = "A"
        hn = anchor_fit.get("holdout_nmad")
        trig.append(f"anchors: n={anchor_fit.get('n_used')} offset {anchor_fit.get('offset_m', 0):+.2f} m" + (f", hold-out NMAD {hn:.2f} m" if isinstance(hn, (int, float)) else ""))
        if isinstance(hn, (int, float)) and hn > 3.0:
            quality = "WARNING" if quality == "WARNING" else "LIMITED"; trig.append("anchor hold-out NMAD > 3 m")
    # object scale: anchors (RANSAC) > DEM residual fit (unvalidated) > none
    anchor_scale_ok = anchors_ok and bool((anchor_fit.get("scale_fit") or {}).get("accepted"))
    if anchor_scale_ok:
        trig.append(f"object scale from object anchors (inliers {anchor_fit['scale_fit'].get('n_inliers')}/{anchor_fit['scale_fit'].get('n_total')})")
        flags.append("OBJECT_SCALE_ANCHORS")
    elif scale_fit and scale_fit.get("accepted"):
        # scene-level scale is the weakest metric claim we make -> never better than LIMITED
        quality = "WARNING" if quality == "WARNING" else "LIMITED"
        trig.append(f"object scale from DEM residual fit (n={scale_fit.get('n_cells')}, NMAD={scale_fit.get('residual_nmad_m'):.2f} m) — unvalidated")
        flags.append("OBJECT_SCALE_DEM_FIT")
    else:
        quality = "WARNING"; trig.append("no object scale available: DSM = terrain layer only; object heights remain relative"); flags.append("NO_OBJECT_SCALE")
        if anchor_fit and (anchor_fit.get("scale_fit") or {}).get("reason"):
            trig.append("anchor scale " + str(anchor_fit["scale_fit"]["reason"]))
    has_scale = anchor_scale_ok or bool(scale_fit and scale_fit.get("accepted"))
    return TierDecision(tier, True, has_scale, True, out_vcrs, quality, trig, flags, notes)
