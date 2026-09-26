"""Calibration tier decision and quality state (Phase 5 §10, Phase 6 §10).

Tiers: R relative · H metric height above ground (fine-tuned metric nDSM model installed, no usable DEM) · T absolute elevation: DEM + DEM-calibrated model detail · A anchor-refined (terrain offset and/or detail gain).
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


def decide(*, georeferenced: bool, dem_found: bool, terrain_stats: dict[str, Any] | None, scale_fit: dict[str, Any] | None, anchor_fit: dict[str, Any] | None, out_vcrs: str, datum_ok: bool, consistency: dict[str, Any] | None, cfg: dict[str, Any], is_metric: bool = False) -> TierDecision:
    """scale_fit: the DEM-band detail fusion report ({"accepted": bool, "tiles_with_detail": n, ...}).
    anchor_fit: the anchor report ({"accepted": ground-offset accepted, "gain_fit": {"accepted": ...}, ...})."""
    if not georeferenced:
        return TierDecision("R", False, False, False, None, "UNVALIDATED", ["no georeferencing"], [], ["Relative surface structure only; no scale, no elevation."])
    flags: list[str] = []
    trig: list[str] = []
    if is_metric:
        notes: list[str] = ["Object heights come from the DepthWizard fine-tuned metric nDSM model (Depth Anything V2 Small, tiled at ~0.5 m); its accuracy was measured on held-out Swiss regions (see model card), not on this scene."]
    else:
        notes = ["Sub-DEM-posting detail comes from a zero-shot relative model (Depth Anything V2 Small, tiled) whose per-tile scale is calibrated against the DEM; no fine-tuned metric model installed (tier H unavailable)."]
    if not dem_found:
        if is_metric:
            return TierDecision("H", True, True, False, None, "LIMITED", ["no DEM covers the AOI: metric heights above ground only, no absolute elevation"], ["NO_DEM", "METRIC_NDSM_MODEL"], notes)
        return TierDecision("R", True, False, False, None, "WARNING", ["no DEM covers the AOI"], ["NO_DEM"], notes + ["Horizontal scale known (GSD); no absolute elevation possible without a DEM or anchors."])
    if not datum_ok:
        if is_metric:
            return TierDecision("H", True, True, False, None, "LIMITED", ["vertical datum transform unsafe: metric heights above ground only"], ["DATUM_UNSAFE", "METRIC_NDSM_MODEL"], notes)
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
    # with a metric model terrain = DSM - model heights (the ground-anchor terrain offset is not applied), so only an
    # accepted anchor fit on the DSM itself makes the result anchor-refined
    anchors_ok = bool(anchor_fit and anchor_fit.get("accepted")) and not is_metric
    gain_ok = bool(anchor_fit and (anchor_fit.get("gain_fit") or {}).get("accepted"))
    if anchors_ok or gain_ok:
        tier = "A"
    if anchors_ok:
        hn = anchor_fit.get("holdout_nmad")  # type: ignore[union-attr]
        trig.append(f"ground anchors: n={anchor_fit.get('n_used')} terrain offset {anchor_fit.get('offset_m', 0):+.2f} m" + (f", hold-out NMAD {hn:.2f} m" if isinstance(hn, (int, float)) else ""))  # type: ignore[union-attr]
        if isinstance(hn, (int, float)) and hn > 3.0:
            quality = "WARNING" if quality == "WARNING" else "LIMITED"; trig.append("anchor hold-out NMAD > 3 m")
    detail_ok = bool(scale_fit and scale_fit.get("accepted"))
    if detail_ok and gain_ok:
        g = anchor_fit["gain_fit"]  # type: ignore[index]
        trig.append(f"model detail calibrated by anchors (gain {g.get('detail_gain', 1):.2f}, offset {g.get('offset_m', 0):+.2f} m, leave-one-out RMSE {g.get('loo_rmse_m', float('nan')):.2f} m vs {g.get('tier_t_rmse_m', float('nan')):.2f} m tier T)")
        flags.append("OBJECT_SCALE_ANCHORS")
    elif detail_ok and is_metric:
        quality = "WARNING" if quality == "WARNING" else "LIMITED"
        flags.append("METRIC_NDSM_MODEL")
        if anchor_fit:
            g = anchor_fit.get("gain_fit") or {}
            trig.append("object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene); anchors supplied but not used: " + str(g.get("reason", "no gain fit")))
        else:
            trig.append("object heights from the fine-tuned metric nDSM model (validated on held-out regions, not on this scene) — no anchors to confirm them here")
    elif detail_ok:
        # scene-level calibration against the DEM alone is the weakest metric claim we make -> never better than LIMITED
        quality = "WARNING" if quality == "WARNING" else "LIMITED"
        trig.append(f"model detail scaled from the DEM band ({scale_fit.get('tiles_with_detail')}/{scale_fit.get('n_tiles')} tiles, median gain {scale_fit.get('gain_median', 0):.2f} m/unit) — unvalidated without anchors or reference")  # type: ignore[union-attr]
        flags.append("OBJECT_SCALE_DEM_FIT")
        if anchor_fit and (anchor_fit.get("gain_fit") or {}).get("reason"):
            trig.append("anchor gain " + str(anchor_fit["gain_fit"]["reason"]))
    else:
        quality = "WARNING"; trig.append("no model detail calibrated: DSM = DEM relief only (no sub-posting structure)"); flags.append("NO_OBJECT_SCALE")
    return TierDecision(tier, True, True, True, out_vcrs, quality, trig, flags, notes)
