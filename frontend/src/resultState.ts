/**
 * Result state model shared by the app (main.ts) and the offline scene (standalone.ts): one honest label per
 * combination of backend tier and flags.
 */
import type { Result } from "./api";

// ──────────────────────────────────────── Result state model
export type ResultState = "RELATIVE" | "TERRAIN_ONLY" | "TERRAIN_SCALED" | "ANCHOR_REFINED" | "METRIC_MODEL" | "METRIC_HEIGHTS";

export interface StateDisplay {
  label: string;      // short badge label
  full: string;       // longer description
  cssClass: string;   // state-RELATIVE etc.
  hudState: string;   // HUD top-left
  hudTier: string;    // HUD tier line
}

export const STATE_DISPLAY: Record<ResultState, StateDisplay> = {
  RELATIVE: {
    label: "RELATIVE SURFACE",
    full:  "Relative Surface Structure — no units, no scale, no elevation. Upload a GeoTIFF for absolute output.",
    cssClass: "state-RELATIVE",
    hudState: "RELATIVE SURFACE STRUCTURE",
    hudTier:  "Tier R · non-metric",
  },
  TERRAIN_ONLY: {
    label: "DEM RELIEF ONLY",
    full:  "Absolute elevation from the DEM (datum-checked). No model detail could be calibrated for this scene, so sub-30 m structure (buildings, tree crowns) is not added.",
    cssClass: "state-TERRAIN_ONLY",
    hudState: "DEM RELIEF ONLY",
    hudTier:  "",  // filled dynamically
  },
  TERRAIN_SCALED: {
    label: "DEM + CALIBRATED MODEL DETAIL",
    full:  "Absolute DSM: the DEM keeps everything it resolves (≥ 30 m); Depth Anything V2 (tiled) adds the finer detail, scaled per tile against the DEM. Without anchors or a reference this scale is unvalidated — quality ≤ LIMITED.",
    cssClass: "state-TERRAIN_SCALED",
    hudState: "DEM + MODEL DETAIL",
    hudTier:  "",
  },
  ANCHOR_REFINED: {
    label: "DEM + ANCHOR-CALIBRATED DETAIL",
    full:  "Absolute DSM: DEM + tiled model detail whose gain was fitted to ground-control anchors (accepted only when leave-one-out error improves). Ground anchors also correct the terrain datum.",
    cssClass: "state-ANCHOR_REFINED",
    hudState: "ANCHOR-CALIBRATED DSM",
    hudTier:  "",
  },
  METRIC_MODEL: {
    label: "DEM + FINE-TUNED METRIC HEIGHTS",
    full:  "Absolute DSM: the DEM keeps everything ≥ 30 m; the fine-tuned DepthWizard nDSM model (heights in metres, validated on held-out regions) adds building and tree structure. Terrain = DSM − model heights.",
    cssClass: "state-TERRAIN_SCALED",
    hudState: "DEM + METRIC nDSM MODEL",
    hudTier:  "",
  },
  METRIC_HEIGHTS: {
    label: "METRIC HEIGHTS · NO ELEVATION (TIER H)",
    full:  "Heights above ground in metres from the fine-tuned nDSM model. No DEM (or no safe datum) for this area, so there is no absolute elevation — upload a DEM or anchors for tier T/A.",
    cssClass: "state-TERRAIN_ONLY",
    hudState: "METRIC HEIGHTS ABOVE GROUND",
    hudTier:  "",
  },
};

export function resolveState(res: Pick<Result, "mode" | "calibration_tier" | "flags" | "object_scale_source">): ResultState {
  if (res.mode === "B" && res.calibration_tier === "H") return "METRIC_HEIGHTS";
  if (res.mode !== "B" || res.calibration_tier === "R") return "RELATIVE"; // Mode B without DEM / safe datum stays relative
  const flags = res.flags ?? [];
  if (flags.includes("METRIC_NDSM_MODEL") && !flags.includes("OBJECT_SCALE_ANCHORS")) return "METRIC_MODEL";
  if (flags.includes("NO_OBJECT_SCALE")) return "TERRAIN_ONLY";
  if (flags.includes("OBJECT_SCALE_ANCHORS")) return "ANCHOR_REFINED";
  if (flags.includes("OBJECT_SCALE_DEM_FIT")) return "TERRAIN_SCALED";
  // Fallback: if scale source exists use it, else terrain-only
  if (res.object_scale_source?.includes("anchor")) return "ANCHOR_REFINED";
  if (res.object_scale_source) return "TERRAIN_SCALED";
  return "TERRAIN_ONLY";
}
