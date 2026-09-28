"""Measure helicopter landing-zone screening against Swiss LiDAR (docs/landing_zones.md, section Measurement).

Truth: swissSURFACE3D (DSM) and swissALTI3D (DTM), pixel-aligned with the DepthWizard jobs (urban 0.5 m; rural
aggregated 4 x 4 to the 2 m job grid: mean for surfaces, max for obstacle presence and corridor heights).

    python scripts/hlz_study.py [--out docs/landing_zones_results.md]

1. The obstacle buffer and slope margin are CHOSEN on the urban scene with a rule fixed before the sweep, then
   reported unchanged on the rural scene (held out).
2. The roughness limit is the P95 of DepthWizard pad roughness on LiDAR-landable centres.
3. Final numbers per scene and pad size:
   * false clear: share of DepthWizard-feasible pad centres where LiDAR shows an object inside the pad (nDSM > 2.5 m,
     the claimed detection threshold; also > 1.0 m) or a slope beyond the limit. The roughness test is off in these
     rates, so they are upper bounds for the full rule set;
   * recall: share of LiDAR-landable centres (clear at 2.5 m, slope <= 7 deg) that DepthWizard also finds feasible;
   * slope error: DepthWizard pad slope minus LiDAR pad slope on DepthWizard-feasible centres;
   * corridors: every DepthWizard CLEAR / BLOCKED bearing of the reported sites is re-checked on the LiDAR DSM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.disaster.flood import _footprint_sets  # noqa: E402
from core.disaster.landing_zones import _pooled, approach_status, count_in_pad, pad_kernel, plane_fields, screen  # noqa: E402
from core.screening_params import HLZ_CORRIDOR_CELL_M, HLZ_CORRIDOR_WIDTH_FACTOR, HLZ_SIZES, HLZ_SLOPE_ALL_DEG  # noqa: E402

REF = ROOT / "assets" / "reference"
SCENES = {
    "urban": ("b1c4eab4e781", "urban_2682-1247", 1),  # Zurich 2682-1247, 0.5 m, tier A
    "rural": ("4908186d1282", "rural_2621-1202", 4),  # 2621-1202, 2 m, tier T
}
OBJ_T, OBJ_SIGMA = 2.5, 5.515  # model card da-v2-small-ndsm@1.0.0 (the jobs use this model)
STRIDE_M = 2.0
BUFFERS = (0.0, 5.0, 10.0, 15.0)
MARGINS = (0.0, 1.0, 2.0, 3.0)
TARGET = 0.02  # fixed before the sweep: <= 2 % false clear (pad object > 2.5 m, and slope) on the choice scene


def read(p: Path) -> tuple[np.ndarray, object]:
    with rasterio.open(p) as d:
        a = d.read(1).astype(np.float64)
        if d.nodata is not None:
            a[a == d.nodata] = np.nan
        return a, d.transform


def block(a: np.ndarray, f: int, how: str) -> np.ndarray:
    if f == 1:
        return a
    h, w = a.shape[0] // f, a.shape[1] // f
    b = a[: h * f, : w * f].reshape(h, f, w, f)
    return b.mean(axis=(1, 3)) if how == "mean" else b.max(axis=(1, 3))


def truth(job: str, ref: str, f: int, size: int) -> dict:
    """Everything that does not depend on the buffer / margin being evaluated."""
    jd = ROOT / "data" / "jobs" / job
    dsm, t = read(jd / "dsm.tif")
    ndsm, _ = read(jd / "ndsm.tif")
    b_data = json.loads((jd / "buildings.json").read_text(encoding="utf-8")) if (jd / "buildings.json").exists() else {}
    lab, _ = _footprint_sets(jd, b_data, dsm.shape)
    Ld_raw, _ = read(REF / f"swisssurface3d_{ref}_dsm_0.5m.tif")
    Lt_raw, _ = read(REF / f"swissalti3d_{ref}_dtm_0.5m.tif")
    Ld, Lmax, Lnd_max = block(Ld_raw, f, "mean"), block(Ld_raw, f, "max"), block(Ld_raw - Lt_raw, f, "max")
    assert Ld.shape == dsm.shape, (Ld.shape, dsm.shape)
    D = HLZ_SIZES[size]["diameter_m"]
    K, dx, dy, k = pad_kernel(t, D / 2.0)
    Lvalid = np.isfinite(Ld)
    la, lb, lc, _ = plane_fields(Ld, Lvalid, K, dx, dy)
    st = max(1, int(round(STRIDE_M / abs(t.a))))
    sub = np.zeros(dsm.shape, bool)
    sub[::st, ::st] = True
    sub &= count_in_pad(~Lvalid, K, k, outside=True) == 0
    f_c = max(1, int(HLZ_CORRIDOR_CELL_M // abs(t.a)))
    Hd, inv_d = _pooled(np.where(np.isfinite(Lmax), Lmax, -np.inf), ~np.isfinite(Lmax), f_c)
    return {
        "dsm": dsm, "ndsm": ndsm, "b": lab > 0 if lab is not None else None, "t": t, "size": size, "D": D, "sub": sub,
        "l_obj25": count_in_pad(Lvalid & (np.nan_to_num(Lnd_max) > 2.5), K, k, outside=False) > 0,
        "l_obj10": count_in_pad(Lvalid & (np.nan_to_num(Lnd_max) > 1.0), K, k, outside=False) > 0,
        "l_slope": np.degrees(np.arctan(np.hypot(la, lb))), "lc": lc, "Hd": Hd, "inv_d": inv_d, "t_c": t @ rasterio.Affine.scale(f_c),
    }


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def evaluate(T: dict, buffer_m: float, margin_deg: float, *, roughness: float | None = None, corridors: bool = True) -> dict:
    kw = dict(size=T["size"], object_threshold_m=OBJ_T, object_sigma_m=OBJ_SIGMA, object_buffer_m=buffer_m, slope_margin_deg=margin_deg)
    t0 = time.perf_counter()
    dw_nr = screen(T["dsm"], T["ndsm"], T["b"], None, T["t"], roughness_max_m=np.inf, **kw)
    runtime = time.perf_counter() - t0
    dw = dw_nr if roughness is None else screen(T["dsm"], T["ndsm"], T["b"], None, T["t"], roughness_max_m=roughness, **kw)
    feas = T["sub"] & dw_nr["feasible"]
    land = T["sub"] & ~T["l_obj25"] & (T["l_slope"] <= HLZ_SLOPE_ALL_DEG)
    err = (dw_nr["slope"] - T["l_slope"])[feas]
    D = T["D"]
    n_clear = n_false_clear = n_blocked = n_false_blocked = n_sites_clear = n_sites_none_true = 0
    for s in dw["sites"] if corridors else []:
        tr = {a["bearingDeg"]: a["status"] for a in approach_status(T["Hd"], T["inv_d"], T["t_c"], s["x"], s["y"], float(T["lc"][s["row"], s["col"]]), D / 2.0, HLZ_CORRIDOR_WIDTH_FACTOR * D)}
        for a in s["approaches"]:
            if a["status"] == "CLEAR":
                n_clear += 1
                n_false_clear += tr[a["bearingDeg"]] == "BLOCKED"
            elif a["status"] == "BLOCKED":
                n_blocked += 1
                n_false_blocked += tr[a["bearingDeg"]] == "CLEAR"
        if s["clearBearings"]:
            n_sites_clear += 1
            n_sites_none_true += all(tr[b_] == "BLOCKED" for b_ in s["clearBearings"])
    return {
        "size": T["size"], "D": D, "runtime_s": runtime, "n_eval": int(T["sub"].sum()), "n_feas": int(feas.sum()), "n_land": int(land.sum()),
        "fc25": _mean(T["l_obj25"][feas]), "fc10": _mean(T["l_obj10"][feas]), "fc_slope": _mean(T["l_slope"][feas] > dw_nr["slopeMaxDeg"]),
        "recall": _mean(dw_nr["feasible"][land]),
        "slope_bias": float(np.mean(err)) if err.size else float("nan"),
        "slope_nmad": float(1.4826 * np.median(np.abs(err - np.median(err)))) if err.size else float("nan"),
        "slope_p95": float(np.percentile(np.abs(err), 95)) if err.size else float("nan"),
        "rough_land": dw_nr["roughness"][land], "sites": len(dw["sites"]),
        "clear": n_clear, "false_clear": n_false_clear, "blocked": n_blocked, "false_blocked": n_false_blocked,
        "sites_clear": n_sites_clear, "sites_none_true": n_sites_none_true,
    }


def pc(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f} %"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "landing_zones_results.md"))
    a = ap.parse_args()
    TT = {name: {z: truth(job, ref, f, z) for z in (1, 3)} for name, (job, ref, f) in SCENES.items()}
    L = ["# Landing-zone screening: measured against Swiss LiDAR", "",
         "Generated by `scripts/hlz_study.py`. Truth: swissSURFACE3D / swissALTI3D. DW = DepthWizard. Urban = Zurich 2682-1247 at 0.5 m; "
         "rural = 2621-1202 at 2 m. Pad centres are sampled every 2 m. In the pad tables the roughness test is off, so the false-clear rates are upper bounds.", ""]

    L += ["## 1. Choice of obstacle buffer and slope margin (urban scene only)", "",
          f"Rule fixed before the sweep: the smallest buffer, then the smallest margin, giving false clear <= {TARGET:.0%} for both 'object > 2.5 m in pad' and 'slope over the limit', for Size 1 and Size 3.", "",
          "| Buffer (m) | Margin (deg) | Size | False clear, object > 2.5 m | False clear, slope | Recall |", "|---|---|---|---|---|---|"]
    chosen = None
    for b in BUFFERS:
        for m in MARGINS:
            ev = [evaluate(TT["urban"][z], b, m, corridors=False) for z in (1, 3)]
            L += [f"| {b:g} | {m:g} | {e['size']} | {pc(e['fc25'])} | {pc(e['fc_slope'])} | {pc(e['recall'])} |" for e in ev]
            if chosen is None and all(e["fc25"] <= TARGET and e["fc_slope"] <= TARGET for e in ev):
                chosen = (b, m)
    L += ["", f"**Urban choice: buffer {chosen[0]:g} m, slope margin {chosen[1]:g} deg.**" if chosen else "**No combination met the target; the largest values were used.**", ""]
    b, m = chosen if chosen else (BUFFERS[-1], MARGINS[-1])

    # 1b. the urban scene is almost flat, so it cannot constrain the slope margin; check it on rural at the urban buffer
    L += ["## 1b. Slope margin check on the rural scene (at the urban buffer)", "",
          f"Rule: the smallest margin with slope false clear <= {TARGET:.0%} on both scenes for Size 1 and Size 3. The object rate is shown for information; the buffer is not re-chosen here.", "",
          "| Margin (deg) | Size | False clear, object > 2.5 m | False clear, slope | Recall |", "|---|---|---|---|---|"]
    m_final = None
    for mm in MARGINS + (4.0,):
        ev = [evaluate(TT["rural"][z], b, mm, corridors=False) for z in (1, 3)]
        L += [f"| {mm:g} | {e['size']} | {pc(e['fc25'])} | {pc(e['fc_slope'])} | {pc(e['recall'])} |" for e in ev]
        if m_final is None and mm >= m and all(e["fc_slope"] <= TARGET for e in ev):
            m_final = mm
    m = m_final if m_final is not None else 4.0
    L += ["", f"**Final: buffer {b:g} m (chosen on urban), slope margin {m:g} deg (chosen on rural; no independent scene tests it).**", ""]

    rough = np.concatenate([evaluate(TT[n][1], b, m, corridors=False)["rough_land"] for n in TT])
    r95 = float(np.percentile(rough, 95))
    L += ["## 2. Roughness calibration", "",
          f"DW pad RMS residual on LiDAR-landable Size-1 centres, both scenes, at the chosen setting (n = {rough.size}): P50 {np.percentile(rough, 50):.2f} m, "
          f"P90 {np.percentile(rough, 90):.2f} m, **P95 {r95:.2f} m**, P99 {np.percentile(rough, 99):.2f} m.", ""]

    L += ["## 3. Final rule set: pad centres", "",
          "| Scene | Size | DW feasible / evaluated | LiDAR landable | False clear: object > 2.5 m | Object > 1.0 m | Slope | Recall | Slope error bias / NMAD / P95 abs (deg) | Runtime (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    C = ["", "## 4. Final rule set: approach corridors of reported sites, re-checked on the LiDAR DSM", "",
         "| Scene | Size | Sites | DW CLEAR bearings | LiDAR BLOCKED (false clear) | DW BLOCKED bearings | LiDAR CLEAR (conservative) | Sites with a CLEAR bearing | ...of which none is clear on LiDAR |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, label in (("urban", "urban (buffer chosen here)"), ("rural", "rural (buffer held out; slope margin chosen here)")):
        for z in (1, 3):
            e = evaluate(TT[name][z], b, m, roughness=r95)
            L.append(f"| {label} | {z} ({e['D']:.0f} m) | {e['n_feas']} / {e['n_eval']} | {e['n_land']} | {pc(e['fc25'])} | {pc(e['fc10'])} | {pc(e['fc_slope'])} | {pc(e['recall'])} | "
                     f"{e['slope_bias']:+.2f} / {e['slope_nmad']:.2f} / {e['slope_p95']:.2f} | {e['runtime_s']:.1f} |")
            C.append(f"| {label} | {z} | {e['sites']} | {e['clear']} | {e['false_clear']} | {e['blocked']} | {e['false_blocked']} | {e['sites_clear']} | {e['sites_none_true']} |")
    L += C
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
