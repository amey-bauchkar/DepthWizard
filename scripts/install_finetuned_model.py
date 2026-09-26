"""Install the fine-tuned metric nDSM model produced by notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb.

Usage:  python scripts/install_finetuned_model.py depthwizard_ndsm_model.zip [--version 1.0.0] [--models-dir models]

Verifies the weights' SHA-256 against training_report.json, checks they load into Depth Anything V2 Small and that they
are not the unchanged baseline, copies them to models/da-v2-small-ndsm/<version>/, writes a model card whose
validation section is the report's MEASURED held-out test metrics, and registers the model in models/INDEX.json.
The app uses it automatically for Mode B (model_metric in configs/default.yaml); restart the server afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = "da-v2-small-ndsm"
WEIGHTS = "depth_anything_v2_ndsm_s.pth"
BASELINE_SHA = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(td)
        w, rep_p = Path(td) / WEIGHTS, Path(td) / "training_report.json"
        if not w.exists() or not rep_p.exists():
            print(f"ERROR: zip must contain {WEIGHTS} and training_report.json", file=sys.stderr)
            return 1
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        got = sha256(w)
        if got != rep.get("sha256"):
            print(f"ERROR: sha256 mismatch {got} != report {rep.get('sha256')}", file=sys.stderr)
            return 1
        if got == BASELINE_SHA:
            print("ERROR: these are the unchanged baseline weights, not a fine-tuned model", file=sys.stderr)
            return 1
        if rep.get("output_quantity") != "metric_ndsm_metres":
            print(f"ERROR: report output_quantity is {rep.get('output_quantity')!r}", file=sys.stderr)
            return 1
        import torch

        sys.path.insert(0, str(ROOT / "ml" / "registry" / "vendor"))
        from depth_anything_v2.dpt import DepthAnythingV2

        m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)  # raises if incompatible
        dst_dir = a.models_dir / NAME / a.version
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(w, dst_dir / WEIGHTS)
        shutil.copy2(rep_p, dst_dir / "training_report.json")
    t = rep.get("test_metrics_finetuned", {})
    b = rep.get("test_metrics_zeroshot_oracle_affine", {})
    card = {
        "name": NAME,
        "version": a.version,
        "role": "Fine-tuned metric nDSM model (tier H): height above ground in metres from 0.5 m RGB tiles; used for Mode B tiled inference.",
        "architecture": "Depth Anything V2 Small: DINOv2 ViT-S/14 encoder, DPT decoder, features=64, out_channels=[48,96,192,384]",
        "parameters_millions": 24.8,
        "base_model": rep.get("base_model"),
        "training_data": rep.get("data"),
        "training_gsd_m": rep.get("training_gsd_m"),
        "training_tiles": {k: len(v) for k, v in rep.get("tiles", {}).items()},
        "excluded": f"DepthWizard validation tiles {rep.get('excluded_validation_tiles')} and everything within {rep.get('exclusion_km')} km",
        "output_quantity": "metric_ndsm_metres",
        "metric": True,
        "calibration_tier": "H",
        "validation": {
            "protocol": "held-out REGIONS (spatially separate from training), full-tile 518 px sliding window at 0.5 m vs swisstopo LiDAR nDSM",
            "test_regions": sorted({s.split("(")[-1].rstrip(")") for s in rep.get("tiles", {}).get("test", [])}),
            "finetuned": {k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()},
            "zeroshot_with_oracle_affine": {k: round(v, 3) if isinstance(v, float) else v for k, v in b.items()},
        },
        "known_limitations": [
            "Trained on Switzerland only (swisstopo); accuracy on Indian imagery/sensors is unmeasured.",
            "First-surface model: sees canopy and roof tops; ground under dense canopy is inferred.",
            "Trained at 0.5 m nadir orthophotos; very different GSD or strongly off-nadir imagery degrade it.",
            "RGB and LiDAR acquisitions differ by up to max_year_gap years (construction/tree growth = label noise).",
        ],
        "licence": "Apache-2.0 (Depth Anything V2 Small derivative); training data (c) swisstopo OGD",
        "weights_sha256": rep.get("sha256"),
        "trained": {"created": rep.get("created"), "gpu": rep.get("gpu"), "torch": rep.get("torch"), "iters": (rep.get("config") or {}).get("iters")},
        "preprocess": {"input_size": 518, "size_multiple": 14, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    card_p = a.models_dir / NAME / a.version / "model_card.json"
    card_p.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    idx_p = a.models_dir / "INDEX.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"models": {}}
    idx["models"].setdefault(NAME, {})[a.version] = {
        "file": f"{NAME}/{a.version}/{WEIGHTS}",
        "sha256": rep["sha256"],
        "architecture": card["architecture"],
        "licence": card["licence"],
        "output": "metric_ndsm_metres",
        "source": "local fine-tuning: notebooks/DepthWizard_FineTune_nDSM_Colab.ipynb",
        "card": f"{NAME}/{a.version}/model_card.json",
    }
    idx_p.write_text(json.dumps(idx, indent=2) + "\n", encoding="utf-8")
    print(f"installed {NAME}@{a.version} -> {card_p.parent}")
    print(f"held-out test nDSM RMSE: fine-tuned {t.get('RMSE', float('nan')):.2f} m vs zero-shot+oracle {b.get('RMSE', float('nan')):.2f} m")
    print("restart the server; Mode B now uses the metric model (disable with DW_MODEL_METRIC_ENABLED=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
