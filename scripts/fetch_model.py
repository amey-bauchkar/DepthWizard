"""Download model weights listed in models/INDEX.json and verify their SHA-256.

Usage: python scripts/fetch_model.py [--name da-v2-small-baseline] [--version 1.0.0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="da-v2-small-baseline")
    ap.add_argument("--version", default="1.0.0")
    a = ap.parse_args()
    idx = json.loads((MODELS / "INDEX.json").read_text(encoding="utf-8"))
    entry = idx["models"][a.name][a.version]
    dst = MODELS / entry["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256(dst) == entry["sha256"]:
        print(f"OK (already present, hash verified): {dst}")
        return 0
    print(f"Downloading {entry['source']} -> {dst}")
    urllib.request.urlretrieve(entry["source"], dst)  # noqa: S310 (fixed, documented URL)
    got = sha256(dst)
    if got != entry["sha256"]:
        dst.unlink(missing_ok=True)
        print(f"ERROR: checksum mismatch {got} != {entry['sha256']}", file=sys.stderr)
        return 1
    print(f"OK: {dst} ({dst.stat().st_size/1e6:.1f} MB) sha256={got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
