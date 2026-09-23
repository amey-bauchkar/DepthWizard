"""Model registry: resolves (name, version) -> weights path + model card, verifies checksum, records provenance.

INDEX.json schema:
{
  "models": {
    "<name>": {"<version>": {"file": "relative/path.pth", "sha256": "...", "architecture": "...", "licence": "...",
                             "output": "relative_inverse_depth", "card": "model_card.json"}}
  }
}
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.errors import ModelUnavailableError


@dataclass
class ModelCard:
    name: str
    version: str
    architecture: str
    weights_path: Path | None
    sha256_expected: str | None
    sha256_actual: str | None
    licence: str
    output_quantity: str  # e.g. "relative_inverse_depth" -- NEVER "metres" for the baseline
    source: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["weights_path"] = str(self.weights_path) if self.weights_path else None
        return d


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.index_path = self.models_dir / "INDEX.json"

    def _index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            raise ModelUnavailableError(f"model index not found: {self.index_path}")
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def resolve(self, name: str, version: str, *, verify_hash: bool = True) -> ModelCard:
        if name == "stub":
            return ModelCard("stub", "0", "brightness-stub (deterministic test model)", None, None, None, "n/a", "relative_brightness", "internal")
        idx = self._index()
        try:
            entry = idx["models"][name][version]
        except KeyError as e:
            raise ModelUnavailableError(f"model {name}@{version} not in {self.index_path}") from e
        wp = self.models_dir / entry["file"]
        if not wp.exists():
            raise ModelUnavailableError(f"weights file missing: {wp} (run scripts/fetch_model.py)")
        actual = sha256_file(wp)
        expected = entry.get("sha256")
        if verify_hash and expected and actual != expected:
            raise ModelUnavailableError(f"weights checksum mismatch for {wp}: {actual} != {expected}")
        card_extra: dict[str, Any] = {}
        card_file = self.models_dir / entry.get("card", "") if entry.get("card") else None
        if card_file and card_file.exists():
            card_extra = json.loads(card_file.read_text(encoding="utf-8"))
        return ModelCard(
            name=name,
            version=version,
            architecture=entry.get("architecture", "unknown"),
            weights_path=wp,
            sha256_expected=expected,
            sha256_actual=actual,
            licence=entry.get("licence", "unknown"),
            output_quantity=entry.get("output", "relative_inverse_depth"),
            source=entry.get("source", "unknown"),
            extra=card_extra,
        )
