"""DepthWizard Metric nDSM multi-task model (Phase 5 §7, Phase 6 §5/§6).

Adapts Depth Anything V2 with:
1. Metric nDSM Head: Regresses height above ground in metres [0..250m].
2. Auxiliary Semantic Head: Predicts 6 land-cover classes (Ground, Building, Tree, Road, Water, Other).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parents[1] / "registry" / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from depth_anything_v2.dpt import DepthAnythingV2


class MetricNDSMDepthAnything(nn.Module):
    """Multi-task adaptation of Depth Anything V2 for metric height and land-cover semantics."""

    def __init__(
        self,
        encoder: str = "vits",
        features: int = 64,
        out_channels: list[int] | None = None,
        num_classes: int = 6,
        max_height_m: float = 250.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [48, 96, 192, 384] if encoder == "vits" else [256, 512, 1024, 1024]
        self.max_height_m = max_height_m

        # Base Depth Anything V2 model
        self.da_v2 = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=False,
            use_clstoken=False,
        )

        # Auxiliary semantic head: projects DPT features to 6 semantic classes
        head_features_2 = 32
        self.semantic_head = nn.Sequential(
            nn.Conv2d(features // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def load_pretrained_baseline(self, weights_path: Path | str) -> None:
        """Loads official Depth Anything V2 pre-trained checkpoint."""
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.da_v2.load_state_dict(sd, strict=True)

    def forward(
        self,
        x: torch.Tensor,
        return_semantics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.
        Args:
            x: (B, 3, H, W) normalized input images.
            return_semantics: whether to compute auxiliary semantic logits.
        Returns:
            (pred_ndsm, pred_semantics)
              - pred_ndsm: (B, H, W) float32 height in metres (strictly >= 0.0)
              - pred_semantics: (B, num_classes, H, W) or None
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.da_v2.pretrained.get_intermediate_layers(
            x,
            self.da_v2.intermediate_layer_idx[self.da_v2.encoder],
            return_class_token=True,
        )

        # Forward through DPT Head
        dpt_head = self.da_v2.depth_head
        out = []
        for i, feat in enumerate(features):
            feat = feat[0]
            feat = feat.permute(0, 2, 1).reshape((feat.shape[0], feat.shape[-1], patch_h, patch_w))
            feat = dpt_head.projects[i](feat)
            feat = dpt_head.resize_layers[i](feat)
            out.append(feat)

        layer_1, layer_2, layer_3, layer_4 = out
        path_4 = dpt_head.scratch.refinenet4(dpt_head.scratch.layer4_rn(layer_4), size=dpt_head.scratch.layer3_rn(layer_3).shape[2:])
        path_3 = dpt_head.scratch.refinenet3(path_4, dpt_head.scratch.layer3_rn(layer_3), size=dpt_head.scratch.layer2_rn(layer_2).shape[2:])
        path_2 = dpt_head.scratch.refinenet2(path_3, dpt_head.scratch.layer2_rn(layer_2), size=dpt_head.scratch.layer1_rn(layer_1).shape[2:])
        path_1 = dpt_head.scratch.refinenet1(path_2, dpt_head.scratch.layer1_rn(layer_1))

        # Output conv 1 + upsample
        out_f1 = dpt_head.scratch.output_conv1(path_1)
        out_up = F.interpolate(out_f1, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)

        # Depth output
        depth = dpt_head.scratch.output_conv2(out_up)
        depth = F.relu(depth).squeeze(1)

        # Auxiliary Semantics
        sem_logits = None
        if return_semantics:
            sem_logits = self.semantic_head(out_up)

        return depth, sem_logits

    def export_dav2_state_dict(self) -> dict[str, Any]:
        """Exports weights matching standard DepthAnythingV2 checkpoint format for seamless DepthWizard loading."""
        return self.da_v2.state_dict()
