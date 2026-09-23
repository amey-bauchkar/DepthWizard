"""Unit tests for the new zero-compromise metric training pipeline modules."""
from __future__ import annotations

import numpy as np
import torch
import pytest

from ml.train.losses import EdgeAwareGradientLoss, TwoZoneCompoundLoss
from ml.train.model import MetricNDSMDepthAnything


def test_edge_aware_gradient_loss():
    loss_fn = EdgeAwareGradientLoss()
    pred = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    target = torch.ones((2, 1, 32, 32), dtype=torch.float32)
    img = torch.rand((2, 3, 32, 32), dtype=torch.float32)

    # Both flat -> gradient diff should be 0.0
    l = loss_fn(pred, target, img)
    assert float(l.item()) == pytest.approx(0.0, abs=1e-5)

    # Introduce step edge in target
    target[:, :, :, 16:] = 10.0
    l_step = loss_fn(pred, target, img)
    assert l_step.item() > 0.0


def test_two_zone_compound_loss():
    criterion = TwoZoneCompoundLoss(ground_threshold_m=0.5)

    pred = torch.zeros((2, 64, 64), dtype=torch.float32)
    target = torch.zeros((2, 64, 64), dtype=torch.float32)

    # Elevate one building to 20m
    target[0, 10:25, 10:25] = 20.0
    pred[0, 10:25, 10:25] = 18.5  # close estimate

    sem_pred = torch.rand((2, 6, 64, 64), dtype=torch.float32)
    sem_target = torch.zeros((2, 64, 64), dtype=torch.long)
    sem_target[0, 10:25, 10:25] = 1  # building class

    losses = criterion(pred, target, sem_pred, sem_target)
    assert "total_loss" in losses
    assert "loss_ground" in losses
    assert "loss_obj" in losses
    assert "loss_edge" in losses
    assert losses["total_loss"].item() > 0.0


def test_metric_model_instantiation():
    model = MetricNDSMDepthAnything(encoder="vits")
    assert model.da_v2.encoder == "vits"
    assert hasattr(model, "semantic_head")

    # Verify dummy forward pass
    dummy = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    depth, sem = model(dummy)
    assert depth.shape == (1, 224, 224)
    assert sem.shape == (1, 6, 224, 224)
    assert torch.all(depth >= 0.0), "Predicted heights must be non-negative"
