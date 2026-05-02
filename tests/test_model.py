"""
Tests for src/model.py (SPSQ-CNN architecture).
"""

from __future__ import annotations

import torch
import pytest

from src.model import SPSQConvNet, count_params
from src.utils import EPOCH_SAMPLES, N_CHANNELS


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_single_sample(self):
        model = SPSQConvNet()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, EPOCH_SAMPLES, N_CHANNELS))
        assert y.shape == (1,)

    def test_batch_32(self):
        model = SPSQConvNet()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(32, EPOCH_SAMPLES, N_CHANNELS))
        assert y.shape == (32,)

    def test_output_dtype_float32(self):
        model = SPSQConvNet()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(4, EPOCH_SAMPLES, N_CHANNELS))
        assert y.dtype == torch.float32

    def test_output_in_unit_interval(self):
        """Sigmoid output must be in [0, 1]."""
        model = SPSQConvNet()
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(64, EPOCH_SAMPLES, N_CHANNELS))
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0


# ---------------------------------------------------------------------------
# Parameter count — must match Shukla et al. Table I exactly
# ---------------------------------------------------------------------------

class TestParamCount:
    def test_trainable_params(self):
        n = sum(p.numel() for p in SPSQConvNet().parameters())
        assert n == 45_649, f"trainable params: expected 45,649, got {n}"

    def test_paper_param_count(self):
        """count_params() (trainable + BN running stats) must equal Table I."""
        n = count_params(SPSQConvNet())
        assert n == 45_809, f"paper param count: expected 45,809, got {n}"


# ---------------------------------------------------------------------------
# Training behaviour
# ---------------------------------------------------------------------------

class TestTraining:
    def test_gradients_flow(self):
        model = SPSQConvNet()
        x = torch.randn(8, EPOCH_SAMPLES, N_CHANNELS)
        y = torch.zeros(8)
        loss = torch.nn.functional.binary_cross_entropy(model(x), y)
        loss.backward()
        assert model.conv2d.weight.grad is not None
        assert model.conv2d.weight.grad.abs().sum() > 0

    def test_dropout_active_in_train_mode(self):
        """Two passes in train mode must differ (dropout active)."""
        model = SPSQConvNet()
        model.train()
        x = torch.randn(32, EPOCH_SAMPLES, N_CHANNELS)
        torch.manual_seed(0)
        y1 = model(x)
        torch.manual_seed(1)
        y2 = model(x)
        assert not torch.allclose(y1, y2)

    def test_deterministic_in_eval_mode(self):
        """Two passes in eval mode must be identical (dropout inactive)."""
        model = SPSQConvNet()
        model.eval()
        x = torch.randn(16, EPOCH_SAMPLES, N_CHANNELS)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))
