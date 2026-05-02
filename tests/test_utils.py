"""
Tests for src/utils.py — device detection and seed reproducibility.
"""

import numpy as np
import torch
import pytest

from src.utils import get_device, set_seed, DEFAULT_SEED


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid_type(self):
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_tensor_moves_to_device(self):
        """A tensor can actually be allocated on the returned device."""
        device = get_device()
        t = torch.zeros(4, device=device)
        assert t.device.type == device.type


class TestSetSeed:
    def test_numpy_reproducible(self):
        set_seed(DEFAULT_SEED)
        a = np.random.rand(10)
        set_seed(DEFAULT_SEED)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducible(self):
        set_seed(DEFAULT_SEED)
        a = torch.rand(10)
        set_seed(DEFAULT_SEED)
        b = torch.rand(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = torch.rand(10)
        set_seed(1)
        b = torch.rand(10)
        assert not torch.equal(a, b)
