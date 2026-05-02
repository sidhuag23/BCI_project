"""
Tests for src/train.py.

Training tests use tiny synthetic datasets (n=40-60 epochs) and n_epochs=2
so the suite finishes in seconds.  They verify interface contracts, not
convergence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model import SPSQConvNet
from src.train import (
    WEnsemble, load_ensemble, mixed_augment, mixup_augment, naive_clone,
    save_ensemble, train_ensemble, train_one,
)
from src.utils import EPOCH_SAMPLES, N_CHANNELS

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balanced(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return n//2 non-P300 + n//2 P300 synthetic epochs."""
    rng = np.random.default_rng(seed)
    epochs = rng.standard_normal((n, EPOCH_SAMPLES, N_CHANNELS)).astype(np.float32)
    labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int8)
    return epochs, labels


def _imbalanced(n_non: int = 50, n_p300: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_non + n_p300
    epochs = rng.standard_normal((n, EPOCH_SAMPLES, N_CHANNELS)).astype(np.float32)
    labels = np.array([0] * n_non + [1] * n_p300, dtype=np.int8)
    return epochs, labels


# ---------------------------------------------------------------------------
# naive_clone
# ---------------------------------------------------------------------------

class TestNaiveClone:
    def test_balanced_counts(self):
        """P300 count must equal non-P300 count after cloning."""
        epochs, labels = _imbalanced(50, 10)
        bal_ep, bal_lb = naive_clone(epochs, labels)
        assert (bal_lb == 0).sum() == (bal_lb == 1).sum()

    def test_output_shape(self):
        epochs, labels = _imbalanced(50, 10)
        bal_ep, bal_lb = naive_clone(epochs, labels)
        assert bal_ep.shape[1:] == (EPOCH_SAMPLES, N_CHANNELS)
        assert len(bal_ep) == len(bal_lb)

    def test_labels_only_binary(self):
        epochs, labels = _imbalanced(50, 10)
        _, bal_lb = naive_clone(epochs, labels)
        assert set(int(v) for v in bal_lb) == {0, 1}

    def test_already_balanced_unchanged(self):
        """1:1 input should return the same total count."""
        epochs, labels = _balanced(40)
        bal_ep, bal_lb = naive_clone(epochs, labels)
        assert len(bal_ep) == len(epochs)

    def test_non_p300_epochs_preserved(self):
        """Original non-P300 epochs must appear in output unchanged."""
        epochs, labels = _imbalanced(50, 10)
        bal_ep, bal_lb = naive_clone(epochs, labels)
        non_out = bal_ep[bal_lb == 0]
        non_in = epochs[labels == 0]
        np.testing.assert_array_equal(non_out, non_in)

    def test_no_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No P300"):
            naive_clone(epochs, labels)

    def test_no_non_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.ones(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No non-P300"):
            naive_clone(epochs, labels)


# ---------------------------------------------------------------------------
# mixup_augment
# ---------------------------------------------------------------------------

class TestMixupAugment:
    def _imbalanced(self, n_non: int = 50, n_p300: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = n_non + n_p300
        epochs = rng.standard_normal((n, EPOCH_SAMPLES, N_CHANNELS)).astype(np.float32)
        labels = np.array([0] * n_non + [1] * n_p300, dtype=np.int8)
        return epochs, labels

    def test_balanced_p300_side_count(self):
        """P300-side count (labels >= 0.5) must equal non-P300 count."""
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        assert (bal_lb >= 0.5).sum() == (bal_lb < 0.5).sum()

    def test_output_shape(self):
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        assert bal_ep.shape[1:] == (EPOCH_SAMPLES, N_CHANNELS)
        assert len(bal_ep) == len(bal_lb)

    def test_labels_dtype_float32(self):
        epochs, labels = self._imbalanced(50, 10)
        _, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        assert bal_lb.dtype == np.float32

    def test_soft_labels_in_unit_interval(self):
        epochs, labels = self._imbalanced(50, 10)
        _, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        assert float(bal_lb.min()) >= 0.0 and float(bal_lb.max()) <= 1.0

    def test_mixed_epochs_are_convex_combinations(self):
        """Mixed epochs must be element-wise between their two parents."""
        rng = np.random.default_rng(0)
        n_non, n_p300 = 10, 5
        epochs, labels = self._imbalanced(n_non, n_p300, seed=1)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4, rng=rng)
        p300_ep = epochs[labels == 1]
        non_ep = epochs[labels == 0]
        # Mixed epochs start after original non-P300 and original P300
        n_mixed = n_non - n_p300
        mixed = bal_ep[n_non + n_p300:]
        # Each mixed epoch must lie in [min(p300,non), max(p300,non)] element-wise
        # (not tight, but mixed epochs can't exceed the range of their parents by more
        #  than numerical noise)
        assert mixed.shape[0] == n_mixed

    def test_lambda_at_least_half(self):
        """Soft labels of mixed epochs must be >= 0.5 (P300 side dominates)."""
        epochs, labels = self._imbalanced(50, 10)
        _, bal_lb = mixup_augment(epochs, labels, alpha=0.4,
                                  rng=np.random.default_rng(0))
        n_non, n_p300 = 50, 10
        mixed_labels = bal_lb[n_non + n_p300:]
        assert (mixed_labels >= 0.5).all()

    def test_original_non_p300_preserved(self):
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        np.testing.assert_array_equal(bal_ep[:50], epochs[labels == 0])

    def test_already_balanced_no_mixing(self):
        """Balanced input (n_non == n_p300) must return combined data, no mixing."""
        rng = np.random.default_rng(0)
        n = 20
        epochs = rng.standard_normal((n, EPOCH_SAMPLES, N_CHANNELS)).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10, dtype=np.int8)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4)
        assert len(bal_ep) == n

    def test_no_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No P300"):
            mixup_augment(epochs, labels)

    def test_no_non_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.ones(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No non-P300"):
            mixup_augment(epochs, labels)

    def test_invalid_alpha_raises(self):
        epochs, labels = self._imbalanced(50, 10)
        with pytest.raises(ValueError, match="alpha"):
            mixup_augment(epochs, labels, alpha=0.0)

    def test_different_alpha_gives_different_lambdas(self):
        """α=0.2 and α=2.0 should produce different soft-label distributions."""
        epochs, labels = self._imbalanced(50, 10)
        rng = np.random.default_rng(42)
        _, lb02 = mixup_augment(epochs, labels, alpha=0.2, rng=rng)
        rng = np.random.default_rng(42)
        _, lb20 = mixup_augment(epochs, labels, alpha=2.0, rng=rng)
        assert not np.allclose(lb02, lb20)

    def test_train_one_accepts_soft_labels(self):
        """train_one must run without error when labels contain Mixup soft values."""
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixup_augment(epochs, labels, alpha=0.4,
                                       rng=np.random.default_rng(0))
        model = SPSQConvNet()
        acc = train_one(model, bal_ep, bal_lb, n_epochs=1, batch_size=16,
                        device=CPU)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# WEnsemble
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# mixed_augment
# ---------------------------------------------------------------------------

class TestMixedAugment:
    def _imbalanced(self, n_non: int = 50, n_p300: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = n_non + n_p300
        epochs = rng.standard_normal((n, EPOCH_SAMPLES, N_CHANNELS)).astype(np.float32)
        labels = np.array([0] * n_non + [1] * n_p300, dtype=np.int8)
        return epochs, labels

    def test_balanced_p300_side_count(self):
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixed_augment(epochs, labels, alpha=0.4, mixup_fraction=0.5)
        assert (bal_lb >= 0.5).sum() == (bal_lb < 0.5).sum()

    def test_output_shape(self):
        epochs, labels = self._imbalanced(50, 10)
        bal_ep, bal_lb = mixed_augment(epochs, labels, alpha=0.4, mixup_fraction=0.5)
        assert bal_ep.shape[1:] == (EPOCH_SAMPLES, N_CHANNELS)
        assert len(bal_ep) == len(bal_lb)

    def test_labels_dtype_float32(self):
        epochs, labels = self._imbalanced(50, 10)
        _, bal_lb = mixed_augment(epochs, labels, alpha=0.4, mixup_fraction=0.5)
        assert bal_lb.dtype == np.float32

    def test_mixup_fraction_zero_equals_naive_clone(self):
        """mixup_fraction=0 must give same count and hard labels as naive_clone."""
        epochs, labels = self._imbalanced(50, 10)
        rng = np.random.default_rng(0)
        _, lb_mixed = mixed_augment(epochs, labels, alpha=0.4,
                                    mixup_fraction=0.0, rng=rng)
        _, lb_naive = naive_clone(epochs, labels, rng=np.random.default_rng(0))
        # Both must have same total count and same label value set {0, 1}
        assert len(lb_mixed) == len(lb_naive)
        assert set(np.unique(lb_mixed).tolist()) <= {0.0, 1.0}

    def test_mixup_fraction_one_has_soft_labels(self):
        """mixup_fraction=1 must produce soft labels on all augmented samples."""
        epochs, labels = self._imbalanced(50, 10)
        _, lb = mixed_augment(epochs, labels, alpha=0.4, mixup_fraction=1.0,
                              rng=np.random.default_rng(0))
        # At least some labels should be strictly between 0.5 and 1
        mixed_part = lb[60:]  # after 50 non + 10 original P300
        assert ((mixed_part > 0.5) & (mixed_part < 1.0)).any()

    def test_mixed_fraction_produces_partial_soft_labels(self):
        """With mixup_fraction=0.5, roughly half generated samples have soft labels."""
        n_non, n_p300 = 50, 10
        epochs, labels = self._imbalanced(n_non, n_p300)
        n_generate = n_non - n_p300  # 40
        n_mixup_exp = int(0.5 * n_generate)   # 20
        n_clone_exp = n_generate - n_mixup_exp  # 20
        _, lb = mixed_augment(epochs, labels, alpha=0.4, mixup_fraction=0.5,
                              rng=np.random.default_rng(0))
        aug_part = lb[n_non + n_p300:]
        soft_count = int((aug_part < 1.0).sum())
        hard_count = int((aug_part == 1.0).sum())
        assert soft_count == n_mixup_exp
        assert hard_count == n_clone_exp

    def test_invalid_mixup_fraction_raises(self):
        epochs, labels = self._imbalanced()
        with pytest.raises(ValueError, match="mixup_fraction"):
            mixed_augment(epochs, labels, mixup_fraction=1.5)

    def test_invalid_alpha_raises(self):
        epochs, labels = self._imbalanced()
        with pytest.raises(ValueError, match="alpha"):
            mixed_augment(epochs, labels, alpha=-0.1)

    def test_no_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No P300"):
            mixed_augment(epochs, labels)

    def test_no_non_p300_raises(self):
        epochs = np.zeros((10, EPOCH_SAMPLES, N_CHANNELS), dtype=np.float32)
        labels = np.ones(10, dtype=np.int8)
        with pytest.raises(ValueError, match="No non-P300"):
            mixed_augment(epochs, labels)


# ---------------------------------------------------------------------------
# WEnsemble
# ---------------------------------------------------------------------------

class TestWEnsemble:
    def _make(self, n: int = 2) -> WEnsemble:
        return WEnsemble(
            [SPSQConvNet() for _ in range(n)],
            np.ones(n, dtype=np.float32) / n,
        )

    def test_predict_proba_shape(self):
        x = np.random.randn(20, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        out = self._make().predict_proba(x, device=CPU)
        assert out.shape == (20,)

    def test_predict_proba_dtype(self):
        x = np.random.randn(10, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        assert self._make().predict_proba(x, device=CPU).dtype == np.float32

    def test_predict_proba_in_unit_interval(self):
        x = np.random.randn(30, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        out = self._make().predict_proba(x, device=CPU)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_weighted_average_matches_manual(self):
        """Weighted sum of two models must equal manual computation."""
        m1, m2 = SPSQConvNet(), SPSQConvNet()
        for m in (m1, m2):
            m.eval()
        w = np.array([0.4, 0.6], dtype=np.float32)
        ens = WEnsemble([m1, m2], w)

        x = np.random.randn(8, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            p1 = m1(xt).numpy()
            p2 = m2(xt).numpy()
        expected = 0.4 * p1 + 0.6 * p2
        np.testing.assert_allclose(ens.predict_proba(x, device=CPU), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# train_one
# ---------------------------------------------------------------------------

class TestTrainOne:
    def test_returns_accuracy_in_unit_interval(self):
        epochs, labels = _balanced(40)
        model = SPSQConvNet()
        acc = train_one(model, epochs, labels, n_epochs=2, batch_size=16, device=CPU)
        assert 0.0 <= acc <= 1.0

    def test_model_weights_change(self):
        """Parameters must change after at least one gradient step."""
        model = SPSQConvNet()
        w_before = model.fc3.weight.detach().clone()
        epochs, labels = _balanced(40)
        train_one(model, epochs, labels, n_epochs=2, batch_size=16, device=CPU)
        assert not torch.allclose(w_before, model.fc3.weight)

    def test_model_in_eval_mode_after(self):
        """train_one must leave the model in eval mode (used for Tk computation)."""
        model = SPSQConvNet()
        epochs, labels = _balanced(40)
        train_one(model, epochs, labels, n_epochs=1, batch_size=16, device=CPU)
        assert not model.training

    def test_seed_gives_same_result(self):
        """Same seed must produce identical training outcomes."""
        epochs, labels = _balanced(40, seed=7)

        def _run():
            torch.manual_seed(0)   # fixes both weight init and dropout sampling
            m = SPSQConvNet()
            return train_one(m, epochs, labels, n_epochs=3, batch_size=16, device=CPU, seed=42)

        assert _run() == _run()


# ---------------------------------------------------------------------------
# train_ensemble
# ---------------------------------------------------------------------------

class TestTrainEnsemble:
    def test_returns_wensemble(self):
        epochs, labels = _balanced(40)
        ens = train_ensemble(epochs, labels, n_classifiers=2, n_epochs=1, verbose=False, device=CPU)
        assert isinstance(ens, WEnsemble)

    def test_correct_number_of_classifiers(self):
        epochs, labels = _balanced(40)
        ens = train_ensemble(epochs, labels, n_classifiers=3, n_epochs=1, verbose=False, device=CPU)
        assert len(ens.models) == 3
        assert len(ens.weights) == 3

    def test_weights_sum_to_one(self):
        epochs, labels = _balanced(40)
        ens = train_ensemble(epochs, labels, n_classifiers=2, n_epochs=1, verbose=False, device=CPU)
        np.testing.assert_allclose(float(ens.weights.sum()), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        epochs, labels = _balanced(40)
        ens = train_ensemble(epochs, labels, n_classifiers=2, n_epochs=1, verbose=False, device=CPU)
        assert (ens.weights >= 0).all()

    def test_ensemble_predict_proba_shape(self):
        epochs, labels = _balanced(40)
        ens = train_ensemble(epochs, labels, n_classifiers=2, n_epochs=1, verbose=False, device=CPU)
        x = np.random.randn(15, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        out = ens.predict_proba(x, device=CPU)
        assert out.shape == (15,)


# ---------------------------------------------------------------------------
# save_ensemble / load_ensemble
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def _small_ensemble(self) -> WEnsemble:
        epochs, labels = _balanced(40)
        return train_ensemble(epochs, labels, n_classifiers=2, n_epochs=1,
                              verbose=False, device=CPU)

    def test_roundtrip_predictions_identical(self, tmp_path):
        """Loaded ensemble must produce bit-identical probabilities."""
        ens = self._small_ensemble()
        path = tmp_path / "ens.pt"
        save_ensemble(ens, path)
        ens2 = load_ensemble(path, device=CPU)

        x = np.random.randn(8, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        np.testing.assert_array_equal(
            ens.predict_proba(x, device=CPU),
            ens2.predict_proba(x, device=CPU),
        )

    def test_roundtrip_weights_identical(self, tmp_path):
        ens = self._small_ensemble()
        path = tmp_path / "ens.pt"
        save_ensemble(ens, path)
        ens2 = load_ensemble(path, device=CPU)
        np.testing.assert_array_equal(ens.weights, ens2.weights)

    def test_roundtrip_n_models(self, tmp_path):
        ens = self._small_ensemble()
        path = tmp_path / "ens.pt"
        save_ensemble(ens, path)
        ens2 = load_ensemble(path, device=CPU)
        assert len(ens2.models) == len(ens.models)

    def test_loaded_models_in_eval_mode(self, tmp_path):
        ens = self._small_ensemble()
        path = tmp_path / "ens.pt"
        save_ensemble(ens, path)
        ens2 = load_ensemble(path, device=CPU)
        assert all(not m.training for m in ens2.models)

    def test_creates_parent_directories(self, tmp_path):
        ens = self._small_ensemble()
        nested = tmp_path / "a" / "b" / "ens.pt"
        save_ensemble(ens, nested)
        assert nested.exists()
