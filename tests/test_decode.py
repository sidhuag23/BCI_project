"""
Tests for src/decode.py.

All tests use synthetic per-epoch data constructed in memory — no .mat files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decode import (
    CHAR_MATRIX,
    _avg_scores,
    accuracy_vs_reps,
    character_accuracy,
    decode_characters,
    true_char_indices_from_epochs,
)
from src.utils import COL_CODES, N_COLS, N_FLASH_CODES, N_REPETITIONS, N_ROWS, ROW_CODES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_data(
    char_positions: list[tuple[int, int]],
    n_reps: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic per-epoch data with a perfect P300 signal.

    For each character at (row, col), P300 probability = 1.0 for the target
    row code and target column code, 0.0 for all other codes.

    Returns (probs, stimulus_codes, char_indices, true_char_indices).
    """
    probs, codes, char_idx, true_chars = [], [], [], []
    for c, (row, col) in enumerate(char_positions):
        target_col_code = COL_CODES[col]   # 1-6
        target_row_code = ROW_CODES[row]   # 7-12
        true_chars.append(row * N_COLS + col)
        for _ in range(n_reps):
            for code in range(1, N_FLASH_CODES + 1):
                is_target = code in (target_col_code, target_row_code)
                probs.append(1.0 if is_target else 0.0)
                codes.append(code)
                char_idx.append(c)
    return (
        np.array(probs, dtype=np.float32),
        np.array(codes, dtype=np.int8),
        np.array(char_idx, dtype=np.int16),
        np.array(true_chars, dtype=np.int16),
    )


def _labeled_data(
    char_positions: list[tuple[int, int]],
    n_reps: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like _perfect_data but also returns labels (1 for target codes, 0 otherwise).
    Returns (labels, stimulus_codes, char_indices, true_char_indices).
    """
    labels, codes, char_idx, true_chars = [], [], [], []
    for c, (row, col) in enumerate(char_positions):
        target_col_code = COL_CODES[col]
        target_row_code = ROW_CODES[row]
        true_chars.append(row * N_COLS + col)
        for _ in range(n_reps):
            for code in range(1, N_FLASH_CODES + 1):
                is_target = code in (target_col_code, target_row_code)
                labels.append(1 if is_target else 0)
                codes.append(code)
                char_idx.append(c)
    return (
        np.array(labels, dtype=np.int8),
        np.array(codes, dtype=np.int8),
        np.array(char_idx, dtype=np.int16),
        np.array(true_chars, dtype=np.int16),
    )


# A small set of diverse (row, col) positions for tests
_TEST_POSITIONS = [(0, 0), (1, 3), (3, 5), (5, 2), (2, 1)]


# ---------------------------------------------------------------------------
# CHAR_MATRIX
# ---------------------------------------------------------------------------

class TestCharMatrix:
    def test_shape(self):
        assert CHAR_MATRIX.shape == (N_ROWS, N_COLS)

    def test_all_unique(self):
        assert len(set(CHAR_MATRIX.ravel())) == N_ROWS * N_COLS

    def test_known_corners(self):
        assert CHAR_MATRIX[0, 0] == "A"
        assert CHAR_MATRIX[0, 5] == "F"
        assert CHAR_MATRIX[5, 0] == "5"
        assert CHAR_MATRIX[5, 5] == "_"


# ---------------------------------------------------------------------------
# _avg_scores
# ---------------------------------------------------------------------------

class TestAvgScores:
    def test_shape(self):
        probs = np.ones(N_FLASH_CODES, dtype=np.float32)
        codes = np.arange(1, N_FLASH_CODES + 1, dtype=np.int32)
        assert _avg_scores(probs, codes, None).shape == (N_FLASH_CODES,)

    def test_single_rep_equals_value(self):
        """With one occurrence per code, score == prob value."""
        probs = np.arange(1, N_FLASH_CODES + 1, dtype=np.float32) / N_FLASH_CODES
        codes = np.arange(1, N_FLASH_CODES + 1, dtype=np.int32)
        scores = _avg_scores(probs, codes, None)
        np.testing.assert_allclose(scores, probs, atol=1e-6)

    def test_max_reps_truncation(self):
        """max_reps=1 with 3 reps should use only the first occurrence."""
        codes = np.tile(np.arange(1, N_FLASH_CODES + 1), 3).astype(np.int32)
        probs = np.zeros(len(codes), dtype=np.float32)
        probs[:N_FLASH_CODES] = 1.0   # only first rep has signal
        scores_full = _avg_scores(probs, codes, max_reps=None)
        scores_1 = _avg_scores(probs, codes, max_reps=1)
        # With all reps: mean(1,0,0)=0.333; with max_reps=1: mean(1)=1.0
        assert scores_1[0] > scores_full[0]

    def test_missing_code_gives_zero(self):
        codes = np.array([1, 2], dtype=np.int32)
        probs = np.array([0.5, 0.5], dtype=np.float32)
        scores = _avg_scores(probs, codes, None)
        # Codes 3-12 not present → 0
        assert scores[2:].sum() == 0.0


# ---------------------------------------------------------------------------
# decode_characters
# ---------------------------------------------------------------------------

class TestDecodeCharacters:
    def test_perfect_signal_all_correct(self):
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS, n_reps=1)
        decoded = decode_characters(probs, codes, char_idx, len(_TEST_POSITIONS))
        np.testing.assert_array_equal(decoded, true)

    def test_output_shape(self):
        probs, codes, char_idx, _ = _perfect_data(_TEST_POSITIONS)
        decoded = decode_characters(probs, codes, char_idx, len(_TEST_POSITIONS))
        assert decoded.shape == (len(_TEST_POSITIONS),)

    def test_output_dtype_int16(self):
        probs, codes, char_idx, _ = _perfect_data(_TEST_POSITIONS)
        decoded = decode_characters(probs, codes, char_idx, len(_TEST_POSITIONS))
        assert decoded.dtype == np.int16

    def test_output_in_valid_range(self):
        """All decoded positions must be in [0, N_ROWS*N_COLS)."""
        probs, codes, char_idx, _ = _perfect_data(_TEST_POSITIONS)
        decoded = decode_characters(probs, codes, char_idx, len(_TEST_POSITIONS))
        assert (decoded >= 0).all() and (decoded < N_ROWS * N_COLS).all()

    def test_max_reps_1_still_correct_perfect_signal(self):
        """Even with max_reps=1, a perfect signal yields correct decodes."""
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS, n_reps=5)
        decoded = decode_characters(probs, codes, char_idx, len(_TEST_POSITIONS), max_reps=1)
        np.testing.assert_array_equal(decoded, true)

    def test_max_reps_reduces_data_used(self):
        """Removing all reps except the first should still work on clean signal."""
        positions = [(0, 0), (2, 3)]
        probs, codes, char_idx, true = _perfect_data(positions, n_reps=10)
        for r in (1, 3, 5, 10):
            decoded = decode_characters(probs, codes, char_idx, len(positions), max_reps=r)
            np.testing.assert_array_equal(decoded, true, err_msg=f"max_reps={r}")

    def test_uniform_probs_returns_some_index(self):
        """All-equal probs (0.5) shouldn't crash; argmax picks index 0."""
        n = 3
        n_epochs = n * N_FLASH_CODES
        probs = np.full(n_epochs, 0.5, dtype=np.float32)
        codes = np.tile(np.arange(1, N_FLASH_CODES + 1), n).astype(np.int8)
        char_idx = np.repeat(np.arange(n), N_FLASH_CODES).astype(np.int16)
        decoded = decode_characters(probs, codes, char_idx, n)
        assert decoded.shape == (n,)


# ---------------------------------------------------------------------------
# character_accuracy
# ---------------------------------------------------------------------------

class TestCharacterAccuracy:
    def test_perfect(self):
        true = np.array([0, 5, 10, 17], dtype=np.int16)
        assert character_accuracy(true, true) == 1.0

    def test_zero(self):
        true = np.array([0, 1, 2, 3], dtype=np.int16)
        wrong = np.array([4, 5, 6, 7], dtype=np.int16)
        assert character_accuracy(wrong, true) == 0.0

    def test_half(self):
        true = np.array([0, 1, 2, 3], dtype=np.int16)
        half = np.array([0, 1, 6, 7], dtype=np.int16)
        assert character_accuracy(half, true) == 0.5


# ---------------------------------------------------------------------------
# accuracy_vs_reps
# ---------------------------------------------------------------------------

class TestAccuracyVsReps:
    def test_returns_dict_with_correct_keys(self):
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS, n_reps=3)
        result = accuracy_vs_reps(probs, codes, char_idx, true, len(_TEST_POSITIONS),
                                  rep_counts=[1, 2, 3])
        assert set(result.keys()) == {1, 2, 3}

    def test_perfect_signal_accuracy_one_at_all_reps(self):
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS, n_reps=5)
        result = accuracy_vs_reps(probs, codes, char_idx, true, len(_TEST_POSITIONS),
                                  rep_counts=range(1, 6))
        for r, acc in result.items():
            assert acc == 1.0, f"rep={r}: expected 1.0, got {acc}"

    def test_default_rep_counts(self):
        """No rep_counts arg → keys 1..N_REPETITIONS."""
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS,
                                                     n_reps=N_REPETITIONS)
        result = accuracy_vs_reps(probs, codes, char_idx, true, len(_TEST_POSITIONS))
        assert set(result.keys()) == set(range(1, N_REPETITIONS + 1))

    def test_accuracy_non_decreasing_on_clean_signal(self):
        """More repetitions should not hurt on a perfect signal."""
        probs, codes, char_idx, true = _perfect_data(_TEST_POSITIONS,
                                                     n_reps=N_REPETITIONS)
        result = accuracy_vs_reps(probs, codes, char_idx, true, len(_TEST_POSITIONS))
        accs = [result[r] for r in sorted(result)]
        assert all(accs[i] <= accs[i + 1] + 1e-9 for i in range(len(accs) - 1))


# ---------------------------------------------------------------------------
# true_char_indices_from_epochs
# ---------------------------------------------------------------------------

class TestTrueCharIndicesFromEpochs:
    def test_recovers_correct_indices(self):
        labels, codes, char_idx, true = _labeled_data(_TEST_POSITIONS)
        recovered = true_char_indices_from_epochs(labels, codes, char_idx,
                                                  len(_TEST_POSITIONS))
        np.testing.assert_array_equal(recovered, true)

    def test_output_shape(self):
        labels, codes, char_idx, _ = _labeled_data(_TEST_POSITIONS)
        out = true_char_indices_from_epochs(labels, codes, char_idx,
                                            len(_TEST_POSITIONS))
        assert out.shape == (len(_TEST_POSITIONS),)

    def test_output_dtype_int16(self):
        labels, codes, char_idx, _ = _labeled_data(_TEST_POSITIONS)
        out = true_char_indices_from_epochs(labels, codes, char_idx,
                                            len(_TEST_POSITIONS))
        assert out.dtype == np.int16

    def test_no_p300_labels_raises(self):
        labels = np.zeros(N_FLASH_CODES, dtype=np.int8)
        codes = np.arange(1, N_FLASH_CODES + 1, dtype=np.int8)
        char_idx = np.zeros(N_FLASH_CODES, dtype=np.int16)
        with pytest.raises(ValueError, match="no P300 target"):
            true_char_indices_from_epochs(labels, codes, char_idx, 1)
