"""
Tests for src/data_loader.py.

All tests use synthetic .mat files created in pytest tmp_path — no real data required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from src.data_loader import EpochData, _ensure_3d, _find_onsets, load_subject
from src.utils import EPOCH_SAMPLES, N_CHANNELS, N_FLASH_CODES, N_REPETITIONS


# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------

def _make_mat(
    tmp_path: Path,
    n_chars: int = 3,
    n_reps: int = 2,
    flash_stride: int = 60,
    target_codes: tuple[int, ...] = (1, 7),
    include_stim_type: bool = True,
) -> Path:
    """
    Write a minimal synthetic .mat file.

    Layout per character:
        codes 1-12 repeat n_reps times, each code active for 10 samples,
        followed by (flash_stride - 10) silence samples.
    Extra EPOCH_SAMPLES of silence are appended so the last epoch fits.

    flash_stride must be >= 10 (active samples) + enough silence that
    back-to-back epochs don't corrupt each other in tests.
    """
    n_time = n_chars * n_reps * N_FLASH_CODES * flash_stride + EPOCH_SAMPLES  # padding at end

    rng = np.random.default_rng(seed=42)
    # shape: (n_chars, n_time, N_CHANNELS) — standard BCI Comp III format
    signal = rng.standard_normal((n_chars, n_time, N_CHANNELS)).astype(np.float32)
    stim_code = np.zeros((n_chars, n_time), dtype=np.int32)
    stim_type = np.zeros((n_chars, n_time), dtype=np.int32)

    for c in range(n_chars):
        for rep in range(n_reps):
            for i, code in enumerate(range(1, N_FLASH_CODES + 1)):
                onset = (rep * N_FLASH_CODES + i) * flash_stride
                stim_code[c, onset : onset + 10] = code
                if code in target_codes and include_stim_type:
                    stim_type[c, onset : onset + 10] = 1

    mat_path = tmp_path / "synthetic.mat"
    mat_dict: dict = {"Signal": signal, "StimulusCode": stim_code}
    if include_stim_type:
        mat_dict["StimulusType"] = stim_type
    scipy.io.savemat(str(mat_path), mat_dict)
    return mat_path


# ---------------------------------------------------------------------------
# _find_onsets
# ---------------------------------------------------------------------------

class TestFindOnsets:
    def test_basic_two_flashes(self):
        code = np.array([0, 0, 3, 3, 3, 0, 0, 7, 7, 0])
        np.testing.assert_array_equal(_find_onsets(code), [2, 7])

    def test_onset_at_sample_zero(self):
        """Flash that begins at index 0 must be detected."""
        code = np.array([5, 5, 0, 0, 2, 2])
        np.testing.assert_array_equal(_find_onsets(code), [0, 4])

    def test_no_flashes(self):
        assert len(_find_onsets(np.zeros(30, dtype=int))) == 0

    def test_all_active(self):
        """One continuous block = one onset at index 0."""
        code = np.ones(20, dtype=int) * 4
        np.testing.assert_array_equal(_find_onsets(code), [0])

    def test_realistic_count(self):
        """12 codes × 3 reps with stride 50 should yield exactly 36 onsets."""
        n_codes, n_reps, stride = 12, 3, 50
        code = np.zeros(n_codes * n_reps * stride, dtype=np.int32)
        for rep in range(n_reps):
            for i in range(n_codes):
                t = (rep * n_codes + i) * stride
                code[t : t + 10] = i + 1
        assert len(_find_onsets(code)) == n_codes * n_reps


# ---------------------------------------------------------------------------
# _ensure_3d
# ---------------------------------------------------------------------------

class TestEnsure3d:
    def test_already_3d_passes_through(self):
        sig = np.zeros((5, 500, N_CHANNELS), dtype=np.float32)
        sc = np.zeros((5, 500), dtype=np.int32)
        sig_out, sc_out, _ = _ensure_3d(sig, sc, None)
        assert sig_out.shape == (5, 500, N_CHANNELS)
        assert sc_out.shape == (5, 500)

    def test_2d_gets_char_axis(self):
        """squeeze_me removes the n_chars=1 axis; _ensure_3d must restore it."""
        sig = np.zeros((500, N_CHANNELS), dtype=np.float32)
        sc = np.zeros(500, dtype=np.int32)
        sig_out, sc_out, _ = _ensure_3d(sig, sc, None)
        assert sig_out.shape == (1, 500, N_CHANNELS)
        assert sc_out.shape == (1, 500)

    def test_transposed_channel_axis_corrected(self):
        """(n_chars, N_CHANNELS, n_time) should be transposed to (n_chars, n_time, N_CHANNELS)."""
        sig = np.zeros((3, N_CHANNELS, 500), dtype=np.float32)
        sc = np.zeros((3, 500), dtype=np.int32)
        sig_out, _, _ = _ensure_3d(sig, sc, None)
        assert sig_out.shape == (3, 500, N_CHANNELS)


# ---------------------------------------------------------------------------
# load_subject
# ---------------------------------------------------------------------------

class TestLoadSubject:
    def test_returns_epoch_data(self, tmp_path):
        mat = _make_mat(tmp_path)
        data = load_subject(mat, "synth")
        assert isinstance(data, EpochData)

    def test_epoch_shape(self, tmp_path):
        n_chars, n_reps = 4, 3
        mat = _make_mat(tmp_path, n_chars=n_chars, n_reps=n_reps)
        data = load_subject(mat)
        expected = n_chars * n_reps * N_FLASH_CODES
        assert data.epochs.shape == (expected, EPOCH_SAMPLES, N_CHANNELS)

    def test_epoch_dtype_float32(self, tmp_path):
        data = load_subject(_make_mat(tmp_path))
        assert data.epochs.dtype == np.float32

    def test_stimulus_codes_range(self, tmp_path):
        data = load_subject(_make_mat(tmp_path))
        codes = set(int(c) for c in data.stimulus_codes)
        assert codes == set(range(1, N_FLASH_CODES + 1))

    def test_p300_count_two_targets_per_rep(self, tmp_path):
        """Each repetition has exactly 2 target flashes (one row, one col)."""
        n_chars, n_reps = 5, 3
        mat = _make_mat(tmp_path, n_chars=n_chars, n_reps=n_reps, target_codes=(2, 9))
        data = load_subject(mat)
        assert data.n_p300 == n_chars * n_reps * 2

    def test_total_p300_plus_non_p300_equals_total(self, tmp_path):
        data = load_subject(_make_mat(tmp_path))
        assert data.n_p300 + data.n_non_p300 == data.n_epochs

    def test_labels_minus_one_when_no_stim_type(self, tmp_path):
        mat = _make_mat(tmp_path, include_stim_type=False)
        data = load_subject(mat)
        assert np.all(data.labels == -1)

    def test_char_indices_cover_all_chars(self, tmp_path):
        n_chars = 6
        data = load_subject(_make_mat(tmp_path, n_chars=n_chars))
        assert set(int(i) for i in data.char_indices) == set(range(n_chars))

    def test_epochs_per_char_equals_reps_times_codes(self, tmp_path):
        n_chars, n_reps = 3, 4
        data = load_subject(_make_mat(tmp_path, n_chars=n_chars, n_reps=n_reps))
        for c in range(n_chars):
            count = int((data.char_indices == c).sum())
            assert count == n_reps * N_FLASH_CODES

    def test_subject_id_stored(self, tmp_path):
        data = load_subject(_make_mat(tmp_path), subject_id="A_train")
        assert data.subject_id == "A_train"
