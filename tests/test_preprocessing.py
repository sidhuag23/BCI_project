"""
Tests for src/preprocessing.py.

Spectral tests use 10-second signals (2400 samples at 240 Hz) to give the
0.1 Hz high-pass sufficient settling time.  The middle half of each filtered
signal is measured to avoid any remaining edge transients.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from src.preprocessing import _build_bandpass_sos, apply_bandpass, preprocess
from src.utils import EPOCH_SAMPLES, N_CHANNELS, SAMPLING_RATE

# ---------------------------------------------------------------------------
# Constants for spectral tests
# ---------------------------------------------------------------------------

# Realistic per-character signal length (~30 s at 240 Hz).
# The reflect-pad in apply_bandpass uses up to 12,000 samples; with a 7,200-
# sample signal, it pads 7,199 samples on each side — enough for the 0.1 Hz
# pole (period 2,400 samples) to settle.
_LONG_N: int = 7200
# Skip first/last quarter to avoid any remaining edge transients
_TRIM: int = _LONG_N // 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _long_sine(freq_hz: float, n_chans: int = N_CHANNELS, n_epochs: int = 1) -> np.ndarray:
    """Return (n_epochs, _LONG_N, n_chans) array of a single-frequency sine."""
    t = np.linspace(0, _LONG_N / SAMPLING_RATE, _LONG_N, endpoint=False)
    sine = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return np.broadcast_to(
        sine[np.newaxis, :, np.newaxis], (n_epochs, _LONG_N, n_chans)
    ).copy()


def _rms_mid(x: np.ndarray) -> float:
    """RMS of the middle portion (after _TRIM trim on each end) of axis=1."""
    return float(np.sqrt(np.mean(x[:, _TRIM:-_TRIM, :] ** 2)))


# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------

class TestBuildBandpassSos:
    def test_sos_shape(self):
        """4th-order bandpass → 4 second-order sections."""
        sos = _build_bandpass_sos(float(SAMPLING_RATE))
        assert sos.shape == (4, 6)

    def test_different_fs_gives_different_coefficients(self):
        assert not np.allclose(
            _build_bandpass_sos(240.0),
            _build_bandpass_sos(256.0),
        )


# ---------------------------------------------------------------------------
# Shape and dtype (short arrays fine — we're not checking spectral content)
# ---------------------------------------------------------------------------

class TestApplyBandpassShape:
    def test_shape_preserved_on_short_array(self):
        x = np.random.randn(8, EPOCH_SAMPLES, N_CHANNELS).astype(np.float32)
        assert apply_bandpass(x).shape == x.shape

    def test_shape_preserved_on_long_array(self):
        x = np.random.randn(3, _LONG_N, N_CHANNELS).astype(np.float32)
        assert apply_bandpass(x).shape == x.shape

    def test_output_dtype_float32(self):
        x = np.random.randn(4, _LONG_N, N_CHANNELS).astype(np.float64)
        assert apply_bandpass(x).dtype == np.float32

    def test_zero_input_gives_zero_output(self):
        x = np.zeros((2, _LONG_N, N_CHANNELS), dtype=np.float32)
        np.testing.assert_allclose(apply_bandpass(x), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Spectral behaviour — tested on long signals only
# ---------------------------------------------------------------------------

class TestBandpassSpectral:
    def test_attenuates_50hz(self):
        """50 Hz >> 10 Hz cutoff: expect > 40 dB attenuation (measured ~47 dB)."""
        x = _long_sine(50.0)
        filt = apply_bandpass(x)
        assert _rms_mid(filt) < _rms_mid(x) * 0.01, (
            f"50 Hz: in={_rms_mid(x):.4f}, out={_rms_mid(filt):.5f}"
        )

    def test_attenuates_30hz(self):
        """30 Hz > 10 Hz cutoff: expect > 33 dB attenuation."""
        x = _long_sine(30.0)
        filt = apply_bandpass(x)
        assert _rms_mid(filt) < _rms_mid(x) * 0.022

    def test_attenuates_20hz(self):
        """20 Hz > 10 Hz cutoff."""
        x = _long_sine(20.0)
        filt = apply_bandpass(x)
        assert _rms_mid(filt) < _rms_mid(x) * 0.005

    def test_passes_5hz(self):
        """5 Hz is inside the 0.1-10 Hz passband: expect < 2 dB loss."""
        x = _long_sine(5.0)
        filt = apply_bandpass(x)
        assert _rms_mid(filt) > _rms_mid(x) * 0.80, (
            f"5 Hz: in={_rms_mid(x):.4f}, out={_rms_mid(filt):.4f}"
        )

    def test_passes_2hz(self):
        """2 Hz is inside the passband."""
        x = _long_sine(2.0)
        filt = apply_bandpass(x)
        assert _rms_mid(filt) > _rms_mid(x) * 0.80


# ---------------------------------------------------------------------------
# Per-channel independence
# ---------------------------------------------------------------------------

class TestChannelIndependence:
    def test_active_channel_does_not_bleed_into_silent_channels(self):
        x = np.zeros((2, _LONG_N, N_CHANNELS), dtype=np.float32)
        t = np.linspace(0, _LONG_N / SAMPLING_RATE, _LONG_N, endpoint=False).astype(np.float32)
        x[:, :, 0] = np.sin(2 * np.pi * 5 * t)

        filt = apply_bandpass(x)
        np.testing.assert_allclose(filt[:, :, 1:], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# preprocess() — integration test with synthetic ContinuousData
# ---------------------------------------------------------------------------

class TestPreprocess:
    def _make_continuous(self, n_chars: int = 2, n_reps: int = 1,
                         flash_stride: int = 60) -> "ContinuousData":  # type: ignore[name-defined]
        """Create a minimal ContinuousData object in memory (no .mat file)."""
        from src.data_loader import ContinuousData

        n_time = n_chars * n_reps * 12 * flash_stride + EPOCH_SAMPLES
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((n_chars, n_time, N_CHANNELS)).astype(np.float32)
        stim_code = np.zeros((n_chars, n_time), dtype=np.int32)
        stim_type = np.zeros((n_chars, n_time), dtype=np.int32)

        for c in range(n_chars):
            for rep in range(n_reps):
                for i, code in enumerate(range(1, 13)):
                    onset = (rep * 12 + i) * flash_stride
                    stim_code[c, onset : onset + 10] = code
                    if code in (1, 7):
                        stim_type[c, onset : onset + 10] = 1

        return ContinuousData(
            signal=signal, stimulus_code=stim_code, stimulus_type=stim_type,
            n_chars=n_chars, subject_id="synth", has_labels=True,
        )

    def test_returns_epoch_data(self):
        from src.data_loader import EpochData
        data = preprocess(self._make_continuous())
        assert isinstance(data, EpochData)

    def test_epoch_shape(self):
        n_chars, n_reps = 3, 2
        data = preprocess(self._make_continuous(n_chars=n_chars, n_reps=n_reps))
        expected = n_chars * n_reps * 12
        assert data.epochs.shape == (expected, EPOCH_SAMPLES, N_CHANNELS)

    def test_epoch_dtype_float32(self):
        assert preprocess(self._make_continuous()).epochs.dtype == np.float32

    def test_labels_preserved(self):
        """Label counts must match those from epoch_continuous (unfiltered)."""
        from src.data_loader import epoch_continuous
        cont = self._make_continuous(n_chars=3, n_reps=2)
        unfiltered = epoch_continuous(cont)
        filtered = preprocess(cont)
        assert filtered.n_p300 == unfiltered.n_p300
        assert filtered.n_non_p300 == unfiltered.n_non_p300
