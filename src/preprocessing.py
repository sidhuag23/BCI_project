"""
Signal preprocessing for SPSQ-CNN P300 detection.

Pipeline (Shukla et al. 2024, Sec. II-B):
  1. Bandpass filter: 4th-order Chebyshev Type I, 0.1-10 Hz, per channel.
     Zero-phase implementation (sosfiltfilt) applied to the CONTINUOUS EEG
     (per character), NOT to individual 160-sample epochs.
  2. Epoch: detect flash onsets from StimulusCode, cut 160-sample windows
     from the already-filtered continuous signal.

Why continuous-then-epoch matters:
  The 0.1 Hz high-pass has a time-constant of ~1.6 seconds (384 samples at
  240 Hz).  Applied to a 160-sample epoch it produces transients larger than
  the signal itself, corrupting every epoch.  Applied to the full per-character
  recording (~7-8 k samples) it settles cleanly and only affects the first
  and last few epochs of each character block.

Entry point for the training pipeline:
    from src.data_loader import load_continuous
    from src.preprocessing import preprocess

    data = preprocess(load_continuous("data/Subject_A_Train.mat", "A_train"))
    # data.epochs: float32 (15300, 160, 64) — filtered and epoched
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import cheby1, sosfiltfilt

from src.utils import EPOCH_SAMPLES, N_CHANNELS, SAMPLING_RATE

# Chebyshev Type I spec (Shukla et al., Sec. II-B)
_CHEBY_ORDER: int = 4
_CHEBY_RIPPLE_DB: float = 0.5   # max passband ripple — paper unspecified; 0.5 dB is standard
_BANDPASS_LOW: float = 0.1      # Hz
_BANDPASS_HIGH: float = 10.0    # Hz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(raw: "ContinuousData", fs: float = float(SAMPLING_RATE)) -> "EpochData":  # type: ignore[name-defined]
    """
    Full preprocessing pipeline: bandpass-filter continuous EEG, then epoch.

    This is the main entry-point for training.  Pass the result of
    data_loader.load_continuous() here.

    Args:
        raw: ContinuousData from load_continuous().
        fs:  Sampling rate in Hz.

    Returns:
        EpochData with filtered epochs, same labels/codes/char_indices as
        the unfiltered version produced by data_loader.load_subject().
    """
    from src.data_loader import ContinuousData, epoch_continuous

    filtered_signal = apply_bandpass(raw.signal, fs)

    filtered_raw = ContinuousData(
        signal=filtered_signal,
        stimulus_code=raw.stimulus_code,
        stimulus_type=raw.stimulus_type,
        n_chars=raw.n_chars,
        subject_id=raw.subject_id,
        has_labels=raw.has_labels,
    )
    return epoch_continuous(filtered_raw)


def apply_bandpass(
    signal: np.ndarray,
    fs: float = float(SAMPLING_RATE),
) -> np.ndarray:
    """
    Apply 4th-order Chebyshev Type I bandpass 0.1-10 Hz along axis=1 (time).

    The filter has poles very close to the unit circle (0.1 Hz / 120 Hz Nyquist
    = 0.00083 normalised).  scipy's sosfiltfilt uses sosfilt_zi to seed initial
    conditions; for near-unit-circle poles, that linear system is ill-conditioned
    and produces huge transients.  Fix: reflect-pad the signal by up to 5 periods
    of the lowest cutoff frequency (~12,000 samples at 240 Hz) before filtering,
    then trim the padding away.  This gives the filter enough context to settle.

    Expects signal.ndim == 3 with time on axis=1:  (n, n_time, C).
    Typically called on continuous per-character EEG (n_time ≈ 7000 samples).

    Args:
        signal: float32 array (n, n_time, C) with time on axis=1.
        fs:     Sampling frequency in Hz.

    Returns:
        Filtered array, same shape, dtype float32.
    """
    sos = _build_bandpass_sos(fs)

    # Reflect-pad: 5 periods of the lowest cutoff, clamped to signal length - 1
    pad = min(signal.shape[1] - 1, int(5 * fs / _BANDPASS_LOW))
    padded = np.pad(signal, [(0, 0), (pad, pad), (0, 0)], mode="reflect")
    filtered = sosfiltfilt(sos, padded, axis=1)
    return filtered[:, pad:-pad, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bandpass_sos(fs: float) -> np.ndarray:
    """4th-order Chebyshev Type I bandpass in second-order sections form."""
    return cheby1(
        N=_CHEBY_ORDER,
        rp=_CHEBY_RIPPLE_DB,
        Wn=[_BANDPASS_LOW, _BANDPASS_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_erp_comparison(
    epochs: np.ndarray,
    labels: np.ndarray,
    fs: float = float(SAMPLING_RATE),
    save_path: Path | str | None = None,
    title: str = "Target vs Non-Target ERP",
) -> None:
    """
    Grand-average ERP plot for P300 (label=1) vs non-P300 (label=0) epochs.

    Averages across all channels to produce a single global trace per class.
    A clear positive deflection at ~300 ms in the target trace confirms that
    preprocessing preserved the ERP morphology.

    Args:
        epochs:    (n_epochs, EPOCH_SAMPLES, N_CHANNELS) filtered float array.
        labels:    (n_epochs,) int — 0 or 1.
        fs:        Sampling rate in Hz.
        save_path: Save PNG here; if None, show interactively.
        title:     Figure title.
    """
    import matplotlib.pyplot as plt

    time_ms = np.arange(EPOCH_SAMPLES) / fs * 1000

    p300_mask = labels == 1
    non_mask = labels == 0

    if p300_mask.sum() == 0 or non_mask.sum() == 0:
        raise ValueError("Need at least one P300 and one non-P300 epoch to plot.")

    # Grand average across epochs, then mean across channels → (EPOCH_SAMPLES,)
    p300_mean = epochs[p300_mask].mean(axis=0).mean(axis=1)
    non_mean = epochs[non_mask].mean(axis=0).mean(axis=1)

    # Standard error for shading
    p300_se = epochs[p300_mask].mean(axis=2).std(axis=0) / np.sqrt(p300_mask.sum())
    non_se = epochs[non_mask].mean(axis=2).std(axis=0) / np.sqrt(non_mask.sum())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_ms, p300_mean, color="tab:red", lw=1.8,
            label=f"P300 target (n={p300_mask.sum()})")
    ax.fill_between(time_ms, p300_mean - p300_se, p300_mean + p300_se,
                    color="tab:red", alpha=0.2)
    ax.plot(time_ms, non_mean, color="tab:blue", lw=1.8,
            label=f"Non-target (n={non_mask.sum()})")
    ax.fill_between(time_ms, non_mean - non_se, non_mean + non_se,
                    color="tab:blue", alpha=0.2)
    ax.axvline(300, color="gray", ls="--", lw=1.0, label="300 ms")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Time after stimulus (ms)")
    ax.set_ylabel("Mean amplitude (µV, grand-avg across 64 ch)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, (EPOCH_SAMPLES - 1) / fs * 1000)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved ERP plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI  —  python -m src.preprocessing data/Subject_A_Train.mat [--save ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from src.data_loader import load_continuous

    ap = argparse.ArgumentParser(
        description="Filter a subject's EEG and plot the P300 grand-average ERP."
    )
    ap.add_argument("mat_path", help="e.g. data/Subject_A_Train.mat")
    ap.add_argument("--subject-id", default="")
    ap.add_argument("--save", default=None, help="Save PNG here, e.g. results/erp_A.png")
    args = ap.parse_args()

    print(f"Loading {args.mat_path}")
    raw = load_continuous(args.mat_path, args.subject_id)

    if not raw.has_labels:
        print("No P300 labels in this file (test set?) -- cannot plot ERP.", file=sys.stderr)
        sys.exit(1)

    print(f"Filtering {raw.n_chars} characters continuous signal")
    data = preprocess(raw)

    print(f"Epochs: {data.epochs.shape}  dtype: {data.epochs.dtype}")
    print(f"P300: {data.n_p300}  Non-P300: {data.n_non_p300}")
    print("Plotting ERP …")

    plot_erp_comparison(
        data.epochs,
        data.labels,
        title=f"Target vs Non-Target ERP — {args.subject_id or args.mat_path}",
        save_path=args.save,
    )
