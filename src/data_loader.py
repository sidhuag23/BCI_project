"""
Data loader for BCI Competition III Dataset II (.mat format).

Each .mat file stores continuous EEG organised per character:
    Signal        (n_chars, n_time, 64)  float
    StimulusCode  (n_chars, n_time)      int  0=no stimulus, 1-12=flash code
    StimulusType  (n_chars, n_time)      int  1=target (P300), 0=non-target

Two loading modes
-----------------
load_continuous(path) → ContinuousData
    Returns the raw per-character signal arrays without epoching.
    Use this when you need to filter the continuous signal BEFORE epoching
    (required for the 0.1 Hz high-pass — see preprocessing.py).

load_subject(path) → EpochData
    Convenience wrapper: load_continuous + epoch (no filtering).
    Suitable for quick inspection; the training pipeline uses
    preprocessing.preprocess(load_continuous(...)) instead.

Epoch extraction detects 0→nonzero transitions in StimulusCode (flash onsets)
and cuts a 667 ms / EPOCH_SAMPLES=160-sample window from each onset.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

from src.utils import (
    EPOCH_SAMPLES,
    N_CHANNELS,
    N_FLASH_CODES,
    N_REPETITIONS,
    N_TEST_CHARS,
    N_TRAIN_CHARS,
)

# Number of samples the stimulus is active (~100 ms at 240 Hz).
# Used when reading the StimulusType label within the flash window.
_FLASH_ACTIVE_SAMPLES: int = 24


@dataclass
class ContinuousData:
    """Raw per-character continuous EEG before epoching."""

    signal: np.ndarray           # float32  (n_chars, n_time, N_CHANNELS)
    stimulus_code: np.ndarray    # int32    (n_chars, n_time)
    stimulus_type: np.ndarray    # int32    (n_chars, n_time)  — all-zero if unlabelled
    n_chars: int
    subject_id: str
    has_labels: bool             # False when StimulusType is absent or all-zero


@dataclass
class EpochData:
    """All epochs extracted from one subject's .mat file."""

    epochs: np.ndarray          # float32  (n_epochs, EPOCH_SAMPLES, N_CHANNELS)
    labels: np.ndarray          # int8     (n_epochs,)  0=non-P300, 1=P300, -1=unknown
    stimulus_codes: np.ndarray  # int8     (n_epochs,)  1-12
    char_indices: np.ndarray    # int16    (n_epochs,)  0-indexed character number
    n_chars: int
    subject_id: str

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def n_p300(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def n_non_p300(self) -> int:
        return int((self.labels == 0).sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_continuous(mat_path: Path | str, subject_id: str = "") -> ContinuousData:
    """
    Load a .mat file and return raw continuous per-character arrays.

    This is the preferred entry-point for the training pipeline.
    Pass the result to preprocessing.preprocess() to filter-then-epoch
    in the correct order (Shukla et al., Sec. II-B).

    Args:
        mat_path:   Path to Subject_{A,B}_{Train,Test}.mat
        subject_id: Human-readable label (e.g. "A_train").

    Returns:
        ContinuousData with signal (n_chars, n_time, N_CHANNELS) and
        matching stimulus arrays.
    """
    mat_path = Path(mat_path)
    raw = _loadmat(mat_path)

    signal = np.asarray(raw["Signal"], dtype=np.float32)
    stim_code = np.asarray(raw["StimulusCode"], dtype=np.int32)
    stim_type_raw = np.asarray(
        raw.get("StimulusType", np.zeros_like(stim_code)), dtype=np.int32
    )
    has_labels = bool(stim_type_raw.any())

    signal, stim_code, stim_type_raw = _ensure_3d(signal, stim_code, stim_type_raw)

    return ContinuousData(
        signal=signal,
        stimulus_code=stim_code,
        stimulus_type=stim_type_raw,
        n_chars=signal.shape[0],
        subject_id=subject_id,
        has_labels=has_labels,
    )


def epoch_continuous(cont: ContinuousData) -> EpochData:
    """
    Extract epochs from a ContinuousData object (no filtering).

    Shared epoching logic used by both load_subject() and
    preprocessing.preprocess().  Call preprocessing.preprocess() instead
    of this function when you want filtered epochs.
    """
    epochs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    codes_list: list[int] = []
    char_idx_list: list[int] = []

    n_time = cont.signal.shape[1]

    for c in range(cont.n_chars):
        for onset in _find_onsets(cont.stimulus_code[c]):
            if onset + EPOCH_SAMPLES > n_time:
                continue
            epochs_list.append(cont.signal[c, onset : onset + EPOCH_SAMPLES, :])
            codes_list.append(int(cont.stimulus_code[c, onset]))

            if cont.has_labels:
                win_end = min(onset + _FLASH_ACTIVE_SAMPLES, n_time)
                label = int(cont.stimulus_type[c, onset:win_end].max())
            else:
                label = -1
            labels_list.append(label)
            char_idx_list.append(c)

    return EpochData(
        epochs=np.stack(epochs_list).astype(np.float32),
        labels=np.array(labels_list, dtype=np.int8),
        stimulus_codes=np.array(codes_list, dtype=np.int8),
        char_indices=np.array(char_idx_list, dtype=np.int16),
        n_chars=cont.n_chars,
        subject_id=cont.subject_id,
    )


def load_subject(mat_path: Path | str, subject_id: str = "") -> EpochData:
    """
    Convenience wrapper: load .mat and epoch without filtering.

    For the training pipeline, use preprocessing.preprocess(load_continuous(...))
    so that bandpass filtering happens on the continuous signal before epoching.
    This function is kept for quick inspection and backward compatibility with tests.
    """
    return epoch_continuous(load_continuous(mat_path, subject_id))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _loadmat(path: Path) -> dict:
    """Wrap scipy.io.loadmat with a clear error for HDF5 (v7.3) files."""
    try:
        return scipy.io.loadmat(str(path), squeeze_me=True)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{path.name} appears to be a MATLAB v7.3 (HDF5) file. "
            "Install mat73 (`uv add mat73`) and replace this call with "
            "`mat73.loadmat(path)`."
        ) from exc


def _ensure_3d(
    signal: np.ndarray,
    stim_code: np.ndarray,
    stim_type: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Normalize all arrays to:
        signal     (n_chars, n_time, N_CHANNELS)
        stim_code  (n_chars, n_time)
        stim_type  (n_chars, n_time) or None

    scipy's squeeze_me=True may remove the character axis when n_chars=1,
    producing 2-D arrays.  The channel axis may also be transposed.
    """
    if signal.ndim == 2:
        # (n_time, n_channels) — squeeze_me removed the char axis
        signal = signal[np.newaxis]       # (1, n_time, n_channels)
        stim_code = stim_code.reshape(1, -1)
        if stim_type is not None:
            stim_type = stim_type.reshape(1, -1)

    # Fix transposed channel axis: expect last dim == N_CHANNELS
    if signal.ndim == 3 and signal.shape[-1] != N_CHANNELS:
        if signal.shape[1] == N_CHANNELS:
            signal = signal.transpose(0, 2, 1)  # (n_chars, n_channels, n_time) → correct

    return signal, stim_code, stim_type


def _find_onsets(stimulus_code_1d: np.ndarray) -> np.ndarray:
    """
    Return sample indices where a new flash begins (0→nonzero transition).

    Consecutive nonzero samples belonging to the same flash produce a single
    onset at the first nonzero sample.
    """
    code = stimulus_code_1d.ravel().astype(np.int32)
    padded = np.empty(len(code) + 1, dtype=np.int32)
    padded[0] = 0
    padded[1:] = code
    onset_mask = (padded[:-1] == 0) & (padded[1:] != 0)
    return np.where(onset_mask)[0]


# ---------------------------------------------------------------------------
# CLI smoke test  —  python -m src.data_loader <path/to/Subject_X_Train.mat>
# ---------------------------------------------------------------------------

def _run_smoke_test(mat_path: str, subject_id: str) -> None:
    data = load_subject(mat_path, subject_id)

    print(f"\n=== {data.subject_id or mat_path} ===")
    print(f"Characters   : {data.n_chars}")
    print(f"Total epochs : {data.n_epochs}")
    print(f"Epoch shape  : {data.epochs.shape}")
    print(f"P300 epochs  : {data.n_p300}")
    print(f"Non-P300     : {data.n_non_p300}")
    print(f"Stim codes   : {sorted(set(int(c) for c in data.stimulus_codes))}")
    label_set = sorted(set(int(l) for l in data.labels))
    print(f"Label values : {label_set}  {'(unlabelled test set)' if label_set == [-1] else ''}")

    errors: list[str] = []

    expected_total = data.n_chars * N_REPETITIONS * N_FLASH_CODES
    if data.n_epochs != expected_total:
        errors.append(
            f"Epoch count {data.n_epochs} != expected "
            f"{data.n_chars}×{N_REPETITIONS}×{N_FLASH_CODES}={expected_total}"
        )

    if data.epochs.shape[1:] != (EPOCH_SAMPLES, N_CHANNELS):
        errors.append(
            f"Epoch shape {data.epochs.shape} — expected (*, {EPOCH_SAMPLES}, {N_CHANNELS})"
        )

    if data.n_chars == N_TRAIN_CHARS and data.n_p300 > 0:
        expected_p300 = N_TRAIN_CHARS * N_REPETITIONS * 2  # 1 row + 1 col per rep
        if data.n_p300 != expected_p300:
            errors.append(
                f"P300 count {data.n_p300} != expected {expected_p300} "
                f"({N_TRAIN_CHARS}chars × {N_REPETITIONS}reps × 2 target-flashes)"
            )

    codes_present = set(int(c) for c in data.stimulus_codes)
    expected_codes = set(range(1, N_FLASH_CODES + 1))
    if codes_present != expected_codes:
        errors.append(f"Stim codes present {codes_present} != expected {expected_codes}")

    if errors:
        for e in errors:
            print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll sanity checks PASSED.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Smoke-test the data loader on a BCI Comp III .mat file."
    )
    ap.add_argument("mat_path", help="e.g. data/Subject_A_Train.mat")
    ap.add_argument("--subject-id", default="", help="Label for output, e.g. A_train")
    args = ap.parse_args()
    _run_smoke_test(args.mat_path, args.subject_id)
