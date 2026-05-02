"""
Character decoding for the P300 speller (Shukla et al. 2024, Eq. 2-4).

Decoding procedure for each character:
  1. s_k = mean P300 probability over R repetitions for flash code k      (Eq. 2)
  2. col* = argmax_{k in COL_CODES 1..6}   s_k                           (Eq. 3)
  3. row* = argmax_{k in ROW_CODES 7..12}  s_k                           (Eq. 4)
  4. char_idx = row* * N_COLS + col*   (0-indexed position in 6×6 grid)

Entry points:
    from src.decode import decode_characters, accuracy_vs_reps

    decoded = decode_characters(probs, data.stimulus_codes, data.char_indices, data.n_chars)
    accs = accuracy_vs_reps(probs, data.stimulus_codes, data.char_indices,
                            true_indices, data.n_chars)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.utils import (
    COL_CODES,
    N_COLS,
    N_FLASH_CODES,
    N_REPETITIONS,
    N_ROWS,
    ROW_CODES,
)

# ---------------------------------------------------------------------------
# BCI Competition III Dataset II — standard 6×6 character matrix
# Row codes 7-12 → rows 0-5; column codes 1-6 → cols 0-5
# ---------------------------------------------------------------------------
CHAR_MATRIX: np.ndarray = np.array(
    [
        ["A", "B", "C", "D", "E", "F"],
        ["G", "H", "I", "J", "K", "L"],
        ["M", "N", "O", "P", "Q", "R"],
        ["S", "T", "U", "V", "W", "X"],
        ["Y", "Z", "1", "2", "3", "4"],
        ["5", "6", "7", "8", "9", "_"],
    ],
    dtype="<U1",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_characters(
    probs: np.ndarray,
    stimulus_codes: np.ndarray,
    char_indices: np.ndarray,
    n_chars: int,
    max_reps: int | None = None,
) -> np.ndarray:
    """
    Decode one character per entry from per-epoch P300 probabilities.

    For each character:
      s_k  = mean(probs where code == k), first max_reps occurrences  (Eq. 2)
      col* = argmax s_k  for k in COL_CODES                           (Eq. 3)
      row* = argmax s_k  for k in ROW_CODES                           (Eq. 4)

    Args:
        probs:          (n_epochs,) float32 — P300 probability per epoch.
        stimulus_codes: (n_epochs,) int — flash codes 1-12.
        char_indices:   (n_epochs,) int — 0-indexed character id per epoch.
        n_chars:        Number of characters to decode.
        max_reps:       Restrict to first max_reps repetitions (None = all).

    Returns:
        (n_chars,) int16 — decoded 0-indexed character position in the 6×6 grid.
    """
    decoded = np.empty(n_chars, dtype=np.int16)
    codes_int = stimulus_codes.astype(np.int32)

    for c in range(n_chars):
        mask = char_indices == c
        scores = _avg_scores(probs[mask], codes_int[mask], max_reps)

        col_scores = scores[np.array(COL_CODES) - 1]   # indices 0..5
        row_scores = scores[np.array(ROW_CODES) - 1]   # indices 6..11

        best_col = int(np.argmax(col_scores))
        best_row = int(np.argmax(row_scores))
        decoded[c] = best_row * N_COLS + best_col

    return decoded


def accuracy_vs_reps(
    probs: np.ndarray,
    stimulus_codes: np.ndarray,
    char_indices: np.ndarray,
    true_char_indices: np.ndarray,
    n_chars: int,
    rep_counts: Sequence[int] | None = None,
) -> dict[int, float]:
    """
    Character accuracy at each repetition count (for the per-rep accuracy curve).

    Args:
        probs:             (n_epochs,) float32 — P300 probabilities.
        stimulus_codes:    (n_epochs,) int — flash codes 1-12.
        char_indices:      (n_epochs,) int — character id per epoch.
        true_char_indices: (n_chars,) int — ground-truth char positions.
        n_chars:           Number of characters.
        rep_counts:        Repetition counts to evaluate (default 1..N_REPETITIONS).

    Returns:
        {n_reps: accuracy} ordered dict.
    """
    if rep_counts is None:
        rep_counts = range(1, N_REPETITIONS + 1)

    return {
        r: character_accuracy(
            decode_characters(probs, stimulus_codes, char_indices, n_chars, max_reps=r),
            true_char_indices,
        )
        for r in rep_counts
    }


def character_accuracy(
    decoded: np.ndarray,
    true_char_indices: np.ndarray,
) -> float:
    """
    Fraction of correctly decoded characters.

    Args:
        decoded:           (n_chars,) int — output of decode_characters().
        true_char_indices: (n_chars,) int — ground-truth 0-indexed char positions.

    Returns:
        Accuracy in [0, 1].
    """
    return float((decoded == true_char_indices).mean())


def true_char_indices_from_epochs(
    labels: np.ndarray,
    stimulus_codes: np.ndarray,
    char_indices: np.ndarray,
    n_chars: int,
) -> np.ndarray:
    """
    Recover the ground-truth character index for each character from P300 labels.

    For character c, the P300 (label=1) epochs identify which column code (1-6)
    and which row code (7-12) are the targets.  Their intersection gives the
    true character:  char_idx = (row_code - 7) * N_COLS + (col_code - 1).

    Args:
        labels:         (n_epochs,) int — 0/1/-1 per epoch.
        stimulus_codes: (n_epochs,) int — flash codes 1-12.
        char_indices:   (n_epochs,) int — character id per epoch.
        n_chars:        Total number of characters.

    Returns:
        (n_chars,) int16 — 0-indexed character position in the 6×6 grid.

    Raises:
        ValueError: if no P300 target epoch is found for a character.
    """
    result = np.empty(n_chars, dtype=np.int16)
    codes_int = stimulus_codes.astype(np.int32)

    for c in range(n_chars):
        mask = (char_indices == c) & (labels == 1)
        c_codes = codes_int[mask]

        col_hits = c_codes[(c_codes >= 1) & (c_codes <= 6)]
        row_hits = c_codes[(c_codes >= 7) & (c_codes <= 12)]

        if len(col_hits) == 0 or len(row_hits) == 0:
            raise ValueError(
                f"Character {c}: no P300 target found "
                f"(col_hits={len(col_hits)}, row_hits={len(row_hits)}). "
                "This happens on the unlabelled test set (has_labels=False)."
            )

        target_col = int(col_hits[0]) - 1   # 0-indexed column
        target_row = int(row_hits[0]) - 7   # 0-indexed row
        result[c] = target_row * N_COLS + target_col

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _avg_scores(
    probs: np.ndarray,
    codes: np.ndarray,
    max_reps: int | None,
) -> np.ndarray:
    """
    Return (N_FLASH_CODES,) array: scores[k-1] = mean prob for code k,
    using only the first max_reps occurrences of each code.
    """
    scores = np.zeros(N_FLASH_CODES, dtype=np.float32)
    for code in range(1, N_FLASH_CODES + 1):
        p = probs[codes == code]
        if max_reps is not None:
            p = p[:max_reps]
        if len(p) > 0:
            scores[code - 1] = float(p.mean())
    return scores
