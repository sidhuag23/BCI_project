"""
Shared utilities: device selection, reproducible seeding, and project-wide constants.
"""

import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device detection: CUDA → MPS → CPU
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device without hardcoding any backend."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Paper-derived constants (Shukla et al. 2024, Sec. II)
# ---------------------------------------------------------------------------

# EEG recording
SAMPLING_RATE: int = 240          # Hz
N_CHANNELS: int = 64
N_ROWS: int = 6
N_COLS: int = 6
N_FLASH_CODES: int = N_ROWS + N_COLS   # 12

# Epoch window: 0-667 ms post-stimulus → 160 samples at 240 Hz
EPOCH_SAMPLES: int = 160          # int(0.667 * 240) = 160
EPOCH_END_MS: float = 667.0

# Speller paradigm counts (per character)
N_REPETITIONS: int = 15
N_EPOCHS_PER_CHAR: int = N_FLASH_CODES * N_REPETITIONS   # 180

# Training set sizes (both subjects)
N_TRAIN_CHARS: int = 85
N_TEST_CHARS: int = 100

# Class imbalance: 1 P300 per flash × 2 target flashes (1 row + 1 col)
# → 2 P300 : 10 Non-P300 per repetition → 1:5 ratio
P300_RATIO: int = 5               # Non-P300 : P300

# Ensemble
N_CLASSIFIERS: int = 5

# Training hyperparameters (Shukla et al., Sec. II-B)
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 32
DROPOUT_P: float = 0.8

# StimulusCode assignment (BCI Comp III Dataset II)
# Codes 1-6 → column intensifications
# Codes 7-12 → row intensifications
COL_CODES: tuple[int, ...] = tuple(range(1, 7))    # 1..6
ROW_CODES: tuple[int, ...] = tuple(range(7, 13))   # 7..12

# Default random seed
DEFAULT_SEED: int = 42
