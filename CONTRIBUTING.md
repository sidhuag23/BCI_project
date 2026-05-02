# Contributing / Teammate Setup

## First-time setup

1. **Clone the repo**

   ```bash
   git clone <repo-url>
   cd bci-p300-mixup
   ```

2. **Install uv** (if not already installed)

   ```bash
   pip install uv
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

   - On Windows, uv will pull PyTorch with CUDA 12.1 from the PyTorch wheel index.
   - On macOS (including Apple Silicon M-series), uv pulls standard PyPI torch, which
     includes MPS support. No extra steps needed.

4. **Add data files**

   Place the BCI Competition III Dataset II `.mat` files in `data/` (not committed to git):
   ```
   data/Subject_A_Train.mat
   data/Subject_A_Test.mat
   data/Subject_B_Train.mat
   data/Subject_B_Test.mat
   ```

5. **Verify setup**

   ```bash
   # Run tests (should pass even without data files)
   uv run pytest

   # Check that your device is detected correctly
   uv run python -c "from src.utils import get_device; print(get_device())"
   ```

## Day-to-day workflow

- Run any script with `uv run python scripts/<script>.py` or activate the venv first.
- The same `python train.py` command works on both machines — device is auto-detected.
- Never commit files in `data/` or `results/`.

## Code conventions

- Type hints on all function signatures.
- No magic numbers — all paper-derived constants live in `src/utils.py`.
- Run `pytest` before opening a PR.
