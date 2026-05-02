# Mixup-Enhanced WE-SPSQ-CNN for P300 Character Recognition

Reproduction and extension of Shukla et al. (2024) on BCI Competition III Dataset II.
Baseline: Weighted Ensemble SPSQ-CNN. Extension: Mixup augmentation replacing naive P300 cloning.

**Authors:** Ashik M Biju, Sidhu A G, Athira A — Digital University Kerala

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install uv (once)
pip install uv

# Install project dependencies
uv sync

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Place the BCI Competition III Dataset II `.mat` files in `data/`:
```
data/
├── Subject_A_Train.mat
├── Subject_A_Test.mat
├── Subject_B_Train.mat
└── Subject_B_Test.mat
```

## Running

```bash
# Reproduce WE-SPSQ-CNN baseline
python scripts/run_baseline.py

# Run Mixup-augmented experiments
python scripts/run_mixup.py

# Generate comparison tables and plots
python scripts/compare_results.py
```

## Running tests

```bash
pytest
```

## Device support

Automatically selects CUDA (Windows/Linux) → MPS (Apple Silicon) → CPU at runtime.
No configuration needed.

## Project structure

```
src/            Core modules (data loading, preprocessing, model, training, evaluation)
scripts/        Entry-point scripts for each experiment
tests/          pytest unit tests
data/           .mat files (gitignored)
results/        Checkpoints and result CSVs (gitignored)
papers/         Reference PDFs
```

## Reference

P. K. Shukla, H. Cecotti, Y. K. Meena — "Towards Effective Deep Neural Network Approach
for Multi-Trial P300-based Character Recognition in Brain-Computer Interfaces", arXiv 2410.08561, 2024.
