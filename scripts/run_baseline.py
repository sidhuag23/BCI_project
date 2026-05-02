"""
WE-SPSQ-CNN baseline reproduction — Shukla et al. 2024.

Trains on a labelled .mat file, optionally evaluates on the test set if a
ground-truth answers file is supplied.

Usage
-----
# Training + resubstitution accuracy only (no test answers needed):
    uv run python scripts/run_baseline.py \\
        --train data/Subject_A_Train.mat --subject A

# Full reproduction (train + test accuracy):
    uv run python scripts/run_baseline.py \\
        --train  data/Subject_A_Train.mat \\
        --test   data/Subject_A_Test.mat \\
        --answers data/Subject_A_answers.txt \\
        --subject A \\
        --checkpoint results/ensemble_A.pt \\
        --save results/baseline_A.csv

Answers file format
-------------------
A plain-text file containing the N ground-truth characters, either as a
single string ("ABCDE...") or one character per line.
Characters must be uppercase; space is written as "_".
(BCI Competition III Dataset II true labels are available from the
competition organisers at http://www.bbci.de/competition/iii/.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import load_continuous
from src.decode import (
    CHAR_MATRIX,
    accuracy_vs_reps,
    true_char_indices_from_epochs,
)
from src.preprocessing import preprocess
from src.train import (
    WEnsemble,
    load_ensemble,
    mixed_augment,
    mixup_augment,
    naive_clone,
    save_ensemble,
    train_ensemble,
)
from src.utils import DEFAULT_SEED, get_device, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_answers(path: Path) -> np.ndarray:
    """
    Load test-set ground-truth characters from a text file.

    Accepts a single string of N chars or one char per line.
    Returns (n_chars,) int16 of 0-indexed grid positions.
    """
    raw = path.read_text(encoding="utf-8").strip()
    chars = [ch for ch in raw.replace("\n", "").replace(" ", "").upper() if ch]

    char_to_idx = {ch: i for i, ch in enumerate(CHAR_MATRIX.ravel())}
    indices = []
    for ch in chars:
        if ch not in char_to_idx:
            raise ValueError(
                f"Unknown character '{ch}' in {path}. "
                f"Valid: {sorted(char_to_idx.keys())}"
            )
        indices.append(char_to_idx[ch])
    return np.array(indices, dtype=np.int16)


def _print_table(
    subject: str,
    train_accs: dict[int, float],
    test_accs: dict[int, float],
    rep_counts: tuple[int, ...],
) -> None:
    have_test = bool(test_accs)
    cols = ["Reps", "Train (resub)"] + (["Test"] if have_test else [])
    w = 16

    print(f"\n{'='*50}")
    print(f"  Subject {subject} — WE-SPSQ-CNN baseline")
    print(f"{'='*50}")
    print("  " + "  ".join(f"{c:>{w}}" for c in cols))
    print("  " + "  ".join("-" * w for _ in cols))
    for r in rep_counts:
        row = [f"{r:>{w}}", f"{train_accs.get(r, float('nan')):>{w-1}.1%}"]
        if have_test:
            row.append(f"{test_accs.get(r, float('nan')):>{w-1}.1%}")
        print("  " + "  ".join(row))

    # Paper reference (Table V, Subject A — Shukla et al. 2024)
    paper_a = {5: 0.765, 10: 0.875, 15: 0.945}
    paper_b = {5: 0.620, 10: 0.800, 15: 0.910}
    paper = paper_a if subject.upper().startswith("A") else paper_b
    print(f"\n  Paper Table V reference (Subject {subject.upper()[0]}):")
    for r in rep_counts:
        if r in paper:
            print(f"    {r:>2} reps: {paper[r]:.1%}")
    print()


def _save_csv(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, accs in [("train_resub", result["train"]), ("test", result["test"])]:
        for r, acc in accs.items():
            rows.append({
                "subject": result["subject"],
                "augment": result.get("augment", "naive"),
                "alpha": result.get("alpha", None),
                "split": split, "reps": r, "accuracy": acc,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    train_path: Path,
    subject: str,
    *,
    test_path: Path | None = None,
    answers_path: Path | None = None,
    augment: str = "naive",
    alpha: float = 0.4,
    mixup_fraction: float = 0.5,
    n_epochs: int = 100,
    n_classifiers: int = 5,
    checkpoint: Path | None = None,
    save_csv: Path | None = None,
    seed: int = DEFAULT_SEED,
    rep_counts: tuple[int, ...] = (5, 10, 15),
) -> dict:
    """Run the full WE-SPSQ-CNN pipeline for one subject."""
    if augment not in ("naive", "mixup", "mixed"):
        raise ValueError(f"augment must be 'naive', 'mixup', or 'mixed', got '{augment}'")

    device = get_device()
    set_seed(seed)
    print(f"Device  : {device}")
    print(f"Subject : {subject}")
    aug_info = {"mixup": f"  alpha={alpha}", "mixed": f"  alpha={alpha}  mixup_fraction={mixup_fraction}"}.get(augment, "")
    print(f"Augment : {augment}{aug_info}")

    # ---- Load and preprocess training data ----------------------------------
    print(f"\nLoading {train_path} ...")
    raw = load_continuous(train_path, subject)
    print(f"Preprocessing {raw.n_chars} characters ...")
    train_data = preprocess(raw)
    print(
        f"Epochs  : {train_data.epochs.shape}  "
        f"P300={train_data.n_p300}  non-P300={train_data.n_non_p300}"
    )

    # ---- Balance / augment classes ------------------------------------------
    rng = np.random.default_rng(seed)
    if augment == "mixup":
        epochs_bal, labels_bal = mixup_augment(
            train_data.epochs, train_data.labels, alpha=alpha, rng=rng
        )
    elif augment == "mixed":
        epochs_bal, labels_bal = mixed_augment(
            train_data.epochs, train_data.labels,
            alpha=alpha, mixup_fraction=mixup_fraction, rng=rng,
        )
    else:
        epochs_bal, labels_bal = naive_clone(train_data.epochs, train_data.labels, rng=rng)
    n_p300_bal = int((labels_bal >= 0.5).sum())
    print(
        f"Balanced: {len(epochs_bal)} epochs  "
        f"(P300-side={n_p300_bal}, non-P300={len(epochs_bal)-n_p300_bal})"
    )

    # ---- Train or load checkpoint -------------------------------------------
    if checkpoint is not None and checkpoint.exists():
        print(f"\nLoading ensemble from {checkpoint} ...")
        ensemble = load_ensemble(checkpoint, device=device)
    else:
        print(
            f"\nTraining {n_classifiers} classifiers × {n_epochs} epochs "
            f"(device={device}) ..."
        )
        ensemble = train_ensemble(
            epochs_bal, labels_bal,
            n_classifiers=n_classifiers,
            n_epochs=n_epochs,
            device=device,
            base_seed=seed,
            verbose=True,
        )
        if checkpoint is not None:
            save_ensemble(ensemble, checkpoint)
            print(f"Ensemble saved to {checkpoint}")

    # ---- Training-set (resubstitution) accuracy -----------------------------
    print("\nComputing training-set probabilities ...")
    train_probs = ensemble.predict_proba(train_data.epochs, device=device)

    true_train = true_char_indices_from_epochs(
        train_data.labels,
        train_data.stimulus_codes,
        train_data.char_indices,
        train_data.n_chars,
    )
    train_accs = accuracy_vs_reps(
        train_probs, train_data.stimulus_codes,
        train_data.char_indices, true_train, train_data.n_chars,
        rep_counts=rep_counts,
    )

    # ---- Test-set accuracy (optional) ---------------------------------------
    test_accs: dict[int, float] = {}
    if test_path is not None:
        if answers_path is None:
            print(
                "\nNOTE: --test provided but no --answers file. "
                "Skipping test-set evaluation."
            )
        else:
            print(f"\nLoading test set {test_path} ...")
            test_raw = load_continuous(test_path, subject + "_test")
            test_data = preprocess(test_raw)
            true_test = _load_answers(answers_path)

            if len(true_test) != test_data.n_chars:
                raise ValueError(
                    f"Answers file has {len(true_test)} entries but test set "
                    f"has {test_data.n_chars} characters."
                )

            print("Computing test-set probabilities ...")
            test_probs = ensemble.predict_proba(test_data.epochs, device=device)
            test_accs = accuracy_vs_reps(
                test_probs, test_data.stimulus_codes,
                test_data.char_indices, true_test, test_data.n_chars,
                rep_counts=rep_counts,
            )

    # ---- Print and optionally save ------------------------------------------
    _print_table(subject, train_accs, test_accs, rep_counts)

    result = {
        "subject": subject, "augment": augment, "alpha": alpha,
        "mixup_fraction": mixup_fraction,
        "train": train_accs, "test": test_accs,
    }
    if save_csv is not None:
        _save_csv(result, save_csv)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="WE-SPSQ-CNN baseline reproduction run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", required=True, type=Path,
                    help="Training .mat file, e.g. data/Subject_A_Train.mat")
    ap.add_argument("--subject", default="",
                    help="Subject label used in output, e.g. A")
    ap.add_argument("--test", type=Path, default=None,
                    help="Test .mat file (optional)")
    ap.add_argument("--answers", type=Path, default=None,
                    help="Plain-text file with test ground-truth characters")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Save/load ensemble here, e.g. results/ensemble_A.pt")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save accuracy CSV here, e.g. results/baseline_A.csv")
    ap.add_argument("--n-epochs", type=int, default=100,
                    help="Training epochs per classifier")
    ap.add_argument("--n-classifiers", type=int, default=5,
                    help="Ensemble size")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--reps", nargs="+", type=int, default=[5, 10, 15],
                    help="Repetition counts to report")
    ap.add_argument("--augment", choices=["naive", "mixup", "mixed"], default="naive",
                    help="Class-balance strategy")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Beta(alpha,alpha) concentration for Mixup/mixed")
    ap.add_argument("--mixup-fraction", type=float, default=0.5,
                    help="Fraction of generated samples from Mixup for 'mixed' strategy")
    args = ap.parse_args()

    run(
        train_path=args.train,
        subject=args.subject,
        test_path=args.test,
        answers_path=args.answers,
        augment=args.augment,
        alpha=args.alpha,
        mixup_fraction=args.mixup_fraction,
        n_epochs=args.n_epochs,
        n_classifiers=args.n_classifiers,
        checkpoint=args.checkpoint,
        save_csv=args.save,
        seed=args.seed,
        rep_counts=tuple(args.reps),
    )
