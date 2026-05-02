"""
Generate per-repetition accuracy curves and comparison tables from trained
ensemble checkpoints.

Evaluates each checkpoint at repetitions 1-15, then produces:
  • accuracy_curves_{subject}_{split}.png  — accuracy vs. n_reps plot
  • comparison_table.txt                  — Table V/VI-style numbers
  • sanity_report.txt                     — baseline vs. paper ±3% check

Usage
-----
# Training-set evaluation only (no test answers needed):
    uv run python scripts/plot_results.py \\
        --train-a data/Subject_A_Train.mat \\
        --train-b data/Subject_B_Train.mat \\
        --results-dir results/experiments \\
        --outdir results/figures

# With test-set evaluation:
    uv run python scripts/plot_results.py \\
        --train-a  data/Subject_A_Train.mat \\
        --test-a   data/Subject_A_Test.mat \\
        --answers-a data/Subject_A_answers.txt \\
        --train-b  data/Subject_B_Train.mat \\
        --test-b   data/Subject_B_Test.mat \\
        --answers-b data/Subject_B_answers.txt \\
        --results-dir results/experiments \\
        --outdir results/figures

Checkpoints are expected at:
    {results_dir}/ens_{subject}_{variant_key}.pt
e.g. results/experiments/ens_A_naive.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_continuous
from src.decode import (
    CHAR_MATRIX,
    accuracy_vs_reps,
    true_char_indices_from_epochs,
)
from src.preprocessing import preprocess
from src.train import load_ensemble
from src.utils import N_REPETITIONS, get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = [
    {"key": "naive",    "label": "Baseline (naive clone)",   "color": "steelblue",   "ls": "-"},
    {"key": "mixup_02", "label": "Mixup alpha=0.2",          "color": "darkorange",  "ls": "--"},
    {"key": "mixup_04", "label": "Mixup alpha=0.4",          "color": "forestgreen", "ls": "-."},
    {"key": "mixed_04", "label": "Mixed 50% Mixup alpha=0.4","color": "crimson",     "ls": ":"},
]

PAPER_REF: dict[str, dict[int, float]] = {
    "A": {5: 0.765, 10: 0.875, 15: 0.945},
    "B": {5: 0.620, 10: 0.800, 15: 0.910},
}

ALL_REPS = list(range(1, N_REPETITIONS + 1))
REPORT_REPS = [5, 10, 15]
TOLERANCE = 0.03   # ±3 pp for sanity check

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_answers(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8").strip()
    chars = [ch for ch in raw.replace("\n", "").replace(" ", "").upper() if ch]
    char_to_idx = {ch: i for i, ch in enumerate(CHAR_MATRIX.ravel())}
    return np.array([char_to_idx[ch] for ch in chars], dtype=np.int16)


def load_subject_data(
    train_path: Path,
    subject: str,
    test_path: Path | None = None,
    answers_path: Path | None = None,
) -> dict:
    """Preprocess train (and optionally test) data for one subject."""
    print(f"  Loading {train_path} ...")
    train_raw = load_continuous(train_path, subject)
    train_data = preprocess(train_raw)
    true_train = true_char_indices_from_epochs(
        train_data.labels, train_data.stimulus_codes,
        train_data.char_indices, train_data.n_chars,
    )
    result = {"train": (train_data, true_train)}

    if test_path is not None and answers_path is not None:
        print(f"  Loading {test_path} ...")
        test_raw = load_continuous(test_path, subject + "_test")
        test_data = preprocess(test_raw)
        true_test = _load_answers(answers_path)
        result["test"] = (test_data, true_test)

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_checkpoint(
    ckpt_path: Path,
    data: object,
    true_chars: np.ndarray,
    device,
) -> dict[int, float]:
    """Load checkpoint and return accuracy at every rep count 1-15."""
    ens = load_ensemble(ckpt_path, device=device)
    probs = ens.predict_proba(data.epochs, device=device)
    return accuracy_vs_reps(
        probs, data.stimulus_codes, data.char_indices,
        true_chars, data.n_chars,
        rep_counts=ALL_REPS,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_curves(
    results: dict[str, dict[int, float]],
    subject: str,
    split: str,
    paper_ref: dict[int, float],
    save_path: Path,
) -> None:
    """
    Plot accuracy vs. repetition count for all variants.

    results: {variant_key: {n_reps: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Paper reference — dashed grey line through known points
    if paper_ref:
        rep_pts = sorted(paper_ref)
        acc_pts = [paper_ref[r] * 100 for r in rep_pts]
        ax.plot(rep_pts, acc_pts, "k--", lw=1.5, label="Paper (Shukla 2024)",
                marker="D", markersize=5, zorder=5)

    # Our variants
    for var in VARIANTS:
        key = var["key"]
        if key not in results:
            continue
        accs = [results[key].get(r, float("nan")) * 100 for r in ALL_REPS]
        ax.plot(ALL_REPS, accs,
                color=var["color"], ls=var["ls"], lw=1.8,
                marker="o", markersize=3, label=var["label"])

    ax.set_xlabel("Number of repetitions")
    ax.set_ylabel("Character accuracy (%)")
    ax.set_title(f"Subject {subject} — {split.capitalize()} accuracy vs. repetitions")
    ax.set_xlim(0.5, N_REPETITIONS + 0.5)
    ax.set_xticks(ALL_REPS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------

def format_comparison_table(
    all_results: dict[str, dict[str, dict[int, float]]],
) -> str:
    """
    Format a Table V/VI-style comparison for all subjects.

    all_results: {subject: {variant_key: {n_reps: accuracy}}}
    """
    variant_labels = {v["key"]: v["label"] for v in VARIANTS}
    col_w = 9
    label_w = 28
    lines = []

    for subject, subj_results in sorted(all_results.items()):
        lines.append(f"\nSubject {subject}")
        lines.append("=" * (label_w + (col_w + 2) * len(REPORT_REPS) + 2))
        hdr = f"  {'Method':<{label_w}}" + "".join(f"  {r:>{col_w}}r" for r in REPORT_REPS)
        lines.append(hdr)
        lines.append("  " + "-" * (label_w + (col_w + 2) * len(REPORT_REPS)))

        paper = PAPER_REF.get(subject.upper()[0], {})
        if paper:
            row = f"  {'Paper (Shukla 2024)':<{label_w}}"
            row += "".join(
                f"  {paper.get(r, float('nan')):>{col_w}.1%}" for r in REPORT_REPS
            )
            lines.append(row)

        for var in VARIANTS:
            key = var["key"]
            if key not in subj_results:
                continue
            accs = subj_results[key]
            row = f"  {variant_labels[key]:<{label_w}}"
            row += "".join(
                f"  {accs.get(r, float('nan')):>{col_w}.1%}" for r in REPORT_REPS
            )
            lines.append(row)

    return "\n".join(lines)


def format_sanity_report(
    all_results: dict[str, dict[str, dict[int, float]]],
) -> str:
    """
    Check whether our baseline (naive clone) is within ±3 pp of the paper.
    """
    lines = [
        "Reproduction Sanity Report",
        "=" * 50,
        f"Criterion: baseline within +/-{TOLERANCE*100:.0f} pp of paper (Shukla 2024)",
    ]

    overall_pass = True
    for subject, subj_results in sorted(all_results.items()):
        paper = PAPER_REF.get(subject.upper()[0], {})
        naive_accs = subj_results.get("naive", {})
        if not naive_accs or not paper:
            continue

        lines.append(f"\nSubject {subject}:")
        lines.append(f"  {'Reps':>5}  {'Paper':>7}  {'Ours':>7}  {'Delta':>7}  Pass?")
        lines.append("  " + "-" * 42)

        subj_pass = True
        for r in REPORT_REPS:
            p = paper.get(r, float("nan"))
            o = naive_accs.get(r, float("nan"))
            delta = o - p
            passed = abs(delta) <= TOLERANCE
            if not passed:
                subj_pass = False
                overall_pass = False
            tick = "PASS" if passed else "FAIL"
            lines.append(
                f"  {r:>5}  {p:>7.1%}  {o:>7.1%}  {delta:>+7.1%}  {tick}"
            )

        lines.append(f"  Subject {subject}: {'PASS' if subj_pass else 'FAIL'}")

    lines.append("\n" + ("=" * 50))
    lines.append(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    subjects: list[dict],
    results_dir: Path,
    outdir: Path,
    splits: list[str],
) -> None:
    device = get_device()
    outdir.mkdir(parents=True, exist_ok=True)

    all_train: dict[str, dict[str, dict[int, float]]] = {}
    all_test: dict[str, dict[str, dict[int, float]]] = {}

    for subj in subjects:
        s = subj["subject"]
        print(f"\nSubject {s}")
        print("-" * 40)
        subj_data = load_subject_data(
            subj["train_path"], s,
            subj.get("test_path"), subj.get("answers_path"),
        )

        all_train[s] = {}
        all_test[s] = {}

        for var in VARIANTS:
            ckpt = results_dir / f"ens_{s}_{var['key']}.pt"
            if not ckpt.exists():
                print(f"  [{var['key']}] checkpoint not found: {ckpt} — skipping")
                continue
            print(f"  Evaluating {var['key']} ...")

            train_data, true_train = subj_data["train"]
            all_train[s][var["key"]] = eval_checkpoint(ckpt, train_data, true_train, device)

            if "test" in subj_data:
                test_data, true_test = subj_data["test"]
                all_test[s][var["key"]] = eval_checkpoint(ckpt, test_data, true_test, device)

        # Plots
        for split, split_results in [("train", all_train), ("test", all_test)]:
            if split not in splits:
                continue
            if not split_results.get(s):
                continue
            plot_accuracy_curves(
                split_results[s], s, split,
                paper_ref=PAPER_REF.get(s.upper()[0], {}),
                save_path=outdir / f"accuracy_curves_{s}_{split}.png",
            )

    # Comparison table
    for split, split_results, fname in [
        ("train", all_train, "comparison_table_train.txt"),
        ("test",  all_test,  "comparison_table_test.txt"),
    ]:
        if split not in splits:
            continue
        if not any(split_results.values()):
            continue
        table = format_comparison_table(split_results)
        print(table)
        path = outdir / fname
        path.write_text(table, encoding="utf-8")
        print(f"\nSaved {path}")

    # Sanity report (always from train split — that's what we measure baseline on)
    if all_train:
        report = format_sanity_report(all_train)
        print("\n" + report)
        rpath = outdir / "sanity_report.txt"
        rpath.write_text(report, encoding="utf-8")
        print(f"Saved {rpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot per-repetition accuracy curves from trained checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train-a", type=Path, default=None)
    ap.add_argument("--train-b", type=Path, default=None)
    ap.add_argument("--test-a",  type=Path, default=None)
    ap.add_argument("--answers-a", type=Path, default=None)
    ap.add_argument("--test-b",  type=Path, default=None)
    ap.add_argument("--answers-b", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results/experiments"),
                    help="Directory containing ens_{subject}_{variant}.pt checkpoints")
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"),
                    help="Output directory for plots and text reports")
    ap.add_argument("--splits", nargs="+", choices=["train", "test"],
                    default=["train"],
                    help="Which splits to plot (train = resubstitution)")
    args = ap.parse_args()

    subjects = []
    if args.train_a is not None:
        subjects.append({
            "subject": "A",
            "train_path": args.train_a,
            "test_path": args.test_a,
            "answers_path": args.answers_a,
        })
    if args.train_b is not None:
        subjects.append({
            "subject": "B",
            "train_path": args.train_b,
            "test_path": args.test_b,
            "answers_path": args.answers_b,
        })

    if not subjects:
        ap.error("Provide at least one of --train-a / --train-b.")

    run(
        subjects=subjects,
        results_dir=args.results_dir,
        outdir=args.outdir,
        splits=args.splits,
    )
