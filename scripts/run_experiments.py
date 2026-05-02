"""
Run all augmentation variants for one or both subjects and produce a
comparison table (paper Table V/VI format).

Runs four variants × up to two subjects; each variant is checkpointed so
re-running the script skips training and goes straight to reporting.

Usage
-----
# Both subjects, training only (no test answers yet):
    uv run python scripts/run_experiments.py \\
        --train-a data/Subject_A_Train.mat \\
        --train-b data/Subject_B_Train.mat \\
        --outdir  results/experiments \\
        --n-epochs 100

# With test-set evaluation (add answers files):
    uv run python scripts/run_experiments.py \\
        --train-a data/Subject_A_Train.mat \\
        --train-b data/Subject_B_Train.mat \\
        --test-a  data/Subject_A_Test.mat --answers-a data/Subject_A_answers.txt \\
        --test-b  data/Subject_B_Test.mat --answers-b data/Subject_B_answers.txt \\
        --outdir  results/experiments \\
        --n-epochs 100

Variants run
------------
  naive       — WE-SPSQ-CNN baseline (naive P300 cloning)
  mixup_02    — Mixup α=0.2
  mixup_04    — Mixup α=0.4  (default for paper experiments)
  mixed_04    — Mixed strategy 50% Mixup / 50% naive, α=0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.run_baseline import run

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

VARIANTS: list[dict] = [
    {
        "key": "naive",
        "label": "Baseline (naive clone)",
        "augment": "naive",
        "alpha": 0.4,
        "mixup_fraction": 0.5,
    },
    {
        "key": "mixup_02",
        "label": "Mixup alpha=0.2",
        "augment": "mixup",
        "alpha": 0.2,
        "mixup_fraction": 0.5,
    },
    {
        "key": "mixup_04",
        "label": "Mixup alpha=0.4",
        "augment": "mixup",
        "alpha": 0.4,
        "mixup_fraction": 0.5,
    },
    {
        "key": "mixed_04",
        "label": "Mixed 50% Mixup alpha=0.4",
        "augment": "mixed",
        "alpha": 0.4,
        "mixup_fraction": 0.5,
    },
]

PAPER_REF: dict[str, dict[int, float]] = {
    "A": {5: 0.765, 10: 0.875, 15: 0.945},
    "B": {5: 0.620, 10: 0.800, 15: 0.910},
}

REP_COUNTS = (5, 10, 15)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(
    subjects: list[dict],
    outdir: Path,
    n_epochs: int,
    n_classifiers: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run every variant for every subject; return combined results DataFrame.

    subjects: list of dicts with keys train_path, subject, test_path (opt), answers_path (opt).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for subj in subjects:
        subj_key = subj["subject"]
        print(f"\n{'#'*60}")
        print(f"# Subject {subj_key}")
        print(f"{'#'*60}")

        for var in VARIANTS:
            ckpt = outdir / f"ens_{subj_key}_{var['key']}.pt"
            csv_path = outdir / f"{subj_key}_{var['key']}.csv"

            print(f"\n--- {var['label']} ---")
            result = run(
                train_path=subj["train_path"],
                subject=subj_key,
                test_path=subj.get("test_path"),
                answers_path=subj.get("answers_path"),
                augment=var["augment"],
                alpha=var["alpha"],
                mixup_fraction=var["mixup_fraction"],
                n_epochs=n_epochs,
                n_classifiers=n_classifiers,
                checkpoint=ckpt,
                save_csv=csv_path,
                seed=seed,
                rep_counts=REP_COUNTS,
            )

            for split in ("train", "test"):
                for r, acc in result[split].items():
                    all_rows.append({
                        "subject": subj_key,
                        "variant": var["key"],
                        "label": var["label"],
                        "split": split,
                        "reps": r,
                        "accuracy": acc,
                    })

    df = pd.DataFrame(all_rows)
    combined_path = outdir / "all_results.csv"
    df.to_csv(combined_path, index=False)
    print(f"\nAll results saved to {combined_path}")
    return df


def print_comparison_table(df: pd.DataFrame, split: str = "train") -> None:
    """Print a paper-style comparison table for the given split."""
    sub_df = df[df["split"] == split]
    if sub_df.empty:
        return

    subjects = sorted(sub_df["subject"].unique())
    variants = [v["key"] for v in VARIANTS]
    variant_labels = {v["key"]: v["label"] for v in VARIANTS}

    col_w = 10
    label_w = 26

    for subj in subjects:
        print(f"\n{'='*70}")
        print(f"  Subject {subj} — {split} accuracy")
        print(f"{'='*70}")

        # Header
        header = f"  {'Method':<{label_w}}" + "".join(f"  {r:>{col_w}} rep" for r in REP_COUNTS)
        print(header)
        print("  " + "-" * (label_w + (col_w + 6) * len(REP_COUNTS)))

        # Paper reference row
        paper = PAPER_REF.get(subj.upper()[0], {})
        if paper:
            row = f"  {'Paper (Shukla 2024)':<{label_w}}"
            for r in REP_COUNTS:
                row += f"  {paper.get(r, float('nan')):>{col_w}.1%}"
            print(row)

        # Our results
        for vk in variants:
            vrows = sub_df[(sub_df["subject"] == subj) & (sub_df["variant"] == vk)]
            if vrows.empty:
                continue
            label = variant_labels[vk]
            row = f"  {label:<{label_w}}"
            for r in REP_COUNTS:
                acc = vrows[vrows["reps"] == r]["accuracy"]
                val = float(acc.iloc[0]) if not acc.empty else float("nan")
                row += f"  {val:>{col_w}.1%}"
            print(row)

        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run all augmentation variants and produce comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train-a", type=Path, default=None,
                    help="Subject A training .mat file")
    ap.add_argument("--train-b", type=Path, default=None,
                    help="Subject B training .mat file")
    ap.add_argument("--test-a", type=Path, default=None)
    ap.add_argument("--answers-a", type=Path, default=None,
                    help="Subject A test answers text file")
    ap.add_argument("--test-b", type=Path, default=None)
    ap.add_argument("--answers-b", type=Path, default=None,
                    help="Subject B test answers text file")
    ap.add_argument("--outdir", type=Path, default=Path("results/experiments"),
                    help="Output directory for checkpoints and CSVs")
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--n-classifiers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="Which split to show in comparison table")
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

    df = run_all(
        subjects=subjects,
        outdir=args.outdir,
        n_epochs=args.n_epochs,
        n_classifiers=args.n_classifiers,
        seed=args.seed,
    )

    print_comparison_table(df, split=args.split)
    if args.split == "train" and "test" in df["split"].values:
        print_comparison_table(df, split="test")
