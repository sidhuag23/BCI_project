"""
Generate publication-quality PDF figures for the BCI P300 Mixup report.

Outputs (report_figures/):
  character_accuracy_vs_repetitions.pdf
  roc_curves.pdf
  alpha_ablation.pdf
  confusion_matrices.pdf
  sota_comparison.pdf
  mixup_signal_example.pdf
  training_loss_curves.pdf  <- SKIPPED (loss not logged)

Also writes IMPLEMENTATION_SUMMARY.md in the project root.

ROC / confusion matrices use the TRAINING set (only labeled data available).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_continuous
from src.decode import CHAR_MATRIX, accuracy_vs_reps, true_char_indices_from_epochs
from src.preprocessing import preprocess
from src.train import load_ensemble
from src.utils import EPOCH_SAMPLES, SAMPLING_RATE, N_REPETITIONS, get_device

# ── Constants ────────────────────────────────────────────────────────────────

OUTDIR = Path("report_figures")
RESULTS_DIR = Path("results/experiments")
ALL_REPS = list(range(1, N_REPETITIONS + 1))
PAPER_REF = {"A": {5: 76.5, 10: 87.5, 15: 94.5}, "B": {5: 62.0, 10: 80.0, 15: 91.0}}

VARIANTS = [
    {"key": "naive",    "label": "Baseline (Naive Clone)", "color": "#2166ac"},
    {"key": "mixup_02", "label": "Mixup α=0.2",           "color": "#d6604d"},
    {"key": "mixup_04", "label": "Mixup α=0.4",           "color": "#228B22"},
    {"key": "mixed_04", "label": "Mixed 50% (Ours)",       "color": "#7B2D8B"},
]

SAVE_KW = {"bbox_inches": "tight", "dpi": 300}


# ── Style ────────────────────────────────────────────────────────────────────

def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "savefig.facecolor": "white",
    })


# ── Utilities ────────────────────────────────────────────────────────────────

def _load_answers(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8").strip()
    chars = [c for c in raw.replace("\n", "").replace(" ", "").upper() if c]
    c2i = {c: i for i, c in enumerate(CHAR_MATRIX.ravel())}
    return np.array([c2i[c] for c in chars], dtype=np.int16)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    """ROC curve and AUC without sklearn."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    idx = np.argsort(-y_score, kind="stable")
    y_s = y_true[idx]
    sc_s = y_score[idx]

    distinct = np.where(np.diff(sc_s) != 0)[0]
    t_idx = np.r_[distinct, len(y_s) - 1]

    cum_tp = np.cumsum(y_s)[t_idx]
    cum_fp = (1 + t_idx) - cum_tp

    tpr = np.r_[0.0, cum_tp / n_pos, 1.0]
    fpr = np.r_[0.0, cum_fp / n_neg, 1.0]

    # Ensure fpr is monotone for trapz
    order = np.argsort(fpr, kind="stable")
    fpr, tpr = fpr[order], tpr[order]
    auc_val = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc_val


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]]."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_data() -> dict:
    cfg = {
        "A": ("data/Subject_A_Train.mat", "data/Subject_A_Test.mat", "data/Subject_A_answers.txt"),
        "B": ("data/Subject_B_Train.mat", "data/Subject_B_Test.mat", "data/Subject_B_answers.txt"),
    }
    result = {}
    for subj, (tr, te, ans) in cfg.items():
        print(f"  [{subj}] loading train …")
        train_data = preprocess(load_continuous(tr, subj))
        true_train = true_char_indices_from_epochs(
            train_data.labels, train_data.stimulus_codes,
            train_data.char_indices, train_data.n_chars,
        )
        print(f"  [{subj}] loading test …")
        test_data = preprocess(load_continuous(te, subj + "_test"))
        true_test = _load_answers(Path(ans))
        result[subj] = {
            "train": train_data, "true_train": true_train,
            "test": test_data,  "true_test": true_test,
        }
    return result


def compute_all_metrics(all_data: dict, device) -> dict:
    """Load each checkpoint; compute predictions, accuracy curves, ROC, CM."""
    metrics: dict = {}
    for subj, data in all_data.items():
        metrics[subj] = {}
        for var in VARIANTS:
            key = var["key"]
            ckpt = RESULTS_DIR / f"ens_{subj}_{key}.pt"
            if not ckpt.exists():
                print(f"  SKIP {ckpt}")
                continue
            print(f"  [{subj}/{key}] inference …", flush=True)
            ens = load_ensemble(ckpt, device=device)

            tr = all_data[subj]["train"]
            te = all_data[subj]["test"]

            tr_probs = ens.predict_proba(tr.epochs, device=device)
            te_probs = ens.predict_proba(te.epochs, device=device)

            tr_accs = accuracy_vs_reps(tr_probs, tr.stimulus_codes, tr.char_indices,
                                       all_data[subj]["true_train"], tr.n_chars)
            te_accs = accuracy_vs_reps(te_probs, te.stimulus_codes, te.char_indices,
                                       all_data[subj]["true_test"], te.n_chars)

            y_ep = tr.labels.astype(float)
            fpr, tpr, auc_val = _roc_auc(y_ep, tr_probs)

            y_pred_ep = (tr_probs >= 0.5).astype(int)
            cm = _confusion_matrix(y_ep, y_pred_ep)
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            total = tn + fp + fn + tp
            acc_ep  = (tp + tn) / total
            prec_ep = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_ep  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_ep   = 2 * prec_ep * rec_ep / (prec_ep + rec_ep) if (prec_ep + rec_ep) > 0 else 0.0

            metrics[subj][key] = {
                "tr_probs": tr_probs, "te_probs": te_probs,
                "tr_accs": tr_accs,   "te_accs": te_accs,
                "fpr": fpr, "tpr": tpr, "auc": auc_val,
                "cm": cm,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "acc_ep": acc_ep, "prec": prec_ep,
                "rec": rec_ep,    "f1": f1_ep,
            }
    return metrics


# ── Plot 1 ───────────────────────────────────────────────────────────────────

def plot1(metrics: dict, all_data: dict) -> str:
    print("\nPlot 1: character_accuracy_vs_repetitions.pdf")
    fig, ax = plt.subplots(figsize=(10, 6))

    bl_curves, mx_curves = [], []

    for subj, (subj_color_bl, subj_color_mx) in [
        ("A", ("#2166ac", "#d73027")),
        ("B", ("#74add1", "#fc8d59")),
    ]:
        if "naive" not in metrics[subj] or "mixup_04" not in metrics[subj]:
            continue
        bl = [metrics[subj]["naive"]["te_accs"][r] * 100 for r in ALL_REPS]
        mx = [metrics[subj]["mixup_04"]["te_accs"][r] * 100 for r in ALL_REPS]
        bl_curves.append(bl)
        mx_curves.append(mx)

        ax.plot(ALL_REPS, bl, color=subj_color_bl, lw=1.6, marker="o", ms=3,
                label=f"Baseline Subject {subj}")
        ax.plot(ALL_REPS, mx, color=subj_color_mx, lw=1.6, marker="s", ms=3,
                label=f"Mixup α=0.4 Subject {subj}")

    if len(bl_curves) == 2:
        bm = np.mean(bl_curves, axis=0)
        mm = np.mean(mx_curves, axis=0)
        ax.plot(ALL_REPS, bm, color="#2166ac", lw=2.8, ls="--", label="Baseline Mean", zorder=5)
        ax.plot(ALL_REPS, mm, color="#d73027", lw=2.8, ls="-",  label="Mixup α=0.4 Mean", zorder=5)

    ax.set_xlabel("Number of repetitions")
    ax.set_ylabel("Character recognition accuracy (%)")
    ax.set_title("Character recognition accuracy across repetitions")
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(ALL_REPS)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    out = OUTDIR / "character_accuracy_vs_repetitions.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── Plot 2 ───────────────────────────────────────────────────────────────────

def plot2(metrics: dict) -> str:
    print("\nPlot 2: roc_curves.pdf")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, subj in zip(axes, ["A", "B"]):
        ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1.2, ls="--", label="Random (AUC=0.500)")
        for var in [
            {"key": "naive",    "label": "Baseline",   "color": "#2166ac"},
            {"key": "mixup_04", "label": "Mixup α=0.4","color": "#d73027"},
        ]:
            k = var["key"]
            if k not in metrics[subj]:
                continue
            m = metrics[subj][k]
            ax.plot(m["fpr"], m["tpr"], color=var["color"], lw=2.0,
                    label=f"{var['label']} (AUC={m['auc']:.3f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Subject {subj}")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.legend(loc="lower right")

    fig.suptitle("ROC curves: Baseline vs Mixup\n"
                 "(single-trial P300 detection, training-set resubstitution)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTDIR / "roc_curves.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── Plot 3 ───────────────────────────────────────────────────────────────────

def plot3(metrics: dict, all_data: dict) -> str:
    print("\nPlot 3: alpha_ablation.pdf")
    strategies = [
        ("naive",    "Baseline\n(α→0)"),
        ("mixup_02", "Mixup\nα=0.2"),
        ("mixup_04", "Mixup\nα=0.4"),
        ("mixed_04", "Mixed 50%\nα=0.4"),
    ]
    rep_list = [5, 10, 15]
    rep_colors = ["#9ecae1", "#4292c6", "#08519c"]
    bar_w = 0.22
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, subj in zip(axes, ["A", "B"]):
        for i, (r, col) in enumerate(zip(rep_list, rep_colors)):
            vals = []
            for key, _ in strategies:
                if key in metrics[subj]:
                    vals.append(metrics[subj][key]["te_accs"].get(r, float("nan")) * 100)
                else:
                    vals.append(float("nan"))
            bars = ax.bar(x + (i - 1) * bar_w, vals, bar_w * 0.9,
                          label=f"{r} reps", color=col, alpha=0.9)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                            f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([s[1] for s in strategies], fontsize=8.5)
        ax.set_xlabel("Augmentation strategy")
        ax.set_ylabel("Character accuracy (%)" if subj == "A" else "")
        ax.set_title(f"Subject {subj}")
        ax.set_ylim(0, 112)
        if subj == "A":
            ax.legend(fontsize=9)

    fig.suptitle("Effect of Mixup alpha on character recognition accuracy", fontsize=13)
    plt.tight_layout()
    out = OUTDIR / "alpha_ablation.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── Plot 4 ───────────────────────────────────────────────────────────────────

def plot4(metrics: dict) -> str:
    print("\nPlot 4: confusion_matrices.pdf")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    col_labels = ["Non-P300", "P300"]

    for row, subj in enumerate(["A", "B"]):
        for col, (key, title) in enumerate([
            ("naive",    "Baseline (Naive Clone)"),
            ("mixup_04", "Mixup α=0.4"),
        ]):
            ax = axes[row][col]
            if key not in metrics[subj]:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            cm = metrics[subj][key]["cm"]
            total = cm.sum()
            thresh = cm.max() / 2.0

            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, aspect="auto",
                           vmin=0, vmax=cm.max())

            cell_labels = [["TN", "FP"], ["FN", "TP"]]
            for ii in range(2):
                for jj in range(2):
                    v = cm[ii, jj]
                    pct = v / total * 100
                    c = "white" if v > thresh else "#222222"
                    ax.text(jj, ii, f"{cell_labels[ii][jj]}\n{v:,}\n({pct:.1f}%)",
                            ha="center", va="center", fontsize=9, color=c, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(col_labels, fontsize=9)
            ax.set_yticklabels(col_labels, fontsize=9)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(f"Subject {subj} — {title}", fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("P300 detection confusion matrices\n(epoch-level, training-set resubstitution)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTDIR / "confusion_matrices.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── Plot 6 ───────────────────────────────────────────────────────────────────

def plot6(metrics: dict) -> str:
    print("\nPlot 6: sota_comparison.pdf")

    # Mean across subjects for our methods
    def _mean(key, r):
        vals = [metrics[s].get(key, {}).get("te_accs", {}).get(r, float("nan")) * 100
                for s in ["A", "B"]]
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")

    # (label, acc_5r, acc_10r, acc_15r, is_ours)
    methods = [
        ("ESVM (Rakotomamonjy 2008)",       None,  None,  96.5, False),
        ("FLD Ensemble (Salvaris 2009)",     None,  None,  96.5, False),
        ("CNN (Cecotti 2011)",               None,  88.5,  88.5, False),
        ("ST-CapsNet (Wang 2023)",           None,  None,  98.0, False),
        ("WE-SPSQ-CNN (Shukla 2024)",       69.25, 83.75, 92.75, False),
        ("Ours — Baseline",                 _mean("naive",    5), _mean("naive",   10), _mean("naive",   15), True),
        ("Ours — Mixed 50% (Proposed)",     _mean("mixed_04", 5), _mean("mixed_04",10), _mean("mixed_04",15), True),
    ]

    rep_labels = ["5 reps", "10 reps", "15 reps"]
    rep_colors = ["#9ecae1", "#4292c6", "#08519c"]
    bar_h = 0.22
    n = len(methods)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (rl, rc) in enumerate(zip(rep_labels, rep_colors)):
        vals = [m[i + 1] for m in methods]
        pos = y + (i - 1) * bar_h
        bars = ax.barh(pos, [v if v is not None else 0 for v in vals],
                       bar_h * 0.9, label=rl, color=rc, alpha=0.9)
        for bar, val in zip(bars, vals):
            if val is not None and not np.isnan(val):
                ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=7.5)

    # Highlight our methods
    our_idx = [i for i, m in enumerate(methods) if m[4]]
    for i in our_idx:
        ax.axhspan(i - 0.45, i + 0.45, alpha=0.07, color="#d73027", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in methods], fontsize=9)
    ax.set_xlabel("Character recognition accuracy (%)")
    ax.set_title("Comparison with state-of-the-art at 5, 10, and 15 repetitions\n"
                 "(BCI Competition III Dataset II, mean across subjects where available)",
                 fontsize=12)
    ax.set_xlim(0, 108)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    out = OUTDIR / "sota_comparison.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── Plot 7 ───────────────────────────────────────────────────────────────────

def plot7(all_data: dict) -> str:
    print("\nPlot 7: mixup_signal_example.pdf")
    tr = all_data["A"]["train"]

    p300_idx = np.where(tr.labels == 1)[0]
    non_idx  = np.where(tr.labels == 0)[0]

    rng = np.random.default_rng(42)
    x1 = tr.epochs[rng.choice(p300_idx)]   # (160, 64) P300
    x2 = tr.epochs[rng.choice(non_idx)]    # (160, 64) Non-P300
    lam = 0.7
    xm = lam * x1 + (1 - lam) * x2

    # Channel with highest variance across P300 epochs (proxy for Pz)
    # var(axis=(0,1)) collapses over epochs and time → shape (64,)
    ch = int(np.argmax(tr.epochs[p300_idx].var(axis=(0, 1))))
    t_ms = np.arange(EPOCH_SAMPLES) / SAMPLING_RATE * 1000

    titles = [
        (x1[:, ch], f"P300 epoch x1  (y1 = 1.0, ch {ch})", "#2166ac"),
        (x2[:, ch], f"Non-P300 epoch x2  (y2 = 0.0, ch {ch})", "#d6604d"),
        (xm[:, ch], f"Mixup blend  lam*x1 + (1-lam)*x2   lam={lam},  soft label={lam:.1f}", "#228B22"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (sig, lab, col) in zip(axes, titles):
        ax.plot(t_ms, sig, color=col, lw=1.4)
        ax.axvline(300, color="#888888", lw=1.0, ls="--", alpha=0.7)
        ax.axhline(0,   color="#cccccc", lw=0.7)
        ax.set_title(lab, fontsize=11, color=col, fontweight="bold")
        ax.set_ylabel("Amplitude (µV)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Time after stimulus (ms)")
    axes[-1].text(302, axes[-1].get_ylim()[0], "300 ms", fontsize=8, color="#666666")
    fig.suptitle(f"Mixup-generated synthetic EEG sample (Subject A, channel {ch})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUTDIR / "mixup_signal_example.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  -> {out}")
    return str(out)


# ── IMPLEMENTATION_SUMMARY.md ────────────────────────────────────────────────

def write_summary(metrics: dict, all_data: dict, fig_paths: dict) -> str:
    def pct(val):
        return f"{val*100:.1f}%" if not np.isnan(val) else "—"

    # Build results tables
    def result_table(split_key):
        lines = []
        lines.append("| Method | Subj | 1r | 5r | 10r | 15r |")
        lines.append("|---|---|---|---|---|---|")
        for var in VARIANTS:
            k = var["key"]
            for subj in ["A", "B"]:
                if k not in metrics[subj]:
                    continue
                accs = metrics[subj][k][split_key]
                r1  = pct(accs.get(1, float("nan")))
                r5  = pct(accs.get(5, float("nan")))
                r10 = pct(accs.get(10, float("nan")))
                r15 = pct(accs.get(15, float("nan")))
                lines.append(f"| {var['label']} | {subj} | {r1} | {r5} | {r10} | {r15} |")
        return "\n".join(lines)

    def single_trial_table():
        lines = []
        lines.append("| Method | Subj | Accuracy | Precision | Recall | F1 | AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        for var in VARIANTS:
            k = var["key"]
            for subj in ["A", "B"]:
                if k not in metrics[subj]:
                    continue
                m = metrics[subj][k]
                lines.append(
                    f"| {var['label']} | {subj} "
                    f"| {m['acc_ep']*100:.1f}% "
                    f"| {m['prec']*100:.1f}% "
                    f"| {m['rec']*100:.1f}% "
                    f"| {m['f1']*100:.1f}% "
                    f"| {m['auc']:.3f} |"
                )
        return "\n".join(lines)

    def cm_table():
        lines = []
        lines.append("| Method | Subj | TP | TN | FP | FN |")
        lines.append("|---|---|---|---|---|---|")
        for var in VARIANTS:
            k = var["key"]
            for subj in ["A", "B"]:
                if k not in metrics[subj]:
                    continue
                m = metrics[subj][k]
                lines.append(
                    f"| {var['label']} | {subj} "
                    f"| {m['tp']:,} | {m['tn']:,} | {m['fp']:,} | {m['fn']:,} |"
                )
        return "\n".join(lines)

    # Paper reference comparison at key rep counts
    ref_lines = ["| Method | Subj | 5r | 10r | 15r |", "|---|---|---|---|---|"]
    for subj in ["A", "B"]:
        ref = PAPER_REF[subj]
        ref_lines.append(f"| Paper (Shukla 2024) | {subj} | {ref[5]}% | {ref[10]}% | {ref[15]}% |")
    for var in VARIANTS:
        k = var["key"]
        for subj in ["A", "B"]:
            if k not in metrics[subj]:
                continue
            accs = metrics[subj][k]["te_accs"]
            ref_lines.append(
                f"| {var['label']} | {subj} "
                f"| {pct(accs.get(5,float('nan')))} "
                f"| {pct(accs.get(10,float('nan')))} "
                f"| {pct(accs.get(15,float('nan')))} |"
            )
    ref_table = "\n".join(ref_lines)

    content = f"""# Implementation Summary — BCI P300 Mixup Enhanced

**Authors:** Ashik M Biju, Sidhu A G, Athira A
**Institution:** School of Computer Science & Engineering, Digital University Kerala
**Paper reproduced:** Shukla, Cecotti & Meena (2024), *Towards Effective Deep Neural Network Approach for Multi-Trial P300-based Character Recognition in Brain-Computer Interfaces*

---

## 1. Project Structure

```
BCI_Mixup_Enhanced/
├── src/
│   ├── model.py          # SPSQConvNet (45,809 params), count_params()
│   ├── data_loader.py    # load_continuous(), epoch_continuous(), load_subject()
│   ├── preprocessing.py  # preprocess() — Chebyshev bandpass + epoch
│   ├── train.py          # naive_clone(), mixup_augment(), mixed_augment(),
│   │                     # WEnsemble, train_one(), train_ensemble()
│   ├── decode.py         # decode_characters(), accuracy_vs_reps()
│   └── utils.py          # constants, get_device(), set_seed()
├── scripts/
│   ├── run_baseline.py         # single-subject CLI
│   ├── run_experiments.py      # all 4 variants × 2 subjects
│   ├── plot_results.py         # PNG accuracy curves + sanity report
│   └── generate_report_figures.py  # this script → PDF figures
├── data/
│   ├── Subject_A_Train.mat / Subject_A_Test.mat / Subject_A_answers.txt
│   └── Subject_B_Train.mat / Subject_B_Test.mat / Subject_B_answers.txt
├── results/experiments/
│   ├── ens_{{A,B}}_{{naive,mixup_02,mixup_04,mixed_04}}.pt  (8 checkpoints)
│   └── {{A,B}}_{{naive,mixup_02,mixup_04,mixed_04}}.csv     (accuracy CSVs)
└── report_figures/     ← generated by this script
```

---

## 2. Implementation Details

### Framework & Hardware
- **Language:** Python 3.12
- **Deep learning:** PyTorch ≥ 2.2 (CUDA 12.1) — paper used Keras/TensorFlow
- **Hardware:** NVIDIA RTX 3050 4 GB, Ryzen 7 6000, 16 GB RAM, Windows 11

### CNN Architecture — SPSQ-CNN (exact match to paper Table I)

| Layer | Config | Output shape | Params |
|---|---|---|---|
| BatchNorm1d | 64 channels | (B, 160, 64) | 128 |
| Reshape | add channel dim | (B, 1, 160, 64) | 0 |
| Conv2d | 1→32, kernel (1×64) | (B, 32, 160, 1) | 2,080 |
| Reshape | squeeze last dim | (B, 32, 160) | 0 |
| Conv1d | 32→16, kernel 20, stride 20 | (B, 16, 8) | 10,256 |
| BatchNorm1d | 16 channels | (B, 16, 8) | 32 |
| LeakyReLU | — | (B, 16, 8) | 0 |
| Flatten | — | (B, 128) | 0 |
| Linear + Tanh + Dropout | 128→128 | (B, 128) | 16,512 |
| Linear + Tanh + Dropout | 128→128 | (B, 128) | 16,512 |
| Linear + Sigmoid | 128→1 | (B, 1) | 129 |
| **Total (trainable + BN buffers)** | | | **45,809** |

**Architecture notes:**
- Input: (batch, 160, 64) — time × channels
- Conv2d performs spatial filtering across all 64 channels simultaneously
- Conv1d stride=20 decimates 160 time steps → 8 (no explicit downsample layer)
- Dropout p=0.8 (higher than typical, consistent with paper)
- Activation: Tanh in FC layers, Sigmoid output → P300 probability in [0,1]

### Preprocessing Pipeline

1. **Bandpass filter:** 4th-order Chebyshev Type I, 0.1–10 Hz, zero-phase (sosfiltfilt)
   Applied to **continuous per-character EEG** (~7,000 samples) before epoching.
   Reflect-pad = 5 × (fs / 0.1) = 12,000 samples to handle near-unit-circle poles.
2. **Epoch extraction:** Detect 0→nonzero transitions in StimulusCode; cut 160-sample (667 ms) windows.
3. **Label assignment:** Check StimulusType in first 24 samples (~100 ms) of each window.

**Note:** The paper's mention of "decimation every 14th sample" was NOT reproduced identically. The Conv1d(kernel=20, stride=20) performs learned temporal compression internally. This may explain the 5r gap for Subject A.

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1×10⁻³ |
| Batch size | 32 |
| Epochs per classifier | 100 |
| Ensemble size | 5 classifiers |
| Dropout | p = 0.8 |
| Random seed | 42 |
| Loss function | BCELoss (accepts soft labels natively) |

### Ensemble Details
- 5 SPSQ-CNN classifiers, seeds 42, 43, 44, 45, 46
- Each trained on the full balanced training set (not subsets)
- Weight: Wₖ = Tₖ / Σ Tᵢ  where Tₖ = training accuracy of classifier k
- Inference: weighted sum of predicted P300 probabilities

### Mixup Implementation

| Aspect | Detail |
|---|---|
| Pairing | Cross-class: one P300 + one Non-P300 per pair |
| λ distribution | Beta(α, α), clamped: λ = max(λ, 1−λ) → λ ∈ [0.5, 1.0) |
| α values tested | 0.2, 0.4 |
| n_generate | n_non − n_p300 = 12,750 − 2,550 = 10,200 samples |
| Soft label | y_new = λ (P300 always carries ≥ 50% weight) |
| Loss | BCELoss with soft y — no code change needed |
| Insertion point | Replaces naive_clone() only; everything else unchanged |

**Mixed 50% strategy (our contribution):**
5,100 Mixup samples (soft labels) + 5,100 naive clones (hard labels) = 10,200 generated

---

## 3. Dataset

| Parameter | Value |
|---|---|
| Dataset | BCI Competition III Dataset II |
| Subjects | 2 (A and B) |
| Sampling rate | 240 Hz |
| EEG channels | 64 |
| Training characters | 85 per subject |
| Test characters | 100 per subject |
| Repetitions | 15 |
| Flashes per repetition | 12 (6 rows + 6 columns) |
| Total training epochs | 85 × 15 × 12 = 15,300 |
| P300 training epochs | 85 × 15 × 2 = 2,550 |
| Non-P300 training epochs | 12,750 |
| Class imbalance ratio | 5:1 |

---

## 4. Character Recognition Results (Test Set)

### Paper vs Our Results

{ref_table}

### Full Accuracy Curve — Test Set (all repetitions 1–15)

{result_table("te_accs")}

### Training Set (Resubstitution)

{result_table("tr_accs")}

*All resubstitution values are 100% — this is expected and not informative (model tested on its own training data).*

---

## 5. Single-Trial P300 Detection (Training Set, Epoch Level)

*These metrics measure epoch-level P300 vs Non-P300 discrimination on the training set (the only set with epoch-level labels).*

{single_trial_table()}

### Confusion Matrix Entries

{cm_table()}

---

## 6. Final Accuracy Claims (Defensible)

| Claim | Value | Evidence |
|---|---|---|
| Baseline matches paper at 15r (Subj A) | 95.0% (paper: 94.5%) | test CSV |
| Baseline matches paper at 10r (Subj A) | 88.0% (paper: 87.5%) | test CSV |
| Baseline exceeds paper at all reps (Subj B) | 79–93% vs 62–91% | test CSV |
| Mixup (α=0.2) best at 15r Subj A | 97.0% | test CSV |
| Mixed 50% best at 15r Subj B | 95.0% | test CSV |
| Mixed 50% achieves highest avg 15r | 96.0% mean | computed |
| Best augmentation at 5r Subj B | Mixed 50%: 80.0% | test CSV |
| Mixup does NOT improve 5r Subj A | 54–59% vs 60% baseline | test CSV |

---

## 7. Key Findings

- **Baseline successfully reproduced.** At 10r and 15r, our PyTorch implementation matches the paper within 0.5–1.0 percentage points for Subject A and exceeds the paper by 12–17 pp for Subject B. The correct order of Chebyshev filtering → epoching (not epoch → filter) is critical.

- **Mixup improves peak accuracy at 15 repetitions.** Both Mixup α=0.2 and Mixed 50% reach 97% for Subject A and 95% for Subject B at 15r — each +2 pp over the naive clone baseline.

- **Mixed 50% is the best single strategy for Subject B.** It achieves the highest or joint-highest accuracy at every repetition count (80% / 93% / 95%), outperforming both pure Mixup and the baseline simultaneously.

- **Mixup degrades accuracy at 5 repetitions for Subject A.** Pure Mixup (54%) loses 6 pp vs baseline (60%). The soft-label calibration that helps at high repetitions hurts when only 5 noisy epochs are available. The Mixed 50% hybrid (59%) partially recovers this.

- **All methods exceed the paper on Subject B** — indicating our PyTorch preprocessing (especially the reflect-padded Chebyshev filter applied to continuous EEG) produces superior feature quality compared to the original implementation.

---

## 8. Known Issues and Limitations

| Issue | Description |
|---|---|
| Sanity report shows "FAIL" | Compares our **resubstitution training** accuracy (100%) against the **paper's test** accuracy. This is an apples-to-oranges setup error in `plot_results.py`. The correct comparison (our test vs paper test) shows the baseline is within ±1 pp at 10r/15r. |
| Subject A at 5r gap | Our baseline (60%) is 16.5 pp below the paper (76.5%). Likely causes: different random seed, PyTorch vs Keras numerics, or paper's preprocessing decimation not exactly replicated. |
| No epoch-level test metrics | The competition test set has no StimulusType labels, so ROC curves and confusion matrices can only be computed on the training set (resubstitution). These results are optimistic. |
| No training loss logging | Loss was printed to console but not saved to file. Training dynamics (Plot 5) cannot be reconstructed without retraining with logging enabled. |
| Mixed strategy not in original paper | The Mixed 50% hybrid is our own contribution and has no direct baseline from Shukla et al. to compare against. |
| No cross-validation | All training uses the full 85-character training set; no k-fold CV was performed. Single-run variance is unknown. |
| Framework difference | Paper used Keras/TensorFlow; we used PyTorch. Small numerical differences in BatchNorm, weight initialization, and Adam implementation may cause result variation. |

---

## 9. Generated Figures

| Figure | File | Description |
|---|---|---|
| Plot 1 | {fig_paths.get("p1", "N/A")} | Character accuracy vs repetitions (Baseline vs Mixup) |
| Plot 2 | {fig_paths.get("p2", "N/A")} | ROC curves (training set, both subjects) |
| Plot 3 | {fig_paths.get("p3", "N/A")} | Alpha ablation — augmentation strategy comparison |
| Plot 4 | {fig_paths.get("p4", "N/A")} | Confusion matrices (training set, 2×2 grid) |
| Plot 5 | SKIPPED | Training loss curves — not logged during training |
| Plot 6 | {fig_paths.get("p6", "N/A")} | State-of-the-art comparison |
| Plot 7 | {fig_paths.get("p7", "N/A")} | Mixup signal example (EEG waveforms) |

---

*Generated: 2026-05-09*
*Code: BCI_Mixup_Enhanced / scripts/generate_report_figures.py*
"""

    out = Path("IMPLEMENTATION_SUMMARY.md")
    out.write_text(content, encoding="utf-8")
    print(f"\nSummary written to {out}")
    return str(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _setup_style()
    OUTDIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("\n=== Loading data ===")
    all_data = load_all_data()

    print("\n=== Computing metrics ===")
    metrics = compute_all_metrics(all_data, device)

    print("\n=== Generating figures ===")
    fp: dict[str, str] = {}
    fp["p1"] = plot1(metrics, all_data)
    fp["p2"] = plot2(metrics)
    fp["p3"] = plot3(metrics, all_data)
    fp["p4"] = plot4(metrics)
    fp["p6"] = plot6(metrics)
    fp["p7"] = plot7(all_data)

    print("\n=== Writing summary ===")
    write_summary(metrics, all_data, fp)

    print("\n=== Done ===")
    print(f"Figures in: {OUTDIR.resolve()}")
    print(f"Summary:    {Path('IMPLEMENTATION_SUMMARY.md').resolve()}")


if __name__ == "__main__":
    main()
