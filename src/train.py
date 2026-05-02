"""
Training pipeline for WE-SPSQ-CNN (Shukla et al. 2024, Sec. II-C).

Entry points:
    from src.train import naive_clone, train_ensemble

    epochs_bal, labels_bal = naive_clone(data.epochs, data.labels)
    ensemble = train_ensemble(epochs_bal, labels_bal)
    probs = ensemble.predict_proba(test_epochs)

Weighted ensemble (Eq. 1):  Wk = Tk / sum(Ti)
  where Tk is classifier k's training-set accuracy after convergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import SPSQConvNet
from src.utils import BATCH_SIZE, DEFAULT_SEED, LEARNING_RATE, N_CLASSIFIERS, get_device


# ---------------------------------------------------------------------------
# Class-balance augmentation
# ---------------------------------------------------------------------------

def naive_clone(
    epochs: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance P300 vs non-P300 by repeating (cloning) P300 epochs.

    BCI competition data has ~5:1 non-P300:P300 imbalance.  Each P300 epoch
    is repeated floor(n_non / n_p300) times; if n_non is not a multiple, the
    remainder is filled from the start of the P300 pool (or sampled without
    replacement when rng is supplied).

    Args:
        epochs: (n, EPOCH_SAMPLES, N_CHANNELS) float32
        labels: (n,) int — values 0 or 1
        rng:    numpy Generator for reproducible remainder sampling (optional)

    Returns:
        (balanced_epochs, balanced_labels) — combined but NOT shuffled.
        DataLoader handles per-epoch shuffling during training.
    """
    p300_mask = labels == 1
    non_mask = labels == 0

    p300_ep = epochs[p300_mask]
    non_ep = epochs[non_mask]
    n_p300, n_non = len(p300_ep), len(non_ep)

    if n_p300 == 0:
        raise ValueError("No P300 epochs to clone.")
    if n_non == 0:
        raise ValueError("No non-P300 epochs.")

    full_reps = n_non // n_p300
    remainder = n_non % n_p300

    if remainder > 0:
        extra_idx = (
            rng.choice(n_p300, size=remainder, replace=False)
            if rng is not None
            else np.arange(remainder)
        )
        p300_bal = np.concatenate(
            [np.tile(p300_ep, (full_reps, 1, 1)), p300_ep[extra_idx]], axis=0
        )
    else:
        p300_bal = np.tile(p300_ep, (full_reps, 1, 1))

    bal_epochs = np.concatenate([non_ep, p300_bal], axis=0)
    bal_labels = np.concatenate(
        [np.zeros(n_non, dtype=labels.dtype), np.ones(len(p300_bal), dtype=labels.dtype)]
    )
    return bal_epochs, bal_labels


def mixup_augment(
    epochs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace naive P300 cloning with Mixup (Zhang et al. 2018, ICLR).

    Generates (n_non - n_p300) synthetic epochs by linearly interpolating
    randomly paired P300 and non-P300 epochs:

        x_new = λ · x_p300 + (1 - λ) · x_non,   λ ~ Beta(α, α)
        y_new = λ                                  (soft label in [0.5, 1])

    λ is clamped to max(λ, 1-λ) so P300 always carries the dominant weight.

    The combined output is:
        • All original non-P300 epochs         (hard label 0.0)
        • All original P300 epochs             (hard label 1.0)
        • (n_non - n_p300) Mixup P300 epochs   (soft label λ ∈ [0.5, 1))

    BCELoss accepts soft labels natively so no change to the training loop
    is needed.

    Args:
        epochs: (n, EPOCH_SAMPLES, N_CHANNELS) float32
        labels: (n,) int — values 0 or 1
        alpha:  Beta distribution concentration; paper tests 0.2 and 0.4.
                Smaller α → more extreme λ (closer to 0 or 1).
        rng:    numpy Generator for reproducibility.

    Returns:
        (bal_epochs, bal_labels) — combined but NOT shuffled.
        bal_labels dtype float32 with values in [0.0, 1.0].
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    if rng is None:
        rng = np.random.default_rng()

    p300_mask = labels == 1
    non_mask = labels == 0

    p300_ep = epochs[p300_mask]
    non_ep = epochs[non_mask]
    n_p300, n_non = len(p300_ep), len(non_ep)

    if n_p300 == 0:
        raise ValueError("No P300 epochs for Mixup.")
    if n_non == 0:
        raise ValueError("No non-P300 epochs for Mixup.")

    n_generate = n_non - n_p300

    if n_generate < 0:
        raise ValueError(
            f"n_non ({n_non}) < n_p300 ({n_p300}): Mixup expects more non-P300 than P300."
        )

    if n_generate == 0:
        bal_epochs = np.concatenate([non_ep, p300_ep], axis=0)
        bal_labels = np.concatenate([
            np.zeros(n_non, dtype=np.float32),
            np.ones(n_p300, dtype=np.float32),
        ])
        return bal_epochs, bal_labels

    # λ ~ Beta(α, α), clamped so P300 side always has weight ≥ 0.5
    lam = rng.beta(alpha, alpha, size=n_generate).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)

    p300_idx = rng.integers(0, n_p300, size=n_generate)
    non_idx = rng.integers(0, n_non, size=n_generate)

    lam3 = lam[:, np.newaxis, np.newaxis]  # (n_generate, 1, 1) for broadcasting
    mixed_ep = (lam3 * p300_ep[p300_idx] + (1.0 - lam3) * non_ep[non_idx]).astype(np.float32)

    bal_epochs = np.concatenate([non_ep, p300_ep, mixed_ep], axis=0)
    bal_labels = np.concatenate([
        np.zeros(n_non, dtype=np.float32),
        np.ones(n_p300, dtype=np.float32),
        lam,
    ])
    return bal_epochs, bal_labels


def mixed_augment(
    epochs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4,
    mixup_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance classes using a mixture of Mixup and naive cloning (mixed strategy).

    Of the (n_non - n_p300) generated samples:
        floor(mixup_fraction × n_generate) → Mixup (soft label λ ∈ [0.5, 1])
        n_generate - n_mixup               → naive clone (hard label 1.0)

    Boundary cases:
        mixup_fraction=0.0  →  pure naive cloning  (equivalent to naive_clone)
        mixup_fraction=1.0  →  pure Mixup          (equivalent to mixup_augment)

    Args:
        epochs:          (n, EPOCH_SAMPLES, N_CHANNELS) float32
        labels:          (n,) int — values 0 or 1
        alpha:           Beta(α, α) concentration for the Mixup portion
        mixup_fraction:  Fraction of generated samples from Mixup, in [0, 1]
        rng:             numpy Generator for reproducibility

    Returns:
        (bal_epochs, bal_labels) — NOT shuffled; labels are float32.
    """
    if not 0.0 <= mixup_fraction <= 1.0:
        raise ValueError(f"mixup_fraction must be in [0, 1], got {mixup_fraction}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if rng is None:
        rng = np.random.default_rng()

    p300_mask = labels == 1
    non_mask = labels == 0
    p300_ep = epochs[p300_mask]
    non_ep = epochs[non_mask]
    n_p300, n_non = len(p300_ep), len(non_ep)

    if n_p300 == 0:
        raise ValueError("No P300 epochs for augmentation.")
    if n_non == 0:
        raise ValueError("No non-P300 epochs for augmentation.")

    n_generate = n_non - n_p300
    if n_generate < 0:
        raise ValueError(
            f"n_non ({n_non}) < n_p300 ({n_p300}): expects more non-P300 than P300."
        )

    if n_generate == 0:
        bal_epochs = np.concatenate([non_ep, p300_ep], axis=0)
        bal_labels = np.concatenate([
            np.zeros(n_non, dtype=np.float32),
            np.ones(n_p300, dtype=np.float32),
        ])
        return bal_epochs, bal_labels

    n_mixup_gen = int(mixup_fraction * n_generate)
    n_clone_gen = n_generate - n_mixup_gen

    aug_ep_parts: list[np.ndarray] = []
    aug_lb_parts: list[np.ndarray] = []

    if n_mixup_gen > 0:
        lam = rng.beta(alpha, alpha, size=n_mixup_gen).astype(np.float32)
        lam = np.maximum(lam, 1.0 - lam)
        p300_idx = rng.integers(0, n_p300, size=n_mixup_gen)
        non_idx = rng.integers(0, n_non, size=n_mixup_gen)
        lam3 = lam[:, np.newaxis, np.newaxis]
        aug_ep_parts.append(
            (lam3 * p300_ep[p300_idx] + (1.0 - lam3) * non_ep[non_idx]).astype(np.float32)
        )
        aug_lb_parts.append(lam)

    if n_clone_gen > 0:
        clone_idx = np.arange(n_clone_gen) % n_p300
        aug_ep_parts.append(p300_ep[clone_idx])
        aug_lb_parts.append(np.ones(n_clone_gen, dtype=np.float32))

    bal_epochs = np.concatenate([non_ep, p300_ep] + aug_ep_parts, axis=0)
    bal_labels = np.concatenate(
        [np.zeros(n_non, dtype=np.float32), np.ones(n_p300, dtype=np.float32)] + aug_lb_parts
    )
    return bal_epochs, bal_labels


# ---------------------------------------------------------------------------
# Weighted Ensemble
# ---------------------------------------------------------------------------

class WEnsemble:
    """
    Weighted Ensemble of SPSQ-CNN classifiers (WE-SPSQ-CNN).

    Combines per-classifier P300 probabilities using weights proportional to
    each classifier's training-set accuracy (Shukla et al., Eq. 1):
        Wk = Tk / sum(Ti)
    """

    def __init__(self, models: list[SPSQConvNet], weights: np.ndarray) -> None:
        """
        Args:
            models:  Trained SPSQConvNet instances.
            weights: (n_classifiers,) non-negative weights summing to 1.0.
        """
        self.models = models
        self.weights = weights  # float32 (n_classifiers,)

    def predict_proba(
        self,
        epochs: np.ndarray,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """
        Weighted-average P300 probability over all ensemble members.

        Args:
            epochs:     (n, EPOCH_SAMPLES, N_CHANNELS) float32
            batch_size: Inference batch size (limits GPU memory use).
            device:     Inference device (default: get_device()).

        Returns:
            (n,) float32 probabilities in [0, 1]
        """
        if device is None:
            device = get_device()

        out = np.zeros(len(epochs), dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(epochs).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for model, w in zip(self.models, self.weights):
            model.eval()
            model.to(device)
            preds: list[np.ndarray] = []
            with torch.no_grad():
                for (xb,) in loader:
                    preds.append(model(xb.to(device)).cpu().numpy())
            out += float(w) * np.concatenate(preds)

        return out


# ---------------------------------------------------------------------------
# Single-classifier training
# ---------------------------------------------------------------------------

def train_one(
    model: SPSQConvNet,
    epochs: np.ndarray,
    labels: np.ndarray,
    *,
    n_epochs: int = 100,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> float:
    """
    Train a single SPSQ-CNN in-place and return training accuracy Tk.

    Args:
        model:     SPSQConvNet — modified in-place.
        epochs:    (n, EPOCH_SAMPLES, N_CHANNELS) float32, already balanced.
        labels:    (n,) int — values 0 or 1.
        n_epochs:  Training epochs.
        lr:        Adam learning rate.
        batch_size: Mini-batch size.
        device:    Training device (default: get_device()).
        seed:      DataLoader shuffle seed for reproducibility.
        verbose:   Print average loss every 10 epochs.

    Returns:
        Training accuracy Tk in [0, 1].
    """
    if device is None:
        device = get_device()

    model.to(device)

    X = torch.from_numpy(epochs).float()
    y = torch.from_numpy(labels.astype(np.float32))
    dataset = TensorDataset(X, y)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for ep in range(n_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}/{n_epochs}  loss={total_loss / len(loader):.4f}")

    # Training accuracy for ensemble weight Tk.
    # yb.round() handles both hard labels (0/1) and Mixup soft labels (λ ∈ [0.5,1]).
    model.eval()
    correct = 0
    eval_loader = DataLoader(dataset, batch_size=batch_size * 4, shuffle=False)
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += ((model(xb) >= 0.5).float() == yb.round()).sum().item()
    return correct / len(dataset)


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------

def train_ensemble(
    epochs: np.ndarray,
    labels: np.ndarray,
    *,
    n_classifiers: int = N_CLASSIFIERS,
    n_epochs: int = 100,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
    base_seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> WEnsemble:
    """
    Train WE-SPSQ-CNN: n_classifiers base models with deterministic seeding.

    Each classifier is seeded as base_seed + k so the ensemble is reproducible
    but every member sees a different mini-batch shuffle order.

    Args:
        epochs:        (n, EPOCH_SAMPLES, N_CHANNELS) float32 — pass the output
                       of naive_clone() so classes are already balanced.
        labels:        (n,) int — values 0 or 1.
        n_classifiers: Ensemble size (default 5, from paper).
        n_epochs:      Training epochs per classifier.
        lr:            Adam learning rate.
        batch_size:    Mini-batch size.
        device:        Training device (default: get_device()).
        base_seed:     Seed k = base_seed + k for classifier k.
        verbose:       Print per-classifier Tk and final weights.

    Returns:
        WEnsemble with trained models and normalised weights Wk = Tk / sum(Ti).
    """
    if device is None:
        device = get_device()

    models: list[SPSQConvNet] = []
    accs: list[float] = []

    for k in range(n_classifiers):
        if verbose:
            print(f"Training classifier {k + 1}/{n_classifiers} ...", flush=True)
        model = SPSQConvNet()
        acc = train_one(
            model, epochs, labels,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            device=device, seed=base_seed + k, verbose=False,
        )
        models.append(model)
        accs.append(acc)
        if verbose:
            print(f"  Tk = {acc:.4f}")

    accs_arr = np.array(accs, dtype=np.float32)
    weights = accs_arr / accs_arr.sum()

    if verbose:
        print(f"Ensemble weights: {np.round(weights, 4)}")

    return WEnsemble(models, weights)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_ensemble(ensemble: WEnsemble, path: Path | str) -> None:
    """
    Save a WEnsemble to a .pt file.

    Stores each classifier's state_dict and the weight vector.
    Safe to load with weights_only=True.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_states": [m.cpu().state_dict() for m in ensemble.models],
            "weights": torch.from_numpy(ensemble.weights),
        },
        path,
    )


def load_ensemble(path: Path | str, device: torch.device | None = None) -> WEnsemble:
    """
    Load a WEnsemble from a .pt checkpoint produced by save_ensemble().

    Args:
        path:   Path to the .pt file.
        device: Device to map models to (default: get_device()).

    Returns:
        WEnsemble in eval mode on the requested device.
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    models: list[SPSQConvNet] = []
    for state in ckpt["model_states"]:
        m = SPSQConvNet()
        m.load_state_dict(state)
        m.to(device).eval()
        models.append(m)
    return WEnsemble(models, ckpt["weights"].numpy())
