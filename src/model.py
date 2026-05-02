"""
SPSQ-CNN from Shukla et al. 2024, Table I.

Architecture (input → output shape, trainable params):
  BatchNorm1d(64)              (batch, 160, 64)  →  same         128
  Conv2d(1→32, k=(1,64))      (batch, 1, 160, 64) → (batch,32,160,1) 2,080
  Conv1d(32→16, k=20, s=20)   (batch, 32, 160)  → (batch,16,8)  10,256
  BatchNorm1d(16)              same                               32
  LeakyReLU
  Flatten                      (batch, 128)
  Linear(128→128) + Tanh + Dropout(0.8)                          16,512
  Linear(128→128) + Tanh + Dropout(0.8)                          16,512
  Linear(128→1)  + Sigmoid                                       129
  ────────────────────────────────────────────────────
  Trainable total                                                 45,649
  + BN running stats (mean+var, no num_batches_tracked)            + 160
  Paper Table I total                                             45,809
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils import DROPOUT_P, N_CHANNELS


class SPSQConvNet(nn.Module):
    """Single SPSQ-CNN classifier — one member of the WE-SPSQ-CNN ensemble."""

    def __init__(self, dropout_p: float = DROPOUT_P) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(N_CHANNELS)                     # over 64 channels
        self.conv2d = nn.Conv2d(1, 32, kernel_size=(1, N_CHANNELS))
        self.conv1d = nn.Conv1d(32, 16, kernel_size=20, stride=20) # 160→8 time steps
        self.bn2 = nn.BatchNorm1d(16)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, EPOCH_SAMPLES, N_CHANNELS) float32
        Returns:
            (batch,) P300 probabilities in [0, 1]
        """
        # BN1: (batch,160,64) → permute → (batch,64,160) → BN → permute back
        x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 2D spatial filter — collapse all 64 channels at each time step
        x = x.unsqueeze(1)              # (batch, 1, 160, 64)
        x = self.conv2d(x)              # (batch, 32, 160, 1)
        x = x.squeeze(-1)               # (batch, 32, 160)

        # 1D temporal convolution with stride 20 → 8 time steps
        x = self.conv1d(x)              # (batch, 16, 8)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = x.flatten(1)                # (batch, 128)
        x = self.dropout(torch.tanh(self.fc1(x)))
        x = self.dropout(torch.tanh(self.fc2(x)))
        return self.fc3(x).squeeze(-1).sigmoid()


def count_params(model: nn.Module) -> int:
    """
    Count parameters using the same convention as Shukla et al. Table I:
    trainable params + BN running_mean + BN running_var (num_batches_tracked excluded).
    Returns 45,809 for a freshly constructed SPSQConvNet.
    """
    n = sum(p.numel() for p in model.parameters())
    for name, buf in model.named_buffers():
        if "num_batches_tracked" not in name:
            n += buf.numel()
    return n
