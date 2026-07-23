"""Multi-layer temporal network for OL posture / technique classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oline_cv.nn.features import FEATURE_DIM, POSTURE_CLASSES


class ResidualTemporalBlock(nn.Module):
    """Dilated temporal conv + MLP residual block."""

    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        padding = dilation
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=3, padding=padding, dilation=dilation, groups=1
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.conv(h)
        h = h.transpose(1, 2)
        # Match length if dilation padding added extra
        if h.size(1) != x.size(1):
            h = h[:, : x.size(1)]
        x = x + h
        h = self.ff(self.norm2(x))
        return x + h


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D)
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class OLTechniqueNet(nn.Module):
    """Deep temporal network over pose feature windows.

    Layers:
      1. Input projection
      2–5. Residual temporal blocks (dilations 1,2,4,8)
      6. Attention pooling
      7–8. MLP classifier head
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden: int = 128,
        num_blocks: int = 4,
        num_classes: int = len(POSTURE_CLASSES),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        dilations = [2**i for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden, dilation=d, dropout=dropout) for d in dilations]
        )
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits (B, C)"""
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h)
        return self.head(h)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    @torch.inference_mode()
    def predict_label(self, x: torch.Tensor) -> list[str]:
        ids = self.forward(x).argmax(dim=-1).tolist()
        return [POSTURE_CLASSES[i] for i in ids]


def build_model(hidden: int = 128, num_blocks: int = 4) -> OLTechniqueNet:
    return OLTechniqueNet(hidden=hidden, num_blocks=num_blocks)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
