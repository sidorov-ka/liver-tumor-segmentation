from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


TensorOrDS = torch.Tensor | Sequence[torch.Tensor]


class MultiWindowRefinementLoss(nn.Module):
    """Default nnU-Net (Dice + CE) plus Tversky on tumor logits to reduce under-segmentation."""

    def __init__(
        self,
        base_loss: nn.Module,
        tumor_label: int = 2,
        tversky_weight: float = 0.25,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.tumor_label = int(tumor_label)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.smooth = float(smooth)
        self.last_components: dict[str, float] = {}

    @staticmethod
    def _full_res(x: TensorOrDS) -> torch.Tensor:
        return x[0] if isinstance(x, (list, tuple)) else x

    @staticmethod
    def _full_target(t: TensorOrDS) -> torch.Tensor:
        return t[0] if isinstance(t, (list, tuple)) else t

    def _tversky_tumor(self, logits: torch.Tensor, target_b1: torch.Tensor) -> torch.Tensor:
        """Tversky on softmax tumor vs binary tumor mask (higher beta penalizes FN)."""
        probs = torch.softmax(logits, dim=1)
        if self.tumor_label >= probs.shape[1]:
            raise IndexError(f"tumor_label {self.tumor_label} out of range for {probs.shape[1]} classes")
        p = probs[:, self.tumor_label]
        if target_b1.ndim == 5 and target_b1.shape[1] == 1:
            tgt = target_b1[:, 0]
        else:
            tgt = target_b1
        t = (tgt == self.tumor_label).float()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()
        tp = (p * t).sum()
        a, b = self.tversky_alpha, self.tversky_beta
        denom = tp + a * fp + b * fn + self.smooth
        tversky = (tp + self.smooth) / denom
        return 1.0 - tversky

    def forward(self, net_output: TensorOrDS, target: TensorOrDS) -> torch.Tensor:
        base = self.base_loss(net_output, target)
        logits = self._full_res(net_output)
        tgt = self._full_target(target)
        tv = self._tversky_tumor(logits, tgt)
        total = base + self.tversky_weight * tv
        self.last_components = {
            "base": float(base.detach().cpu()),
            "tversky_tumor": float(tv.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total
