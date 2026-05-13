"""
Uncertainty-guided 2D U-Net for ROI tumor refinement — independent of coarse_to_fine / multiview.

Five input channels: three fixed HU windows + nnU-Net tumor probability + normalized Bernoulli entropy U(p).

- ``UncertaintyUNet2d``: single head — tumor logit (backward compatible).
- ``UncertaintyDualHeadUNet2d``: tumor head + error head (where baseline disagrees with GT at train time).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn

from coarse_to_fine.model import TinyUNet2d


class UncertaintyUNet2d(TinyUNet2d):
    """Five channels: three window-normalized CT maps + tumor prob + normalized entropy."""

    def __init__(self, base: int = 32):
        super().__init__(in_channels=5, base=base)


class UncertaintyDualHeadUNet2d(TinyUNet2d):
    """
    Same backbone as ``UncertaintyUNet2d``; second 1×1 head predicts baseline error (binary).

    Forward returns ``(logit_tumor, logit_error)``. Tumor head uses the same ``out_conv`` name as
    single-head checkpoints for partial weight loading.
    """

    def __init__(self, base: int = 32):
        super().__init__(in_channels=5, base=base)
        self.out_error = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d1 = self.forward_features(x)
        return self.out_conv(d1), self.out_error(d1)


def build_uncertainty_model(base: int, use_error_head: bool) -> Union[UncertaintyUNet2d, UncertaintyDualHeadUNet2d]:
    if use_error_head:
        return UncertaintyDualHeadUNet2d(base=base)
    return UncertaintyUNet2d(base=base)
