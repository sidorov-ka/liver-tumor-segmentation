"""
Uncertainty-guided 2D U-Net for ROI tumor refinement — independent of coarse_to_fine / multiview.

Five input channels: three fixed HU windows + nnU-Net tumor probability + normalized entropy U(p).
Output: one logit map (sigmoid → refined probability).
"""

from __future__ import annotations

from coarse_to_fine.model import TinyUNet2d


class UncertaintyUNet2d(TinyUNet2d):
    """Five channels: three window-normalized CT maps + tumor prob + normalized entropy."""

    def __init__(self, base: int = 32):
        super().__init__(in_channels=5, base=base)
