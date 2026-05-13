"""
Multiview (multi-window) 2D U-Net for ROI tumor refinement — **independent** of coarse_to_fine.

Architecture follows the usual multi-branch / fusion idea: **three fixed HU windows** (three views)
plus the **nnU-Net tumor probability** as an extra channel, fused in a single encoder–decoder
(see e.g. *Deep Multi-View Fusion Network for Lung Nodule Segmentation*, IEEE TMI / similar;
multi-window CT inputs for nodule segmentation).

Train with ``scripts/2d/train_multiview.py``; **do not** load weights from
``train_coarse_to_fine`` / TinyUNet2d(2-channel).
"""

from __future__ import annotations

from coarse_to_fine.model import TinyUNet2d


class MultiviewUNet2d(TinyUNet2d):
    """
    Four input channels: three window-normalized CT maps + tumor probability map.
    Output: one logit map (apply sigmoid for probability in [0, 1]).
    """

    def __init__(self, base: int = 32):
        super().__init__(in_channels=4, base=base)
