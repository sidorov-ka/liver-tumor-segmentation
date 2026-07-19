from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from common.loss_utils import full_resolution, soft_skeletonize, target_masks


TensorOrDeepSupervision = torch.Tensor | Sequence[torch.Tensor]


class SoftClDiceLoss(nn.Module):
    """Wrap Dice+CE with soft center-line Dice (topology-aware) on tumor."""

    def __init__(
        self,
        base_loss: nn.Module,
        tumor_label: int = 2,
        liver_label: int = 1,
        cldice_weight: float = 0.10,
        skeleton_iterations: int = 10,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.tumor_label = int(tumor_label)
        self.liver_label = int(liver_label)
        self.cldice_weight = float(cldice_weight)
        self.skeleton_iterations = max(1, int(skeleton_iterations))
        self.smooth = float(smooth)
        self.last_components: dict[str, float] = {}

    def _cldice_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        prob = (prob * valid).clamp(0.0, 1.0)
        tumor = (tumor * valid).clamp(0.0, 1.0)
        spatial_dims = tuple(range(1, prob.ndim))
        has_tumor = tumor.sum(dim=spatial_dims) > 0
        if not torch.any(has_tumor):
            return prob.new_zeros(())

        pred_skel = soft_skeletonize(prob, iterations=self.skeleton_iterations)
        gt_skel = soft_skeletonize(tumor, iterations=self.skeleton_iterations)

        tprec = (pred_skel * tumor).sum(dim=spatial_dims) / (
            pred_skel.sum(dim=spatial_dims) + self.smooth
        )
        tsens = (gt_skel * prob).sum(dim=spatial_dims) / (
            gt_skel.sum(dim=spatial_dims) + self.smooth
        )
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return (1.0 - cldice)[has_tumor].mean()

    def forward(
        self,
        outputs: TensorOrDeepSupervision,
        targets: TensorOrDeepSupervision,
    ) -> torch.Tensor:
        base_loss = self.base_loss(outputs, targets)
        loss = base_loss
        cldice_loss = loss.new_zeros(())

        if self.cldice_weight > 0:
            logits = full_resolution(outputs)
            target = full_resolution(targets)
            tumor, _, _, valid = target_masks(
                target,
                tumor_label=self.tumor_label,
                liver_label=self.liver_label,
            )
            prob = torch.softmax(logits, dim=1)[:, self.tumor_label]
            cldice_loss = self._cldice_loss(prob, tumor, valid)
            loss = loss + self.cldice_weight * cldice_loss

        self.last_components = {
            "base_loss": float(base_loss.detach().cpu()),
            "cldice_loss": float(cldice_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return loss
