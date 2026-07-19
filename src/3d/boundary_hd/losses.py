from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from common.loss_utils import full_resolution, soft_dilate, soft_erode, target_masks


TensorOrDeepSupervision = torch.Tensor | Sequence[torch.Tensor]


class SoftHausdorffLoss(nn.Module):
    """Wrap Dice+CE with a soft bidirectional Hausdorff-inspired term.

    Approximates surface distance via morphological soft erode/dilate layers
    (differentiable proxy for HD; evaluation still uses true HD95).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        tumor_label: int = 2,
        liver_label: int = 1,
        hd_weight: float = 0.10,
        distance_iterations: int = 8,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.tumor_label = int(tumor_label)
        self.liver_label = int(liver_label)
        self.hd_weight = float(hd_weight)
        self.distance_iterations = max(1, int(distance_iterations))
        self.smooth = float(smooth)
        self.last_components: dict[str, float] = {}

    def _soft_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        return (soft_dilate(mask) - soft_erode(mask)).clamp(0.0, 1.0)

    def _approx_distance(self, mask: torch.Tensor) -> torch.Tensor:
        """Banded distance map via repeated soft erosion of the complement."""
        complement = (1.0 - mask).clamp(0.0, 1.0)
        distance = torch.zeros_like(mask)
        current = complement
        for step in range(1, self.distance_iterations + 1):
            eroded = soft_erode(current)
            band = (current - eroded).clamp_min(0.0)
            distance = distance + band * float(step)
            current = eroded
        # Interior of the object gets 0; exterior gets increasing distance.
        return distance * complement

    def _hausdorff_term(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        prob = (prob * valid).clamp(0.0, 1.0)
        tumor = (tumor * valid).clamp(0.0, 1.0)
        pred_boundary = self._soft_boundary(prob) * valid
        gt_boundary = self._soft_boundary(tumor) * valid
        dist_gt = self._approx_distance(tumor)
        dist_pred = self._approx_distance(prob)

        spatial_dims = tuple(range(1, prob.ndim))
        pred_to_gt = (pred_boundary * dist_gt).sum(dim=spatial_dims) / (
            pred_boundary.sum(dim=spatial_dims) + self.smooth
        )
        gt_to_pred = (gt_boundary * dist_pred).sum(dim=spatial_dims) / (
            gt_boundary.sum(dim=spatial_dims) + self.smooth
        )
        has_gt = tumor.sum(dim=spatial_dims) > 0
        if not torch.any(has_gt):
            return prob.new_zeros(())
        return (0.5 * (pred_to_gt + gt_to_pred))[has_gt].mean()

    def forward(
        self,
        outputs: TensorOrDeepSupervision,
        targets: TensorOrDeepSupervision,
    ) -> torch.Tensor:
        base_loss = self.base_loss(outputs, targets)
        loss = base_loss
        hd_loss = loss.new_zeros(())

        if self.hd_weight > 0:
            logits = full_resolution(outputs)
            target = full_resolution(targets)
            tumor, _, _, valid = target_masks(
                target,
                tumor_label=self.tumor_label,
                liver_label=self.liver_label,
            )
            prob = torch.softmax(logits, dim=1)[:, self.tumor_label]
            hd_loss = self._hausdorff_term(prob, tumor, valid)
            loss = loss + self.hd_weight * hd_loss

        self.last_components = {
            "base_loss": float(base_loss.detach().cpu()),
            "soft_hausdorff_loss": float(hd_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return loss
