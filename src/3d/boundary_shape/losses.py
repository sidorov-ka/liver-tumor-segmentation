from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


TensorOrDeepSupervision = torch.Tensor | Sequence[torch.Tensor]


class BoundaryOversegmentationLoss(nn.Module):
    """Add boundary and over-segmentation penalties to nnU-Net's default loss."""

    def __init__(
        self,
        base_loss: nn.Module,
        tumor_label: int = 2,
        boundary_weight: float = 0.25,
        overseg_weight: float = 0.05,
        boundary_radius: int = 2,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.tumor_label = int(tumor_label)
        self.boundary_weight = float(boundary_weight)
        self.overseg_weight = float(overseg_weight)
        self.boundary_radius = int(boundary_radius)
        self.smooth = float(smooth)

    @staticmethod
    def _full_resolution(x: TensorOrDeepSupervision) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def _tumor_target(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim >= 2 and target.shape[1] == 1:
            target = target[:, 0]
        valid = target >= 0
        tumor = (target == self.tumor_label).float()
        return tumor, valid.float()

    def _boundary_ring(
        self,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        radius = max(self.boundary_radius, 1)
        kernel_size = 2 * radius + 1
        tumor_5d = tumor[:, None]
        dilated = F.max_pool3d(
            tumor_5d,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        eroded = 1.0 - F.max_pool3d(
            1.0 - tumor_5d,
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        ring = (dilated - eroded).clamp_(0.0, 1.0)[:, 0]
        return ring * valid

    def _boundary_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        ring = self._boundary_ring(tumor, valid)
        if torch.count_nonzero(ring).item() == 0:
            return prob.new_zeros(())

        bce = F.binary_cross_entropy(prob, tumor, reduction="none")
        bce = (bce * ring).sum() / ring.sum().clamp_min(self.smooth)

        pred_ring = prob * ring
        target_ring = tumor * ring
        intersection = (pred_ring * target_ring).sum(dim=(1, 2, 3))
        denominator = (
            pred_ring.sum(dim=(1, 2, 3))
            + target_ring.sum(dim=(1, 2, 3))
        )
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        )
        return bce + dice_loss.mean()

    def _oversegmentation_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        prob = prob * valid
        tumor = tumor * valid

        pred_volume = prob.sum(dim=(1, 2, 3))
        target_volume = tumor.sum(dim=(1, 2, 3))
        valid_volume = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)

        has_tumor = target_volume > 0
        excess_ratio = (
            F.relu(pred_volume - target_volume)
            / target_volume.clamp_min(1.0)
        )
        empty_case_ratio = pred_volume / valid_volume
        penalty = torch.where(has_tumor, excess_ratio.square(), empty_case_ratio)
        return penalty.mean()

    def forward(
        self,
        outputs: TensorOrDeepSupervision,
        targets: TensorOrDeepSupervision,
    ) -> torch.Tensor:
        loss = self.base_loss(outputs, targets)

        logits = self._full_resolution(outputs)
        target = self._full_resolution(targets)
        tumor, valid = self._tumor_target(target)
        prob = torch.softmax(logits, dim=1)[:, self.tumor_label]

        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(prob, tumor, valid)
            loss = loss + self.boundary_weight * boundary_loss
        if self.overseg_weight > 0:
            overseg_loss = self._oversegmentation_loss(prob, tumor, valid)
            loss = loss + self.overseg_weight * overseg_loss
        return loss
