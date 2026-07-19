from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from common.loss_utils import dilate, full_resolution, target_masks


TensorOrDeepSupervision = torch.Tensor | Sequence[torch.Tensor]


class AnatomicalLoss(nn.Module):
    """Wrap Dice+CE with boundary-ring, liver-constrained FP, and Tversky terms.

    All additive terms are active from epoch 0 (no curriculum / size gates).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        tumor_label: int = 2,
        liver_label: int = 1,
        boundary_weight: float = 0.10,
        overseg_weight: float = 0.05,
        outside_liver_fp_weight: float = 4.0,
        inside_liver_fp_weight: float = 0.5,
        boundary_radius: int = 2,
        outside_liver_ignore_radius: int = 2,
        inside_liver_ignore_radius: int = 4,
        outside_liver_topk_fraction: float = 0.01,
        inside_liver_topk_fraction: float = 0.002,
        tversky_guard_weight: float = 0.05,
        tversky_guard_alpha: float = 0.30,
        tversky_guard_beta: float = 0.70,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.tumor_label = int(tumor_label)
        self.liver_label = int(liver_label)
        self.boundary_weight = float(boundary_weight)
        self.overseg_weight = float(overseg_weight)
        self.outside_liver_fp_weight = float(outside_liver_fp_weight)
        self.inside_liver_fp_weight = float(inside_liver_fp_weight)
        self.boundary_radius = int(boundary_radius)
        self.outside_liver_ignore_radius = int(outside_liver_ignore_radius)
        self.inside_liver_ignore_radius = int(inside_liver_ignore_radius)
        self.outside_liver_topk_fraction = float(outside_liver_topk_fraction)
        self.inside_liver_topk_fraction = float(inside_liver_topk_fraction)
        self.tversky_guard_weight = float(tversky_guard_weight)
        self.tversky_guard_alpha = float(tversky_guard_alpha)
        self.tversky_guard_beta = float(tversky_guard_beta)
        self.smooth = float(smooth)
        self.last_components: dict[str, float] = {}

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

    def _hard_negative_loss(
        self,
        tumor_logits: torch.Tensor,
        prob: torch.Tensor,
        mask: torch.Tensor,
        topk_fraction: float,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        topk_fraction = max(0.0, min(1.0, float(topk_fraction)))
        if topk_fraction <= 0:
            return prob.new_zeros(())

        for batch_idx in range(prob.shape[0]):
            candidate_mask = mask[batch_idx].bool()
            candidate_count = int(candidate_mask.sum().item())
            if candidate_count == 0:
                continue
            candidate_prob = prob[batch_idx][candidate_mask]
            candidate_logits = tumor_logits[batch_idx][candidate_mask]
            k = max(1, int(candidate_count * topk_fraction))
            k = min(k, candidate_count)
            _, hard_indices = torch.topk(candidate_prob, k=k, largest=True, sorted=False)
            hard_logits = candidate_logits[hard_indices]
            losses.append(F.softplus(hard_logits).mean())

        if not losses:
            return prob.new_zeros(())
        return torch.stack(losses).mean()

    def _boundary_loss(
        self,
        tumor_logits: torch.Tensor,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        ring = self._boundary_ring(tumor, valid)
        if torch.count_nonzero(ring).item() == 0:
            return prob.new_zeros(())

        bce = F.binary_cross_entropy_with_logits(tumor_logits, tumor, reduction="none")
        spatial_dims = tuple(range(1, prob.ndim))
        ring_sum = ring.sum(dim=spatial_dims)
        has_ring = ring_sum > 0
        bce = (bce * ring).sum(dim=spatial_dims) / ring_sum.clamp_min(self.smooth)

        pred_ring = prob * ring
        target_ring = tumor * ring
        intersection = (pred_ring * target_ring).sum(dim=spatial_dims)
        denominator = pred_ring.sum(dim=spatial_dims) + target_ring.sum(dim=spatial_dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        )
        return (bce + dice_loss)[has_ring].mean()

    def _false_positive_loss(
        self,
        tumor_logits: torch.Tensor,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        background: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prob = prob * valid
        tumor = tumor * valid
        outside_ignore = dilate(tumor, self.outside_liver_ignore_radius)
        inside_ignore = dilate(tumor, self.inside_liver_ignore_radius)
        outside_mask = background * (1.0 - outside_ignore) * valid
        inside_mask = liver * (1.0 - inside_ignore) * valid
        outside_loss = self._hard_negative_loss(
            tumor_logits,
            prob,
            outside_mask,
            self.outside_liver_topk_fraction,
        )
        inside_loss = self._hard_negative_loss(
            tumor_logits,
            prob,
            inside_mask,
            self.inside_liver_topk_fraction,
        )
        return outside_loss, inside_loss

    def _tversky_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        alpha = max(0.0, float(self.tversky_guard_alpha))
        beta = max(0.0, float(self.tversky_guard_beta))
        if alpha == 0 and beta == 0:
            return prob.new_zeros(())

        prob = prob * valid
        tumor = tumor * valid
        non_tumor = (1.0 - tumor) * valid
        spatial_dims = tuple(range(1, prob.ndim))
        true_positive = (prob * tumor).sum(dim=spatial_dims)
        false_positive = (prob * non_tumor).sum(dim=spatial_dims)
        false_negative = ((1.0 - prob) * tumor).sum(dim=spatial_dims)
        has_tumor = tumor.sum(dim=spatial_dims) > 0
        if not torch.any(has_tumor):
            return prob.new_zeros(())

        denominator = (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + self.smooth
        )
        tversky = (true_positive + self.smooth) / denominator
        return (1.0 - tversky)[has_tumor].mean()

    def forward(
        self,
        outputs: TensorOrDeepSupervision,
        targets: TensorOrDeepSupervision,
    ) -> torch.Tensor:
        base_loss = self.base_loss(outputs, targets)
        loss = base_loss

        logits = full_resolution(outputs)
        target = full_resolution(targets)
        tumor, liver, background, valid = target_masks(
            target,
            tumor_label=self.tumor_label,
            liver_label=self.liver_label,
        )
        prob = torch.softmax(logits, dim=1)[:, self.tumor_label]
        other_logits = torch.cat(
            (
                logits[:, : self.tumor_label],
                logits[:, self.tumor_label + 1 :],
            ),
            dim=1,
        )
        tumor_logits = logits[:, self.tumor_label] - torch.logsumexp(other_logits, dim=1)

        boundary_loss = loss.new_zeros(())
        outside_fp_loss = loss.new_zeros(())
        inside_fp_loss = loss.new_zeros(())
        tversky_loss = loss.new_zeros(())

        if self.boundary_weight > 0:
            boundary_loss = self._boundary_loss(tumor_logits, prob, tumor, valid)
            loss = loss + self.boundary_weight * boundary_loss

        if self.overseg_weight > 0 and (
            self.outside_liver_fp_weight > 0 or self.inside_liver_fp_weight > 0
        ):
            outside_fp_loss, inside_fp_loss = self._false_positive_loss(
                tumor_logits,
                prob,
                tumor,
                liver,
                background,
                valid,
            )
            fp_loss = (
                self.outside_liver_fp_weight * outside_fp_loss
                + self.inside_liver_fp_weight * inside_fp_loss
            )
            loss = loss + self.overseg_weight * fp_loss

        if self.tversky_guard_weight > 0:
            tversky_loss = self._tversky_loss(prob, tumor, valid)
            loss = loss + self.tversky_guard_weight * tversky_loss

        self.last_components = {
            "base_loss": float(base_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "outside_fp_loss": float(outside_fp_loss.detach().cpu()),
            "inside_fp_loss": float(inside_fp_loss.detach().cpu()),
            "tversky_loss": float(tversky_loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        }
        return loss
