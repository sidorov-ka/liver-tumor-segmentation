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
        liver_label: int = 1,
        boundary_weight: float = 0.25,
        overseg_weight: float = 0.05,
        outside_liver_fp_weight: float = 0.2,
        inside_liver_fp_weight: float = 0.1,
        boundary_radius: int = 2,
        outside_liver_ignore_radius: int = 2,
        inside_liver_ignore_radius: int = 4,
        outside_liver_topk_fraction: float = 0.01,
        inside_liver_topk_fraction: float = 0.003,
        inside_liver_volume_guard_threshold: float = 0.02,
        inside_liver_volume_guard_min_scale: float = 0.10,
        tversky_guard_weight: float = 0.05,
        tversky_guard_alpha: float = 0.30,
        tversky_guard_beta: float = 0.70,
        adaptive_large_tumor_threshold: float = 0.02,
        adaptive_large_tumor_max_threshold: float = 0.10,
        adaptive_fp_min_scale: float = 1.0,
        adaptive_ignore_extra_radius: int = 0,
        under_volume_guard_weight: float = 0.0,
        under_volume_guard_threshold: float = 0.05,
        under_volume_guard_fraction: float = 0.85,
        custom_loss_gate_threshold: float = 0.04,
        custom_loss_gate_temperature: float = 0.015,
        custom_loss_gate_min_scale: float = 0.0,
        under_volume_inverse_gate: bool = False,
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
        self.inside_liver_volume_guard_threshold = float(
            inside_liver_volume_guard_threshold
        )
        self.inside_liver_volume_guard_min_scale = float(
            inside_liver_volume_guard_min_scale
        )
        self.tversky_guard_weight = float(tversky_guard_weight)
        self.tversky_guard_alpha = float(tversky_guard_alpha)
        self.tversky_guard_beta = float(tversky_guard_beta)
        self.adaptive_large_tumor_threshold = float(adaptive_large_tumor_threshold)
        self.adaptive_large_tumor_max_threshold = float(
            adaptive_large_tumor_max_threshold
        )
        self.adaptive_fp_min_scale = float(adaptive_fp_min_scale)
        self.adaptive_ignore_extra_radius = int(adaptive_ignore_extra_radius)
        self.under_volume_guard_weight = float(under_volume_guard_weight)
        self.under_volume_guard_threshold = float(under_volume_guard_threshold)
        self.under_volume_guard_fraction = float(under_volume_guard_fraction)
        self.custom_loss_gate_threshold = float(custom_loss_gate_threshold)
        self.custom_loss_gate_temperature = float(custom_loss_gate_temperature)
        self.custom_loss_gate_min_scale = float(custom_loss_gate_min_scale)
        self.under_volume_inverse_gate = bool(under_volume_inverse_gate)
        self.smooth = float(smooth)
        self.boundary_loss_scale = 1.0
        self.fp_loss_scale = 1.0
        self.last_components: dict[str, float] = {}

    def set_custom_loss_scale(self, scale: float) -> None:
        clipped = max(0.0, min(1.0, float(scale)))
        self.set_custom_loss_scales(clipped, clipped)

    def set_custom_loss_scales(
        self,
        boundary: float,
        fp: float,
    ) -> None:
        self.boundary_loss_scale = max(0.0, min(1.0, float(boundary)))
        self.fp_loss_scale = max(0.0, min(1.0, float(fp)))

    def set_adaptive_fp_min_scale(self, scale: float) -> None:
        """Lower bound for FP hard-negative scaling when adaptive tumor factor is 1 (runtime curriculum)."""
        self.adaptive_fp_min_scale = float(max(0.0, min(1.0, scale)))

    @staticmethod
    def _full_resolution(x: TensorOrDeepSupervision) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    def _target_masks(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if target.ndim >= 2 and target.shape[1] == 1:
            target = target[:, 0]
        valid = target >= 0
        tumor = (target == self.tumor_label).float()
        liver = (target == self.liver_label).float()
        background = (target == 0).float()
        return tumor, liver, background, valid.float()

    @staticmethod
    def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        kernel_size = 2 * radius + 1
        return F.max_pool3d(
            mask[:, None],
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )[:, 0]

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
        sample_weights: torch.Tensor | None = None,
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
            loss = F.softplus(hard_logits).mean()
            if sample_weights is not None:
                loss = loss * sample_weights[batch_idx].to(loss.device, loss.dtype)
            losses.append(loss)

        if not losses:
            return prob.new_zeros(())
        return torch.stack(losses).mean()

    def _inside_liver_volume_guard(
        self,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce in-liver FP pressure when tumor occupies a large foreground share."""
        threshold = max(0.0, float(self.inside_liver_volume_guard_threshold))
        min_scale = max(0.0, min(1.0, float(self.inside_liver_volume_guard_min_scale)))
        if threshold <= 0:
            return tumor.new_ones((tumor.shape[0],))

        spatial_dims = tuple(range(1, tumor.ndim))
        tumor_voxels = (tumor * valid).sum(dim=spatial_dims)
        foreground_voxels = ((tumor + liver).clamp_(0.0, 1.0) * valid).sum(
            dim=spatial_dims
        )
        tumor_fraction = tumor_voxels / foreground_voxels.clamp_min(self.smooth)
        raw_scale = threshold / tumor_fraction.clamp_min(self.smooth)
        return torch.where(
            tumor_voxels > 0,
            raw_scale.clamp(min=min_scale, max=1.0),
            torch.ones_like(raw_scale),
        )

    def _tumor_foreground_fraction(
        self,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_dims = tuple(range(1, tumor.ndim))
        tumor_voxels = (tumor * valid).sum(dim=spatial_dims)
        foreground_voxels = ((tumor + liver).clamp(0.0, 1.0) * valid).sum(
            dim=spatial_dims
        )
        tumor_fraction = tumor_voxels / foreground_voxels.clamp_min(self.smooth)
        return tumor_voxels, foreground_voxels, tumor_fraction

    def _custom_loss_gate(
        self,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep custom BoundaryOverseg terms on small tumors and fade them on large ones."""
        tumor_voxels, _, tumor_fraction = self._tumor_foreground_fraction(
            tumor,
            liver,
            valid,
        )
        threshold = max(0.0, float(self.custom_loss_gate_threshold))
        temperature = max(float(self.custom_loss_gate_temperature), self.smooth)
        min_scale = max(0.0, min(1.0, float(self.custom_loss_gate_min_scale)))
        large_gate = torch.sigmoid((tumor_fraction - threshold) / temperature)
        custom_gate = 1.0 - large_gate * (1.0 - min_scale)
        return torch.where(
            tumor_voxels > 0,
            custom_gate,
            torch.ones_like(custom_gate),
        ), tumor_fraction

    def _adaptive_large_tumor_factor(
        self,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """0 for ordinary patches, 1 for high tumor-burden patches."""
        tumor_voxels, _, tumor_fraction = self._tumor_foreground_fraction(
            tumor,
            liver,
            valid,
        )
        threshold = max(0.0, float(self.adaptive_large_tumor_threshold))
        max_threshold = max(threshold, float(self.adaptive_large_tumor_max_threshold))
        if max_threshold <= threshold:
            factor = (tumor_fraction > threshold).to(tumor_fraction.dtype)
        else:
            factor = (tumor_fraction - threshold) / (max_threshold - threshold)
        factor = factor.clamp(0.0, 1.0)
        return torch.where(tumor_voxels > 0, factor, torch.zeros_like(factor))

    def _adaptive_fp_scale(
        self,
        adaptive_factor: torch.Tensor,
    ) -> torch.Tensor:
        min_scale = max(0.0, min(1.0, float(self.adaptive_fp_min_scale)))
        return 1.0 - adaptive_factor * (1.0 - min_scale)

    def _adaptive_dilate(
        self,
        tumor: torch.Tensor,
        base_radius: int,
        adaptive_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extra_radius = max(0, int(self.adaptive_ignore_extra_radius))
        if extra_radius == 0 or tumor.shape[0] == 0:
            radius = max(0, int(base_radius))
            return self._dilate(tumor, radius), tumor.new_full(
                (tumor.shape[0],),
                float(radius),
            )

        masks: list[torch.Tensor] = []
        radii: list[float] = []
        for batch_idx in range(tumor.shape[0]):
            factor = float(adaptive_factor[batch_idx].detach().cpu())
            radius = max(0, int(base_radius)) + int(round(extra_radius * factor))
            masks.append(self._dilate(tumor[batch_idx : batch_idx + 1], radius))
            radii.append(float(radius))
        return torch.cat(masks, dim=0), tumor.new_tensor(radii)

    def _boundary_loss(
        self,
        tumor_logits: torch.Tensor,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
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
        denominator = (
            pred_ring.sum(dim=spatial_dims)
            + target_ring.sum(dim=spatial_dims)
        )
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (denominator + self.smooth)
        )
        per_sample_loss = bce + dice_loss
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(
                per_sample_loss.device,
                per_sample_loss.dtype,
            )
        return per_sample_loss[has_ring].mean()

    def _hard_false_positive_loss(
        self,
        tumor_logits: torch.Tensor,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        background: torch.Tensor,
        valid: torch.Tensor,
        custom_loss_gate: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        prob = prob * valid
        tumor = tumor * valid

        adaptive_factor = self._adaptive_large_tumor_factor(tumor, liver, valid)
        adaptive_fp_scale = self._adaptive_fp_scale(adaptive_factor) * custom_loss_gate
        outside_ignore, outside_ignore_radius = self._adaptive_dilate(
            tumor,
            self.outside_liver_ignore_radius,
            adaptive_factor,
        )
        inside_ignore, inside_ignore_radius = self._adaptive_dilate(
            tumor,
            self.inside_liver_ignore_radius,
            adaptive_factor,
        )
        outside_ignore = outside_ignore.clamp_(0.0, 1.0)
        inside_ignore = inside_ignore.clamp_(0.0, 1.0)
        outside_liver_mask = background * (1.0 - outside_ignore) * valid
        inside_liver_mask = liver * (1.0 - inside_ignore) * valid

        outside_liver_loss = self._hard_negative_loss(
            tumor_logits,
            prob,
            outside_liver_mask,
            self.outside_liver_topk_fraction,
            sample_weights=adaptive_fp_scale,
        )
        inside_liver_scale = (
            self._inside_liver_volume_guard(tumor, liver, valid) * adaptive_fp_scale
        )
        inside_liver_loss = self._hard_negative_loss(
            tumor_logits,
            prob,
            inside_liver_mask,
            self.inside_liver_topk_fraction,
            sample_weights=inside_liver_scale,
        )
        return (
            outside_liver_loss,
            inside_liver_loss,
            inside_liver_scale.mean(),
            adaptive_factor.mean(),
            adaptive_fp_scale.mean(),
            outside_ignore_radius.mean(),
            inside_ignore_radius.mean(),
        )

    def _tversky_guard_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        valid: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
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
        tumor_voxels = tumor.sum(dim=spatial_dims)
        has_tumor = tumor_voxels > 0
        if not torch.any(has_tumor):
            return prob.new_zeros(())

        denominator = (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + self.smooth
        )
        tversky = (true_positive + self.smooth) / denominator
        per_sample_loss = 1.0 - tversky
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(
                per_sample_loss.device,
                per_sample_loss.dtype,
            )
        return per_sample_loss[has_tumor].mean()

    def _under_volume_guard_loss(
        self,
        prob: torch.Tensor,
        tumor: torch.Tensor,
        liver: torch.Tensor,
        valid: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        threshold = max(0.0, float(self.under_volume_guard_threshold))
        target_fraction = max(0.0, float(self.under_volume_guard_fraction))
        if threshold <= 0 or target_fraction <= 0:
            return prob.new_zeros(())

        tumor_voxels, _, tumor_fraction = self._tumor_foreground_fraction(
            tumor,
            liver,
            valid,
        )
        active = (tumor_voxels > 0) & (tumor_fraction >= threshold)
        if not torch.any(active):
            return prob.new_zeros(())

        spatial_dims = tuple(range(1, prob.ndim))
        predicted_voxels = (prob * valid).sum(dim=spatial_dims)
        required_voxels = target_fraction * tumor_voxels
        relative_deficit = (
            (required_voxels - predicted_voxels).clamp_min(0.0)
            / tumor_voxels.clamp_min(self.smooth)
        )
        per_sample_loss = relative_deficit.square()
        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights.to(
                per_sample_loss.device,
                per_sample_loss.dtype,
            )
        return per_sample_loss[active].mean()

    def forward(
        self,
        outputs: TensorOrDeepSupervision,
        targets: TensorOrDeepSupervision,
    ) -> torch.Tensor:
        base_loss = self.base_loss(outputs, targets)
        loss = base_loss

        boundary_scale = self.boundary_loss_scale
        fp_scale = self.fp_loss_scale

        logits = self._full_resolution(outputs)
        target = self._full_resolution(targets)
        tumor, liver, background, valid = self._target_masks(target)
        custom_loss_gate, tumor_fraction = self._custom_loss_gate(tumor, liver, valid)
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
        outside_liver_loss = loss.new_zeros(())
        inside_liver_loss = loss.new_zeros(())
        inside_liver_volume_scale = loss.new_ones(())
        custom_loss_gate_mean = custom_loss_gate.mean()
        tumor_fraction_mean = tumor_fraction.mean()
        adaptive_large_tumor_factor = loss.new_zeros(())
        adaptive_fp_scale = loss.new_ones(())
        outside_ignore_radius = loss.new_tensor(float(self.outside_liver_ignore_radius))
        inside_ignore_radius = loss.new_tensor(float(self.inside_liver_ignore_radius))
        tversky_guard_loss = loss.new_zeros(())
        under_volume_guard_loss = loss.new_zeros(())

        if self.boundary_weight > 0 and boundary_scale > 0:
            boundary_loss = self._boundary_loss(
                tumor_logits,
                prob,
                tumor,
                valid,
                sample_weights=custom_loss_gate,
            )
            loss = loss + boundary_scale * self.boundary_weight * boundary_loss
        if (
            self.overseg_weight > 0
            and fp_scale > 0
            and (self.outside_liver_fp_weight > 0 or self.inside_liver_fp_weight > 0)
        ):
            (
                outside_liver_loss,
                inside_liver_loss,
                inside_liver_volume_scale,
                adaptive_large_tumor_factor,
                adaptive_fp_scale,
                outside_ignore_radius,
                inside_ignore_radius,
            ) = self._hard_false_positive_loss(
                tumor_logits,
                prob,
                tumor,
                liver,
                background,
                valid,
                custom_loss_gate,
            )
            false_positive_loss = (
                self.outside_liver_fp_weight * outside_liver_loss
                + self.inside_liver_fp_weight * inside_liver_loss
            )
            loss = loss + fp_scale * self.overseg_weight * false_positive_loss
        if self.tversky_guard_weight > 0 and fp_scale > 0:
            tversky_guard_loss = self._tversky_guard_loss(
                prob,
                tumor,
                valid,
                sample_weights=custom_loss_gate,
            )
            loss = loss + fp_scale * self.tversky_guard_weight * tversky_guard_loss
        if self.under_volume_guard_weight > 0 and fp_scale > 0:
            if self.under_volume_inverse_gate:
                uv_weights = (1.0 - custom_loss_gate).clamp(0.0, 1.0)
            else:
                uv_weights = custom_loss_gate
            under_volume_guard_loss = self._under_volume_guard_loss(
                prob,
                tumor,
                liver,
                valid,
                sample_weights=uv_weights,
            )
            loss = (
                loss
                + fp_scale * self.under_volume_guard_weight * under_volume_guard_loss
            )

        self.last_components = {
            "base_loss": float(base_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "outside_fp_loss": float(outside_liver_loss.detach().cpu()),
            "inside_fp_loss": float(inside_liver_loss.detach().cpu()),
            "inside_fp_volume_scale": float(inside_liver_volume_scale.detach().cpu()),
            "custom_loss_gate": float(custom_loss_gate_mean.detach().cpu()),
            "tumor_fraction": float(tumor_fraction_mean.detach().cpu()),
            "adaptive_large_tumor_factor": float(
                adaptive_large_tumor_factor.detach().cpu()
            ),
            "adaptive_fp_min_scale_floor": float(self.adaptive_fp_min_scale),
            "adaptive_fp_scale": float(adaptive_fp_scale.detach().cpu()),
            "outside_ignore_radius": float(outside_ignore_radius.detach().cpu()),
            "inside_ignore_radius": float(inside_ignore_radius.detach().cpu()),
            "tversky_guard_loss": float(tversky_guard_loss.detach().cpu()),
            "under_volume_guard_loss": float(under_volume_guard_loss.detach().cpu()),
            "under_volume_inverse_gate": float(self.under_volume_inverse_gate),
            "total_loss": float(loss.detach().cpu()),
            "boundary_scale": boundary_scale,
            "fp_scale": fp_scale,
        }
        return loss
