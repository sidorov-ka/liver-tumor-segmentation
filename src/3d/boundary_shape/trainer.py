from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_150,
)

from boundary_shape.config import BoundaryOversegConfig
from boundary_shape.losses import BoundaryOversegmentationLoss


class nnUNetTrainer_150_BoundaryOverseg_50epochs(nnUNetTrainer_150):
    """Fine-tune the 150-epoch 3D baseline with a boundary/overseg loss."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.boundary_overseg_config = BoundaryOversegConfig.from_env()
        self.num_epochs = self.boundary_overseg_config.num_epochs
        self.initial_lr = self.boundary_overseg_config.initial_lr

    def _build_loss(self):
        base_loss = super()._build_loss()
        config = self.boundary_overseg_config
        return BoundaryOversegmentationLoss(
            base_loss=base_loss,
            tumor_label=config.tumor_label,
            liver_label=config.liver_label,
            boundary_weight=config.boundary_weight,
            overseg_weight=config.overseg_weight,
            outside_liver_fp_weight=config.outside_liver_fp_weight,
            inside_liver_fp_weight=config.inside_liver_fp_weight,
            boundary_radius=config.boundary_radius,
            outside_liver_ignore_radius=config.outside_liver_ignore_radius,
            inside_liver_ignore_radius=config.inside_liver_ignore_radius,
            outside_liver_topk_fraction=config.outside_liver_topk_fraction,
            inside_liver_topk_fraction=config.inside_liver_topk_fraction,
            inside_liver_volume_guard_threshold=config.inside_liver_volume_guard_threshold,
            inside_liver_volume_guard_min_scale=config.inside_liver_volume_guard_min_scale,
            tversky_guard_weight=config.tversky_guard_weight,
            tversky_guard_alpha=config.tversky_guard_alpha,
            tversky_guard_beta=config.tversky_guard_beta,
            adaptive_large_tumor_threshold=config.adaptive_large_tumor_threshold,
            adaptive_large_tumor_max_threshold=(
                config.adaptive_large_tumor_max_threshold
            ),
            adaptive_fp_min_scale=config.adaptive_fp_min_scale,
            adaptive_ignore_extra_radius=config.adaptive_ignore_extra_radius,
            under_volume_guard_weight=config.under_volume_guard_weight,
            under_volume_guard_threshold=config.under_volume_guard_threshold,
            under_volume_guard_fraction=config.under_volume_guard_fraction,
        )

    def initialize(self):
        super().initialize()
        config = self.boundary_overseg_config
        self.print_to_log_file(
            "BoundaryOverseg fine-tune: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            f"tumor_label={config.tumor_label}, "
            f"liver_label={config.liver_label}, "
            f"boundary_weight={config.boundary_weight}, "
            f"overseg_weight={config.overseg_weight}, "
            f"outside_liver_fp_weight={config.outside_liver_fp_weight}, "
            f"inside_liver_fp_weight={config.inside_liver_fp_weight}, "
            f"boundary_radius={config.boundary_radius}, "
            f"outside_liver_ignore_radius={config.outside_liver_ignore_radius}, "
            f"inside_liver_ignore_radius={config.inside_liver_ignore_radius}, "
            f"outside_liver_topk_fraction={config.outside_liver_topk_fraction}, "
            f"inside_liver_topk_fraction={config.inside_liver_topk_fraction}, "
            f"inside_liver_volume_guard_threshold={config.inside_liver_volume_guard_threshold}, "
            f"inside_liver_volume_guard_min_scale={config.inside_liver_volume_guard_min_scale}, "
            f"tversky_guard_weight={config.tversky_guard_weight}, "
            f"tversky_guard_alpha={config.tversky_guard_alpha}, "
            f"tversky_guard_beta={config.tversky_guard_beta}, "
            f"adaptive_large_tumor_threshold={config.adaptive_large_tumor_threshold}, "
            f"adaptive_large_tumor_max_threshold={config.adaptive_large_tumor_max_threshold}, "
            f"adaptive_fp_min_scale={config.adaptive_fp_min_scale}, "
            f"adaptive_ignore_extra_radius={config.adaptive_ignore_extra_radius}, "
            f"under_volume_guard_weight={config.under_volume_guard_weight}, "
            f"under_volume_guard_threshold={config.under_volume_guard_threshold}, "
            f"under_volume_guard_fraction={config.under_volume_guard_fraction}, "
            f"boundary_start_epoch={config.boundary_start_epoch}, "
            f"fp_start_epoch={config.fp_start_epoch}, "
            f"custom_loss_ramp_epochs={config.custom_loss_ramp_epochs}"
        )

    @staticmethod
    def _ramp_scale(current_epoch: int, start_epoch: int, ramp_epochs: int) -> float:
        if current_epoch < start_epoch:
            return 0.0
        return min(1.0, (current_epoch - start_epoch + 1) / max(ramp_epochs, 1))

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        config = self.boundary_overseg_config
        boundary_scale = self._ramp_scale(
            self.current_epoch,
            config.boundary_start_epoch,
            config.custom_loss_ramp_epochs,
        )
        fp_scale = self._ramp_scale(
            self.current_epoch,
            config.fp_start_epoch,
            config.custom_loss_ramp_epochs,
        )
        self.loss.set_custom_loss_scales(boundary_scale, fp_scale)
        self.print_to_log_file(
            "BoundaryOverseg custom loss scales: "
            f"boundary={boundary_scale:.4f}, fp={fp_scale:.4f}"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        components = getattr(self.loss, "last_components", None)
        if not components:
            return
        self.print_to_log_file(
            "BoundaryOverseg loss components: "
            + ", ".join(f"{key}={value:.4f}" for key, value in components.items())
        )
