from __future__ import annotations

import torch
from nnunet_trainer_500_compat import nnUNetTrainer_500

from anatomical.config import AnatomicalConfig
from anatomical.losses import AnatomicalLoss


class nnUNetTrainer_500_Anatomical(nnUNetTrainer_500):
    """Arm 2: anatomy-constrained additive loss, train from scratch."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.anatomical_config = AnatomicalConfig.from_env()
        self.num_epochs = self.anatomical_config.num_epochs
        self.initial_lr = self.anatomical_config.initial_lr

    def _build_loss(self):
        base_loss = super()._build_loss()
        config = self.anatomical_config
        return AnatomicalLoss(
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
            tversky_guard_weight=config.tversky_guard_weight,
            tversky_guard_alpha=config.tversky_guard_alpha,
            tversky_guard_beta=config.tversky_guard_beta,
        )

    def initialize(self):
        super().initialize()
        config = self.anatomical_config
        self.print_to_log_file(
            "Anatomical arm: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            f"boundary_weight={config.boundary_weight}, "
            f"overseg_weight={config.overseg_weight}, "
            f"tversky_weight={config.tversky_guard_weight}, "
            "curriculum=off"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        components = getattr(self.loss, "last_components", None)
        if not components:
            return
        self.print_to_log_file(
            "Anatomical loss components: "
            + ", ".join(f"{key}={value:.4f}" for key, value in components.items())
        )
