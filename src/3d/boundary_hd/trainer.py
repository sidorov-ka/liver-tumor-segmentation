from __future__ import annotations

import torch
from nnunet_trainer_500_compat import nnUNetTrainer_500

from boundary_hd.config import BoundaryHDConfig
from boundary_hd.losses import SoftHausdorffLoss


class nnUNetTrainer_500_BoundaryHD(nnUNetTrainer_500):
    """Arm 3: soft Hausdorff additive loss, train from scratch."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.boundary_hd_config = BoundaryHDConfig.from_env()
        self.num_epochs = self.boundary_hd_config.num_epochs
        self.initial_lr = self.boundary_hd_config.initial_lr

    def _build_loss(self):
        base_loss = super()._build_loss()
        config = self.boundary_hd_config
        return SoftHausdorffLoss(
            base_loss=base_loss,
            tumor_label=config.tumor_label,
            liver_label=config.liver_label,
            hd_weight=config.hd_weight,
            distance_iterations=config.distance_iterations,
        )

    def initialize(self):
        super().initialize()
        config = self.boundary_hd_config
        self.print_to_log_file(
            "BoundaryHD arm: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            f"hd_weight={config.hd_weight}, "
            f"distance_iterations={config.distance_iterations}, "
            "curriculum=off"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        components = getattr(self.loss, "last_components", None)
        if not components:
            return
        self.print_to_log_file(
            "BoundaryHD loss components: "
            + ", ".join(f"{key}={value:.4f}" for key, value in components.items())
        )
