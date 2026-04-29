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
            boundary_weight=config.boundary_weight,
            overseg_weight=config.overseg_weight,
            boundary_radius=config.boundary_radius,
        )

    def initialize(self):
        super().initialize()
        config = self.boundary_overseg_config
        self.print_to_log_file(
            "BoundaryOverseg fine-tune: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            f"tumor_label={config.tumor_label}, "
            f"boundary_weight={config.boundary_weight}, "
            f"overseg_weight={config.overseg_weight}, "
            f"boundary_radius={config.boundary_radius}"
        )
