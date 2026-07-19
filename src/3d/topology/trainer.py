from __future__ import annotations

import torch
from nnunet_trainer_500_compat import nnUNetTrainer_500

from topology.config import TopologyConfig
from topology.losses import SoftClDiceLoss


class nnUNetTrainer_500_Topology(nnUNetTrainer_500):
    """Arm 4: soft clDice topology loss, train from scratch."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.topology_config = TopologyConfig.from_env()
        self.num_epochs = self.topology_config.num_epochs
        self.initial_lr = self.topology_config.initial_lr

    def _build_loss(self):
        base_loss = super()._build_loss()
        config = self.topology_config
        return SoftClDiceLoss(
            base_loss=base_loss,
            tumor_label=config.tumor_label,
            liver_label=config.liver_label,
            cldice_weight=config.cldice_weight,
            skeleton_iterations=config.skeleton_iterations,
        )

    def initialize(self):
        super().initialize()
        config = self.topology_config
        self.print_to_log_file(
            "Topology arm: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            f"cldice_weight={config.cldice_weight}, "
            f"skeleton_iterations={config.skeleton_iterations}, "
            "curriculum=off"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        components = getattr(self.loss, "last_components", None)
        if not components:
            return
        self.print_to_log_file(
            "Topology loss components: "
            + ", ".join(f"{key}={value:.4f}" for key, value in components.items())
        )
