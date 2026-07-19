from __future__ import annotations

import torch
from nnunet_trainer_500_compat import nnUNetTrainer_500

from baseline.config import BaselineConfig


class nnUNetTrainer_500_Baseline(nnUNetTrainer_500):
    """Arm 1: default nnU-Net Dice+CE, train from scratch."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.baseline_config = BaselineConfig.from_env()
        self.num_epochs = self.baseline_config.num_epochs
        self.initial_lr = self.baseline_config.initial_lr

    def initialize(self):
        super().initialize()
        config = self.baseline_config
        self.print_to_log_file(
            "Baseline arm: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            "loss=default nnU-Net Dice+CE"
        )
