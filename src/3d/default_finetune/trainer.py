from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_150,
)

from default_finetune.config import DefaultFinetuneConfig


class nnUNetTrainer_150_DefaultFinetune_50epochs(nnUNetTrainer_150):
    """Fine-tune the 150-epoch 3D baseline with the default nnU-Net loss."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.default_finetune_config = DefaultFinetuneConfig.from_env()
        self.num_epochs = self.default_finetune_config.num_epochs
        self.initial_lr = self.default_finetune_config.initial_lr

    def initialize(self):
        super().initialize()
        config = self.default_finetune_config
        self.print_to_log_file(
            "Default fine-tune control: "
            f"epochs={config.num_epochs}, initial_lr={config.initial_lr}, "
            "loss=default nnU-Net Dice+CE"
        )
