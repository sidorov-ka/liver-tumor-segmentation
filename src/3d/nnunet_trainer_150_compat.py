"""nnUNetTrainer with num_epochs=150 — some nnunetv2 wheels omit this class in nnUNetTrainer_Xepochs."""

from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_150(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


__all__ = ["nnUNetTrainer_150"]
