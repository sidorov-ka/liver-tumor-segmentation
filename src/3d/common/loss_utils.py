from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


TensorOrDeepSupervision = torch.Tensor | Sequence[torch.Tensor]


def full_resolution(x: TensorOrDeepSupervision) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def target_masks(
    target: torch.Tensor,
    tumor_label: int = 2,
    liver_label: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim >= 2 and target.shape[1] == 1:
        target = target[:, 0]
    valid = target >= 0
    tumor = (target == tumor_label).float()
    liver = (target == liver_label).float()
    background = (target == 0).float()
    return tumor, liver, background, valid.float()


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool3d(
        mask[:, None],
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )[:, 0]


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    return -F.max_pool3d(-mask[:, None], kernel_size=3, stride=1, padding=1)[:, 0]


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(mask[:, None], kernel_size=3, stride=1, padding=1)[:, 0]


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Soft-clDice skeleton approximation (Shit et al.)."""
    skeleton = torch.zeros_like(mask)
    current = mask
    for _ in range(max(1, int(iterations))):
        opened = soft_open(current)
        delta = (current - opened).clamp_min(0.0)
        skeleton = skeleton + soft_erode(delta)
        current = soft_erode(current)
    return skeleton.clamp(0.0, 1.0)
