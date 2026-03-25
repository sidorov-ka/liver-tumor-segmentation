"""Small helpers: losses, metrics, checkpoint I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """pred, target: (N, 1, H, W) in [0, 1]."""
    pred = pred.contiguous()
    target = target.contiguous()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (2.0 * inter + eps) / denom


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 - dice_coefficient(pred, target, eps=eps).mean()


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5
) -> torch.Tensor:
    """logits: (N, 1, H, W); target: (N, 1, H, W)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    meta: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def numpy_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    inter = float(np.dot(pred, target))
    return float((2.0 * inter + eps) / (pred.sum() + target.sum() + eps))
