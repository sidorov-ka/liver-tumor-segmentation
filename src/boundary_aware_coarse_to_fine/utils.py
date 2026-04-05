"""Losses, entropy from coarse probability, checkpoint I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F


def bernoulli_entropy_numpy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Entropy U(p) = -p log p - (1-p) log(1-p); numerically stable."""
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).astype(np.float32)


def bernoulli_entropy_torch(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Same as :func:`bernoulli_entropy_numpy` for tensors in [0, 1]."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


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


def bce_dice_loss_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bce_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    BCE+Dice restricted to pixels where ``mask > 0.5``.
    ``mask``: (N, 1, H, W). If no masked pixels in a batch item, contribution is zero.
    """
    m = (mask > 0.5).float()
    denom = m.sum(dim=(1, 2, 3))
    if (denom < 1e-6).all():
        return logits.sum() * 0.0

    bce_n = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * m
    bce = (bce_n.sum(dim=(1, 2, 3)) / (denom + eps)).mean()

    prob = torch.sigmoid(logits)
    inter = (prob * target * m).sum(dim=(1, 2, 3))
    psum = (prob * m).sum(dim=(1, 2, 3))
    tsum = (target * m).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (psum + tsum + eps)
    dice_l = 1.0 - dice.mean()

    return bce_weight * bce + (1.0 - bce_weight) * dice_l


def bce_dice_with_optional_ring(
    logits: torch.Tensor,
    target: torch.Tensor,
    ring: torch.Tensor | None,
    bce_weight: float,
    lambda_boundary: float,
) -> torch.Tensor:
    """``loss = main + lambda_boundary * masked`` when ``ring`` is not None and lambda > 0."""
    main = bce_dice_loss(logits, target, bce_weight=bce_weight)
    if ring is None or lambda_boundary <= 0.0:
        return main
    ring_loss = bce_dice_loss_masked(logits, target, ring, bce_weight=bce_weight)
    return main + lambda_boundary * ring_loss


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
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def numpy_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    inter = float(np.dot(pred, target))
    return float((2.0 * inter + eps) / (pred.sum() + target.sum() + eps))
