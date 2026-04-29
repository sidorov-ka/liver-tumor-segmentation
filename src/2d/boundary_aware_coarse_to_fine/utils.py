"""Losses, entropy, checkpoint I/O, FP tumor-component removal (infer)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from multiview.ct_windows import apply_hu_window
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


def bce_term(
    logits: torch.Tensor, target: torch.Tensor, focal_gamma: float = 0.0
) -> torch.Tensor:
    """BCE or focal BCE (gamma>0): (1-p_t)^gamma * BCE per pixel, then mean."""
    if focal_gamma <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, target)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    w = (1.0 - p_t).clamp(min=1e-6).pow(focal_gamma)
    return (w * ce).mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """logits: (N, 1, H, W); target: (N, 1, H, W)."""
    bce = bce_term(logits, target, focal_gamma=focal_gamma)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def bce_dice_loss_masked(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    bce_weight: float = 0.5,
    eps: float = 1e-6,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """
    BCE+Dice restricted to pixels where ``mask > 0.5``.
    ``mask``: (N, 1, H, W). If no masked pixels in a batch item, contribution is zero.
    """
    m = (mask > 0.5).float()
    denom = m.sum(dim=(1, 2, 3))
    if (denom < 1e-6).all():
        return logits.sum() * 0.0

    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal_gamma > 0.0:
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        w = (1.0 - p_t).clamp(min=1e-6).pow(focal_gamma)
        ce = ce * w
    bce_n = ce * m
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
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """``loss = main + lambda_boundary * masked`` when ``ring`` is not None and lambda > 0."""
    main = bce_dice_loss(logits, target, bce_weight=bce_weight, focal_gamma=focal_gamma)
    if ring is None or lambda_boundary <= 0.0:
        return main
    ring_loss = bce_dice_loss_masked(
        logits, target, ring, bce_weight=bce_weight, focal_gamma=focal_gamma
    )
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


@dataclass
class FpComponentRemovalConfig:
    """Whole 3D tumor CC removal before boundary refine (infer).

    **HU (dominant):** mean of the first HU window (narrow, same as training ch0) on ``ct_volume``
    must lie in ``[hu_narrow_mean_min, hu_narrow_mean_max]`` in [0,1] windowed space; otherwise the
    whole CC is removed regardless of nnU-Net softmax.
    """

    enabled: bool = True
    hu_filter_enabled: bool = True
    hu_narrow_mean_min: float = 0.1
    hu_narrow_mean_max: float = 0.9
    mean_prob_below: Optional[float] = None
    max_voxels_small_cc: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FpComponentRemovalConfig":
        if not d:
            return cls(enabled=False)
        return cls(
            enabled=bool(d.get("enabled", True)),
            hu_filter_enabled=bool(d.get("hu_filter_enabled", True)),
            hu_narrow_mean_min=float(d.get("hu_narrow_mean_min", 0.1)),
            hu_narrow_mean_max=float(d.get("hu_narrow_mean_max", 0.9)),
            mean_prob_below=(
                float(d["mean_prob_below"]) if d.get("mean_prob_below") is not None else None
            ),
            max_voxels_small_cc=(
                int(d["max_voxels_small_cc"])
                if d.get("max_voxels_small_cc") is not None
                else None
            ),
        )


def remove_false_positive_tumor_components(
    pred_seg: np.ndarray,
    tumor_prob: Optional[np.ndarray],
    tumor_label: int,
    liver_label: int,
    cfg: FpComponentRemovalConfig,
    *,
    ct_volume: Optional[np.ndarray] = None,
    hu_windows: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, int]:
    """26-connected tumor components; revert failing ones to ``liver_label``.

    Order: (1) HU narrow-window mean on ``ct_volume`` — if outside range, remove; (2) optional mean
    softmax; (3) optional tiny CC by voxel count.
    """
    if not cfg.enabled:
        return pred_seg.copy(), 0

    vol = pred_seg.copy()
    if vol.ndim == 4:
        raise ValueError("pred_seg must be (Z,Y,X) integer labels for fp removal")
    tm = vol == tumor_label
    if not np.any(tm):
        return pred_seg.copy(), 0

    struct = np.ones((3, 3, 3), dtype=bool)
    labeled, n_comp = ndimage.label(tm, structure=struct)
    if n_comp == 0:
        return pred_seg.copy(), 0

    wnarrow: Optional[np.ndarray] = None
    if (
        cfg.hu_filter_enabled
        and ct_volume is not None
        and hu_windows is not None
        and cfg.hu_narrow_mean_min < cfg.hu_narrow_mean_max
    ):
        w0, l0 = hu_windows[0]
        wnarrow = apply_hu_window(ct_volume.astype(np.float32), w0, l0)

    removed = 0
    for k in range(1, n_comp + 1):
        cc = labeled == k
        nv = int(np.count_nonzero(cc))
        kill = False
        if wnarrow is not None:
            m = float(np.mean(wnarrow[cc]))
            if m < cfg.hu_narrow_mean_min or m > cfg.hu_narrow_mean_max:
                kill = True
        if not kill and cfg.max_voxels_small_cc is not None and nv <= cfg.max_voxels_small_cc:
            kill = True
        if (
            not kill
            and cfg.mean_prob_below is not None
            and tumor_prob is not None
            and float(np.mean(tumor_prob[cc])) < cfg.mean_prob_below
        ):
            kill = True
        if kill:
            vol[cc] = liver_label
            removed += 1

    return vol, removed
