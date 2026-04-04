"""Full-volume tumor probability refinement inside coarse-tumor (± optional uncertainty) ROIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from coarse_to_fine.roi import BBox3D, bbox3d_from_mask

from multiview.ct_windows import stack_multi_window
from multiview.suspicious import connected_components_3d

from uncertainty.config import UncertaintyConfig
from uncertainty.uncertainty import binary_entropy_probability, normalize_entropy_01

if TYPE_CHECKING:
    import torch.nn as nn


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def refinement_seed_mask(prob: np.ndarray, cfg: UncertaintyConfig) -> np.ndarray:
    """
    Primary ROI seed: coarse tumor (p >= roi_positive_threshold).
    If ``uncertainty_threshold`` is set, also union voxels with high normalized entropy and
    sufficiently large p (see ``prob_min_for_uncertainty_union``) — entropy is not the sole driver.
    """
    p = prob.astype(np.float32)
    u = binary_entropy_probability(p)
    u01 = normalize_entropy_01(u)
    seed = (p >= float(cfg.roi_positive_threshold)).astype(np.uint8)
    if cfg.uncertainty_threshold is not None:
        t = float(cfg.uncertainty_threshold)
        extra = (u01 > t) & (p >= float(cfg.prob_min_for_uncertainty_union))
        seed = (seed | extra.astype(np.uint8)).astype(np.uint8)
    return seed


def _blend_alpha(cfg: UncertaintyConfig) -> float:
    mode = (cfg.update_mode or "replace").strip().lower()
    if mode == "replace":
        return 1.0
    if mode != "blend":
        raise ValueError(f"update_mode must be 'replace' or 'blend', got {cfg.update_mode!r}")
    return float(cfg.alpha)


def _forward_patch(
    model: "nn.Module",
    ct_patch_hw: np.ndarray,
    prob_patch_hw: np.ndarray,
    multi_hw: np.ndarray,
    crop_hw: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """
    Stack 5 channels: three windows + tumor prob + normalized entropy (per-channel normalize).
    Returns refined tumor probability (h, w).
    """
    h, w = ct_patch_hw.shape
    p = np.clip(prob_patch_hw.astype(np.float32), 0.0, 1.0)
    u = binary_entropy_probability(p)
    u01 = normalize_entropy_01(u)
    chans = [_normalize_slice(multi_hw[i]) for i in range(3)]
    chans.append(_normalize_slice(p))
    chans.append(_normalize_slice(u01))
    tin = np.stack(chans, axis=0)
    t = torch.from_numpy(tin).float().unsqueeze(0).to(device)
    t = F.interpolate(t, size=crop_hw, mode="bilinear", align_corners=False)
    with torch.no_grad():
        logit = model(t)
        pr = torch.sigmoid(logit)
    pr = F.interpolate(pr, size=(h, w), mode="bilinear", align_corners=False)
    return pr[0, 0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def refine_tumor_probability_volume(
    ct_hu: np.ndarray,
    prob_tumor: np.ndarray,
    model: "nn.Module",
    device: torch.device,
    cfg: UncertaintyConfig,
) -> np.ndarray:
    """
    ct_hu: (Z, Y, X) HU, aligned with prob_tumor.
    prob_tumor: (Z, Y, X) nnU-Net tumor class probability.
    Returns updated tumor probability (baseline outside ROI seeds; refined inside component ROIs).
    """
    if ct_hu.shape != prob_tumor.shape:
        raise ValueError(f"ct_hu {ct_hu.shape} != prob_tumor {prob_tumor.shape}")
    out = prob_tumor.astype(np.float32).copy()
    seed = refinement_seed_mask(out, cfg)
    labeled, n = connected_components_3d(seed)
    if n == 0:
        return out

    h_crop, w_crop = cfg.crop_size
    model.eval()
    alpha = _blend_alpha(cfg)

    for label_id in range(1, n + 1):
        comp = labeled == label_id
        nv = int(comp.sum())
        if nv < cfg.min_component_voxels:
            continue
        bbox = bbox3d_from_mask(
            comp.astype(np.float32),
            pad=cfg.roi_pad,
            min_side=cfg.min_roi_side,
        )
        if bbox is None:
            continue
        _refine_one_bbox(
            out,
            ct_hu,
            bbox,
            model,
            device,
            cfg,
            h_crop,
            w_crop,
            alpha,
        )
    return out


def _refine_one_bbox(
    out: np.ndarray,
    ct_hu: np.ndarray,
    bbox: BBox3D,
    model: "nn.Module",
    device: torch.device,
    cfg: UncertaintyConfig,
    h_crop: int,
    w_crop: int,
    alpha: float,
) -> None:
    z0, z1, y0, y1, x0, x1 = bbox.z0, bbox.z1, bbox.y0, bbox.y1, bbox.x0, bbox.x1
    for z in range(z0, z1):
        img2d = ct_hu[z, y0:y1, x0:x1]
        if img2d.size == 0:
            continue
        pr2d = out[z, y0:y1, x0:x1]
        multi = stack_multi_window(img2d, cfg.hu_windows)
        pr_new = _forward_patch(
            model,
            img2d,
            pr2d,
            multi,
            (h_crop, w_crop),
            device,
        )
        if alpha >= 1.0 - 1e-7:
            out[z, y0:y1, x0:x1] = pr_new
        else:
            coarse2d = pr2d.astype(np.float32)
            out[z, y0:y1, x0:x1] = alpha * pr_new + (1.0 - alpha) * coarse2d
