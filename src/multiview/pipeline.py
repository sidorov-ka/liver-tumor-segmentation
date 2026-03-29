"""Full-volume tumor probability refinement inside suspicious ROIs (nnU-Net tumor prob unchanged outside ROIs)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from coarse_to_fine.roi import BBox3D, bbox3d_from_mask

from multiview.config import MultiviewConfig
from multiview.ct_windows import stack_multi_window
from multiview.suspicious import connected_components_3d, suspicious_mask

if TYPE_CHECKING:
    import torch.nn as nn


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _forward_patch(
    model: "nn.Module",
    ct_patch_hw: np.ndarray,
    prob_patch_hw: np.ndarray,
    multi_hw: np.ndarray,
    crop_hw: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """
    ct_patch_hw: (h, w) HU
    prob_patch_hw: (h, w)
    multi_hw: (3, h, w) windowed [0,1]
    Stack 4 channels: three windows + prob (MultiviewUNet2d).
    Returns refined tumor probability (h, w).
    """
    h, w = ct_patch_hw.shape
    chans = [_normalize_slice(multi_hw[i]) for i in range(3)]
    chans.append(_normalize_slice(prob_patch_hw))
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
    cfg: MultiviewConfig,
) -> np.ndarray:
    """
    ct_hu: (Z, Y, X) Hounsfield units, aligned with prob_tumor.
    prob_tumor: (Z, Y, X) coarse tumor class probability (nnU-Net softmax channel).
    Returns updated tumor probability map (coarse outside all ROIs; refined inside component ROIs).
    """
    if ct_hu.shape != prob_tumor.shape:
        raise ValueError(f"ct_hu {ct_hu.shape} != prob_tumor {prob_tumor.shape}")
    out = prob_tumor.astype(np.float32).copy()
    susp = suspicious_mask(prob_tumor, cfg)
    labeled, n = connected_components_3d(susp)
    if n == 0:
        return out

    h_crop, w_crop = cfg.crop_size
    model.eval()

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
        )
    return out


def _refine_one_bbox(
    out: np.ndarray,
    ct_hu: np.ndarray,
    bbox: BBox3D,
    model: "nn.Module",
    device: torch.device,
    cfg: MultiviewConfig,
    h_crop: int,
    w_crop: int,
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
        out[z, y0:y1, x0:x1] = pr_new
