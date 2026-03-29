"""NPZ dataset for multiview training (same export as coarse_to_fine: scripts/export.py).

Builds 4-channel input aligned with ``infer_multiview`` / ``pipeline._forward_patch``:
three HU-window maps (stack_multi_window) + nnU-Net tumor probability, then per-channel
zero-mean / unit-variance normalization on each 2D map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from coarse_to_fine.metrics import parse_case_id_from_npz
from coarse_to_fine.roi import bbox2d_from_mask
from multiview.config import MultiviewConfig
from multiview.ct_windows import stack_multi_window


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _build_multiview_x(
    img_hw: np.ndarray,
    coarse_prob_hw: np.ndarray,
    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> np.ndarray:
    """Returns (4, H, W) float32 — same channel order as inference."""
    multi = stack_multi_window(img_hw.astype(np.float32), hu_windows)
    coarse = np.clip(coarse_prob_hw.astype(np.float32), 0.0, 1.0)
    chans = [_normalize_slice(multi[i]) for i in range(3)]
    chans.append(_normalize_slice(coarse))
    return np.stack(chans, axis=0).astype(np.float32)


def _resize_multiview_gt(
    x4: np.ndarray,
    gt: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size
    t = torch.from_numpy(x4).float().unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    x4_r = t[0].numpy()
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    gt_r = F.interpolate(gt_t, size=(h, w), mode="nearest")[0, 0].numpy()
    return x4_r, gt_r


class MultiviewSliceDataset(Dataset):
    """
    Reads .npz from export (train/ and val/).

    Each sample:
      input: (4, H, W) — three window-normalized CT views + coarse tumor prob (normalized)
      target: (1, H, W) — GT tumor binary
    """

    def __init__(
        self,
        manifest: List[Path],
        mv_cfg: MultiviewConfig,
        crop_size: Tuple[int, int] = (256, 256),
        use_coarse_prob: bool = True,
        roi_aligned: bool = True,
        roi_pad_xy: Tuple[int, int] = (16, 16),
        min_roi_xy: Tuple[int, int] = (32, 32),
    ):
        self.manifest = manifest
        self.mv_cfg = mv_cfg
        self.crop_size = crop_size
        self.use_coarse_prob = use_coarse_prob
        self.roi_aligned = roi_aligned
        self.roi_pad_xy = roi_pad_xy
        self.min_roi_xy = min_roi_xy

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.manifest[idx]
        case_id = parse_case_id_from_npz(str(p))
        z = np.load(p)
        if self.use_coarse_prob and "coarse_tumor_prob" in z.files:
            coarse = z["coarse_tumor_prob"].astype(np.float32)
        else:
            coarse = z["coarse_tumor"].astype(np.float32)
        img = z["image"].astype(np.float32)
        gt = z["gt_tumor"].astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        coarse = np.clip(coarse, 0.0, 1.0)
        gt = (gt > 0.5).astype(np.float32)

        if self.roi_aligned:
            gt_bin = gt > 0.5
            coarse_bin = coarse > 0.5
            union = np.logical_or(gt_bin, coarse_bin).astype(np.float32)
            box = bbox2d_from_mask(union, pad=self.roi_pad_xy, min_side=self.min_roi_xy)
            if box is not None:
                y0, y1, x0, x1 = box
                img = img[y0:y1, x0:x1]
                coarse = coarse[y0:y1, x0:x1]
                gt = gt[y0:y1, x0:x1]

        x4 = _build_multiview_x(img, coarse, self.mv_cfg.hu_windows)
        x4, gt = _resize_multiview_gt(x4, gt, self.crop_size)
        y = gt[None, ...].astype(np.float32)

        return {
            "x": torch.from_numpy(x4),
            "y": torch.from_numpy(y),
            "case_id": case_id,
            "npz_path": str(p),
        }


def load_manifest(export_root: Path) -> Tuple[List[Path], List[Path]]:
    train_dir = export_root / "train"
    val_dir = export_root / "val"
    train = sorted(train_dir.glob("*.npz"))
    val = sorted(val_dir.glob("*.npz"))
    return train, val


def build_multiview_datasets(
    export_root: Path,
    mv_cfg: MultiviewConfig,
    crop_size: Tuple[int, int] = (256, 256),
    use_coarse_prob: bool = True,
    roi_aligned: bool = True,
    roi_pad_xy: Tuple[int, int] = (16, 16),
    min_roi_xy: Tuple[int, int] = (32, 32),
    max_train: Optional[int] = None,
) -> Tuple[MultiviewSliceDataset, MultiviewSliceDataset]:
    train_paths, val_paths = load_manifest(export_root)
    if max_train is not None:
        train_paths = train_paths[:max_train]
    common: Dict[str, Any] = dict(
        mv_cfg=mv_cfg,
        crop_size=crop_size,
        use_coarse_prob=use_coarse_prob,
        roi_aligned=roi_aligned,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
    )
    train_ds = MultiviewSliceDataset(train_paths, **common)
    val_ds = MultiviewSliceDataset(val_paths, **common)
    return train_ds, val_ds
