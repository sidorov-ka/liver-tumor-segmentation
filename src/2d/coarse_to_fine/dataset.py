"""NPZ dataset for coarse_to_fine (exported slices).

Training aligned with inference (scripts/infer_coarse_to_fine.py _refine_roi_slices):
  - Second channel: nnU-Net tumor **probability** (softmax), matching Universal Topology
    Refinement-style refinement of probability maps (Li et al., arXiv:2409.09796). We use real
    nnU-Net probs + 2D ROI crops; full polynomial synthesis from the paper is not implemented.
  - Optional ROI: crop around (GT ∪ coarse) with pad / min side, then **per-patch** ``_normalize_slice``
    on the CT channel (matches ``infer_coarse_to_fine._refine_roi_slices``: normalize ROI only, then resize).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from coarse_to_fine.metrics import parse_case_id_from_npz
from coarse_to_fine.roi import bbox2d_from_mask


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    """Per-slice robust normalization to zero mean / unit variance."""
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _resize_pair(
    img: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize to (H, W) using torch (bilinear for image, nearest for masks)."""
    h, w = size
    t = torch.from_numpy(img)[None, None].float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    img_r = t[0, 0].numpy()
    m1_t = torch.from_numpy(m1)[None, None].float()
    m1_r = F.interpolate(m1_t, size=(h, w), mode="nearest")[0, 0].numpy()
    m2_t = torch.from_numpy(m2)[None, None].float()
    m2_r = F.interpolate(m2_t, size=(h, w), mode="nearest")[0, 0].numpy()
    return img_r, m1_r, m2_r


def _resize_pair_prob(
    img: np.ndarray,
    coarse_prob: np.ndarray,
    gt: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear for image and coarse probability; nearest for GT."""
    h, w = size
    ti = torch.from_numpy(img)[None, None].float()
    ti = F.interpolate(ti, size=(h, w), mode="bilinear", align_corners=False)
    img_r = ti[0, 0].numpy()
    tc = torch.from_numpy(coarse_prob)[None, None].float()
    tc = F.interpolate(tc, size=(h, w), mode="bilinear", align_corners=False)
    coarse_r = tc[0, 0].numpy()
    m2_t = torch.from_numpy(gt)[None, None].float()
    gt_r = F.interpolate(m2_t, size=(h, w), mode="nearest")[0, 0].numpy()
    return img_r, coarse_r, gt_r


class RefinementSliceDataset(Dataset):
    """
    Reads .npz files produced by scripts/export.py.

    Each sample:
      input: (C, H, W) — image + coarse (prob or binary)
      target: (1, H, W) — GT tumor binary
    """

    def __init__(
        self,
        manifest: List[Path],
        crop_size: Tuple[int, int] = (256, 256),
        use_coarse_prob: bool = True,
        roi_aligned: bool = True,
        roi_pad_xy: Tuple[int, int] = (16, 16),
        min_roi_xy: Tuple[int, int] = (32, 32),
    ):
        self.manifest = manifest
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

        # Match infer_coarse_to_fine._refine_roi_slices: normalize CT on the ROI patch only (not full slice).
        img = _normalize_slice(img)

        if self.use_coarse_prob:
            img, coarse, gt = _resize_pair_prob(img, coarse, gt, self.crop_size)
        else:
            img, coarse, gt = _resize_pair(img, coarse, gt, self.crop_size)

        x = np.stack([img, coarse], axis=0).astype(np.float32)
        y = gt[None, ...].astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "case_id": case_id,
            "npz_path": str(p),
        }


def load_manifest(export_root: Path) -> Tuple[List[Path], List[Path]]:
    """Loads train/val .npz from export_root/train and export_root/val."""
    train_dir = export_root / "train"
    val_dir = export_root / "val"
    train = sorted(train_dir.glob("*.npz"))
    val = sorted(val_dir.glob("*.npz"))
    return train, val


def build_datasets(
    export_root: Path,
    crop_size: Tuple[int, int] = (256, 256),
    use_coarse_prob: bool = True,
    roi_aligned: bool = True,
    roi_pad_xy: Tuple[int, int] = (16, 16),
    min_roi_xy: Tuple[int, int] = (32, 32),
    max_train: Optional[int] = None,
) -> Tuple[RefinementSliceDataset, RefinementSliceDataset]:
    train_paths, val_paths = load_manifest(export_root)
    if max_train is not None:
        train_paths = train_paths[:max_train]
    common = dict(
        crop_size=crop_size,
        use_coarse_prob=use_coarse_prob,
        roi_aligned=roi_aligned,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
    )
    train_ds = RefinementSliceDataset(train_paths, **common)
    val_ds = RefinementSliceDataset(val_paths, **common)
    return train_ds, val_ds


def save_manifest_json(
    export_root: Path,
    meta: Dict[str, Any],
) -> None:
    (export_root / "export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
