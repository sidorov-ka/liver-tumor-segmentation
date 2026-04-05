"""NPZ dataset: CT + coarse tumor prob + entropy(coarse prob); optional boundary ring for loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from boundary_aware_coarse_to_fine.boundary import boundary_ring_from_binary_mask
from boundary_aware_coarse_to_fine.metrics import parse_case_id_from_npz
from boundary_aware_coarse_to_fine.roi import bbox2d_from_mask
from boundary_aware_coarse_to_fine.utils import bernoulli_entropy_numpy


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _resize_triplet_prob(
    img: np.ndarray,
    coarse_prob: np.ndarray,
    gt: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear for CT and coarse prob; nearest for GT. Uncertainty from resized coarse prob."""
    h, w = size
    ti = torch.from_numpy(img)[None, None].float()
    ti = F.interpolate(ti, size=(h, w), mode="bilinear", align_corners=False)
    img_r = ti[0, 0].numpy()
    tc = torch.from_numpy(coarse_prob)[None, None].float()
    tc = F.interpolate(tc, size=(h, w), mode="bilinear", align_corners=False)
    coarse_r = tc[0, 0].numpy()
    unc_r = bernoulli_entropy_numpy(coarse_r)
    m2_t = torch.from_numpy(gt)[None, None].float()
    gt_r = F.interpolate(m2_t, size=(h, w), mode="nearest")[0, 0].numpy()
    return img_r, coarse_r, unc_r, gt_r


class BoundaryAwareRefinementSliceDataset(Dataset):
    """
    Reads .npz files produced by scripts/export.py (same as coarse_to_fine).

    Each sample:
      input: (3, H, W) — CT + coarse tumor probability + Bernoulli entropy(coarse prob)
      target: (1, H, W) — GT tumor binary
      ring: (1, H, W) — boundary ring from coarse binary mask (for optional weighted loss)
    """

    def __init__(
        self,
        manifest: List[Path],
        crop_size: Tuple[int, int] = (256, 256),
        use_coarse_prob: bool = True,
        roi_aligned: bool = True,
        roi_pad_xy: Tuple[int, int] = (16, 16),
        min_roi_xy: Tuple[int, int] = (32, 32),
        boundary_dilate_iters: int = 2,
        boundary_erode_iters: int = 2,
        coarse_bin_threshold: float = 0.5,
    ):
        self.manifest = manifest
        self.crop_size = crop_size
        self.use_coarse_prob = use_coarse_prob
        self.roi_aligned = roi_aligned
        self.roi_pad_xy = roi_pad_xy
        self.min_roi_xy = min_roi_xy
        self.boundary_dilate_iters = boundary_dilate_iters
        self.boundary_erode_iters = boundary_erode_iters
        self.coarse_bin_threshold = coarse_bin_threshold

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

        img = _normalize_slice(img)

        if self.use_coarse_prob:
            img, coarse, unc, gt = _resize_triplet_prob(img, coarse, gt, self.crop_size)
        else:
            raise ValueError(
                "boundary_aware_coarse_to_fine expects coarse probability for channel 2–3; "
                "omit --no-coarse-prob or use coarse_to_fine export with coarse_tumor_prob."
            )

        coarse_bin_small = (coarse > self.coarse_bin_threshold).astype(np.float32)
        ring = boundary_ring_from_binary_mask(
            coarse_bin_small,
            dilate_iters=self.boundary_dilate_iters,
            erode_iters=self.boundary_erode_iters,
        )

        x = np.stack([img, coarse, unc], axis=0).astype(np.float32)
        y = gt[None, ...].astype(np.float32)
        ring = ring[None, ...].astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "ring": torch.from_numpy(ring),
            "case_id": case_id,
            "npz_path": str(p),
        }


def load_manifest(export_root: Path) -> Tuple[List[Path], List[Path]]:
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
    boundary_dilate_iters: int = 2,
    boundary_erode_iters: int = 2,
    coarse_bin_threshold: float = 0.5,
) -> Tuple[BoundaryAwareRefinementSliceDataset, BoundaryAwareRefinementSliceDataset]:
    train_paths, val_paths = load_manifest(export_root)
    if max_train is not None:
        train_paths = train_paths[:max_train]
    common = dict(
        crop_size=crop_size,
        use_coarse_prob=use_coarse_prob,
        roi_aligned=roi_aligned,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
        boundary_dilate_iters=boundary_dilate_iters,
        boundary_erode_iters=boundary_erode_iters,
        coarse_bin_threshold=coarse_bin_threshold,
    )
    train_ds = BoundaryAwareRefinementSliceDataset(train_paths, **common)
    val_ds = BoundaryAwareRefinementSliceDataset(val_paths, **common)
    return train_ds, val_ds


def save_manifest_json(export_root: Path, meta: Dict[str, Any]) -> None:
    (export_root / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
