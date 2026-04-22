"""NPZ dataset: three HU windows + coarse prob + entropy; optional boundary ring for loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from boundary_aware_coarse_to_fine.boundary import boundary_ring_from_binary_mask
from boundary_aware_coarse_to_fine.config import DEFAULT_HU_WINDOWS
from boundary_aware_coarse_to_fine.metrics import parse_case_id_from_npz
from boundary_aware_coarse_to_fine.roi import bbox2d_from_mask
from multiview.ct_windows import stack_multi_window
from uncertainty.uncertainty import binary_entropy_probability, normalize_entropy_01


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _resize_multi_coarse_gt(
    multi_hw: np.ndarray,
    coarse_prob: np.ndarray,
    gt: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear for (3,H,W) multi-window and coarse prob; nearest for GT."""
    h, w = size
    t = torch.from_numpy(multi_hw).float().unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    multi_r = t[0].numpy()
    tc = torch.from_numpy(coarse_prob)[None, None].float()
    tc = F.interpolate(tc, size=(h, w), mode="bilinear", align_corners=False)
    coarse_r = tc[0, 0].numpy()
    m2_t = torch.from_numpy(gt)[None, None].float()
    gt_r = F.interpolate(m2_t, size=(h, w), mode="nearest")[0, 0].numpy()
    return multi_r, coarse_r, gt_r


def prepare_input_tensor_5ch(
    img_hu_hw: np.ndarray,
    coarse_prob_hw: np.ndarray,
    crop_hw: Tuple[int, int],
    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> np.ndarray:
    """Align ROI slice to ``crop_hw`` and build (5, H, W) like training (for inference)."""
    gt_dummy = np.zeros(img_hu_hw.shape, dtype=np.float32)
    multi = stack_multi_window(img_hu_hw.astype(np.float32), hu_windows)
    multi_r, coarse_r, _ = _resize_multi_coarse_gt(multi, coarse_prob_hw, gt_dummy, crop_hw)
    return _build_x5_from_resized(multi_r, coarse_r)


class BoundaryAwareRefinementSliceDataset(Dataset):
    """
    Reads .npz files produced by scripts/export.py (same as coarse_to_fine).

    Each sample:
      input: (5, H, W) — three HU-window channels + coarse tumor probability + normalized entropy
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
        hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = DEFAULT_HU_WINDOWS,
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
        self.hu_windows = hu_windows

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

        multi, coarse_r, gt_r = _resize_multi_coarse_gt(
            stack_multi_window(img, self.hu_windows),
            coarse,
            gt,
            self.crop_size,
        )
        x5 = _build_x5_from_resized(multi, coarse_r)

        coarse_bin_small = (coarse_r > self.coarse_bin_threshold).astype(np.float32)
        ring = boundary_ring_from_binary_mask(
            coarse_bin_small,
            dilate_iters=self.boundary_dilate_iters,
            erode_iters=self.boundary_erode_iters,
        )

        y = gt_r[None, ...].astype(np.float32)
        ring = ring[None, ...].astype(np.float32)

        return {
            "x": torch.from_numpy(x5),
            "y": torch.from_numpy(y),
            "ring": torch.from_numpy(ring),
            "case_id": case_id,
            "npz_path": str(p),
        }


def _build_x5_from_resized(
    multi_hw: np.ndarray,
    coarse_prob_hw: np.ndarray,
) -> np.ndarray:
    """Build 5-channel input from already spatially aligned multi (3,H,W) and coarse (H,W)."""
    coarse = np.clip(coarse_prob_hw.astype(np.float32), 0.0, 1.0)
    u = binary_entropy_probability(coarse)
    u01 = normalize_entropy_01(u)
    chans = [_normalize_slice(multi_hw[i]) for i in range(3)]
    chans.append(_normalize_slice(coarse))
    chans.append(_normalize_slice(u01.astype(np.float32)))
    return np.stack(chans, axis=0).astype(np.float32)


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
    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = DEFAULT_HU_WINDOWS,
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
        hu_windows=hu_windows,
    )
    train_ds = BoundaryAwareRefinementSliceDataset(train_paths, **common)
    val_ds = BoundaryAwareRefinementSliceDataset(val_paths, **common)
    return train_ds, val_ds


def save_manifest_json(export_root: Path, meta: Dict[str, Any]) -> None:
    (export_root / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
