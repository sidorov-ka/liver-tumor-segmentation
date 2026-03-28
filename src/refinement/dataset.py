"""NPZ dataset for stage-2 refinement (exported slices)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from refinement.metrics import parse_case_id_from_npz


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


class RefinementSliceDataset(Dataset):
    """
    Reads .npz files produced by scripts/export.py.

    Each sample:
      input: (C, H, W) — image + coarse (+ optional prob)
      target: (1, H, W) — GT tumor binary
    """

    def __init__(
        self,
        manifest: List[Path],
        crop_size: Tuple[int, int] = (256, 256),
        use_coarse_prob: bool = False,
    ):
        self.manifest = manifest
        self.crop_size = crop_size
        self.use_coarse_prob = use_coarse_prob

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
        img = _normalize_slice(img)
        coarse = np.clip(coarse, 0.0, 1.0)
        gt = (gt > 0.5).astype(np.float32)

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
    use_coarse_prob: bool = False,
    max_train: Optional[int] = None,
) -> Tuple[RefinementSliceDataset, RefinementSliceDataset]:
    train_paths, val_paths = load_manifest(export_root)
    if max_train is not None:
        train_paths = train_paths[:max_train]
    train_ds = RefinementSliceDataset(train_paths, crop_size, use_coarse_prob)
    val_ds = RefinementSliceDataset(val_paths, crop_size, use_coarse_prob)
    return train_ds, val_ds


def save_manifest_json(
    export_root: Path,
    meta: Dict[str, Any],
) -> None:
    (export_root / "export_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
