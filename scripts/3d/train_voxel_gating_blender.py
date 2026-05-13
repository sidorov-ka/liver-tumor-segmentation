#!/usr/bin/env python3
"""
Train a tiny voxel-wise blender that fuses two segmentation volumes (A, B) into a final multi-class mask.

Motivation: case-wise rules (pick A or B for whole volume) can improve global Dice but hurt mean Dice badly.
This script trains a simple per-voxel gating/blending model on val (or train) split using GT.

Model:
  - Input features per voxel: one-hot(seg_A) concatenated with one-hot(seg_B)
    => shape (2 * num_classes,)
  - Output: logits over num_classes (linear layer).

This is equivalent to a learned 1x1x1 conv and is fast to train.

Outputs:
  - <out_dir>/blender.pth  (state dict)
  - <out_dir>/meta.json   (paths, split, classes, label set)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, object | None]:
    """Load segmentation array; tolerate non-gzip data with .nii.gz suffix (nnU-Net preprocessed GT quirk)."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj).astype(np.int16, copy=False)
        return arr, img.affine, img.header if hasattr(img, "header") else None
    except Exception:
        raw = path.read_bytes()
        if len(raw) >= 4 and raw[:4] == b"\x1f\x8b\x08\x00":
            raise
        from nibabel import Nifti1Image

        img2 = Nifti1Image.from_bytes(raw)
        arr = np.asanyarray(img2.dataobj).astype(np.int16, copy=False)
        return arr, img2.affine, img2.header if hasattr(img2, "header") else None


def _pred_map(pred_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.name[: -len(".nii.gz")]
        out[cid] = p
    return out


def _split_ids(pre: Path, dataset_folder: str, fold: int, split: str) -> list[str]:
    sp = (pre / dataset_folder / "splits_final.json").resolve()
    if not sp.is_file():
        raise FileNotFoundError(sp)
    splits = json.loads(sp.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold {fold} out of range (n_splits={len(splits)})")
    ids = list(splits[fold][split])
    ids.sort()
    return ids


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    # labels: (Z,Y,X) int
    out = np.zeros((num_classes, *labels.shape), dtype=np.uint8)
    for c in range(num_classes):
        out[c] = (labels == c).astype(np.uint8)
    return out


def _sample_voxels(
    rng: np.random.Generator,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    n_samples: int,
    tumor_label: int,
    tumor_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n, 2*num_classes) uint8
      y: (n,) int64
    """
    # Flatten indices
    flat = gt.reshape(-1)
    n_vox = flat.shape[0]
    n_tum = int((flat == tumor_label).sum())

    n_tum_samp = int(round(n_samples * tumor_frac))
    n_bg_samp = n_samples - n_tum_samp

    idxs: list[np.ndarray] = []
    if n_tum > 0 and n_tum_samp > 0:
        tum_idx = np.flatnonzero(flat == tumor_label)
        take = min(n_tum_samp, tum_idx.shape[0])
        idxs.append(rng.choice(tum_idx, size=take, replace=False))
        n_bg_samp = n_samples - take

    if n_bg_samp > 0:
        # sample from all voxels (includes liver/bg/others); this keeps class balance reasonable
        idxs.append(rng.integers(0, n_vox, size=n_bg_samp, endpoint=False))

    idx = np.concatenate(idxs) if idxs else rng.integers(0, n_vox, size=n_samples, endpoint=False)
    rng.shuffle(idx)

    a = seg_a.reshape(-1)[idx]
    b = seg_b.reshape(-1)[idx]
    y = gt.reshape(-1)[idx].astype(np.int64, copy=False)

    # one-hot features
    X = np.zeros((idx.shape[0], 2 * num_classes), dtype=np.uint8)
    for c in range(num_classes):
        X[:, c] = (a == c)
        X[:, num_classes + c] = (b == c)
    return X, y


class VoxelBlender(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.linear = torch.nn.Linear(2 * self.num_classes, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train voxel-wise blender (one-hot(A)+one-hot(B) -> GT).")
    p.add_argument("--pred-a", type=str, required=True, help="Folder A with *.nii.gz predictions (e.g. adaptive).")
    p.add_argument("--pred-b", type=str, required=True, help="Folder B with *.nii.gz predictions (e.g. size-gated).")
    p.add_argument("--gt-dir", type=str, required=True, help="GT folder with *.nii.gz (e.g. nnUNet_preprocessed/.../gt_segmentations).")
    p.add_argument("--out-dir", type=str, required=True, help="Output folder for blender.pth and meta.json.")
    p.add_argument("--num-classes", type=int, default=3, help="Total number of classes in masks (default 3: bg,liver,tumor).")
    p.add_argument("--tumor-label", type=int, default=2, help="Label id for tumor (default 2).")
    p.add_argument("--dataset-folder", type=str, default="Dataset001_LiverTumor")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", type=str, default="val", choices=("val", "train"), help="Which split to train on.")
    p.add_argument("--nnunet-preprocessed", type=str, default=None, help="Path to nnUNet_preprocessed (for splits_final.json).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples-per-case", type=int, default=50000, help="Random voxels sampled per case per epoch.")
    p.add_argument("--tumor-frac", type=float, default=0.30, help="Fraction of samples drawn from tumor voxels if present.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred_a_dir = Path(args.pred_a).resolve()
    pred_b_dir = Path(args.pred_b).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = Path(args.nnunet_preprocessed or (REPO_ROOT / "nnUNet_preprocessed")).resolve()
    case_ids = _split_ids(pre, args.dataset_folder, int(args.fold), str(args.split))

    map_a = _pred_map(pred_a_dir)
    map_b = _pred_map(pred_b_dir)
    common = [c for c in case_ids if c in map_a and c in map_b and (gt_dir / f"{c}.nii.gz").is_file()]
    if not common:
        raise SystemExit("No common cases found for training split.")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    num_classes = int(args.num_classes)
    tumor_label = int(args.tumor_label)
    model = VoxelBlender(num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = torch.nn.CrossEntropyLoss()

    rng = np.random.default_rng(int(args.seed))

    # simple online training; do not keep full volumes in RAM
    model.train()
    for ep in range(int(args.epochs)):
        ep_losses: list[float] = []
        rng.shuffle(common)
        for cid in common:
            gt, _aff, _hdr = _load_nifti(gt_dir / f"{cid}.nii.gz")
            pa, _, _ = _load_nifti(map_a[cid])
            pb, _, _ = _load_nifti(map_b[cid])
            if gt.shape != pa.shape or gt.shape != pb.shape:
                raise RuntimeError(f"{cid}: shape mismatch gt {gt.shape}, A {pa.shape}, B {pb.shape}")

            X_np, y_np = _sample_voxels(
                rng=rng,
                seg_a=pa,
                seg_b=pb,
                gt=gt,
                num_classes=num_classes,
                n_samples=int(args.samples_per_case),
                tumor_label=tumor_label,
                tumor_frac=float(args.tumor_frac),
            )
            X = torch.from_numpy(X_np.astype(np.float32, copy=False)).to(device)
            y = torch.from_numpy(y_np).to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        print(f"epoch {ep+1}/{args.epochs} loss={mean_loss:.6f} n_cases={len(common)}")

    # save
    ckpt = {
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "tumor_label": tumor_label,
        "pred_a": str(pred_a_dir),
        "pred_b": str(pred_b_dir),
        "gt_dir": str(gt_dir),
        "fold": int(args.fold),
        "split": str(args.split),
    }
    torch.save(ckpt, out_dir / "blender.pth")
    (out_dir / "meta.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    print(f"saved: {(out_dir / 'blender.pth').as_posix()}")


if __name__ == "__main__":
    main()

