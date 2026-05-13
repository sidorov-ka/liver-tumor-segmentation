#!/usr/bin/env python3
"""
Apply a trained voxel-wise blender (trained by ``scripts/3d/train_voxel_gating_blender.py``) to fuse two
prediction folders into a final multi-class segmentation.

This is a tiny learned 1×1×1 model (linear layer) that maps per-voxel features:
  one_hot(seg_A) || one_hot(seg_B)
to class logits.

Input:
  - ``--pred-a``: folder with ``<case_id>.nii.gz`` (e.g. adaptive)
  - ``--pred-b``: folder with ``<case_id>.nii.gz`` (e.g. size-gated)
  - ``--blender``: checkpoint ``blender.pth``

Output:
  - ``-o``: folder with fused ``<case_id>.nii.gz``

Split filtering:
  --split val/train/all uses ``nnUNet_preprocessed/<Dataset>/splits_final.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, header) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(np.int16), affine, header)
    nib.save(out, str(path))


def _pred_map(pred_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.name[: -len(".nii.gz")]
        out[cid] = p
    return out


def _split_ids(pre: Path, dataset_folder: str, fold: int, split: str) -> set[str] | None:
    if split == "all":
        return None
    sp = (pre / dataset_folder / "splits_final.json").resolve()
    if not sp.is_file():
        raise FileNotFoundError(sp)
    splits = json.loads(sp.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold {fold} out of range (n_splits={len(splits)})")
    return set(splits[fold][split])


class VoxelBlender(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.linear = torch.nn.Linear(2 * self.num_classes, self.num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _features(seg_a: np.ndarray, seg_b: np.ndarray, num_classes: int) -> np.ndarray:
    """Return (N, 2*num_classes) float32 for flattened voxels."""
    a = seg_a.reshape(-1)
    b = seg_b.reshape(-1)
    X = np.zeros((a.shape[0], 2 * num_classes), dtype=np.float32)
    for c in range(num_classes):
        X[:, c] = (a == c)
        X[:, num_classes + c] = (b == c)
    return X


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer fused segmentation with voxel-wise blender.")
    p.add_argument("--pred-a", type=str, required=True)
    p.add_argument("--pred-b", type=str, required=True)
    p.add_argument("--blender", type=str, required=True, help="Path to blender.pth")
    p.add_argument("-o", "--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--chunk", type=int, default=2_000_000, help="Voxels per forward chunk (RAM).")
    p.add_argument("--dataset-folder", type=str, default="Dataset001_LiverTumor")
    p.add_argument("--nnunet-preprocessed", type=str, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split", type=str, default="val", choices=("val", "train", "all"))
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    pred_a_dir = Path(args.pred_a).resolve()
    pred_b_dir = Path(args.pred_b).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.blender).resolve(), map_location="cpu", weights_only=False)
    num_classes = int(ckpt["num_classes"])

    model = VoxelBlender(num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    pre = Path(args.nnunet_preprocessed or (REPO_ROOT / "nnUNet_preprocessed")).resolve()
    split_ids = _split_ids(pre, args.dataset_folder, int(args.fold), str(args.split))

    map_a = _pred_map(pred_a_dir)
    map_b = _pred_map(pred_b_dir)
    common = sorted(set(map_a) & set(map_b))
    if split_ids is not None:
        common = [c for c in common if c in split_ids]

    for cid in common:
        sa, aff, hdr = _load_nifti(map_a[cid])
        sb, aff_b, hdr_b = _load_nifti(map_b[cid])
        if sa.shape != sb.shape:
            raise RuntimeError(f"{cid}: shape mismatch {sa.shape} vs {sb.shape}")
        if not np.allclose(aff, aff_b, atol=1e-4):
            raise RuntimeError(f"{cid}: affine mismatch between inputs")

        X = _features(sa, sb, num_classes=num_classes)
        n = X.shape[0]
        out = np.empty((n,), dtype=np.int16)
        bs = int(args.chunk)
        for s in range(0, n, bs):
            xb = torch.from_numpy(X[s : s + bs]).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).to("cpu").numpy().astype(np.int16, copy=False)
            out[s : s + bs] = pred
        fused = out.reshape(sa.shape)
        _save_nifti(out_dir / f"{cid}.nii.gz", fused, aff, hdr)
        print(f"saved {cid}")


if __name__ == "__main__":
    main()
