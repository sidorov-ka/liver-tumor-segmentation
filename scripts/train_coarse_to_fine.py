#!/usr/bin/env python3
"""Train coarse_to_fine (stage-2) U-Net on exported .npz slices.

Default training matches scripts/infer_coarse_to_fine.py: nnU-Net tumor **probability** as 2nd channel and
2D ROI crop (GT ∪ coarse) + resize — see Li et al., Universal Topology Refinement (arXiv:2409.09796)
for probability-map refinement (we do not implement polynomial synthesis from that paper).

Writes under coarse_to_fine_results/ only (not multiview_results; not nnU-Net): training_log_*.txt, validation/summary.json, checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coarse_to_fine.dataset import build_datasets  # noqa: E402
from coarse_to_fine.paths import COARSE_TO_FINE_TASK_DIR, DEFAULT_COARSE_TO_FINE_RESULTS_ROOT  # noqa: E402
from coarse_to_fine.trainer import coarse_to_fine_collate_fn, run_training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train coarse_to_fine network on ROI-exported slices.")
    p.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help="Directory with train/ and val/ .npz from scripts/export.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory for this run (default: "
            f"{DEFAULT_COARSE_TO_FINE_RESULTS_ROOT}/<dataset>/fold_<n>/{COARSE_TO_FINE_TASK_DIR}/run_<timestamp>)."
        ),
    )
    p.add_argument(
        "--dataset-folder",
        type=str,
        default="Dataset001_LiverTumor",
        help="Dataset name segment in the default output path (match nnU-Net / export).",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index in the default output path (match nnU-Net fold used for export).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--crop-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    p.add_argument(
        "--no-coarse-prob",
        action="store_true",
        help="Use binary coarse_tumor mask as 2nd channel. Default: coarse_tumor_prob (softmax), aligned with infer_coarse_to_fine.",
    )
    p.add_argument(
        "--no-roi-align",
        action="store_true",
        help="Legacy: resize the full slice to crop_size. Default: ROI crop around GT∪coarse then resize (aligned with infer_coarse_to_fine bbox crops).",
    )
    p.add_argument(
        "--roi-pad",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("Y", "X"),
        help="Padding for 2D training ROI (default 16 16; infer_coarse_to_fine uses bbox3d pad (2,16,16) on Y/X).",
    )
    p.add_argument(
        "--min-roi",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("Y", "X"),
        help="Minimum ROI height/width before resize (default 32 32).",
    )
    p.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE in BCE+Dice.")
    p.add_argument("--max-train", type=int, default=None, help="Optional cap on training files (debug).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.export_dir)
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            REPO_ROOT
            / DEFAULT_COARSE_TO_FINE_RESULTS_ROOT
            / args.dataset_folder
            / f"fold_{args.fold}"
            / COARSE_TO_FINE_TASK_DIR
            / f"run_{stamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using default --out-dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    use_coarse_prob = not bool(args.no_coarse_prob)
    roi_aligned = not bool(args.no_roi_align)
    roi_pad_xy = (int(args.roi_pad[0]), int(args.roi_pad[1]))
    min_roi_xy = (int(args.min_roi[0]), int(args.min_roi[1]))

    in_channels = 2
    train_ds, val_ds = build_datasets(
        export_dir,
        crop_size=crop_size,
        use_coarse_prob=use_coarse_prob,
        roi_aligned=roi_aligned,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
        max_train=args.max_train,
    )
    if len(train_ds) == 0:
        raise ValueError("train/ has no .npz files — run scripts/export.py first.")
    if len(val_ds) == 0:
        raise ValueError(
            "val/ has no .npz files. Run scripts/export.py with nnU-Net splits_final.json "
            "so validation cases are exported, or add val slices manually."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=coarse_to_fine_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=coarse_to_fine_collate_fn,
    )

    training_args = {
        "export_dir": str(export_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "dataset_folder": args.dataset_folder,
        "fold": args.fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "crop_size": list(crop_size),
        "use_coarse_prob": use_coarse_prob,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "bce_weight": args.bce_weight,
        "max_train": args.max_train,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = run_training(
        train_loader,
        val_loader,
        out_dir,
        in_channels=in_channels,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        bce_weight=args.bce_weight,
        training_args=training_args,
    )

    meta = {
        "in_channels": in_channels,
        "dataset_folder": args.dataset_folder,
        "fold": args.fold,
        "run_layout": f"{DEFAULT_COARSE_TO_FINE_RESULTS_ROOT}/<dataset>/fold_<n>/{COARSE_TO_FINE_TASK_DIR}/run_<timestamp>",
        "crop_size": list(crop_size),
        "use_coarse_prob": use_coarse_prob,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "reference": "UTR-style: prob refinement + ROI crop; Li et al. arXiv:2409.09796 (polynomial synthesis not implemented)",
        "export_dir": str(export_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "validation_summary": str((out_dir / "validation" / "summary.json").resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'meta.json'}. Best checkpoint: {best}")
    print(f"Metrics (nnU-Net–style): {out_dir / 'validation' / 'summary.json'}")


if __name__ == "__main__":
    main()
