#!/usr/bin/env python3
"""Train boundary_aware_coarse_to_fine (stage-2) on export.py .npz: CT + coarse prob + entropy; ROI as coarse_to_fine.

Writes under ``results_boundary_aware_coarse_to_fine/``: checkpoints, ``training_log_*.txt``,
``validation/summary.json``, and by default TensorBoard scalars under ``<out_dir>/tensorboard``.
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

from boundary_aware_coarse_to_fine.dataset import build_datasets  # noqa: E402
from boundary_aware_coarse_to_fine.paths import (  # noqa: E402
    BOUNDARY_AWARE_COARSE_TO_FINE_TASK_DIR,
    DEFAULT_BOUNDARY_AWARE_COARSE_TO_FINE_RESULTS_ROOT,
)
from boundary_aware_coarse_to_fine.trainer import (  # noqa: E402
    boundary_aware_collate_fn,
    run_training,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train boundary_aware_coarse_to_fine (3-channel) on ROI-exported slices."
    )
    p.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help="Directory with train/ and val/ .npz from scripts/export.py (same as coarse_to_fine).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory for this run (default: "
            f"{DEFAULT_BOUNDARY_AWARE_COARSE_TO_FINE_RESULTS_ROOT}/<dataset>/fold_<n>/"
            f"{BOUNDARY_AWARE_COARSE_TO_FINE_TASK_DIR}/run_<timestamp>)."
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
        "--no-roi-align",
        action="store_true",
        help="Legacy: resize the full slice to crop_size. Default: ROI crop around GT∪coarse then resize.",
    )
    p.add_argument(
        "--roi-pad",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("Y", "X"),
        help="Padding for 2D training ROI (default 16 16).",
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
    p.add_argument(
        "--lambda-boundary",
        type=float,
        default=0.0,
        help="If > 0, loss += lambda * BCE+Dice on boundary-ring pixels only (ring from coarse mask).",
    )
    p.add_argument(
        "--boundary-dilate-iters",
        type=int,
        default=2,
        help="Morphology dilations for boundary ring (train + infer default in meta).",
    )
    p.add_argument(
        "--boundary-erode-iters",
        type=int,
        default=2,
        help="Morphology erosions for boundary ring.",
    )
    p.add_argument(
        "--coarse-bin-threshold",
        type=float,
        default=0.5,
        help="Threshold on coarse probability to build binary mask for ring (train).",
    )
    p.add_argument("--max-train", type=int, default=None, help="Optional cap on training files (debug).")
    p.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard scalar logging (default: logs to <out_dir>/tensorboard).",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: <out_dir>/tensorboard).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.export_dir)
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            REPO_ROOT
            / DEFAULT_BOUNDARY_AWARE_COARSE_TO_FINE_RESULTS_ROOT
            / args.dataset_folder
            / f"fold_{args.fold}"
            / BOUNDARY_AWARE_COARSE_TO_FINE_TASK_DIR
            / f"run_{stamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using default --out-dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    roi_aligned = not bool(args.no_roi_align)
    roi_pad_xy = (int(args.roi_pad[0]), int(args.roi_pad[1]))
    min_roi_xy = (int(args.min_roi[0]), int(args.min_roi[1]))

    in_channels = 3
    train_ds, val_ds = build_datasets(
        export_dir,
        crop_size=crop_size,
        use_coarse_prob=True,
        roi_aligned=roi_aligned,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
        max_train=args.max_train,
        boundary_dilate_iters=args.boundary_dilate_iters,
        boundary_erode_iters=args.boundary_erode_iters,
        coarse_bin_threshold=args.coarse_bin_threshold,
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
        collate_fn=boundary_aware_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=boundary_aware_collate_fn,
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
        "use_coarse_prob": True,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "bce_weight": args.bce_weight,
        "lambda_boundary": args.lambda_boundary,
        "boundary_dilate_iters": args.boundary_dilate_iters,
        "boundary_erode_iters": args.boundary_erode_iters,
        "coarse_bin_threshold": args.coarse_bin_threshold,
        "max_train": args.max_train,
        "tensorboard": not args.no_tensorboard,
        "tensorboard_dir": args.tensorboard_dir,
    }

    tensorboard_dir = None
    if not args.no_tensorboard:
        tensorboard_dir = (
            Path(args.tensorboard_dir)
            if args.tensorboard_dir
            else out_dir / "tensorboard"
        )

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
        lambda_boundary=args.lambda_boundary,
        training_args=training_args,
        tensorboard_dir=tensorboard_dir,
    )

    meta = {
        "in_channels": in_channels,
        "adaptive_inference": {
            "enabled": True,
            "small_voxel_threshold": 6000,
            "large_voxel_threshold": 280000,
            "mean_prob_suspicious": 0.22,
            "refine_tau_small": 0.45,
            "refine_tau_large": 0.56,
            "refine_tau_suspicious": 0.62,
            "ring_blend_large": 0.42,
            "ring_blend_suspicious": 0.28,
        },
        "dataset_folder": args.dataset_folder,
        "fold": args.fold,
        "run_layout": (
            f"{DEFAULT_BOUNDARY_AWARE_COARSE_TO_FINE_RESULTS_ROOT}/<dataset>/fold_<n>/"
            f"{BOUNDARY_AWARE_COARSE_TO_FINE_TASK_DIR}/run_<timestamp>"
        ),
        "crop_size": list(crop_size),
        "use_coarse_prob": True,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "bce_weight": args.bce_weight,
        "lambda_boundary": args.lambda_boundary,
        "boundary_dilate_iters": args.boundary_dilate_iters,
        "boundary_erode_iters": args.boundary_erode_iters,
        "coarse_bin_threshold": args.coarse_bin_threshold,
        "export_dir": str(export_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "validation_summary": str((out_dir / "validation" / "summary.json").resolve()),
        "tensorboard_logdir": (
            str(tensorboard_dir.resolve()) if tensorboard_dir is not None else None
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'meta.json'}. Best checkpoint: {best}")
    if tensorboard_dir is not None:
        print(
            f"TensorBoard: tensorboard --logdir {tensorboard_dir.resolve()}"
        )
    print(f"Metrics (nnU-Net–style): {out_dir / 'validation' / 'summary.json'}")


if __name__ == "__main__":
    main()
