#!/usr/bin/env python3
"""Train stage-2 refinement U-Net on exported .npz slices.

Writes under refinement_results/ (not nnUNet_*): training_log_*.txt, validation/summary.json, checkpoints.
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

from refinement.dataset import build_datasets  # noqa: E402
from refinement.trainer import refinement_collate_fn, run_training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train refinement network on ROI-exported slices.")
    p.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help="Directory with train/ and val/ .npz from export_stage1_preds.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output root (default: refinement_results/Dataset001_LiverTumor/fold_0/run_<timestamp>).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--crop-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    p.add_argument(
        "--use-coarse-prob",
        action="store_true",
        help="Use coarse_tumor_prob as 2nd channel (still 2-channel input).",
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
            / "refinement_results"
            / "Dataset001_LiverTumor"
            / "fold_0"
            / f"run_{stamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using default --out-dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)
    crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))

    in_channels = 2
    train_ds, val_ds = build_datasets(
        export_dir,
        crop_size=crop_size,
        use_coarse_prob=args.use_coarse_prob,
        max_train=args.max_train,
    )
    if len(train_ds) == 0:
        raise ValueError("train/ has no .npz files — run export_stage1_preds.py first.")
    if len(val_ds) == 0:
        raise ValueError(
            "val/ has no .npz files. Run export_stage1_preds.py with nnU-Net splits_final.json "
            "so validation cases are exported, or add val slices manually."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=refinement_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=refinement_collate_fn,
    )

    training_args = {
        "export_dir": str(export_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "crop_size": list(crop_size),
        "use_coarse_prob": bool(args.use_coarse_prob),
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
        "crop_size": list(crop_size),
        "use_coarse_prob": bool(args.use_coarse_prob),
        "export_dir": str(export_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "validation_summary": str((out_dir / "validation" / "summary.json").resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'meta.json'}. Best checkpoint: {best}")
    print(f"Metrics (nnU-Net–style): {out_dir / 'validation' / 'summary.json'}")


if __name__ == "__main__":
    main()
