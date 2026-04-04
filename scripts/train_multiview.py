#!/usr/bin/env python3
"""Train MultiviewUNet2d on ``scripts/export.py`` .npz slices (4 ch: 3 HU windows + tumor prob).

Multi-window fusion as in e.g. *Deep Multi-View Fusion Network for Lung Nodule Segmentation* (IEEE TMI).
Normalization matches ``infer_multiview``. Output: ``results_multiview/.../multiview/run_*`` with
``checkpoint_best.pth`` and ``meta.json``. ROI and blend defaults: ``MultiviewConfig`` (override via CLI).
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

from multiview.config import MultiviewConfig, multiview_config_to_json_dict  # noqa: E402
from multiview.dataset import build_multiview_datasets  # noqa: E402
from multiview.paths import DEFAULT_MULTIVIEW_RESULTS_ROOT, MULTIVIEW_TASK_DIR  # noqa: E402
from multiview.trainer import multiview_collate_fn, run_training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MultiviewUNet2d on ROI-exported slices (export.py).")
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
            f"{DEFAULT_MULTIVIEW_RESULTS_ROOT}/<dataset>/fold_<n>/{MULTIVIEW_TASK_DIR}/run_<timestamp>)."
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
    p.add_argument("--base", type=int, default=32, help="UNet base channels (match infer_multiview).")
    p.add_argument("--crop-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    p.add_argument(
        "--no-coarse-prob",
        action="store_true",
        help="Use binary coarse_tumor as 4th channel instead of coarse_tumor_prob.",
    )
    p.add_argument(
        "--roi-mode",
        type=str,
        choices=("infer", "legacy"),
        default="infer",
        help="infer: ROI = suspicious prob band + pad (matches infer_multiview). "
        "legacy: ROI = GT∪coarse (old behavior).",
    )
    p.add_argument(
        "--no-roi-align",
        action="store_true",
        help="Ignore ROI; resize full slice to crop_size (debug / legacy).",
    )
    p.add_argument(
        "--roi-pad",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("Y", "X"),
        help="2D ROI padding (default 16 16).",
    )
    p.add_argument(
        "--min-roi",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("Y", "X"),
        help="Minimum ROI side before resize (default 32 32).",
    )
    p.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE in BCE+Dice.")
    p.add_argument("--max-train", type=int, default=None, help="Optional cap on training files (debug).")
    # MultiviewConfig (HU windows + crop for meta — infer should use same defaults or meta)
    p.add_argument("--prob-lo", type=float, default=None)
    p.add_argument("--prob-hi", type=float, default=None)
    p.add_argument("--min-component-voxels", type=int, default=None)
    p.add_argument("--roi-pad-3d", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--min-roi-side-3d", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--prob-high-band-lo", type=float, default=None)
    p.add_argument("--prob-high-band-hi", type=float, default=None)
    p.add_argument("--refine-blend-mode", type=str, choices=("replace", "blend"), default=None)
    p.add_argument("--refine-alpha", type=float, default=None)
    p.add_argument("--component-size-alpha-threshold-voxels", type=int, default=None)
    p.add_argument("--alpha-blend-small", type=float, default=None)
    p.add_argument("--alpha-blend-large", type=float, default=None)
    p.add_argument("--post-remove-tumor-components-below-voxels", type=int, default=None)
    return p.parse_args()


def _apply_mv_cfg_args(cfg: MultiviewConfig, args: argparse.Namespace) -> None:
    if args.prob_lo is not None:
        cfg.prob_lo = float(args.prob_lo)
    if args.prob_hi is not None:
        cfg.prob_hi = float(args.prob_hi)
    if args.min_component_voxels is not None:
        cfg.min_component_voxels = int(args.min_component_voxels)
    if args.roi_pad_3d is not None:
        cfg.roi_pad = (int(args.roi_pad_3d[0]), int(args.roi_pad_3d[1]), int(args.roi_pad_3d[2]))
    if args.min_roi_side_3d is not None:
        cfg.min_roi_side = (
            int(args.min_roi_side_3d[0]),
            int(args.min_roi_side_3d[1]),
            int(args.min_roi_side_3d[2]),
        )
    cs = getattr(args, "_crop_size_tuple", None)
    if cs is not None:
        cfg.crop_size = (int(cs[0]), int(cs[1]))
    if args.prob_high_band_lo is not None:
        cfg.prob_high_band_lo = float(args.prob_high_band_lo)
    if args.prob_high_band_hi is not None:
        cfg.prob_high_band_hi = float(args.prob_high_band_hi)
    if args.refine_blend_mode is not None:
        cfg.refine_blend_mode = str(args.refine_blend_mode)
    if args.refine_alpha is not None:
        cfg.refine_alpha = float(args.refine_alpha)
    if args.component_size_alpha_threshold_voxels is not None:
        cfg.component_size_alpha_threshold_voxels = int(args.component_size_alpha_threshold_voxels)
    if args.alpha_blend_small is not None:
        cfg.alpha_blend_small = float(args.alpha_blend_small)
    if args.alpha_blend_large is not None:
        cfg.alpha_blend_large = float(args.alpha_blend_large)
    if args.post_remove_tumor_components_below_voxels is not None:
        cfg.post_remove_tumor_components_below_voxels = int(args.post_remove_tumor_components_below_voxels)


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.export_dir)
    crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    setattr(args, "_crop_size_tuple", crop_size)

    mv_cfg = MultiviewConfig()
    _apply_mv_cfg_args(mv_cfg, args)

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = (
            REPO_ROOT
            / DEFAULT_MULTIVIEW_RESULTS_ROOT
            / args.dataset_folder
            / f"fold_{args.fold}"
            / MULTIVIEW_TASK_DIR
            / f"run_{stamp}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using default --out-dir {out_dir}")
    else:
        out_dir = Path(args.out_dir)

    use_coarse_prob = not bool(args.no_coarse_prob)
    roi_aligned = not bool(args.no_roi_align)
    roi_pad_xy = (int(args.roi_pad[0]), int(args.roi_pad[1]))
    min_roi_xy = (int(args.min_roi[0]), int(args.min_roi[1]))

    if args.roi_mode == "infer" and not use_coarse_prob:
        raise SystemExit(
            "--roi-mode infer requires coarse_tumor_prob in .npz (omit --no-coarse-prob). "
            "The suspicious band is defined on softmax tumor probability."
        )

    train_ds, val_ds = build_multiview_datasets(
        export_dir,
        mv_cfg,
        crop_size=crop_size,
        use_coarse_prob=use_coarse_prob,
        roi_aligned=roi_aligned,
        roi_mode=args.roi_mode,
        roi_pad_xy=roi_pad_xy,
        min_roi_xy=min_roi_xy,
        max_train=args.max_train,
    )
    if len(train_ds) == 0:
        hint = ""
        if args.roi_mode == "infer" and roi_aligned:
            hint = (
                " With --roi-mode infer, train/ must contain slices where tumor prob is in "
                f"[{mv_cfg.prob_lo}, {mv_cfg.prob_hi}]. Widen band (--prob-lo/--prob-hi) or use --roi-mode legacy."
            )
        raise ValueError("train/ has no usable .npz files — run scripts/export.py first." + hint)
    if len(val_ds) == 0:
        hint = ""
        if args.roi_mode == "infer" and roi_aligned:
            hint = (
                f" With --roi-mode infer, val/ needs slices with prob in [{mv_cfg.prob_lo}, {mv_cfg.prob_hi}]. "
                "Try --roi-mode legacy or adjust prob band."
            )
        raise ValueError(
            "val/ has no usable .npz files. Run export with splits_final.json, or add val slices." + hint
        )
    print(
        f"Multiview dataset: train={len(train_ds)} val={len(val_ds)} "
        f"(roi_mode={args.roi_mode}, roi_aligned={roi_aligned})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multiview_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=multiview_collate_fn,
    )

    hu_json = [[float(w), float(l)] for w, l in mv_cfg.hu_windows]
    training_args = {
        "export_dir": str(export_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "dataset_folder": args.dataset_folder,
        "fold": args.fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base": args.base,
        "crop_size": list(crop_size),
        "hu_windows": hu_json,
        "use_coarse_prob": use_coarse_prob,
        "roi_mode": args.roi_mode,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "multiview_config": multiview_config_to_json_dict(mv_cfg),
        "bce_weight": args.bce_weight,
        "max_train": args.max_train,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = run_training(
        train_loader,
        val_loader,
        out_dir,
        base=args.base,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        bce_weight=args.bce_weight,
        training_args=training_args,
    )

    meta = {
        "in_channels": 4,
        "dataset_folder": args.dataset_folder,
        "fold": args.fold,
        "run_layout": f"{DEFAULT_MULTIVIEW_RESULTS_ROOT}/<dataset>/fold_<n>/{MULTIVIEW_TASK_DIR}/run_<timestamp>",
        "base": args.base,
        "crop_size": list(crop_size),
        "hu_windows": hu_json,
        "use_coarse_prob": use_coarse_prob,
        "roi_mode": args.roi_mode,
        "roi_aligned": roi_aligned,
        "roi_pad_xy": list(roi_pad_xy),
        "min_roi_xy": list(min_roi_xy),
        "multiview_config": training_args["multiview_config"],
        "export_dir": str(export_dir.resolve()),
        "best_checkpoint": str(best.resolve()),
        "validation_summary": str((out_dir / "validation" / "summary.json").resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'meta.json'}. Best checkpoint: {best}")
    print(f"Metrics: {out_dir / 'validation' / 'summary.json'}")


if __name__ == "__main__":
    main()
