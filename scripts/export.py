#!/usr/bin/env python3
"""
Export nnU-Net v2 (Isensee et al.) stage-1 predictions + paired slices for stage-2 scripts
(coarse_to_fine, multiview, uncertainty).

Tensors align with nnU-Net training; default tile step ``0.75`` matches ``infer_coarse_to_fine``. Re-export if the base model, data, or export options change.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "2d"))

from coarse_to_fine.dataset import save_manifest_json  # noqa: E402

# Same default as scripts/infer_coarse_to_fine.py (nnUNetPredictor tile_step_size)
DEFAULT_TILE_STEP_SIZE = 0.75


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export stage-1 nnU-Net preds for train_coarse_to_fine / train_multiview / train_uncertainty (train/val slices)."
    )
    p.add_argument(
        "--nnunet-raw",
        type=str,
        default=None,
        help="nnUNet_raw root (default: $nnUNet_raw or <repo>/nnUNet_raw)",
    )
    p.add_argument(
        "--nnunet-preprocessed",
        type=str,
        default=None,
        help="nnUNet_preprocessed root (default: $nnUNet_preprocessed or <repo>/nnUNet_preprocessed)",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Trained nnU-Net output folder (contains fold_X, dataset.json, plans.json).",
    )
    p.add_argument(
        "--dataset-folder",
        type=str,
        default="Dataset001_LiverTumor",
        help="Folder name under nnUNet_raw / nnUNet_preprocessed.",
    )
    p.add_argument("--fold", type=int, default=0, help="Fold index for splits_final.json.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pth",
        help="Checkpoint name inside fold_* (e.g. checkpoint_best.pth).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write train/*.npz and val/*.npz.",
    )
    p.add_argument(
        "--all-slices",
        action="store_true",
        help="Save every axial slice; default: only slices with tumor in GT or coarse pred.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "mps"),
    )
    p.add_argument(
        "--tile-step-size",
        type=float,
        default=DEFAULT_TILE_STEP_SIZE,
        help="nnU-Net sliding-window step (default 0.75; same as infer_coarse_to_fine.py).",
    )
    return p.parse_args()


def _repo_nnunet_defaults(repo: Path) -> tuple[str, str, str]:
    raw = os.environ.get("nnUNet_raw", str(repo / "nnUNet_raw"))
    pre = os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed"))
    res = os.environ.get("nnUNet_results", str(repo / "nnUNet_results"))
    return raw, pre, res


def _tumor_label(dataset_json: dict) -> int:
    labels = dataset_json.get("labels", {})
    for name, idx in labels.items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("No 'tumor' entry in dataset.json labels")


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    raw_root = Path(args.nnunet_raw or _repo_nnunet_defaults(repo)[0])
    pre_root = Path(args.nnunet_preprocessed or _repo_nnunet_defaults(repo)[1])
    out_root = Path(args.output_dir)

    default_model = (
        repo
        / "nnUNet_results"
        / "Dataset001_LiverTumor"
        / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
    )
    model_dir = Path(args.model_dir) if args.model_dir else default_model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")

    from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets  # noqa: E402

    dataset_json = load_json(model_dir / "dataset.json")
    tumor_label = _tumor_label(dataset_json)

    splits_path = pre_root / args.dataset_folder / "splits_final.json"
    case_to_files = get_filenames_of_train_images_and_targets(
        str(raw_root / args.dataset_folder),
        dataset_json,
    )
    all_cases = sorted(case_to_files.keys())

    train_cases: list[str] = []
    val_cases: list[str] = []
    if splits_path.is_file():
        splits = load_json(splits_path)
        if args.fold < 0 or args.fold >= len(splits):
            raise IndexError(f"fold {args.fold} out of range (len={len(splits)})")
        train_cases = list(splits[args.fold]["train"])
        val_cases = list(splits[args.fold]["val"])
    else:
        print(
            f"WARNING: {splits_path} not found — exporting ALL training cases into train/ "
            f"(val/ empty). Generate splits with nnU-Net preprocessing for a real val split."
        )
        train_cases = all_cases

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    predictor = nnUNetPredictor(
        tile_step_size=float(args.tile_step_size),
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)

    def export_cases(case_ids: list[str], split_name: str) -> int:
        n_saved = 0
        for case_id in case_ids:
            if case_id not in case_to_files:
                print(f"WARNING: case {case_id} not in nnUNet_raw — skip")
                continue
            entry = case_to_files[case_id]
            image_files = entry["images"]
            label_file = entry["label"]

            data, seg, _props = preprocessor.run_case(
                image_files,
                label_file,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )

            logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
            probs = torch.softmax(logits, dim=0).numpy()
            pred_seg = logits.argmax(dim=0).numpy().astype(np.int16)

            if seg is None:
                raise RuntimeError("Segmentation missing for training export.")
            if seg.ndim == 3:
                seg = seg[np.newaxis, ...]

            _, zd, yd, xd = data.shape
            for z in range(zd):
                gt_z = (seg[0, z] == tumor_label).astype(np.float32)
                coarse_z = (pred_seg[z] == tumor_label).astype(np.float32)
                coarse_prob_z = probs[tumor_label, z].astype(np.float32)
                img_z = data[0, z].astype(np.float32)

                if not args.all_slices:
                    if not (gt_z.any() or coarse_z.any()):
                        continue

                out_name = f"{case_id}_{z:04d}.npz"
                out_path = out_root / split_name / out_name
                np.savez_compressed(
                    out_path,
                    image=img_z,
                    coarse_tumor=coarse_z,
                    coarse_tumor_prob=coarse_prob_z,
                    gt_tumor=gt_z,
                    slice_z=z,
                    case_id=np.array(case_id),
                )
                n_saved += 1
        return n_saved

    n_tr = export_cases(train_cases, "train")
    n_va = export_cases(val_cases, "val")

    save_manifest_json(
        out_root,
        {
            "model_dir": str(model_dir),
            "fold": args.fold,
            "checkpoint": args.checkpoint,
            "tile_step_size": float(args.tile_step_size),
            "dataset_folder": args.dataset_folder,
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "npz_train": n_tr,
            "npz_val": n_va,
            "tumor_label": tumor_label,
            "all_slices": bool(args.all_slices),
        },
    )
    print(f"Done. Saved train={n_tr} val={n_va} slices under {out_root}")


if __name__ == "__main__":
    main()
