#!/usr/bin/env python3
"""
Cache full-volume tumour probability maps (softmax channel) aligned with nnU-Net preprocessed spacing.

Used by ``nnUNetTrainer_150_MultiWindowRefine_50epochs`` (``NNUNET_MW_PROB_DIR``). Each case is saved as::

  <output-dir>/<case_id>.npz   with array ``prob`` shaped (1, Z, Y, X), float16.

Run once per (model-dir, fold, configuration) before multi-window training.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TILE_STEP_SIZE = 0.75


def _repo_defaults(repo: Path) -> tuple[str, str, str]:
    raw = os.environ.get("nnUNet_raw", str(repo / "nnUNet_raw"))
    pre = os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed"))
    res = os.environ.get("nnUNet_results", str(repo / "nnUNet_results"))
    return raw, pre, res


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache tumour prob volumes for multi-window 3D refinement.")
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="nnU-Net folder providing stage-1 logits (e.g. size_gated BoundaryOverseg fold).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write <case_id>.npz (prob key).",
    )
    p.add_argument("--dataset-folder", type=str, default="Dataset001_LiverTumor")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "all"),
        default="all",
        help="Which cases from splits_final.json to cache (default: train+val).",
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    p.add_argument("--tile-step-size", type=float, default=DEFAULT_TILE_STEP_SIZE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    raw_d, pre_d, res_d = _repo_defaults(repo)
    os.environ.setdefault("nnUNet_raw", raw_d)
    os.environ.setdefault("nnUNet_preprocessed", pre_d)
    os.environ.setdefault("nnUNet_results", res_d)

    sys.path.insert(0, str(repo / "scripts"))
    from run_nnunet_with_local_3d_trainers import _register_local_trainers

    _register_local_trainers()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

    dataset_json = load_json(model_dir / "dataset.json")
    tumor_label = None
    for name, idx in dataset_json.get("labels", {}).items():
        if str(name).lower() == "tumor":
            tumor_label = int(idx)
            break
    if tumor_label is None:
        raise KeyError("tumor label not found in dataset.json")

    splits_path = Path(pre_d) / args.dataset_folder / "splits_final.json"
    case_to_files = get_filenames_of_train_images_and_targets(
        str(Path(raw_d) / args.dataset_folder),
        dataset_json,
    )
    all_ids = sorted(case_to_files.keys())
    if args.split == "all":
        case_ids = all_ids
    else:
        if not splits_path.is_file():
            raise FileNotFoundError(f"{splits_path} required for --split {args.split}")
        splits = load_json(str(splits_path))
        fold = int(args.fold)
        if fold < 0 or fold >= len(splits):
            raise IndexError(f"fold {fold} out of range")
        case_ids = list(splits[fold][args.split])

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
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

    for case_id in case_ids:
        if case_id not in case_to_files:
            print(f"skip {case_id} (not in nnUNet_raw)")
            continue
        out_npz = out_dir / f"{case_id}.npz"
        if out_npz.is_file():
            print(f"skip {case_id} (exists)")
            continue
        entry = case_to_files[case_id]
        image_files = entry["images"]
        label_file = entry["label"]
        data, _seg, props = preprocessor.run_case(
            image_files,
            label_file,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        prob = torch.softmax(logits, dim=0)[tumor_label].numpy().astype(np.float16)
        prob = prob[np.newaxis, ...]
        np.savez_compressed(out_npz, prob=prob)
        print(f"saved {case_id} {prob.shape}")
        del data, props, logits
        gc.collect()


if __name__ == "__main__":
    main()
