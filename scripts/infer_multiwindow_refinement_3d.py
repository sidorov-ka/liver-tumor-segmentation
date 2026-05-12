#!/usr/bin/env python3
"""
Full-volume inference for ``nnUNetTrainer_150_MultiWindowRefine_50epochs`` (4-channel input).

Builds the same stack as training: coarse tumour probability (cached ``.npz`` per case) plus three
HU windows from Lim et al. (Diagnostics, 2025): [-1000, 1000], [0, 1000], [400, 1000] HU on
approximately denormalized intensities (dataset fingerprint).

Uses ``nnUNetPredictor`` sliding window (Gaussian, mirroring) like standard nnU-Net 3D inference.
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


def _repo_defaults(repo: Path) -> tuple[str, str]:
    raw = os.environ.get("nnUNet_raw", str(repo / "nnUNet_raw"))
    pre = os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed"))
    return raw, pre


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D multi-window refinement inference (nnU-Net–style).")
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Trained MultiWindowRefine folder (fold_*, checkpoint_best.pth).",
    )
    p.add_argument(
        "--prob-dir",
        type=str,
        required=True,
        help="Per-case ``<case_id>.npz`` with ``prob`` (same as training cache).",
    )
    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="nnUNet_raw-style folder (e.g. .../imagesTr).",
    )
    p.add_argument("-o", "--output", type=str, required=True, help="Output directory for segmentations.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument("--dataset-folder", type=str, default="Dataset001_LiverTumor")
    p.add_argument(
        "--fingerprint-json",
        type=str,
        default=None,
        help="dataset_fingerprint.json (default: under nnUNet_preprocessed).",
    )
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    p.add_argument("--tile-step-size", type=float, default=DEFAULT_TILE_STEP_SIZE)
    p.add_argument("--no-mirroring", action="store_true")
    return p.parse_args()


def _build_data4(
    data_1ch: np.ndarray,
    prob_path: Path,
    fingerprint_json: Path,
) -> torch.Tensor:
    sys.path.insert(0, str(REPO_ROOT / "src" / "3d"))
    from multiwindow.windows import lim_three_windows_from_norm, load_fingerprint_stats

    stats = load_fingerprint_stats(str(fingerprint_json))
    norm = np.asarray(data_1ch[0], dtype=np.float32)
    pr = np.load(str(prob_path), mmap_mode="r")["prob"]
    pr = np.asarray(pr, dtype=np.float32)
    if pr.ndim == 3:
        pr = pr[np.newaxis, ...]
    if pr.shape != data_1ch.shape:
        raise ValueError(f"prob shape {pr.shape} != data {data_1ch.shape} for {prob_path.name}")
    hw = lim_three_windows_from_norm(norm, stats)
    stack = np.concatenate([pr, hw], axis=0).astype(np.float32)
    return torch.from_numpy(stack)


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    raw_d, pre_d = _repo_defaults(repo)
    os.environ.setdefault("nnUNet_raw", raw_d)
    os.environ.setdefault("nnUNet_preprocessed", pre_d)
    res_d = os.environ.get("nnUNet_results", str(repo / "nnUNet_results"))
    os.environ.setdefault("nnUNet_results", res_d)

    sys.path.insert(0, str(repo / "scripts"))
    from run_nnunet_with_local_3d_trainers import _register_local_trainers

    _register_local_trainers()

    model_dir = Path(args.model_dir).resolve()
    prob_dir = Path(args.prob_dir).resolve()
    in_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_json = Path(
        args.fingerprint_json
        or (Path(pre_d) / args.dataset_folder / "dataset_fingerprint.json")
    ).resolve()
    if not fp_json.is_file():
        raise FileNotFoundError(fp_json)

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.utils import (
        create_lists_from_splitted_dataset_folder,
        get_identifiers_from_splitted_dataset_folder,
    )

    dataset_json = load_json(model_dir / "dataset.json")
    file_ending = dataset_json["file_ending"]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=float(args.tile_step_size),
        use_gaussian=True,
        use_mirroring=not args.no_mirroring,
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

    identifiers = get_identifiers_from_splitted_dataset_folder(str(in_dir), file_ending)
    list_of_lists = create_lists_from_splitted_dataset_folder(
        str(in_dir), file_ending, identifiers=identifiers
    )

    for case_id, image_files in zip(identifiers, list_of_lists):
        prob_path = prob_dir / f"{case_id}.npz"
        if not prob_path.is_file():
            print(f"skip {case_id} (no prob cache: {prob_path})")
            continue
        data, _s, props = preprocessor.run_case(
            image_files,
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        data4 = _build_data4(np.asarray(data, dtype=np.float32), prob_path, fp_json)
        logits = predictor.predict_sliding_window_return_logits(data4)
        export_prediction_from_logits(
            logits,
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(out_dir / case_id),
            save_probabilities=False,
        )
        print(f"saved {case_id}")
        del data, props, logits, data4
        gc.collect()


if __name__ == "__main__":
    main()
