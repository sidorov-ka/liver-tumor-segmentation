#!/usr/bin/env python3
"""
Full-volume inference for ``nnUNetTrainer_150_MultiWindowRefine_50epochs`` (4-channel input).

Builds the same stack as training: coarse tumour probability (cached ``.npz`` per case) plus three
HU windows from Lim et al. (Diagnostics, 2025): [-1000, 1000], [0, 1000], [400, 1000] HU on
approximately denormalized intensities (dataset fingerprint).

Uses ``nnUNetPredictor`` sliding window (Gaussian, mirroring) like standard nnU-Net 3D inference.
If ``-o`` is omitted, segmentations are written under ``<model-dir>/fold_<N>/validation/`` (nnU-Net training layout).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for segmentations. "
            "Default: ``<model-dir>/fold_<fold>/validation`` (nnU-Net training layout)."
        ),
    )
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
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "train", "all"),
        help="Which split to run on (from nnUNet_preprocessed/.../splits_final.json). Default: val.",
    )
    return p.parse_args()


def _load_network_and_plans_4ch(
    model_dir: Path,
    fold: int,
    checkpoint_name: str,
    device: torch.device,
):
    """Load checkpoint; build U-Net with 4 inputs (predictor uses ``determine_num_input_channels`` → 1)."""
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset_json = load_json(join(str(model_dir), "dataset.json"))
    plans = load_json(join(str(model_dir), "plans.json"))
    ckpt_path = join(str(model_dir), f"fold_{fold}", checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    inference_allowed_mirroring_axes = (
        checkpoint["inference_allowed_mirroring_axes"]
        if "inference_allowed_mirroring_axes" in checkpoint
        else None
    )
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    trainer_root = join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    trainer_class = recursive_find_python_class(
        trainer_root, trainer_name, "nnunetv2.training.nnUNetTrainer"
    )
    if trainer_class is None:
        raise RuntimeError(
            f"Could not resolve trainer {trainer_name!r} (run from repo after local trainer registration)."
        )
    label_manager = plans_manager.get_label_manager(dataset_json)
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        4,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"])
    network.eval()
    network.to(device)
    return network, plans_manager, configuration_manager, dataset_json, trainer_name, inference_allowed_mirroring_axes


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

    sys.path.insert(0, str(repo / "scripts" / "3d"))
    from run_nnunet_with_local_3d_trainers import _register_local_trainers

    _register_local_trainers()

    model_dir = Path(args.model_dir).resolve()
    prob_dir = Path(args.prob_dir).resolve()
    in_dir = Path(args.input).resolve()
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = (model_dir / f"fold_{args.fold}" / "validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing segmentations to: {out_dir}")

    fp_json = Path(
        args.fingerprint_json
        or (Path(pre_d) / args.dataset_folder / "dataset_fingerprint.json")
    ).resolve()
    if not fp_json.is_file():
        raise FileNotFoundError(fp_json)

    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.utils import (
        create_lists_from_splitted_dataset_folder,
        get_identifiers_from_splitted_dataset_folder,
    )

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=float(args.tile_step_size),
        use_gaussian=True,
        use_mirroring=not args.no_mirroring,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
    )
    (
        network,
        plans_manager,
        configuration_manager,
        dataset_json,
        trainer_name,
        mirror_axes,
    ) = _load_network_and_plans_4ch(model_dir, args.fold, args.checkpoint, device)
    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        None,
        dataset_json,
        trainer_name,
        mirror_axes,
    )
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    file_ending = dataset_json["file_ending"]
    identifiers = get_identifiers_from_splitted_dataset_folder(str(in_dir), file_ending)
    if args.split != "all":
        splits_json = (Path(pre_d) / args.dataset_folder / "splits_final.json").resolve()
        if not splits_json.is_file():
            raise FileNotFoundError(
                f"--split {args.split!r} requires splits_final.json under nnUNet_preprocessed: {splits_json}"
            )
        import json

        splits = json.loads(splits_json.read_text(encoding="utf-8"))
        fold = int(args.fold)
        if fold < 0 or fold >= len(splits):
            raise ValueError(
                f"fold {fold} out of range for splits_final.json (n_splits={len(splits)})"
            )
        allowed = set(splits[fold][args.split])
        identifiers = [i for i in identifiers if i in allowed]
        print(f"Split={args.split}: running {len(identifiers)} cases (fold {fold})")
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
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().contiguous()
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
