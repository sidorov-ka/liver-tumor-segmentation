#!/usr/bin/env python3
"""
Full-volume nnU-Net **2d** validation export: same ``nnUNetPredictor`` + preprocessor as
``infer_coarse_to_fine.py --stage1-only`` (tile step 0.75, Gaussian, mirroring).

Writes flat ``<case_id>.nii.gz`` under ``fold_*/validation/`` (or ``-o``), for example after
training when that folder is missing. Default model is the repo 2d run
``nnUNetTrainer_100epochs__nnUNetPlans__2d``; default ``--split val`` for fold 0.

Optional ``--evaluate`` runs ``scripts/evaluate_segmentations.py`` (pooled tumor Dice/IoU/TP/FP/FN).
Default GT folder: ``nnUNet_preprocessed/<dataset>/gt_segmentations`` if it exists, else
``nnUNet_raw/<dataset>/labelsTr``.

Uses the standard nnU-Net 2d trainer checkpoint layout only (no local 3d trainer wrappers).

Examples::

  .venv/bin/python scripts/2d/run_nnunet2d_validation_export.py --evaluate

  .venv/bin/python scripts/2d/run_nnunet2d_validation_export.py \\
    -o nnUNet_results/Dataset001_LiverTumor/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/validation \\
    --no-skip-heavy-val --evaluate

Set ``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results`` if they differ from repo defaults.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_FOLDER = "Dataset001_LiverTumor"
DEFAULT_TILE_STEP_SIZE = 0.75
DEFAULT_2D_MODEL_DIR = (
    "nnUNet_results/Dataset001_LiverTumor/nnUNetTrainer_100epochs__nnUNetPlans__2d"
)


def _nnunet_preprocessed_default(repo: Path) -> Path:
    return Path(os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed")))


def _resolve_repo_path(repo: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _seg_nifti_path(out_dir: Path, case_id: str, file_ending: str) -> Path:
    fe = file_ending if file_ending.startswith(".") else f".{file_ending}"
    return out_dir / f"{case_id}{fe}"


def _filter_by_split(
    identifiers: list,
    list_of_lists: list,
    split: str,
    fold: int,
    splits_path: Path,
    load_json_fn,
) -> tuple[list, list]:
    if split == "all":
        return identifiers, list_of_lists
    if not splits_path.is_file():
        raise FileNotFoundError(f"--split {split} requires {splits_path}")
    splits = load_json_fn(str(splits_path))
    if fold < 0 or fold >= len(splits):
        raise IndexError(f"fold {fold} out of range (len(splits)={len(splits)})")
    key = "train" if split == "train" else "val"
    allowed = set(splits[fold][key])
    missing_in_input = allowed - set(identifiers)
    if missing_in_input:
        sample = sorted(missing_in_input)[:15]
        print(
            f"WARNING: {len(missing_in_input)} {split} cases not found under -i (skipped): "
            f"{sample}{'...' if len(missing_in_input) > 15 else ''}"
        )
    kept_id: list = []
    kept_lol: list = []
    for cid, lol in zip(identifiers, list_of_lists):
        if cid in allowed:
            kept_id.append(cid)
            kept_lol.append(lol)
    if not kept_id:
        raise SystemExit(f"No cases left after --split {split}: check -i and splits_final.json.")
    print(f"--split {split}: running {len(kept_id)} case(s) (fold {fold}).")
    return kept_id, kept_lol


def _default_eval_gt_dir(repo: Path, dataset_folder: str, pre_root: Path) -> Path:
    cand = pre_root / dataset_folder / "gt_segmentations"
    if cand.is_dir():
        return cand
    return repo / "nnUNet_raw" / dataset_folder / "labelsTr"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export nnU-Net 2d validation segmentations "
            "(same predictor stack as infer_coarse_to_fine --stage1-only)."
        ),
    )
    p.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / DEFAULT_DATASET_FOLDER / "imagesTr"),
        help="Folder with CT volumes (nnU-Net raw naming). Default: nnUNet_raw/.../imagesTr.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Flat folder for <case_id>.nii.gz. Default: <model-dir>/fold_<fold>/validation.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_2D_MODEL_DIR,
        help="Trained nnU-Net folder (dataset.json, plans.json, fold_*).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--split",
        type=str,
        choices=("all", "val", "train"),
        default="val",
        help="Restrict to train or val IDs from splits_final.json (default: val).",
    )
    p.add_argument(
        "--nnunet-preprocessed",
        type=str,
        default=None,
        help=(
            "Root with <dataset-folder>/splits_final.json "
            "(default: $nnUNet_preprocessed or repo/nnUNet_preprocessed)."
        ),
    )
    p.add_argument(
        "--dataset-folder",
        type=str,
        default=DEFAULT_DATASET_FOLDER,
        help="Dataset name under nnUNet_raw / nnUNet_preprocessed.",
    )
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if <case_id>+file_ending already exists under -o.",
    )
    p.add_argument(
        "--skip-heavy-val",
        dest="skip_heavy_val",
        action="store_true",
        help="Skip case_0004 and case_0018. Default: on; use --no-skip-heavy-val to include.",
    )
    p.add_argument(
        "--no-skip-heavy-val",
        dest="skip_heavy_val",
        action="store_false",
        help="Run case_0004 and case_0018 (high RAM).",
    )
    p.set_defaults(skip_heavy_val=True)
    p.add_argument(
        "--exclude-cases",
        type=str,
        default="",
        help="Comma-separated case_ids to skip (in addition to heavy val when enabled).",
    )
    p.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save nnU-Net softmax (<case_id>.npz + .pkl + seg) next to segmentations.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "mps"),
    )
    p.add_argument("--tile-step-size", type=float, default=DEFAULT_TILE_STEP_SIZE)
    p.add_argument("--no-mirroring", action="store_true", help="Disable TTA mirroring.")
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="After export, run scripts/evaluate_segmentations.py on -o vs GT.",
    )
    p.add_argument(
        "--evaluate-gt-dir",
        type=str,
        default=None,
        help="GT folder for --evaluate (default: gt_segmentations if present, else labelsTr).",
    )
    p.add_argument(
        "--evaluate-output-json",
        type=str,
        default=None,
        help="Forward to evaluate_segmentations.py --output-json when using --evaluate.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    model_dir = _resolve_repo_path(repo, args.model_dir)
    in_dir = _resolve_repo_path(repo, args.input)
    if args.output:
        out_dir = _resolve_repo_path(repo, args.output)
    else:
        out_dir = model_dir / f"fold_{args.fold}" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_root = (
        _resolve_repo_path(repo, args.nnunet_preprocessed)
        if args.nnunet_preprocessed
        else _nnunet_preprocessed_default(repo)
    )
    splits_path = pre_root / args.dataset_folder / "splits_final.json"

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
    from nnunetv2.inference.export_prediction import export_prediction_from_logits  # noqa: E402
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
    from nnunetv2.utilities.utils import (  # noqa: E402
        create_lists_from_splitted_dataset_folder,
        get_identifiers_from_splitted_dataset_folder,
    )

    dataset_json = load_json(model_dir / "dataset.json")
    file_ending = dataset_json["file_ending"]

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

    identifiers, list_of_lists = _filter_by_split(
        list(identifiers),
        list(list_of_lists),
        args.split,
        args.fold,
        splits_path,
        load_json,
    )

    exclude: set[str] = set()
    if args.skip_heavy_val:
        exclude.update({"case_0004", "case_0018"})
    if args.exclude_cases.strip():
        exclude.update(x.strip() for x in args.exclude_cases.split(",") if x.strip())
    if exclude:
        kept_id: list = []
        kept_lol: list = []
        for cid, lol in zip(identifiers, list_of_lists):
            if cid in exclude:
                print(f"exclude {cid} (skipped by --skip-heavy-val / --exclude-cases)")
                continue
            kept_id.append(cid)
            kept_lol.append(lol)
        identifiers, list_of_lists = kept_id, kept_lol
        if not identifiers:
            raise SystemExit("No cases left after excluding case IDs.")

    print(f"Writing predictions to {out_dir.resolve()}")

    for case_id, image_files in zip(identifiers, list_of_lists):
        out_seg = _seg_nifti_path(out_dir, case_id, file_ending)
        if args.skip_existing and out_seg.is_file():
            print(f"skip {case_id} (exists: {out_seg.name})")
            continue
        data, _seg, props = preprocessor.run_case(
            image_files,
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        export_prediction_from_logits(
            logits,
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(out_dir / case_id),
            save_probabilities=bool(args.save_probabilities),
        )
        print(f"saved {case_id}")
        del data, props, logits
        gc.collect()

    if args.evaluate:
        gt_dir = (
            _resolve_repo_path(repo, args.evaluate_gt_dir)
            if args.evaluate_gt_dir
            else _default_eval_gt_dir(repo, args.dataset_folder, pre_root)
        )
        if not gt_dir.is_dir():
            raise SystemExit(f"--evaluate: GT directory not found: {gt_dir}")
        ev_script = repo / "scripts" / "evaluate_segmentations.py"
        if not ev_script.is_file():
            raise FileNotFoundError(ev_script)
        cmd = [
            sys.executable,
            str(ev_script),
            "--pred-dir",
            str(out_dir),
            "--gt-dir",
            str(gt_dir),
            "--dataset-json",
            str(model_dir / "dataset.json"),
        ]
        if args.evaluate_output_json:
            cmd.extend(["--output-json", str(_resolve_repo_path(repo, args.evaluate_output_json))])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
