#!/usr/bin/env python3
"""
Two-stage full-volume inference: nnU-Net (stage 1) + refiner (stage 2), nnU-Net export style.

**Fair baseline vs two-stage:** use the same code path for stage 1 as the refiner (this script),
not a separate ``nnUNetv2_predict`` call — defaults (checkpoint, step size, TTA) differ otherwise.

- ``--stage1-only -o DIR``: nnU-Net predictions only (same ``nnUNetPredictor`` + preprocessor as
  the two-stage path).
- ``--split val`` (or ``train``): only cases listed in ``nnUNet_preprocessed/.../splits_final.json``
  for ``--fold`` (typical val metrics; input folder is still ``imagesTr`` or a staging dir).
- ``--skip-existing``: skip cases whose output ``<case_id>.nii.gz`` already exists under ``-o``.
- By default ``case_0004`` and ``case_0018`` are skipped (very large volumes); use ``--no-skip-heavy-val`` to include them.
  ``--exclude-cases`` adds more IDs to skip (same as ``infer_multiview`` / ``infer_uncertainty``).
- ``--save-probabilities``: save nnU-Net softmax (``<case_id>.npz`` + ``<case_id>.pkl`` + stage-1
  ``<case_id>.nii.gz`` in nnU-Net layout). Use ``--prob-dir`` or defaults below.
- Full two-stage: optional ``--export-stage1-to DIR`` writes stage-1 segmentations from the **same**
  forward pass used before coarse_to_fine refinement.

Experiment roots under ``inference_comparison/``: ``baseline/`` (nnU-Net only),
``coarse_to_fine/`` (two-stage), ``multiview/`` (multi-window ROI script).
Set nnUNet_raw, nnUNet_preprocessed, nnUNet_results before running.

Stage-1 ``nnUNetPredictor`` matches ``scripts/export.py`` (tile step **0.75** by default, Gaussian,
mirroring). **Host RAM:** two-stage defaults to float16 coarse logits when ``use_coarse_prob``;
use ``--full-precision-logits`` for float32. Further reduce RAM: smaller ``--tile-step-size`` or
``--no-mirroring``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "2d"))

from coarse_to_fine.model import TinyUNet2d  # noqa: E402
from coarse_to_fine.roi import bbox3d_from_mask, threshold_coarse_tumor  # noqa: E402
from coarse_to_fine.utils import load_checkpoint  # noqa: E402

DEFAULT_INFERENCE_SUBDIR = "two_stage"
# Experiment roots under inference_comparison/: baseline (nnU-Net only), coarse_to_fine (two-stage), multiview (scripts/infer_multiview.py).
DEFAULT_OUTPUT_ROOT_BASELINE = "inference_comparison/baseline"
DEFAULT_OUTPUT_ROOT_COARSE_TO_FINE = "inference_comparison/coarse_to_fine"
DEFAULT_DATASET_FOLDER = "Dataset001_LiverTumor"
# Aligned with scripts/export.py nnUNetPredictor
DEFAULT_TILE_STEP_SIZE = 0.75


def _nnunet_preprocessed_default(repo: Path) -> Path:
    return Path(os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed")))


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
    id_set = set(identifiers)
    missing_in_input = allowed - id_set
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


def _resolve_output_root(repo: Path, output_root: str) -> Path:
    p = Path(output_root)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _resolve_repo_path(repo: Path, path_str: str) -> Path:
    """Paths relative to cwd are resolved against repo root (same convention as --output-root)."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="nnU-Net + optional refiner. For fair comparisons vs two-stage, use --stage1-only "
        "or --export-stage1-to (same stage-1 stack as coarse_to_fine; avoid mixing with nnUNetv2_predict defaults).",
    )
    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Folder with CT volumes only, nnU-Net raw naming. Repo-relative or absolute "
        "(e.g. nnUNet_raw/Dataset001_LiverTumor/imagesTr, or imagesTs for test-only).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output folder for final segmentations (per-case subfolders). If omitted, defaults under "
        f"--output-root (see --stage1-only / two-stage layout).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Repo-relative or absolute root when -o is omitted. Default: "
        f"{DEFAULT_OUTPUT_ROOT_BASELINE} for --stage1-only, "
        f"{DEFAULT_OUTPUT_ROOT_COARSE_TO_FINE} for two-stage.",
    )
    p.add_argument(
        "--dataset-folder",
        type=str,
        default=DEFAULT_DATASET_FOLDER,
        help="Dataset subfolder name under --output-root for default -o paths.",
    )
    p.add_argument(
        "--inference-subdir",
        type=str,
        default=DEFAULT_INFERENCE_SUBDIR,
        help="Last path segment when -o is omitted for two-stage (default: two_stage).",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Trained nnU-Net folder (dataset.json, plans.json, fold_*). "
        "Default: Dataset001_LiverTumor 2d run under repo nnUNet_results.",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--split",
        type=str,
        choices=("all", "val", "train"),
        default="all",
        help="Use only train or val case IDs from nnUNet_preprocessed/.../splits_final.json for --fold. "
        "Default `all`: every case found under -i.",
    )
    p.add_argument(
        "--nnunet-preprocessed",
        type=str,
        default=None,
        help="Directory containing <dataset-folder>/splits_final.json (default: $nnUNet_preprocessed or "
        "<repo>/nnUNet_preprocessed). Used only when --split is val or train.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a case if its output segmentation (<case_id>+file_ending) already exists under -o.",
    )
    p.add_argument(
        "--skip-heavy-val",
        dest="skip_heavy_val",
        action="store_true",
        help="Skip case_0004 and case_0018 (very large volumes). Default: on; use --no-skip-heavy-val to run them.",
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
        help="Comma-separated case_ids to skip in addition to heavy cases (when skip-heavy-val is on).",
    )
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument(
        "--coarse-to-fine-dir",
        type=str,
        default=None,
        dest="coarse_to_fine_dir",
        help="coarse_to_fine run dir with checkpoint_best.pth and meta.json (stage coarse_to_fine). Not used with --stage1-only.",
    )
    p.add_argument(
        "--stage1-only",
        action="store_true",
        help="Only run nnU-Net (stage 1) and write to -o. Same predictor/preprocessing as two-stage; "
        "use this for nnU-Net-only masks comparable to the coarse_to_fine path.",
    )
    p.add_argument(
        "--export-stage1-to",
        type=str,
        default=None,
        help="Two-stage only: also save nnU-Net stage-1 predictions here (same forward pass as coarse_to_fine).",
    )
    p.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save nnU-Net full softmax: <case_id>.npz (array 'probabilities'), <case_id>.pkl, and "
        "stage-1 <case_id>.nii.gz. For --stage1-only defaults to -o; for two-stage defaults to "
        "<out_dir>/nnunet_stage1_softmax (so final refined seg in -o is not overwritten).",
    )
    p.add_argument(
        "--prob-dir",
        type=str,
        default=None,
        help="Override directory for nnU-Net probability export (requires --save-probabilities).",
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
        help="nnU-Net sliding-window step (default 0.75; same as export.py). "
        "Larger values reduce peak host RAM during stage 1.",
    )
    p.add_argument(
        "--no-mirroring",
        action="store_true",
        help="Disable nnU-Net TTA mirroring (less host RAM; slightly faster).",
    )
    p.add_argument(
        "--logits-float16",
        dest="logits_float16",
        action="store_true",
        default=True,
        help="Half-precision coarse logits on CPU when use_coarse_prob (default: on; like export --low-mem).",
    )
    p.add_argument(
        "--full-precision-logits",
        dest="logits_float16",
        action="store_false",
        help="Float32 coarse logits on CPU (more host RAM).",
    )
    return p.parse_args()


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-6
    return (x - m) / s


def _tumor_label(dataset_json: dict) -> int:
    for name, idx in dataset_json.get("labels", {}).items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("tumor label not found in dataset.json")


@torch.no_grad()
def _refine_roi_slices(
    model: torch.nn.Module,
    data: np.ndarray,
    pred_seg: np.ndarray,
    bbox,
    tumor_label: int,
    crop_hw: Tuple[int, int],
    device: torch.device,
    use_coarse_prob: bool,
    logits_for_refine: torch.Tensor | None,
) -> np.ndarray:
    """Returns refined tumor probability volume (Z,Y,X) in [0,1], only bbox filled."""
    z0, z1, y0, y1, x0, x1 = bbox.z0, bbox.z1, bbox.y0, bbox.y1, bbox.x0, bbox.x1
    if use_coarse_prob:
        assert logits_for_refine is not None
        probs = torch.softmax(logits_for_refine.float(), dim=0).cpu().numpy()
    else:
        probs = None
    out = np.zeros_like(pred_seg, dtype=np.float32)
    model.eval()
    h_crop, w_crop = crop_hw
    for z in range(z0, z1):
        img2d = data[0, z, y0:y1, x0:x1]
        if use_coarse_prob and probs is not None:
            coarse2d = probs[tumor_label, z, y0:y1, x0:x1].astype(np.float32)
        else:
            coarse2d = (pred_seg[z, y0:y1, x0:x1] == tumor_label).astype(np.float32)
        nh, nw = img2d.shape
        ti = torch.from_numpy(_normalize_slice(img2d))[None, None].to(device)
        tc = torch.from_numpy(coarse2d)[None, None].to(device)
        # Match train dataset: bilinear for CT and prob; nearest for binary coarse (coarse_to_fine.dataset._resize_pair).
        ti = F.interpolate(ti, size=(h_crop, w_crop), mode="bilinear", align_corners=False)
        if use_coarse_prob:
            tc = F.interpolate(tc, size=(h_crop, w_crop), mode="bilinear", align_corners=False)
        else:
            tc = F.interpolate(tc, size=(h_crop, w_crop), mode="nearest")
        tin = torch.cat([ti, tc], dim=1)
        logit = model(tin)
        pr = torch.sigmoid(logit)
        pr = F.interpolate(pr, size=(nh, nw), mode="bilinear", align_corners=False)
        out[z, y0:y1, x0:x1] = pr[0, 0].cpu().numpy()
    return out


def _seg_to_fake_logits(final_seg: np.ndarray, num_classes: int) -> np.ndarray:
    """Build stiff logits so nnU-Net label_manager argmax recovers final_seg."""
    fl = np.zeros((num_classes,) + final_seg.shape, dtype=np.float32)
    for c in range(num_classes):
        fl[c] = (final_seg == c).astype(np.float32) * 1000.0
    return fl


def main() -> None:
    args = _parse_args()

    if args.prob_dir and not args.save_probabilities:
        raise SystemExit("--prob-dir requires --save-probabilities")

    if args.stage1_only:
        if args.coarse_to_fine_dir:
            raise SystemExit("Do not pass --coarse-to-fine-dir with --stage1-only")
        if args.export_stage1_to:
            raise SystemExit("--export-stage1-to is only for two-stage runs (omit --stage1-only)")
    else:
        if not args.coarse_to_fine_dir:
            raise SystemExit("Two-stage inference requires --coarse-to-fine-dir (or use --stage1-only)")

    repo = REPO_ROOT
    default_model = (
        repo
        / "nnUNet_results"
        / "Dataset001_LiverTumor"
        / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
    )
    model_dir = _resolve_repo_path(repo, args.model_dir) if args.model_dir else default_model
    in_dir = _resolve_repo_path(repo, args.input)

    ref_dir = _resolve_repo_path(repo, args.coarse_to_fine_dir) if args.coarse_to_fine_dir else None
    ckpt_path = ref_dir / "checkpoint_best.pth" if ref_dir else None
    meta_path = ref_dir / "meta.json" if ref_dir else None
    if not args.stage1_only:
        assert ckpt_path is not None and meta_path is not None
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path and meta_path.is_file() else {}
    _cs = meta.get("crop_size", [256, 256])
    crop_size = (int(_cs[0]), int(_cs[1]))
    # Default True: matches train_coarse_to_fine (nnU-Net softmax prob channel). Old meta.json may set false.
    use_coarse_prob = bool(meta.get("use_coarse_prob", True))
    in_channels = int(meta.get("in_channels", 2))
    # 3D ROI padding (Z,Y,X): Y/X match train_coarse_to_fine roi_pad_xy; Z uses small default (per-slice training has no Z pad).
    _rp = meta.get("roi_pad_xy", [16, 16])
    _mr = meta.get("min_roi_xy", [32, 32])
    bbox3d_pad = (2, int(_rp[0]), int(_rp[1]))
    bbox3d_min_side = (1, int(_mr[0]), int(_mr[1]))

    if args.output_root is None:
        default_root = (
            DEFAULT_OUTPUT_ROOT_BASELINE
            if args.stage1_only
            else DEFAULT_OUTPUT_ROOT_COARSE_TO_FINE
        )
        out_root = _resolve_output_root(repo, default_root)
    else:
        out_root = _resolve_output_root(repo, args.output_root)
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if args.stage1_only:
        if args.output:
            out_dir = _resolve_repo_path(repo, args.output)
        else:
            out_dir = out_root / args.dataset_folder / f"baseline_stage1_{stamp}"
    else:
        if args.output:
            out_dir = _resolve_repo_path(repo, args.output)
        else:
            # Two-stage coarse-to-fine defaults under inference_comparison/coarse_to_fine/
            out_dir = (
                out_root
                / args.dataset_folder
                / f"infer_{stamp}"
                / ref_dir.name  # type: ignore[union-attr]
                / args.inference_subdir
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_export_dir = (
        _resolve_repo_path(repo, args.export_stage1_to) if args.export_stage1_to else None
    )
    if stage1_export_dir is not None:
        stage1_export_dir.mkdir(parents=True, exist_ok=True)

    prob_dir: Path | None
    if args.save_probabilities:
        if args.prob_dir:
            prob_dir = _resolve_repo_path(repo, args.prob_dir)
        elif args.stage1_only:
            prob_dir = out_dir
        else:
            prob_dir = out_dir / "nnunet_stage1_softmax"
        prob_dir.mkdir(parents=True, exist_ok=True)
    else:
        prob_dir = None

    print(f"Writing predictions to {out_dir.resolve()}")
    if stage1_export_dir is not None:
        print(f"Also writing stage-1 (nnU-Net) predictions to {stage1_export_dir.resolve()}")
    if args.save_probabilities and prob_dir is not None:
        print(f"nnU-Net softmax (npz + pkl + stage-1 seg) -> {prob_dir.resolve()}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
    from nnunetv2.inference.export_prediction import export_prediction_from_logits  # noqa: E402
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
    from nnunetv2.utilities.utils import (  # noqa: E402
        create_lists_from_splitted_dataset_folder,
        get_identifiers_from_splitted_dataset_folder,
    )

    dataset_json = load_json(model_dir / "dataset.json")
    tumor_label = _tumor_label(dataset_json)
    file_ending = dataset_json["file_ending"]

    # Match scripts/export.py (same stage-1 stack as coarse_to_fine export).
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

    ref_model: torch.nn.Module | None = None
    if not args.stage1_only:
        ref_model = TinyUNet2d(in_channels=in_channels, base=32).to(device)
        load_checkpoint(ckpt_path, ref_model, optimizer=None, map_location=device)  # type: ignore[arg-type]

    identifiers = get_identifiers_from_splitted_dataset_folder(str(in_dir), file_ending)
    list_of_lists = create_lists_from_splitted_dataset_folder(
        str(in_dir), file_ending, identifiers=identifiers
    )

    pre_root = (
        _resolve_repo_path(repo, args.nnunet_preprocessed)
        if args.nnunet_preprocessed
        else _nnunet_preprocessed_default(repo)
    )
    splits_path = pre_root / args.dataset_folder / "splits_final.json"
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

    for case_id, image_files in zip(identifiers, list_of_lists):
        seg_dir_skip = (prob_dir if args.save_probabilities else out_dir) if args.stage1_only else out_dir
        out_seg = _seg_nifti_path(seg_dir_skip, case_id, file_ending)
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
        num_classes = int(logits.shape[0])

        if args.stage1_only:
            export_prediction_from_logits(
                logits,
                props,
                predictor.configuration_manager,
                predictor.plans_manager,
                predictor.dataset_json,
                str((prob_dir if args.save_probabilities else out_dir) / case_id),
                save_probabilities=bool(args.save_probabilities),
            )
            print(f"saved {case_id} (stage-1 only)")
            del data, props, logits
            gc.collect()
            continue

        assert prob_dir is not None or not args.save_probabilities
        if args.save_probabilities:
            export_prediction_from_logits(
                logits.clone(),
                props,
                predictor.configuration_manager,
                predictor.plans_manager,
                predictor.dataset_json,
                str(prob_dir / case_id),  # type: ignore[operator]
                save_probabilities=True,
            )
        if stage1_export_dir is not None:
            need_stage1_dup = (not args.save_probabilities) or (
                prob_dir is not None and stage1_export_dir.resolve() != prob_dir.resolve()
            )
            if need_stage1_dup:
                export_prediction_from_logits(
                    logits.clone() if args.save_probabilities else logits,
                    props,
                    predictor.configuration_manager,
                    predictor.plans_manager,
                    predictor.dataset_json,
                    str(stage1_export_dir / case_id),
                    save_probabilities=False,
                )

        pred_seg = logits.argmax(dim=0).numpy().astype(np.int16)
        coarse_bin = threshold_coarse_tumor(pred_seg, tumor_label)
        bbox = bbox3d_from_mask(
            coarse_bin,
            pad=bbox3d_pad,
            min_side=bbox3d_min_side,
        )

        logits_for_refine: torch.Tensor | None = None
        if use_coarse_prob:
            logits_for_refine = logits.half() if args.logits_float16 else logits
        del logits

        if bbox is None:
            final_seg = pred_seg.copy()
        else:
            refined_prob = _refine_roi_slices(
                ref_model,  # type: ignore[arg-type]
                data,
                pred_seg,
                bbox,
                tumor_label,
                crop_size,
                device,
                use_coarse_prob,
                logits_for_refine,
            )
            final_seg = pred_seg.copy()
            zs, ys, xs = bbox.slices()
            roi_sub = refined_prob[zs, ys, xs] > 0.5
            sub = final_seg[zs, ys, xs]
            sub_new = sub.copy()
            sub_new[roi_sub] = tumor_label
            sub_new[~roi_sub & (sub == tumor_label)] = 1
            final_seg[zs, ys, xs] = sub_new
            del refined_prob
        del logits_for_refine

        fake_logits = _seg_to_fake_logits(final_seg, num_classes)
        export_prediction_from_logits(
            torch.from_numpy(fake_logits),
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(out_dir / case_id),
            save_probabilities=False,
        )
        print(f"saved {case_id}")
        del data, props, fake_logits
        gc.collect()


if __name__ == "__main__":
    main()
