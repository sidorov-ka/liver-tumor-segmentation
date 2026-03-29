#!/usr/bin/env python3
"""
Multiview inference: suspicious-voxel ROIs + multi-window 2D refinement (nnU-Net weights unchanged).

Uses **MultiviewUNet2d** (4 ch: three fixed HU windows + tumor prob) trained **separately** from
coarse_to_fine — see multi-view fusion literature (e.g. *Deep Multi-View Fusion Network for Lung
Nodule Segmentation*). Do not point ``--multiview-dir`` at a coarse_to_fine run; use a checkpoint
produced by ``scripts/train_multiview.py`` (``multiview_results/.../multiview/run_*``).

Runs the same nnU-Net forward on preprocessed data as scripts/infer_coarse_to_fine.py (aligned shapes), then
replaces only the tumor-class probability channel inside ROIs with the multiview network; outside
ROIs the tumor channel stays the nnU-Net softmax. Writes only under ``-o``.
Use ``--skip-heavy-val`` or ``--exclude-cases`` to omit large cases without a separate input folder.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coarse_to_fine.utils import load_checkpoint  # noqa: E402
from multiview.config import MultiviewConfig  # noqa: E402
from multiview.model import MultiviewUNet2d  # noqa: E402
from multiview.pipeline import refine_tumor_probability_volume  # noqa: E402


def _resolve_repo_path(repo: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def _nnunet_preprocessed_default(repo: Path) -> Path:
    return Path(os.environ.get("nnUNet_preprocessed", str(repo / "nnUNet_preprocessed")))


def _tumor_label(dataset_json: dict) -> int:
    for name, idx in dataset_json.get("labels", {}).items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("tumor label not found in dataset.json")


def _cfg_from_meta(cfg: MultiviewConfig, meta: dict) -> None:
    """Defaults from meta.json written by train_multiview (CLI flags override later)."""
    if not meta:
        return
    hw = meta.get("hu_windows")
    if hw and len(hw) == 3:
        cfg.hu_windows = tuple((float(a), float(b)) for a, b in hw)
    cs = meta.get("crop_size")
    if cs and len(cs) == 2:
        cfg.crop_size = (int(cs[0]), int(cs[1]))
    mc = meta.get("multiview_config") or {}
    if "prob_lo" in mc:
        cfg.prob_lo = float(mc["prob_lo"])
    if "prob_hi" in mc:
        cfg.prob_hi = float(mc["prob_hi"])
    if "min_component_voxels" in mc:
        cfg.min_component_voxels = int(mc["min_component_voxels"])
    if "roi_pad" in mc and len(mc["roi_pad"]) == 3:
        cfg.roi_pad = (int(mc["roi_pad"][0]), int(mc["roi_pad"][1]), int(mc["roi_pad"][2]))
    if "min_roi_side" in mc and len(mc["min_roi_side"]) == 3:
        cfg.min_roi_side = (
            int(mc["min_roi_side"][0]),
            int(mc["min_roi_side"][1]),
            int(mc["min_roi_side"][2]),
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiview ROI refinement (MultiviewUNet2d checkpoint, not coarse_to_fine)."
    )
    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="nnUNet-style images folder (e.g. nnUNet_raw/.../imagesTr).",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Trained nnU-Net folder (default: Dataset001_LiverTumor 2d trainer under nnUNet_results).",
    )
    p.add_argument(
        "--multiview-dir",
        type=str,
        required=True,
        dest="multiview_dir",
        help="Multiview training run only (not coarse_to_fine_results): checkpoint_best.pth + meta.json "
        "for MultiviewUNet2d (in_channels=4), e.g. multiview_results/.../multiview/run_<timestamp>.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory for multiview segmentations (e.g. inference_comparison/multiview/...).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--dataset-folder",
        type=str,
        default="Dataset001_LiverTumor",
        help="Under nnUNet_preprocessed for splits_final.json (optional filtering).",
    )
    p.add_argument("--nnunet-preprocessed", type=str, default=None)
    p.add_argument(
        "--split",
        type=str,
        choices=["all", "val", "train"],
        default="all",
        help="Restrict to splits_final.json fold (default: all cases under -i).",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--tile-step-size",
        type=float,
        default=0.75,
        help="nnU-Net sliding-window step (match scripts/infer_coarse_to_fine.py / export).",
    )
    p.add_argument("--no-mirroring", action="store_true")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pth",
        help="nnU-Net trainer checkpoint inside model-dir (match scripts/infer_coarse_to_fine.py).",
    )
    p.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also write case_id.npz + case_id.pkl (softmax) like nnU-Net export.",
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--skip-heavy-val",
        action="store_true",
        help="Skip case_0004 and case_0018 (very large volumes; typical nnU-Net tile count >3000 on fold 0 val).",
    )
    p.add_argument(
        "--exclude-cases",
        type=str,
        default="",
        help="Comma-separated case_ids to skip in addition to --skip-heavy-val (e.g. case_0004,case_0018).",
    )
    # Multiview hyperparameters
    p.add_argument("--prob-lo", type=float, default=None)
    p.add_argument("--prob-hi", type=float, default=None)
    p.add_argument("--min-component-voxels", type=int, default=None)
    p.add_argument("--roi-pad", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--min-roi-side", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    in_dir = _resolve_repo_path(repo, args.input)
    out_dir = _resolve_repo_path(repo, args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = _resolve_repo_path(repo, args.multiview_dir)
    ckpt_path = ref_dir / "checkpoint_best.pth"
    meta_path = ref_dir / "meta.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    default_model = (
        repo
        / "nnUNet_results"
        / "Dataset001_LiverTumor"
        / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
    )
    model_dir = _resolve_repo_path(repo, args.model_dir) if args.model_dir else default_model

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

    meta_in_ch = int(meta.get("in_channels", 4))
    if meta_in_ch != 4:
        print(
            f"WARNING: multiview meta.json has in_channels={meta_in_ch}; MultiviewUNet2d expects 4. "
            "Load may fail unless the checkpoint matches."
        )

    base = int(meta.get("base", 32))
    ref_model = MultiviewUNet2d(base=base).to(device)
    load_checkpoint(ckpt_path, ref_model, optimizer=None, map_location=device)  # type: ignore[arg-type]

    cfg = MultiviewConfig()
    _cfg_from_meta(cfg, meta)
    if args.prob_lo is not None:
        cfg.prob_lo = float(args.prob_lo)
    if args.prob_hi is not None:
        cfg.prob_hi = float(args.prob_hi)
    if args.min_component_voxels is not None:
        cfg.min_component_voxels = int(args.min_component_voxels)
    if args.roi_pad is not None:
        cfg.roi_pad = (int(args.roi_pad[0]), int(args.roi_pad[1]), int(args.roi_pad[2]))
    if args.min_roi_side is not None:
        cfg.min_roi_side = (int(args.min_roi_side[0]), int(args.min_roi_side[1]), int(args.min_roi_side[2]))
    if args.crop_size is not None:
        cfg.crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))

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

    pre_root = (
        _resolve_repo_path(repo, args.nnunet_preprocessed)
        if args.nnunet_preprocessed
        else _nnunet_preprocessed_default(repo)
    )
    splits_path = pre_root / args.dataset_folder / "splits_final.json"

    if args.split != "all":
        if not splits_path.is_file():
            raise FileNotFoundError(f"--split {args.split} requires {splits_path}")
        splits = load_json(str(splits_path))
        if args.fold < 0 or args.fold >= len(splits):
            raise IndexError(f"fold {args.fold} out of range")
        key = "train" if args.split == "train" else "val"
        allowed = set(splits[args.fold][key])
        kept_id: list = []
        kept_lol: list = []
        for cid, lol in zip(identifiers, list_of_lists):
            if cid in allowed:
                kept_id.append(cid)
                kept_lol.append(lol)
        identifiers, list_of_lists = kept_id, kept_lol
        if not identifiers:
            raise SystemExit(f"No cases left after --split {args.split}.")

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
        out_seg = out_dir / f"{case_id}{file_ending}"
        if args.skip_existing and out_seg.is_file():
            print(f"skip {case_id}")
            continue

        data, _seg, props = preprocessor.run_case(
            image_files,
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        probs = torch.softmax(logits.float(), dim=0).numpy()
        prob_tumor = probs[tumor_label].astype(np.float32)

        ct = data[0].astype(np.float32)
        refined_tumor = refine_tumor_probability_volume(ct, prob_tumor, ref_model, device, cfg)

        probs_out = probs.copy()
        probs_out[tumor_label] = refined_tumor
        s = probs_out.sum(axis=0, keepdims=True)
        probs_out = np.divide(probs_out, np.maximum(s, 1e-8))

        logits_out = np.log(np.clip(probs_out, 1e-8, 1.0)).astype(np.float32)

        export_prediction_from_logits(
            torch.from_numpy(logits_out),
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(out_dir / case_id),
            save_probabilities=bool(args.save_probabilities),
        )
        print(f"saved {case_id}")
        del data, props, logits, probs, logits_out, probs_out
        gc.collect()


if __name__ == "__main__":
    main()
