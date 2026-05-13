#!/usr/bin/env python3
"""
Uncertainty-guided inference: coarse-tumor (± optional high-entropy) ROIs + 5-channel 2D refinement.

Uses **UncertaintyUNet2d** / **UncertaintyDualHeadUNet2d** (5 ch: three HU windows + tumor prob + entropy).
Dual-head runs apply an error gate on inference. Train with ``scripts/train_uncertainty.py``; point
``--uncertainty-dir`` at ``results_uncertainty/.../uncertainty/run_*``.

Pipeline: nnU-Net logits → softmax → tumor prob → entropy U(p) → ROI from coarse mask (not uncertainty-only) →
refinement inside ROI → renormalize softmax → export under ``-o`` (e.g. inference_comparison/uncertainty).

If ``--update-mode`` is omitted, the refiner output **replaces** nnU-Net prob inside the ROI (matches the training
objective; use ``--update-mode blend`` to mix with baseline like old defaults in ``meta.json``).

**Host RAM:** default ``--logits-float16`` keeps stage-1 logits in half precision before softmax (same idea as
``infer_coarse_to_fine``). Renormalization updates ``probs`` in place (no duplicate full softmax tensor). Use
``--full-precision-logits`` only if you need float32 logits. Larger ``--tile-step-size`` (e.g. 0.85–1.0) reduces
sliding-window overlap and peak RAM. ``--no-mirroring`` also helps. ``--nnunet-low-vram`` sets
``perform_everything_on_device=False`` (saves GPU VRAM; may use more CPU RAM).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "2d"))

from coarse_to_fine.utils import load_checkpoint  # noqa: E402
from uncertainty.config import (  # noqa: E402
    UncertaintyConfig,
    merge_uncertainty_config_from_meta_dict,
)
from uncertainty.model import build_uncertainty_model  # noqa: E402
from uncertainty.pipeline import refine_tumor_probability_volume  # noqa: E402


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


def _cfg_from_meta(cfg: UncertaintyConfig, meta: dict) -> None:
    if not meta:
        return
    hw = meta.get("hu_windows")
    if hw and len(hw) == 3:
        cfg.hu_windows = tuple((float(a), float(b)) for a, b in hw)
    cs = meta.get("crop_size")
    if cs and len(cs) == 2:
        cfg.crop_size = (int(cs[0]), int(cs[1]))
    mc = meta.get("uncertainty_config") or {}
    merge_uncertainty_config_from_meta_dict(cfg, mc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uncertainty-guided ROI refinement (UncertaintyUNet2d checkpoint)."
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
        help="Trained nnU-Net folder (default: Dataset001_LiverTumor 2d under nnUNet_results).",
    )
    p.add_argument(
        "--uncertainty-dir",
        type=str,
        required=True,
        dest="uncertainty_dir",
        help="Uncertainty training run: checkpoint_best.pth + meta.json (5 ch), e.g. results_uncertainty/.../run_*.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <repo>/inference_comparison/uncertainty).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dataset-folder", type=str, default="Dataset001_LiverTumor")
    p.add_argument("--nnunet-preprocessed", type=str, default=None)
    p.add_argument("--split", type=str, choices=["all", "val", "train"], default="all")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--tile-step-size",
        type=float,
        default=0.75,
        help="nnU-Net sliding-window step (default 0.75; larger values reduce overlap and peak host RAM).",
    )
    p.add_argument("--no-mirroring", action="store_true")
    p.add_argument(
        "--logits-float16",
        dest="logits_float16",
        action="store_true",
        default=True,
        help="Half-precision stage-1 logits on CPU before softmax (default: on; saves host RAM).",
    )
    p.add_argument(
        "--full-precision-logits",
        dest="logits_float16",
        action="store_false",
        help="Float32 stage-1 logits on CPU (more host RAM).",
    )
    p.add_argument(
        "--nnunet-low-vram",
        action="store_true",
        help="Run nnU-Net with perform_everything_on_device=False (saves GPU VRAM; may use more CPU RAM).",
    )
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
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
    p.add_argument("--roi-positive-threshold", type=float, default=None)
    p.add_argument("--uncertainty-threshold", type=float, default=None)
    p.add_argument("--prob-min-for-uncertainty-union", type=float, default=None)
    p.add_argument("--min-component-voxels", type=int, default=None)
    p.add_argument("--roi-pad", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--min-roi-side", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--crop-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument(
        "--update-mode",
        type=str,
        choices=("replace", "blend"),
        default=None,
        help="Omit = replace in ROI (training-aligned metrics). blend = use meta/alpha mixing with nnU-Net prob.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Only for --update-mode blend (weight on refined prob).",
    )
    p.add_argument(
        "--error-gate-floor",
        type=float,
        default=None,
        help="Optional [0,1] floor on sigmoid(error_head) during gating (overrides meta). None = use meta / off.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    in_dir = _resolve_repo_path(repo, args.input)
    out_dir = (
        _resolve_repo_path(repo, args.output)
        if args.output is not None
        else (repo / "inference_comparison" / "uncertainty")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = _resolve_repo_path(repo, args.uncertainty_dir)
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

    meta_in_ch = int(meta.get("in_channels", 5))
    if meta_in_ch != 5:
        print(
            f"WARNING: meta.json has in_channels={meta_in_ch}; uncertainty models expect 5. "
            "Load may fail unless the checkpoint matches."
        )

    cfg = UncertaintyConfig()
    _cfg_from_meta(cfg, meta)
    if "use_error_head" in meta:
        cfg.use_error_head = bool(meta["use_error_head"])

    base = int(meta.get("base", 32))
    ref_model = build_uncertainty_model(base=base, use_error_head=cfg.use_error_head).to(device)
    load_checkpoint(
        ckpt_path,
        ref_model,
        optimizer=None,
        map_location=device,
        strict=False,
    )  # type: ignore[arg-type]

    if args.roi_positive_threshold is not None:
        cfg.roi_positive_threshold = float(args.roi_positive_threshold)
    if args.uncertainty_threshold is not None:
        cfg.uncertainty_threshold = float(args.uncertainty_threshold)
    if args.prob_min_for_uncertainty_union is not None:
        cfg.prob_min_for_uncertainty_union = float(args.prob_min_for_uncertainty_union)
    if args.min_component_voxels is not None:
        cfg.min_component_voxels = int(args.min_component_voxels)
    if args.roi_pad is not None:
        cfg.roi_pad = (int(args.roi_pad[0]), int(args.roi_pad[1]), int(args.roi_pad[2]))
    if args.min_roi_side is not None:
        cfg.min_roi_side = (int(args.min_roi_side[0]), int(args.min_roi_side[1]), int(args.min_roi_side[2]))
    if args.crop_size is not None:
        cfg.crop_size = (int(args.crop_size[0]), int(args.crop_size[1]))
    if args.update_mode is not None:
        cfg.update_mode = str(args.update_mode)
    if args.alpha is not None:
        cfg.alpha = float(args.alpha)
    if args.error_gate_floor is not None:
        cfg.error_gate_floor = float(args.error_gate_floor)

    # Infer-only: training optimizes the refiner logits directly (no blend). Omitting --update-mode avoids
    # using legacy meta defaults (blend) so full-volume metrics match the supervised objective.
    if args.update_mode is None:
        cfg.update_mode = "replace"

    print(
        f"Uncertainty refinement: update_mode={cfg.update_mode!r}"
        + (f", alpha={cfg.alpha}" if cfg.update_mode.strip().lower() == "blend" else "")
    )

    perform_on_device = (device.type == "cuda") and (not args.nnunet_low_vram)
    predictor = nnUNetPredictor(
        tile_step_size=float(args.tile_step_size),
        use_gaussian=True,
        use_mirroring=not args.no_mirroring,
        perform_everything_on_device=perform_on_device,
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
        kept_id = []
        kept_lol = []
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
        if args.logits_float16:
            logits = logits.half()
        probs = torch.softmax(logits.float(), dim=0).numpy().astype(np.float32)
        del logits
        gc.collect()

        prob_tumor = probs[tumor_label]
        ct = data[0].astype(np.float32)
        refined_tumor = refine_tumor_probability_volume(ct, prob_tumor, ref_model, device, cfg)

        probs[tumor_label] = refined_tumor
        s = probs.sum(axis=0, keepdims=True)
        np.divide(probs, np.maximum(s, 1e-8), out=probs)

        logits_out = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)

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
        del data, props, logits_out, probs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
