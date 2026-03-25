#!/usr/bin/env python3
"""
Two-stage inference: nnU-Net (stage 1) + 2D refinement (stage 2) inside tumor ROI.

Writes final segmentations in the same way as nnU-Net (NIfTI in original image space).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from refinement.model import TinyUNet2d  # noqa: E402
from refinement.roi import bbox3d_from_mask, threshold_coarse_tumor  # noqa: E402
from refinement.utils import load_checkpoint  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nnU-Net + refinement two-stage inference.")
    p.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Folder with test images (same naming as nnU-Net, e.g. case_0000.nii.gz).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output folder for final segmentations.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="nnU-Net training output folder (default: repo nnUNet_results 2d run).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    p.add_argument(
        "--refinement-dir",
        type=str,
        required=True,
        help="Folder with refinement checkpoint_best.pth and meta.json from train_refiner.py",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "mps"),
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
    crop_hw: tuple[int, int],
    device: torch.device,
    use_coarse_prob: bool,
    logits: torch.Tensor,
) -> np.ndarray:
    """Returns refined tumor probability volume (Z,Y,X) in [0,1], only bbox filled."""
    z0, z1, y0, y1, x0, x1 = bbox.z0, bbox.z1, bbox.y0, bbox.y1, bbox.x0, bbox.x1
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    out = np.zeros_like(pred_seg, dtype=np.float32)
    model.eval()
    h_crop, w_crop = crop_hw
    for z in range(z0, z1):
        img2d = data[0, z, y0:y1, x0:x1]
        if use_coarse_prob:
            coarse2d = probs[tumor_label, z, y0:y1, x0:x1].astype(np.float32)
        else:
            coarse2d = (pred_seg[z, y0:y1, x0:x1] == tumor_label).astype(np.float32)
        nh, nw = img2d.shape
        ti = torch.from_numpy(_normalize_slice(img2d))[None, None].to(device)
        tc = torch.from_numpy(coarse2d)[None, None].to(device)
        tin = torch.cat([ti, tc], dim=1)
        tin = F.interpolate(tin, size=(h_crop, w_crop), mode="bilinear", align_corners=False)
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
    repo = REPO_ROOT
    default_model = (
        repo
        / "nnUNet_results"
        / "Dataset001_LiverTumor"
        / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
    )
    model_dir = Path(args.model_dir) if args.model_dir else default_model
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = Path(args.refinement_dir)
    ckpt_path = ref_dir / "checkpoint_best.pth"
    meta_path = ref_dir / "meta.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    crop_size = tuple(meta.get("crop_size", [256, 256]))
    use_coarse_prob = bool(meta.get("use_coarse_prob", False))
    in_channels = int(meta.get("in_channels", 2))

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

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

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
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

    ref_model = TinyUNet2d(in_channels=in_channels, base=32).to(device)
    load_checkpoint(ckpt_path, ref_model, optimizer=None, map_location=device)

    identifiers = get_identifiers_from_splitted_dataset_folder(str(in_dir), file_ending)
    list_of_lists = create_lists_from_splitted_dataset_folder(
        str(in_dir), file_ending, identifiers=identifiers
    )

    for case_id, image_files in zip(identifiers, list_of_lists):
        data, seg, props = preprocessor.run_case(
            image_files,
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        num_classes = int(logits.shape[0])
        pred_seg = logits.argmax(dim=0).numpy().astype(np.int16)

        coarse_bin = threshold_coarse_tumor(pred_seg, tumor_label)
        bbox = bbox3d_from_mask(coarse_bin)

        if bbox is None:
            final_seg = pred_seg.copy()
        else:
            refined_prob = _refine_roi_slices(
                ref_model,
                data,
                pred_seg,
                bbox,
                tumor_label,
                crop_size,
                device,
                use_coarse_prob,
                logits,
            )
            final_seg = pred_seg.copy()
            zs, ys, xs = bbox.slices()
            roi_sub = refined_prob[zs, ys, xs] > 0.5
            sub = final_seg[zs, ys, xs]
            sub_new = sub.copy()
            sub_new[roi_sub] = tumor_label
            sub_new[~roi_sub & (sub == tumor_label)] = 1
            final_seg[zs, ys, xs] = sub_new

        fake_logits = _seg_to_fake_logits(final_seg, num_classes)
        out_base = out_dir / case_id
        export_prediction_from_logits(
            torch.from_numpy(fake_logits),
            props,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            str(out_base),
            save_probabilities=False,
        )
        print(f"saved {case_id}")


if __name__ == "__main__":
    main()
