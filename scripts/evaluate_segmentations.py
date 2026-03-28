#!/usr/bin/env python3
"""
Volumetric Dice / IoU for tumor vs nnU-Net-style reference labels.

Expects predictions from ``infer.py`` (flat ``<pred_dir>/<case_id>.nii.gz``) and GT in
``<gt_dir>/<case_id>.nii.gz`` (e.g. ``labelsTr``). Same spacing/shape as usual after nnU-Net export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_nifti(path: Path) -> Tuple[np.ndarray, Any]:
    """Load segmentation array; try gzip fallback if extension is .nii.gz but file is uncompressed."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
        return np.asanyarray(img.dataobj), img
    except Exception:
        raw = path.read_bytes()
        if len(raw) >= 4 and raw[:4] == b"\x1f\x8b\x08\x00":
            raise
        from nibabel import Nifti1Image

        return np.asanyarray(Nifti1Image.from_bytes(raw).dataobj), None


def _tumor_label_from_dataset_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels", {})
    for name, idx in labels.items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("No 'tumor' key in dataset.json labels")


def _find_pred_files(pred_dir: Path) -> Dict[str, Path]:
    """Map case_id -> path. Supports ``dir/<case_id>.nii.gz`` or ``dir/<case_id>/<case_id>.nii.gz``."""
    out: Dict[str, Path] = {}
    for p in sorted(pred_dir.glob("*.nii.gz")):
        case_id = p.name[: -len(".nii.gz")] if p.name.endswith(".nii.gz") else p.stem
        out[case_id] = p
    for sub in sorted(pred_dir.iterdir()):
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob("*.nii.gz")):
            case_id = p.name[: -len(".nii.gz")] if p.name.endswith(".nii.gz") else p.stem
            out[case_id] = p
    return out


def _gt_path(gt_dir: Path, case_id: str, file_ending: str) -> Path:
    fe = file_ending if file_ending.startswith(".") else f".{file_ending}"
    return gt_dir / f"{case_id}{fe}"


def _binary_metrics(pred: np.ndarray, ref: np.ndarray, tumor_label: int) -> Dict[str, float]:
    p = pred == tumor_label
    t = ref == tumor_label
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    denom_dice = 2.0 * tp + fp + fn
    dice = float((2.0 * tp) / denom_dice) if denom_dice > 0 else 1.0
    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou > 0 else 1.0
    return {
        "Dice": dice,
        "IoU": iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "n_pred": tp + fp,
        "n_ref": tp + fn,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tumor Dice/IoU vs reference segmentations (NIfTI).")
    p.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Folder with predictions (*.nii.gz), same layout as infer.py output.",
    )
    p.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Folder with reference labels (e.g. nnUNet_raw/.../labelsTr).",
    )
    p.add_argument(
        "--dataset-json",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json"),
        help="Read tumor class index from labels (default: Dataset001_LiverTumor).",
    )
    p.add_argument(
        "--tumor-label",
        type=int,
        default=None,
        help="Override tumor label id (default: from dataset.json).",
    )
    p.add_argument(
        "--file-ending",
        type=str,
        default=".nii.gz",
        help="Suffix for GT filenames (default .nii.gz).",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write full report to this path (default: print summary only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    if not pred_dir.is_dir():
        raise SystemExit(f"pred-dir not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"gt-dir not found: {gt_dir}")

    if args.tumor_label is not None:
        tumor_label = int(args.tumor_label)
    else:
        dj = Path(args.dataset_json)
        if not dj.is_file():
            raise SystemExit(f"dataset-json not found: {dj} (pass --tumor-label)")
        tumor_label = _tumor_label_from_dataset_json(dj)

    pred_map = _find_pred_files(pred_dir)
    if not pred_map:
        raise SystemExit(f"No *.nii.gz found under {pred_dir}")

    global_tp = global_fp = global_fn = 0.0
    per_case: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for case_id in sorted(pred_map.keys()):
        pred_path = pred_map[case_id]
        gt_path = _gt_path(gt_dir, case_id, args.file_ending)
        if not gt_path.is_file():
            skipped.append(case_id)
            continue

        pred_vol, _ = _load_nifti(pred_path)
        ref_vol, _ = _load_nifti(gt_path)
        if pred_vol.shape != ref_vol.shape:
            raise RuntimeError(
                f"Shape mismatch {case_id}: pred {pred_vol.shape} vs ref {ref_vol.shape}"
            )

        m = _binary_metrics(pred_vol, ref_vol, tumor_label)
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        global_tp += tp
        global_fp += fp
        global_fn += fn
        per_case.append({"case_id": case_id, "metrics": {str(tumor_label): {k: m[k] for k in m}}})

    if not per_case:
        raise SystemExit("No overlapping cases between pred-dir and gt-dir (check filenames).")

    denom_dice = 2.0 * global_tp + global_fp + global_fn
    g_dice = float((2.0 * global_tp) / denom_dice) if denom_dice > 0 else 1.0
    denom_iou = global_tp + global_fp + global_fn
    g_iou = float(global_tp / denom_iou) if denom_iou > 0 else 1.0

    report: Dict[str, Any] = {
        "tumor_label": tumor_label,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n_pred_files": len(pred_map),
        "cases_evaluated": len(per_case),
        "skipped_missing_gt": skipped,
        "global": {
            str(tumor_label): {
                "Dice": g_dice,
                "IoU": g_iou,
                "TP": global_tp,
                "FP": global_fp,
                "FN": global_fn,
            }
        },
        "foreground_dice": g_dice,
        "foreground_iou": g_iou,
        "metric_per_case": per_case,
    }

    print(
        f"Evaluated {len(per_case)} cases (tumor label {tumor_label}). "
        f"Global Dice {g_dice:.4f}  IoU {g_iou:.4f}"
    )
    if skipped:
        print(f"Skipped (no GT file): {len(skipped)} — {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
