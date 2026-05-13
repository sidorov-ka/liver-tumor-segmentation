#!/usr/bin/env python3
"""
Axial CT slice with GT vs prediction tumor contours (PNG in repo).

Example:
  python scripts/visualization/visualize_tumor_slice.py --case case_0022
  python scripts/visualization/visualize_tumor_slice.py --case case_0000 \\
    --pred-dir inference_comparison/multiview_infer_run_2026_04_03

  All val cases for fold 0 (cases that have a pred under --pred-dir):
  python scripts/visualization/visualize_tumor_slice.py \\
    --pred-dir inference_comparison/uncertainty \\
    --split val --fold 0 --output-dir visualizations/uncertainty_val_fold0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _short_pred_label(pred_dir: Path, override: Optional[str]) -> str:
    """Stable short tag for filenames (baseline / coarse_to_fine / multiview / folder name)."""
    if override is not None and str(override).strip():
        return str(override).strip().replace(" ", "_")
    name = pred_dir.name
    lower = name.lower()
    if lower == "baseline":
        return "baseline"
    if lower == "coarse_to_fine":
        return "coarse_to_fine"
    if "uncertainty" in lower:
        return "uncertainty"
    if "multiview" in lower:
        return "multiview"
    if len(name) > 40:
        return name[:37] + "..."
    return name


def _load_nifti(path: Path) -> Tuple[np.ndarray, Any]:
    """Load volume; gzip fallback if extension is .nii.gz but file is uncompressed (same as evaluate_segmentations)."""
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


def _tumor_label(dataset_json: dict) -> int:
    labels = dataset_json.get("labels", {})
    for name, idx in labels.items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("No 'tumor' in dataset.json labels")


def _nnunet_preprocessed_default() -> Path:
    return Path(os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "nnUNet_preprocessed")))


def _split_case_ids(split: str, fold: int, dataset_folder: str) -> List[str]:
    path = _nnunet_preprocessed_default() / dataset_folder / "splits_final.json"
    if not path.is_file():
        raise SystemExit(f"splits_final.json not found: {path}")
    splits = json.loads(path.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(splits):
        raise SystemExit(f"fold {fold} out of range (len={len(splits)})")
    key = "train" if split == "train" else "val"
    return list(splits[fold][key])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save one axial slice: CT + GT tumor (green) vs pred tumor (red dashed)."
    )
    p.add_argument(
        "--case",
        type=str,
        default=None,
        help="case_id, e.g. case_0022. Omit with --split to render all matching cases.",
    )
    p.add_argument(
        "--pred-dir",
        type=str,
        default="inference_comparison/baseline",
        help="Folder with <case_id>.nii.gz (nnU-Net or multiview output).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Repo-relative folder for PNG (default: visualizations/).",
    )
    p.add_argument(
        "--dataset-json",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json"),
    )
    p.add_argument(
        "--slice-z",
        type=int,
        default=None,
        help="Override axial index (axis 0). Default: slice with largest GT tumor area.",
    )
    p.add_argument(
        "--ct-window",
        type=float,
        nargs=2,
        default=[-100.0, 400.0],
        metavar=("LO", "HI"),
        help="HU window for CT display.",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Short tag in output filename (default: baseline / coarse_to_fine / multiview from --pred-dir).",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("val", "train"),
        default=None,
        help="With omitted --case: render every case in this split (requires splits_final.json).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--dataset-folder",
        type=str,
        default="Dataset001_LiverTumor",
        help="Under nnUNet_preprocessed for splits_final.json.",
    )
    return p.parse_args()


def _visualize_one_case(
    case_id: str,
    pred_dir: Path,
    out_dir: Path,
    tumor_label: int,
    fe: str,
    slice_z: Optional[int],
    ct_window: Tuple[float, float],
    label_override: Optional[str],
) -> Path:
    gt_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "labelsTr"
    images_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "imagesTr"

    ct_path = images_dir / f"{case_id}_0000{fe}"
    gt_path = gt_dir / f"{case_id}{fe}"
    pred_path = pred_dir / f"{case_id}{fe}"

    for path, lbl in [(ct_path, "CT"), (gt_path, "GT"), (pred_path, "prediction")]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {lbl}: {path}")

    ct, _ = _load_nifti(ct_path)
    gt, _ = _load_nifti(gt_path)
    pr, _ = _load_nifti(pred_path)

    if ct.shape != gt.shape or ct.shape != pr.shape:
        raise RuntimeError(f"Shape mismatch: CT {ct.shape} GT {gt.shape} pred {pr.shape}")

    gt_t = (gt == tumor_label).astype(np.float32)
    pr_t = (pr == tumor_label).astype(np.float32)

    if slice_z is not None:
        z = int(slice_z)
        if z < 0 or z >= ct.shape[0]:
            raise ValueError(f"slice_z {z} out of range [0, {ct.shape[0]})")
    else:
        sums = gt_t.sum(axis=(1, 2))
        if float(sums.max()) <= 0:
            raise RuntimeError(f"No tumor voxels in GT for {case_id}")
        z = int(np.argmax(sums))

    sl_ct = ct[z].astype(np.float32)
    sl_gt = gt_t[z]
    sl_pr = pr_t[z]

    lo, hi = float(ct_window[0]), float(ct_window[1])
    sl_vis = np.clip(sl_ct, lo, hi)
    sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(sl_vis.T, cmap="gray", origin="lower", aspect="auto")
    ax.contour(sl_gt.T, levels=[0.5], colors="lime", linewidths=2.0, origin="lower")
    ax.contour(sl_pr.T, levels=[0.5], colors="red", linewidths=2.0, origin="lower", linestyles="--")
    label = _short_pred_label(pred_dir, label_override)
    ax.set_title(f"{case_id}  z={z}  green=GT tumor  red={label} (pred)")
    ax.axis("off")
    plt.tight_layout()

    out_png = out_dir / f"{case_id}_z{z}_{label}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> None:
    args = _parse_args()
    pred_dir = Path(args.pred_dir)
    pred_dir = pred_dir if pred_dir.is_absolute() else REPO_ROOT / pred_dir
    pred_dir = pred_dir.resolve()
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dj_path = Path(args.dataset_json)
    if not dj_path.is_file():
        raise SystemExit(f"dataset.json not found: {dj_path}")
    dataset_json = json.loads(dj_path.read_text(encoding="utf-8"))
    tumor_label = _tumor_label(dataset_json)
    fe = dataset_json.get("file_ending", ".nii.gz")
    if not fe.startswith("."):
        fe = f".{fe}"

    if args.case is None:
        if args.split is None:
            raise SystemExit("Pass --case, or use --split val|train without --case for batch mode.")
        case_ids = _split_case_ids(args.split, args.fold, args.dataset_folder)
        fe_suffix = fe if fe.startswith(".") else f".{fe}"
        written = 0
        skipped = 0
        for cid in case_ids:
            pred_path = pred_dir / f"{cid}{fe_suffix}"
            if not pred_path.is_file():
                print(f"skip {cid} (no prediction under pred-dir)", file=sys.stderr)
                skipped += 1
                continue
            try:
                out = _visualize_one_case(
                    cid,
                    pred_dir,
                    out_dir,
                    tumor_label,
                    fe,
                    args.slice_z,
                    (float(args.ct_window[0]), float(args.ct_window[1])),
                    args.label,
                )
                print(f"Wrote {out}")
                written += 1
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                print(f"skip {cid}: {e}", file=sys.stderr)
                skipped += 1
        print(f"Done: {written} PNG(s), skipped {skipped}", file=sys.stderr)
        return

    case_id = args.case.strip()
    if not case_id.startswith("case_"):
        case_id = f"case_{case_id}"

    try:
        out_png = _visualize_one_case(
            case_id,
            pred_dir,
            out_dir,
            tumor_label,
            fe,
            args.slice_z,
            (float(args.ct_window[0]), float(args.ct_window[1])),
            args.label,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
