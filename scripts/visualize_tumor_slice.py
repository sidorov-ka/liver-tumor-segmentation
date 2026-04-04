#!/usr/bin/env python3
"""
Axial CT slice with GT vs prediction tumor contours (PNG in repo).

Example:
  python scripts/visualize_tumor_slice.py --case case_0022
  python scripts/visualize_tumor_slice.py --case case_0000 --pred-dir inference_comparison/multiview_infer_run_2026_04_03
  # Output: one PNG, short name: visualizations/case_0000_z123_multiview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save one axial slice: CT + GT tumor (green) vs pred tumor (red dashed)."
    )
    p.add_argument("--case", type=str, required=True, help="case_id, e.g. case_0022")
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    case_id = args.case.strip()
    if not case_id.startswith("case_"):
        case_id = f"case_{case_id}"

    pred_dir = (REPO_ROOT / args.pred_dir).resolve() if not Path(args.pred_dir).is_absolute() else Path(args.pred_dir)
    out_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dj_path = Path(args.dataset_json)
    if not dj_path.is_file():
        raise SystemExit(f"dataset.json not found: {dj_path}")
    dataset_json = json.loads(dj_path.read_text(encoding="utf-8"))
    tumor_label = _tumor_label(dataset_json)
    fe = dataset_json.get("file_ending", ".nii.gz")
    if not fe.startswith("."):
        fe = f".{fe}"

    gt_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "labelsTr"
    images_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "imagesTr"

    # CT: first channel nnU-Net naming
    ct_path = images_dir / f"{case_id}_0000{fe}"
    gt_path = gt_dir / f"{case_id}{fe}"
    pred_path = pred_dir / f"{case_id}{fe}"

    for path, label in [(ct_path, "CT"), (gt_path, "GT"), (pred_path, "prediction")]:
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    ct, _ = _load_nifti(ct_path)
    gt, _ = _load_nifti(gt_path)
    pr, _ = _load_nifti(pred_path)

    if ct.shape != gt.shape or ct.shape != pr.shape:
        raise SystemExit(f"Shape mismatch: CT {ct.shape} GT {gt.shape} pred {pr.shape}")

    gt_t = (gt == tumor_label).astype(np.float32)
    pr_t = (pr == tumor_label).astype(np.float32)

    if args.slice_z is not None:
        z = int(args.slice_z)
        if z < 0 or z >= ct.shape[0]:
            raise SystemExit(f"--slice-z {z} out of range [0, {ct.shape[0]})")
    else:
        sums = gt_t.sum(axis=(1, 2))
        if float(sums.max()) <= 0:
            raise SystemExit(f"No tumor voxels in GT for {case_id}")
        z = int(np.argmax(sums))

    sl_ct = ct[z].astype(np.float32)
    sl_gt = gt_t[z]
    sl_pr = pr_t[z]

    lo, hi = float(args.ct_window[0]), float(args.ct_window[1])
    sl_vis = np.clip(sl_ct, lo, hi)
    sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(sl_vis.T, cmap="gray", origin="lower", aspect="auto")
    ax.contour(sl_gt.T, levels=[0.5], colors="lime", linewidths=2.0, origin="lower")
    ax.contour(sl_pr.T, levels=[0.5], colors="red", linewidths=2.0, origin="lower", linestyles="--")
    label = _short_pred_label(pred_dir, args.label)
    ax.set_title(f"{case_id}  z={z}  green=GT tumor  red={label} (pred)")
    ax.axis("off")
    plt.tight_layout()

    out_png = out_dir / f"{case_id}_z{z}_{label}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
