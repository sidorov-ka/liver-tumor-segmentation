#!/usr/bin/env python3
"""
Scatter: GT tumor volume (mm³) vs ΔDice (model − baseline), computed from NIfTI
predictions under fold_0/validation and GT segmentations (no metrics JSON).

Example:
  .venv/bin/python scripts/plot_val_delta_vs_gt_volume_from_preds.py \\
    --baseline-pred-dir results_3d_default_finetune/.../fold_0/validation \\
    --model-pred-dir results_3d_boundary_shape_runs/20260509_131406_.../fold_0/validation \\
    --model-label "Adaptive large tumor" \\
    --output-png visualizations/delta_vs_vol_adaptive.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_multiview_delta_vs_volume import load_nifti  # noqa: E402


def _spacing_mm3_from_img(img: Any) -> float:
    if img is None:
        return 1.0
    z = img.header.get_zooms()[:3]
    return float(np.prod(z)) if len(z) >= 3 else 1.0


def _spacing_from_path(path: Path) -> float:
    import nibabel as nib

    img = nib.load(str(path))
    z = img.header.get_zooms()[:3]
    return float(np.prod(z)) if len(z) >= 3 else 1.0


def tumor_vol_mm3_from_arrays(gt: np.ndarray, voxel_vol_mm3: float, tumor_label: int = 2) -> float:
    nvox = int((gt == tumor_label).sum())
    return float(nvox * voxel_vol_mm3)


def dice_tumor(pred: np.ndarray, gt: np.ndarray, tumor_label: int = 2) -> float:
    p = pred == tumor_label
    t = gt == tumor_label
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    denom = 2.0 * tp + fp + fn
    return float((2.0 * tp) / denom) if denom > 0 else 1.0


def _case_ids_in_dir(d: Path) -> List[str]:
    return sorted(p.name[: -len(".nii.gz")] for p in d.glob("*.nii.gz"))


def collect_rows(
    baseline_dir: Path,
    model_dir: Path,
    gt_dir: Path,
    tumor_label: int,
) -> List[Dict[str, Any]]:
    bl_ids = set(_case_ids_in_dir(baseline_dir))
    m_ids = set(_case_ids_in_dir(model_dir))
    common = sorted(bl_ids & m_ids)
    rows: List[Dict[str, Any]] = []
    for cid in common:
        gt_p = gt_dir / f"{cid}.nii.gz"
        bl_p = baseline_dir / f"{cid}.nii.gz"
        m_p = model_dir / f"{cid}.nii.gz"
        if not gt_p.is_file():
            continue
        gt, _gt_img = load_nifti(gt_p)
        bl, _ = load_nifti(bl_p)
        pr, pr_img = load_nifti(m_p)
        if gt.shape != bl.shape or gt.shape != pr.shape:
            raise RuntimeError(f"{cid}: shape mismatch gt {gt.shape} baseline {bl.shape} model {pr.shape}")
        vox = _spacing_mm3_from_img(pr_img) if pr_img is not None else _spacing_from_path(m_p)
        vol = tumor_vol_mm3_from_arrays(gt, vox, tumor_label)
        d_bl = dice_tumor(bl, gt, tumor_label)
        d_m = dice_tumor(pr, gt, tumor_label)
        rows.append(
            {
                "case_id": cid,
                "gt_tumor_vol_mm3": vol,
                "dice_baseline": d_bl,
                "dice_model": d_m,
                "delta_dice": d_m - d_bl,
            }
        )
    rows.sort(key=lambda r: r["gt_tumor_vol_mm3"])
    return rows


def _plot_one_ax(ax: Any, rows: List[Dict[str, Any]], model_label: str, baseline_label: str) -> None:
    from scipy.stats import spearmanr

    if not rows:
        ax.set_visible(False)
        return
    vols = np.array([r["gt_tumor_vol_mm3"] for r in rows], dtype=np.float64)
    deltas = np.array([r["delta_dice"] for r in rows], dtype=np.float64)
    spe = spearmanr(vols, deltas)
    labels = [r["case_id"] for r in rows]
    colors = ["tab:green" if y > 0 else "tab:red" for y in deltas]
    ax.scatter(vols, deltas, c=colors, alpha=0.85, s=60)
    for x, y, lab in zip(vols, deltas, labels):
        ax.annotate(
            lab.replace("case_", ""),
            (x, y),
            fontsize=7,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("GT tumor volume (mm³), log scale")
    ax.set_ylabel(f"ΔDice = {model_label} − {baseline_label}")
    ax.set_title(f"{model_label} vs {baseline_label} (n={len(rows)})")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.99,
        0.01,
        f"Spearman ρ={spe.correlation:.2f}  p={spe.pvalue:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ΔDice vs GT volume from validation NIfTIs (no JSON metrics).")
    p.add_argument(
        "--baseline-pred-dir",
        type=str,
        required=True,
        help="fold_0/validation with baseline predictions (*.nii.gz).",
    )
    p.add_argument(
        "--baseline-label",
        type=str,
        default="Default",
        help="Short name for baseline (y-axis / titles).",
    )
    p.add_argument(
        "--gt-dir",
        type=str,
        default=str(REPO_ROOT / "nnUNet_preprocessed" / "Dataset001_LiverTumor" / "gt_segmentations"),
    )
    p.add_argument(
        "--tumor-label",
        type=int,
        default=2,
        help="Label value for tumor in multi-class masks.",
    )
    p.add_argument(
        "--model-pred-dir",
        action="append",
        dest="model_dirs",
        required=True,
        help="Model validation folder (repeat for multiple subplots).",
    )
    p.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        default=None,
        help="Label per --model-pred-dir (same order). Default: folder name.",
    )
    p.add_argument(
        "--output-png",
        type=str,
        required=True,
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    baseline_dir = (REPO_ROOT / args.baseline_pred_dir).resolve()
    gt_dir = (REPO_ROOT / args.gt_dir).resolve()
    model_dirs = [(REPO_ROOT / d).resolve() for d in args.model_dirs]
    labels = args.model_labels
    if labels is None:
        labels = [d.name if d.name != "validation" else d.parent.name for d in model_dirs]
    if len(labels) != len(model_dirs):
        raise SystemExit("Provide one --model-label per --model-pred-dir.")
    baseline_label = (args.baseline_label or "Baseline").strip()

    fig, axes = plt.subplots(1, len(model_dirs), figsize=(6 * len(model_dirs), 6), squeeze=False)
    axes_flat = axes[0]
    for ax, md, ml in zip(axes_flat, model_dirs, labels):
        rows = collect_rows(baseline_dir, md, gt_dir, args.tumor_label)
        print(f"{ml}: n={len(rows)} cases")
        _plot_one_ax(ax, rows, ml, baseline_label)

    fig.tight_layout()
    out = Path(args.output_png)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
