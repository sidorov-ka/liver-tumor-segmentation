#!/usr/bin/env python3
"""One row: axial / coronal / sagittal — CT + GT (green) + pred tumor (red dashed).

Same style as visualize_case_multislice_contact single slices, with foreground crop,
aspect=equal, and higher DPI for presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    try:
        return np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:
        from nibabel import Nifti1Image

        return np.asanyarray(Nifti1Image.from_bytes(path.read_bytes()).dataobj)


def _extract_2d(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    if plane == "axial":
        return vol[int(idx), :, :].astype(np.float32)
    if plane == "coronal":
        return vol[:, int(idx), :].astype(np.float32)
    return vol[:, :, int(idx)].astype(np.float32)


def _crop_bbox(
    label_slice: np.ndarray,
    liver_label: int,
    tumor_label: int,
    margin: int,
) -> tuple[int, int, int, int]:
    fg = (label_slice == liver_label) | (label_slice == tumor_label)
    if not np.any(fg):
        h, w = label_slice.shape
        return 0, h, 0, w
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    m = max(0, int(margin))
    return (
        max(0, int(rows[0]) - m),
        min(label_slice.shape[0], int(rows[-1]) + m + 1),
        max(0, int(cols[0]) - m),
        min(label_slice.shape[1], int(cols[-1]) + m + 1),
    )


def _crop(sl: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    return sl[r0:r1, c0:c1]


def _parse_planes(spec: str) -> List[Tuple[str, int]]:
    """e.g. axial:132,coronal:146,sagittal:347"""
    out: List[Tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, idx_s = part.split(":", 1)
        plane = name.strip().lower()
        if plane in ("horizontal", "transverse"):
            plane = "axial"
        if plane not in ("axial", "coronal", "sagittal"):
            raise SystemExit(f"Unknown plane {name!r}")
        out.append((plane, int(idx_s.strip())))
    if not out:
        raise SystemExit("Empty --planes")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default="case_0004")
    p.add_argument(
        "--planes",
        type=str,
        default="axial:132,coronal:146,sagittal:347",
        help="Comma-separated plane:index (axial|coronal|sagittal).",
    )
    p.add_argument("--pred-dir", type=str, required=True)
    p.add_argument("--label", type=str, default="adaptive_large")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--crop-margin", type=int, default=48)
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--panel-height", type=float, default=5.0)
    p.add_argument("--ct-window", type=float, nargs=2, default=[-100.0, 400.0])
    p.add_argument(
        "--dataset-json",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json"),
    )
    args = p.parse_args()

    case_id = args.case.strip()
    if not case_id.startswith("case_"):
        case_id = f"case_{case_id}"

    dj = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    labels = dj.get("labels", {})
    tumor_label = int(labels.get("tumor", 2))
    liver_label = int(labels.get("liver", 1))
    fe = dj.get("file_ending", ".nii.gz")
    if not fe.startswith("."):
        fe = f".{fe}"

    pred_dir = Path(args.pred_dir)
    pred_dir = pred_dir if pred_dir.is_absolute() else REPO_ROOT / pred_dir
    images_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "imagesTr"
    gt_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "labelsTr"

    ct = _load_nifti(images_dir / f"{case_id}_0000{fe}")
    gt = _load_nifti(gt_dir / f"{case_id}{fe}")
    pr = _load_nifti(pred_dir / f"{case_id}{fe}")

    plane_specs = _parse_planes(args.planes)
    lo, hi = float(args.ct_window[0]), float(args.ct_window[1])
    tag = args.label.replace(" ", "_")

    panels: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for plane, idx in plane_specs:
        sl_lab = _extract_2d(gt, plane, idx)
        sl_ct = _extract_2d(ct, plane, idx)
        sl_gt = (sl_lab == tumor_label).astype(np.float32)
        sl_pr = (_extract_2d(pr, plane, idx) == tumor_label).astype(np.float32)
        if int(args.crop_margin) > 0:
            bb = _crop_bbox(sl_lab.astype(np.int16), liver_label, tumor_label, args.crop_margin)
            sl_ct = _crop(sl_ct, bb)
            sl_gt = _crop(sl_gt, bb)
            sl_pr = _crop(sl_pr, bb)
        sl_vis = np.clip(sl_ct, lo, hi)
        sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)
        axis_lbl = {"axial": "z", "coronal": "j", "sagittal": "k"}[plane]
        plane_title = {
            "axial": "axial (horizontal)",
            "coronal": "coronal",
            "sagittal": "sagittal",
        }[plane]
        panels.append((plane_title, idx, sl_vis, sl_gt, sl_pr))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    aspects = [p[2].shape[1] / max(p[2].shape[0], 1) for p in panels]
    panel_ws = [float(args.panel_height) * a for a in aspects]
    fig_w = sum(panel_ws) + 0.8
    fig_h = float(args.panel_height) + 1.1

    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    for ax, (plane_title, idx, sl_vis, sl_gt, sl_pr) in zip(axes, panels):
        ax.imshow(sl_vis.T, cmap="gray", origin="lower", aspect="equal", interpolation="nearest")
        ax.contour(sl_gt.T, levels=[0.5], colors="lime", linewidths=2.0, origin="lower")
        ax.contour(
            sl_pr.T,
            levels=[0.5],
            colors="red",
            linewidths=2.0,
            origin="lower",
            linestyles="--",
        )
        axis_lbl = "z" if "axial" in plane_title else ("j" if plane_title == "coronal" else "k")
        ax.set_title(
            f"{case_id}  {plane_title}  {axis_lbl}={idx}  green=GT  red={tag}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(f"{case_id}  green=GT  red={tag}", fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_p = Path(args.output)
    if not out_p.is_absolute():
        out_p = REPO_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_p}", file=sys.stderr)


if __name__ == "__main__":
    main()
