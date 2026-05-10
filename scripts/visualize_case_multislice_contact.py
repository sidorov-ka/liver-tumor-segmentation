#!/usr/bin/env python3
"""Multi 2D slices + contact sheet: CT + GT tumor (green) vs pred tumor (red dashed).

Planes (numpy volume shape (d0, d1, d2), e.g. depth × H × W):
  axial / horizontal — slices ``vol[i, :, :]`` (axis 0 normal); горизонтальные / трансверзальные.
  coronal — ``vol[:, j, :]`` (axis 1 normal).
  sagittal — ``vol[:, :, k]`` (axis 2 normal).

With ``--plane-subdirs hv`` (default): writes under ``<output-dir>/horizontal/`` for axial
and ``<output-dir>/vertical/`` for coronal + sagittal. Use ``--plane-subdirs off`` for a flat folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

Plane = Literal["axial", "horizontal", "coronal", "sagittal"]


def _plane_to_mode(plane: str) -> str:
    p = plane.strip().lower()
    if p in ("horizontal", "axial", "transverse"):
        return "axial"
    if p == "coronal":
        return "coronal"
    if p == "sagittal":
        return "sagittal"
    raise SystemExit(f"Unknown --plane {plane!r} (use axial|horizontal|coronal|sagittal)")


def _load_nifti(path: Path) -> Tuple[np.ndarray, Any]:
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


def _sum_axes_for_plane(gt_t: np.ndarray, mode: str) -> np.ndarray:
    if mode == "axial":
        return gt_t.sum(axis=(1, 2))
    if mode == "coronal":
        return gt_t.sum(axis=(0, 2))
    return gt_t.sum(axis=(0, 1))


def _extract_2d(vol: np.ndarray, mode: str, idx: int) -> np.ndarray:
    if mode == "axial":
        return vol[int(idx), :, :]
    if mode == "coronal":
        return vol[:, int(idx), :]
    return vol[:, :, int(idx)]


def _indices_along_tumor(gt_t: np.ndarray, mode: str, n_slices: int) -> List[int]:
    sums = _sum_axes_for_plane(gt_t, mode)
    nz = np.where(sums > 0)[0]
    if len(nz) == 0:
        raise RuntimeError("No tumor voxels in GT for this case")
    lo, hi = int(nz.min()), int(nz.max())
    if hi <= lo:
        return [lo] * n_slices
    raw = [int(np.clip(round(x), lo, hi)) for x in np.linspace(lo, hi, n_slices)]
    out: List[int] = []
    for v in raw:
        if v not in out:
            out.append(v)
    i = lo
    while len(out) < n_slices and i <= hi:
        if i not in out:
            out.append(i)
        i += 1
    return sorted(out[:n_slices])


def _axis_label(mode: str) -> str:
    return {"axial": "z", "coronal": "j", "sagittal": "k"}[mode]


def _plane_title_suffix(mode: str) -> str:
    return {"axial": "axial (horizontal)", "coronal": "coronal", "sagittal": "sagittal"}[
        mode
    ]


def _output_bucket_for_plane(mode: str) -> str:
    """Subfolder name: horizontal = axial; vertical = coronal + sagittal."""
    return "horizontal" if mode == "axial" else "vertical"


def _save_slice_png(
    idx: int,
    sl_ct: np.ndarray,
    sl_gt: np.ndarray,
    sl_pr: np.ndarray,
    ct_window: Tuple[float, float],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = float(ct_window[0]), float(ct_window[1])
    sl_vis = np.clip(sl_ct.astype(np.float32), lo, hi)
    sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(sl_vis.T, cmap="gray", origin="lower", aspect="auto")
    ax.contour(sl_gt.T, levels=[0.5], colors="lime", linewidths=2.0, origin="lower")
    ax.contour(
        sl_pr.T,
        levels=[0.5],
        colors="red",
        linewidths=2.0,
        origin="lower",
        linestyles="--",
    )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_contact_sheet(
    case_id: str,
    mode: str,
    indices: List[int],
    ct: np.ndarray,
    gt_t: np.ndarray,
    pr_t: np.ndarray,
    ct_window: Tuple[float, float],
    out_path: Path,
    pred_tag: str,
    layout: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(indices)
    if layout == "row":
        nrows, ncols = 1, n
    else:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    lo, hi = float(ct_window[0]), float(ct_window[1])
    ax_lbl = _axis_label(mode)
    for i, idx in enumerate(indices):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        sl_ct = _extract_2d(ct, mode, idx).astype(np.float32)
        sl_gt = _extract_2d(gt_t, mode, idx)
        sl_pr = _extract_2d(pr_t, mode, idx)
        sl_vis = np.clip(sl_ct, lo, hi)
        sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)
        ax.imshow(sl_vis.T, cmap="gray", origin="lower", aspect="auto")
        ax.contour(sl_gt.T, levels=[0.5], colors="lime", linewidths=1.5, origin="lower")
        ax.contour(
            sl_pr.T,
            levels=[0.5],
            colors="red",
            linewidths=1.5,
            origin="lower",
            linestyles="--",
        )
        ax.set_title(f"{ax_lbl}={idx}")
        ax.axis("off")
    for j in range(len(indices), nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")
    plane_human = _plane_title_suffix(mode)
    fig.suptitle(
        f"{case_id}  {plane_human}  green=GT tumor  red={pred_tag} (pred)",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default="case_0004")
    p.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Folder with <case>.nii.gz (e.g. fold_0/validation).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/nnunet3d150_val_case_0004",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Short tag for filenames (default: pred-dir folder name).",
    )
    p.add_argument(
        "--plane",
        type=str,
        default="axial",
        help="axial|horizontal (same), coronal, sagittal",
    )
    p.add_argument(
        "--contact-layout",
        choices=("grid", "row"),
        default="grid",
        help="grid=2×3 style; row=single horizontal strip (1×N)",
    )
    p.add_argument("--n-slices", type=int, default=6)
    p.add_argument(
        "--dataset-json",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json"),
    )
    p.add_argument("--ct-window", type=float, nargs=2, default=[-100.0, 400.0])
    p.add_argument(
        "--plane-subdirs",
        choices=("off", "hv"),
        default="hv",
        help="hv: <output-dir>/horizontal (axial) and /vertical (coronal+sagittal); off: flat",
    )
    args = p.parse_args()

    mode = _plane_to_mode(args.plane)

    case_id = args.case.strip()
    if not case_id.startswith("case_"):
        case_id = f"case_{case_id}"

    dj = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    tumor_label = _tumor_label(dj)
    fe = dj.get("file_ending", ".nii.gz")
    if not fe.startswith("."):
        fe = f".{fe}"

    pred_dir = Path(args.pred_dir).resolve()
    if not pred_dir.is_absolute() and not (REPO_ROOT / args.pred_dir).is_absolute():
        pred_dir = (REPO_ROOT / args.pred_dir).resolve()
    base_out = (REPO_ROOT / args.output_dir).resolve()
    if args.plane_subdirs == "hv":
        out_dir = base_out / _output_bucket_for_plane(mode)
    else:
        out_dir = base_out
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.label or pred_dir.name.replace(" ", "_")
    if len(tag) > 48:
        tag = tag[:45] + "..."

    images_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "imagesTr"
    gt_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "labelsTr"
    ct_path = images_dir / f"{case_id}_0000{fe}"
    gt_path = gt_dir / f"{case_id}{fe}"
    pred_path = pred_dir / f"{case_id}{fe}"

    for path, lbl in [(ct_path, "CT"), (gt_path, "GT"), (pred_path, "pred")]:
        if not path.is_file():
            raise SystemExit(f"Missing {lbl}: {path}")

    ct, _ = _load_nifti(ct_path)
    gt, _ = _load_nifti(gt_path)
    pr, _ = _load_nifti(pred_path)

    if ct.shape != gt.shape or ct.shape != pr.shape:
        raise SystemExit(f"Shape mismatch CT {ct.shape} GT {gt.shape} pred {pr.shape}")

    gt_t = (gt == tumor_label).astype(np.float32)
    pr_t = (pr == tumor_label).astype(np.float32)
    indices = _indices_along_tumor(gt_t, mode, max(1, args.n_slices))

    ct_w = (float(args.ct_window[0]), float(args.ct_window[1]))
    ax_lbl = _axis_label(mode)
    plane_slug = mode if mode != "axial" else "axial_horizontal"
    for idx in indices:
        sl_ct = _extract_2d(ct, mode, idx)
        sl_gt = _extract_2d(gt_t, mode, idx)
        sl_pr = _extract_2d(pr_t, mode, idx)
        title = f"{case_id}  {_plane_title_suffix(mode)}  {ax_lbl}={idx}  green=GT  red={tag}"
        if mode == "axial":
            out_png = out_dir / f"{case_id}_{ax_lbl}{idx}_{tag}.png"
        else:
            out_png = out_dir / f"{case_id}_{plane_slug}_{ax_lbl}{idx}_{tag}.png"
        _save_slice_png(idx, sl_ct, sl_gt, sl_pr, ct_w, out_png, title)
        print(f"Wrote {out_png}", file=sys.stderr)

    layout_suffix = f"_{args.contact_layout}"
    if mode == "axial" and args.contact_layout == "grid":
        contact = out_dir / f"{case_id}_contact_sheet_{tag}.png"
    elif mode == "axial":
        contact = out_dir / f"{case_id}_contact_sheet_axial{layout_suffix}_{tag}.png"
    else:
        contact = out_dir / f"{case_id}_contact_sheet_{plane_slug}{layout_suffix}_{tag}.png"
    _save_contact_sheet(
        case_id,
        mode,
        indices,
        ct,
        gt_t,
        pr_t,
        ct_w,
        contact,
        tag,
        args.contact_layout,
    )
    print(f"Wrote {contact}", file=sys.stderr)


if __name__ == "__main__":
    main()
