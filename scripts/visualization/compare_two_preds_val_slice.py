#!/usr/bin/env python3
"""One axial slice: side-by-side preds vs GT + delta map (tumor class only).

Reads CT from imagesTr (case_XXXX_0000.nii.gz), GT from labelsTr, two preds from
fold_0/validation (or any dirs). Writes a single comparison PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _extract_axial(vol: np.ndarray, z: int) -> np.ndarray:
    return vol[int(z), :, :].astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default="case_0004")
    p.add_argument("--z", type=int, required=True, help="Axial index (same as z= in slice PNGs).")
    p.add_argument(
        "--pred-dir-a",
        type=str,
        required=True,
        help="First pred folder (e.g. default finetune validation).",
    )
    p.add_argument(
        "--pred-dir-b",
        type=str,
        required=True,
        help="Second pred folder (e.g. adaptive large tumor validation).",
    )
    p.add_argument(
        "--label-a",
        type=str,
        default="3D nnUNet",
        help="Short name for first model (legend).",
    )
    p.add_argument(
        "--label-b",
        type=str,
        default="Adaptive large tumor",
        help="Short name for second model (legend).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path.",
    )
    p.add_argument(
        "--dataset-json",
        type=str,
        default=str(REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json"),
    )
    p.add_argument("--ct-window", type=float, nargs=2, default=[-100.0, 400.0])
    p.add_argument("--dpi", type=int, default=300, help="PNG resolution (default 300).")
    p.add_argument(
        "--lw-gt",
        type=float,
        default=1.15,
        help="GT contour linewidth (matplotlib units, scaled with figure).",
    )
    p.add_argument(
        "--lw-pred",
        type=float,
        default=1.2,
        help="Prediction contour linewidth.",
    )
    args = p.parse_args()

    case_id = args.case.strip()
    if not case_id.startswith("case_"):
        case_id = f"case_{case_id}"

    dj = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    tumor_label = _tumor_label(dj)
    fe = dj.get("file_ending", ".nii.gz")
    if not fe.startswith("."):
        fe = f".{fe}"

    images_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "imagesTr"
    gt_dir = REPO_ROOT / "nnUNet_raw" / "Dataset001_LiverTumor" / "labelsTr"
    ct_path = images_dir / f"{case_id}_0000{fe}"
    gt_path = gt_dir / f"{case_id}{fe}"
    pred_a = (Path(args.pred_dir_a).resolve() if Path(args.pred_dir_a).is_absolute() else REPO_ROOT / args.pred_dir_a) / f"{case_id}{fe}"
    pred_b = (Path(args.pred_dir_b).resolve() if Path(args.pred_dir_b).is_absolute() else REPO_ROOT / args.pred_dir_b) / f"{case_id}{fe}"

    for path, lbl in [(ct_path, "CT"), (gt_path, "GT"), (pred_a, "pred-a"), (pred_b, "pred-b")]:
        if not path.is_file():
            raise SystemExit(f"Missing {lbl}: {path}")

    ct, _ = _load_nifti(ct_path)
    gt, _ = _load_nifti(gt_path)
    pr_a, _ = _load_nifti(pred_a)
    pr_b, _ = _load_nifti(pred_b)

    if not (ct.shape == gt.shape == pr_a.shape == pr_b.shape):
        raise SystemExit(
            f"Shape mismatch CT {ct.shape} GT {gt.shape} A {pr_a.shape} B {pr_b.shape}"
        )

    z = int(args.z)
    if z < 0 or z >= ct.shape[0]:
        raise SystemExit(f"z={z} out of range for axis0 depth {ct.shape[0]}")

    lo, hi = float(args.ct_window[0]), float(args.ct_window[1])
    sl_ct = _extract_axial(ct, z)
    sl_gt = (_extract_axial(gt, z) == tumor_label).astype(np.float32)
    sl_a = (_extract_axial(pr_a, z) == tumor_label).astype(np.float32)
    sl_b = (_extract_axial(pr_b, z) == tumor_label).astype(np.float32)

    sl_vis = np.clip(sl_ct, lo, hi)
    sl_vis = (sl_vis - lo) / (hi - lo + 1e-8)

    gain_gt = (sl_b > 0.5) & (sl_a <= 0.5) & (sl_gt > 0.5)
    loss_gt = (sl_b <= 0.5) & (sl_a > 0.5) & (sl_gt > 0.5)
    extra_fp = (sl_b > 0.5) & (sl_a <= 0.5) & (sl_gt <= 0.5)
    removed_fp = (sl_b <= 0.5) & (sl_a > 0.5) & (sl_gt <= 0.5)
    delta_pos = gain_gt | extra_fp
    delta_neg = loss_gt | removed_fp

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
        }
    )

    lw_gt = float(args.lw_gt)
    lw_pr = float(args.lw_pred)

    # Images + caption row under each panel (model / Δ title).
    fig = plt.figure(figsize=(17.5, 6.9))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.1],
        hspace=0.14,
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.19,
        wspace=0.06,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cap_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for cax in cap_axes:
        cax.axis("off")

    gt_color = "#66FF66"
    pred_a_color = "#FF1744"
    pred_b_color = "#FF00E5"
    delta_pos_rgb = (0.15, 0.95, 1.0)
    delta_neg_rgb = (1.0, 0.38, 0.18)

    def _panel(ax, pred_mask: np.ndarray, pred_color: str, linestyle: str) -> None:
        ax.imshow(
            sl_vis.T,
            cmap="gray",
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.contour(
            sl_gt.T,
            levels=[0.5],
            colors=gt_color,
            linewidths=lw_gt,
            origin="lower",
        )
        ax.contour(
            pred_mask.T,
            levels=[0.5],
            colors=pred_color,
            linewidths=lw_pr,
            origin="lower",
            linestyles=linestyle,
        )
        ax.axis("off")

    _panel(axes[0], sl_a, pred_a_color, "--")
    _panel(axes[1], sl_b, pred_b_color, "--")

    axd = axes[2]
    axd.imshow(
        sl_vis.T,
        cmap="gray",
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axd.contour(
        sl_gt.T,
        levels=[0.5],
        colors=gt_color,
        linewidths=lw_gt * 0.95,
        origin="lower",
        linestyles="-",
    )
    # Delta overlays (B relative to A)
    h, w = sl_vis.shape
    rgba = np.zeros((w, h, 4), dtype=np.float32)

    def _stamp(mask: np.ndarray, color: Tuple[float, float, float], alpha: float) -> None:
        m = mask.T  # contour/imshow use transposed indexing
        for c in range(3):
            rgba[:, :, c] += m.astype(np.float32) * color[c] * alpha
        rgba[:, :, 3] += m.astype(np.float32) * alpha

    _stamp(delta_pos, delta_pos_rgb, 0.52)
    _stamp(delta_neg, delta_neg_rgb, 0.48)
    rgba[:, :, 3] = np.clip(rgba[:, :, 3], 0.0, 1.0)
    axd.imshow(rgba, origin="lower", aspect="auto", interpolation="nearest")

    axd.axis("off")

    la = args.label_a
    lb = args.label_b
    cap_axes[0].text(
        0.5,
        0.55,
        la,
        transform=cap_axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
    )
    cap_axes[1].text(
        0.5,
        0.55,
        lb,
        transform=cap_axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
    )
    cap_axes[2].text(
        0.5,
        0.55,
        f"Δ: {lb} относительно {la}",
        transform=cap_axes[2].transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
    )

    legend_elems = [
        Line2D(
            [0],
            [0],
            color=gt_color,
            lw=2.8,
            solid_capstyle="round",
            label="Граница GT (факт)",
        ),
        Line2D(
            [0],
            [0],
            color=pred_a_color,
            lw=2.8,
            linestyle=(0, (5, 3)),
            solid_capstyle="round",
            label="Граница предсказания A",
        ),
        Line2D(
            [0],
            [0],
            color=pred_b_color,
            lw=2.8,
            linestyle=(0, (5, 3)),
            solid_capstyle="round",
            label="Граница предсказания B",
        ),
        Patch(
            facecolor=delta_pos_rgb,
            edgecolor="none",
            label="Δ⁺: B добавляет опухоль относительно A",
        ),
        Patch(
            facecolor=delta_neg_rgb,
            edgecolor="none",
            label="Δ⁻: B убирает опухоль относительно A",
        ),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=5,
        fontsize=7.0,
        frameon=True,
        framealpha=0.95,
        borderpad=0.4,
        labelspacing=0.35,
        columnspacing=0.65,
        bbox_to_anchor=(0.5, 0.03),
        bbox_transform=fig.transFigure,
    )
    fig.suptitle(f"{case_id}  axial  z={z}", fontsize=12.5, y=0.98)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)

    n_gain = int(gain_gt.sum())
    n_loss = int(loss_gt.sum())
    n_xfp = int(extra_fp.sum())
    n_rfp = int(removed_fp.sum())
    print(
        f"Wrote {out_p}\n"
        f"  voxels on slice: gain_in_gt={n_gain} loss_in_gt={n_loss} "
        f"extra_fp={n_xfp} removed_fp={n_rfp}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
