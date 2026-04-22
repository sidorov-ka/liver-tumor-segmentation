#!/usr/bin/env python3
"""
Scatter: GT tumor volume (mm³) vs ΔDice (model − baseline), plus printed statistics.

  python scripts/plot_multiview_delta_vs_volume.py
  python scripts/plot_multiview_delta_vs_volume.py --output-json visualizations/multiview_delta_stats.json

  Boundary-aware vs baseline:
  python scripts/plot_multiview_delta_vs_volume.py \\
    --model-metrics inference_comparison/boundary_aware_coarse_to_fine/val_metrics_fold0.json \\
    --model-label "Boundary-aware" \\
    --output-png visualizations/boundary_delta_vs_gt_volume.png \\
    --output-json visualizations/boundary_delta_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_nifti(path: Path) -> Tuple[np.ndarray, Any]:
    import nibabel as nib

    try:
        img = nib.load(str(path))
        return np.asanyarray(img.dataobj), img
    except Exception:
        from nibabel import Nifti1Image

        raw = path.read_bytes()
        return np.asanyarray(Nifti1Image.from_bytes(raw).dataobj), None


def tumor_vol_mm3(path: Path, tumor_label: int = 2) -> float:
    data, img = load_nifti(path)
    nvox = int((data == tumor_label).sum())
    if img is not None:
        z = img.header.get_zooms()[:3]
        vox = float(np.prod(z)) if len(z) >= 3 else 1.0
    else:
        vox = 1.0
    return float(nvox * vox)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ΔDice vs GT tumor volume; print stats.")
    p.add_argument(
        "--model-metrics",
        "--multiview-metrics",
        dest="model_metrics",
        type=str,
        default="inference_comparison/multiview_infer_run_2026_04_03/metrics.json",
        help="JSON with metric_per_case (same schema as val_metrics_fold0.json).",
    )
    p.add_argument(
        "--model-label",
        type=str,
        default="Multiview",
        help="Name for plot title and y-axis (e.g. Multiview, Boundary-aware).",
    )
    p.add_argument(
        "--baseline-metrics",
        type=str,
        default="inference_comparison/baseline/val_metrics_fold0.json",
    )
    p.add_argument(
        "--gt-dir",
        type=str,
        default="nnUNet_raw/Dataset001_LiverTumor/labelsTr",
    )
    p.add_argument(
        "--output-png",
        type=str,
        default="visualizations/multiview_delta_vs_gt_volume.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write stats dict as JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = (REPO_ROOT / args.model_metrics).resolve()
    bl_path = (REPO_ROOT / args.baseline_metrics).resolve()
    gt_dir = (REPO_ROOT / args.gt_dir).resolve()
    model_label = (args.model_label or "Model").strip()

    m_data = json.loads(model_path.read_text(encoding="utf-8"))
    bl = json.loads(bl_path.read_text(encoding="utf-8"))
    m_d = {x["case_id"]: x["metrics"]["2"]["Dice"] for x in m_data["metric_per_case"]}
    bl_d = {x["case_id"]: x["metrics"]["2"]["Dice"] for x in bl["metric_per_case"]}
    common = sorted(set(m_d) & set(bl_d))

    rows: List[Dict[str, Any]] = []
    for cid in common:
        p = gt_dir / f"{cid}.nii.gz"
        if not p.is_file():
            continue
        vol = tumor_vol_mm3(p)
        d_bl = float(bl_d[cid])
        d_m = float(m_d[cid])
        rows.append(
            {
                "case_id": cid,
                "gt_tumor_vol_mm3": vol,
                "dice_baseline": d_bl,
                "dice_model": d_m,
                "delta_dice": d_m - d_bl,
            }
        )

    if not rows:
        raise SystemExit("No cases with GT labels — check paths.")

    rows.sort(key=lambda r: r["gt_tumor_vol_mm3"])
    vols = np.array([r["gt_tumor_vol_mm3"] for r in rows], dtype=np.float64)
    deltas = np.array([r["delta_dice"] for r in rows], dtype=np.float64)

    from scipy.stats import spearmanr

    spe = spearmanr(vols, deltas)

    # Quartiles on volume
    q = np.quantile(vols, [0, 0.25, 0.5, 0.75, 1.0])
    quartile_stats: List[Dict[str, Any]] = []
    labels = ("Q1_smallest", "Q2", "Q3", "Q4_largest")
    for i, name in enumerate(labels):
        lo, hi = q[i], q[i + 1] if i < 3 else q[i + 1] + 1e-6
        sel = [r for r in rows if lo <= r["gt_tumor_vol_mm3"] < hi]
        ds = [r["delta_dice"] for r in sel]
        quartile_stats.append(
            {
                "name": name,
                "vol_mm3_range": [float(lo), float(hi)],
                "n": len(sel),
                "mean_delta_dice": float(np.mean(ds)) if ds else None,
                "median_delta_dice": float(np.median(ds)) if ds else None,
                "n_better": sum(1 for d in ds if d > 0),
                "n_worse": sum(1 for d in ds if d < 0),
            }
        )

    med = float(np.median(vols))
    small = [r for r in rows if r["gt_tumor_vol_mm3"] < med]
    large = [r for r in rows if r["gt_tumor_vol_mm3"] >= med]
    ds_s = [r["delta_dice"] for r in small]
    ds_l = [r["delta_dice"] for r in large]

    median_split = {
        "median_gt_vol_mm3": med,
        "smaller_half": {
            "n": len(small),
            "mean_delta_dice": float(np.mean(ds_s)),
            "median_delta_dice": float(np.median(ds_s)),
            "n_better": sum(1 for d in ds_s if d > 0),
            "n_worse": sum(1 for d in ds_s if d < 0),
        },
        "larger_half": {
            "n": len(large),
            "mean_delta_dice": float(np.mean(ds_l)),
            "median_delta_dice": float(np.median(ds_l)),
            "n_better": sum(1 for d in ds_l if d > 0),
            "n_worse": sum(1 for d in ds_l if d < 0),
        },
    }

    stats: Dict[str, Any] = {
        "n_cases": len(rows),
        "model_label": model_label,
        "model_metrics": str(model_path),
        "baseline_metrics": str(bl_path),
        "gt_dir": str(gt_dir),
        "delta_dice": {
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        },
        "gt_tumor_vol_mm3": {
            "min": float(np.min(vols)),
            "max": float(np.max(vols)),
            "quartiles": [float(x) for x in q],
        },
        "spearman_rho_volume_vs_delta": float(spe.correlation),
        "spearman_p_value": float(spe.pvalue),
        "quartiles_by_volume": quartile_stats,
        "median_volume_split": median_split,
        "per_case": rows,
    }

    # --- print ---
    print("=" * 72)
    print(f"{model_label} vs baseline — ΔDice vs GT tumor volume")
    print("=" * 72)
    print(f"Cases: {stats['n_cases']}  (intersection model + baseline + GT file)")
    print(f"ΔDice ({model_label} − baseline): mean={stats['delta_dice']['mean']:+.4f}  "
          f"median={stats['delta_dice']['median']:+.4f}  "
          f"std={stats['delta_dice']['std']:.4f}  "
          f"min={stats['delta_dice']['min']:+.4f}  max={stats['delta_dice']['max']:+.4f}")
    print(f"GT vol (mm³): min={stats['gt_tumor_vol_mm3']['min']:.0f}  "
          f"max={stats['gt_tumor_vol_mm3']['max']:.0f}")
    print(f"  quartiles: {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f} / {q[4]:.0f}")
    print()
    print(f"Spearman ρ(volume, ΔDice) = {spe.correlation:.3f}  p = {spe.pvalue:.4f}")
    print()
    print("By volume quartile (GT):")
    for qs in quartile_stats:
        print(
            f"  {qs['name']}: n={qs['n']}  "
            f"mean Δ={qs['mean_delta_dice']:+.4f}  median Δ={qs['median_delta_dice']:+.4f}  "
            f"better={qs['n_better']}  worse={qs['n_worse']}"
        )
    print()
    print(f"Median volume split (median = {med:.0f} mm³):")
    print(
        f"  smaller: n={median_split['smaller_half']['n']}  "
        f"mean Δ={median_split['smaller_half']['mean_delta_dice']:+.4f}  "
        f"better/worse={median_split['smaller_half']['n_better']}/{median_split['smaller_half']['n_worse']}"
    )
    print(
        f"  larger:  n={median_split['larger_half']['n']}  "
        f"mean Δ={median_split['larger_half']['mean_delta_dice']:+.4f}  "
        f"better/worse={median_split['larger_half']['n_better']}/{median_split['larger_half']['n_worse']}"
    )
    print("=" * 72)

    if args.output_json:
        out_j = Path(args.output_json)
        if not out_j.is_absolute():
            out_j = REPO_ROOT / out_j
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats JSON: {out_j}")

    # --- plot ---
    xs = vols
    ys = deltas
    labels = [r["case_id"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:green" if y > 0 else "tab:red" for y in ys]
    ax.scatter(xs, ys, c=colors, alpha=0.85, s=60)
    for x, y, lab in zip(xs, ys, labels):
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
    ax.set_ylabel(f"ΔDice = {model_label} − baseline")
    ax.set_title(f"{model_label} vs baseline by tumor size (n=%d)" % len(rows))
    ax.grid(True, alpha=0.3)
    fig.text(
        0.99,
        0.01,
        f"Spearman ρ={spe.correlation:.2f}  p={spe.pvalue:.3f}",
        ha="right",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout()
    out_png = Path(args.output_png)
    if not out_png.is_absolute():
        out_png = REPO_ROOT / out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
