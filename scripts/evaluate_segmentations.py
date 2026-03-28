#!/usr/bin/env python3
"""
Compare nnU-Net-style prediction folders to labelsTr (tumor Dice / IoU).

Default JSON report: inference_comparison/coarse_to_fine/val_metrics_fold<FOLD>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from refinement.utils import numpy_dice  # noqa: E402


def _iou(tp: float, fp: float, fn: float) -> float:
    d = tp + fp + fn
    return float(tp / d) if d > 0 else 1.0


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    from nibabel import Nifti1Image
    from nibabel.filebasedimages import ImageFileError

    try:
        return np.asarray(nib.load(str(path)).dataobj)
    except ImageFileError:
        # Some datasets use .nii.gz names for uncompressed NIfTI (not gzip).
        return np.asarray(Nifti1Image.from_bytes(path.read_bytes()).dataobj)


def _tumor_label(dataset_json: dict) -> int:
    for name, idx in dataset_json.get("labels", {}).items():
        if str(name).lower() == "tumor":
            return int(idx)
    raise KeyError("No 'tumor' in dataset.json labels")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Val metrics: tumor Dice/IoU vs labelsTr.")
    p.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Ground truth (e.g. nnUNet_raw/Dataset001_LiverTumor/labelsTr).",
    )
    p.add_argument("--dataset-json", type=str, default=None)
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--splits-json", type=str, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--pred",
        action="append",
        dest="preds",
        metavar="NAME=DIR",
        required=True,
        help="Repeat for each run: baseline=/path nnunet_only=/path coarse_to_fine=/path",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Default: inference_comparison/coarse_to_fine/val_metrics_fold<FOLD>.json (under repo).",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON (print table only).",
    )
    return p.parse_args()


def _parse_pred_kv(pairs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"Expected NAME=DIR, got: {s}")
        n, d = s.split("=", 1)
        out.append((n.strip(), Path(d.strip())))
    return out


def main() -> None:
    args = _parse_args()
    repo = REPO_ROOT
    labels_dir = Path(args.labels_dir)
    if not labels_dir.is_absolute():
        labels_dir = repo / labels_dir

    dj_path = Path(args.dataset_json or repo / "nnUNet_raw" / "Dataset001_LiverTumor" / "dataset.json")
    if not dj_path.is_absolute():
        dj_path = repo / dj_path
    dj = json.loads(dj_path.read_text(encoding="utf-8"))
    fe = dj.get("file_ending", ".nii.gz")
    tlab = _tumor_label(dj)

    if args.cases:
        case_ids = list(args.cases)
    else:
        sp = Path(args.splits_json or repo / "nnUNet_preprocessed" / "Dataset001_LiverTumor" / "splits_final.json")
        if not sp.is_absolute():
            sp = repo / sp
        splits = json.loads(sp.read_text(encoding="utf-8"))
        case_ids = list(splits[args.fold]["val"])

    named = _parse_pred_kv(args.preds)
    rows: list[dict] = []
    summary: dict[str, dict[str, float]] = {}

    for name, pred_root in named:
        if not pred_root.is_absolute():
            pred_root = repo / pred_root
        if not pred_root.is_dir():
            raise FileNotFoundError(f"Not a directory: {pred_root}")
        dices: list[float] = []
        ious: list[float] = []
        for cid in case_ids:
            gt_p = labels_dir / f"{cid}{fe}"
            pr_p = pred_root / f"{cid}{fe}"
            if not gt_p.is_file():
                print(f"WARNING: missing GT {gt_p}", flush=True)
                continue
            if not pr_p.is_file():
                print(f"WARNING: missing pred {pr_p} ({name})", flush=True)
                continue
            gt, pr = _load_nifti(gt_p), _load_nifti(pr_p)
            if gt.shape != pr.shape:
                print(
                    f"WARNING: shape mismatch {cid} gt{gt.shape} pred{pr.shape} — skip",
                    flush=True,
                )
                continue
            g = (gt == tlab).astype(np.float64).ravel()
            pv = (pr == tlab).astype(np.float64).ravel()
            tp = float(np.dot(pv, g))
            fp = float(np.dot(pv, 1 - g))
            fn = float(np.dot(1 - pv, g))
            di = numpy_dice(pv, g)
            io = _iou(tp, fp, fn)
            dices.append(di)
            ious.append(io)
            rows.append({"name": name, "case_id": cid, "dice_tumor": di, "iou_tumor": io})
        summary[name] = {
            "n_eval": float(len(dices)),
            "mean_dice_tumor": float(np.mean(dices)) if dices else float("nan"),
            "mean_iou_tumor": float(np.mean(ious)) if ious else float("nan"),
        }

    print("\n=== Mean tumor (val cases) ===")
    for name, m in summary.items():
        print(
            f"  {name}: n={int(m['n_eval'])}  "
            f"Dice={m['mean_dice_tumor']:.4f}  IoU={m['mean_iou_tumor']:.4f}"
        )
    if len(summary) == 2:
        k = list(summary.keys())
        d0, d1 = summary[k[0]]["mean_dice_tumor"], summary[k[1]]["mean_dice_tumor"]
        if not (np.isnan(d0) or np.isnan(d1)):
            print(f"\nΔ mean Dice ({k[1]} − {k[0]}): {d1 - d0:+.4f}")

    if args.no_json:
        return

    json_path = args.output_json
    if json_path is None:
        json_path = repo / "inference_comparison" / "coarse_to_fine" / f"val_metrics_fold{args.fold}.json"
    else:
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = repo / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels_dir": str(labels_dir.resolve()),
        "dataset_json": str(dj_path.resolve()),
        "tumor_label_value": tlab,
        "fold": args.fold,
        "val_cases": case_ids,
        "summary": summary,
        "per_case": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path.resolve()}")


if __name__ == "__main__":
    main()
