#!/usr/bin/env python3
"""
Fuse two nnU-Net softmax exports (``--save-probabilities`` from
``infer_coarse_to_fine`` / nnU-Net) and write hard segmentations (NIfTI).

Each input directory must contain ``<case_id>.npz`` (``probabilities``,
shape C×Z×Y×X in patient space) and ``<case_id>.pkl``. Both runs need the
same preprocessing so arrays align.

Example::

    python scripts/3d/infer_fuse_softmax_blend.py \\
      --model-dir <...__3d_fullres> \\
      --softmax-dir-a <run_a/nnunet_stage1_softmax> \\
      --softmax-dir-b <run_b/nnunet_stage1_softmax> \\
      --lambda 0.5 -o fused_segmentations/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _configuration_name_from_checkpoint(model_dir: Path) -> str:
    import torch

    for fold_dir in sorted(model_dir.glob("fold_*")):
        for ck_name in ("checkpoint_best.pth", "checkpoint_final.pth"):
            ck_path = fold_dir / ck_name
            if ck_path.is_file():
                ckpt = torch.load(
                    ck_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                return str(ckpt["init_args"]["configuration"])
    raise FileNotFoundError(
        f"Could not find checkpoint_best/final under {model_dir}/fold_*/ "
        "— pass a trained nnU-Net model directory."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Blend two nnU-Net softmax npz exports and save fused "
            "segmentations."
        )
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help=(
            "Trained nnU-Net folder with dataset.json and plans.json "
            "(label handling + writer)."
        ),
    )
    p.add_argument(
        "--softmax-dir-a",
        type=Path,
        required=True,
        help="Dir with <case>.npz (probabilities) + <case>.pkl from model A.",
    )
    p.add_argument(
        "--softmax-dir-b",
        type=Path,
        required=True,
        help="Dir with <case>.npz (probabilities) + <case>.pkl from model B.",
    )
    p.add_argument(
        "--lambda",
        type=float,
        default=0.5,
        dest="lam",
        metavar="L",
        help="Weight for A: p = L*p_A + (1-L)*p_B (default 0.5).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for fused NIfTI.",
    )
    p.add_argument(
        "--cases",
        type=str,
        default="",
        help=(
            "Comma-separated case IDs; default = intersection of *.npz "
            "stems in both dirs."
        ),
    )
    p.add_argument(
        "--props-from",
        choices=("a", "b"),
        default="a",
        help="Which side's .pkl to use for write_seg (default: a).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output segmentation already exists.",
    )
    return p.parse_args()


def _case_ids_from_dir(d: Path) -> set[str]:
    return {p.stem for p in d.glob("*.npz")}


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.lam <= 1.0):
        raise SystemExit("--lambda must be in [0, 1]")
    model_dir = args.model_dir.resolve()
    dir_a = args.softmax_dir_a.resolve()
    dir_b = args.softmax_dir_b.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from batchgenerators.utilities.file_and_folder_operations import (
        load_json,
        load_pickle,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset_json = load_json(str(model_dir / "dataset.json"))
    plans = load_json(str(model_dir / "plans.json"))
    plans_manager = PlansManager(plans)
    _ = _configuration_name_from_checkpoint(model_dir)
    label_manager = plans_manager.get_label_manager(dataset_json)
    file_ending = dataset_json["file_ending"]
    rw = plans_manager.image_reader_writer_class()

    if args.cases.strip():
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(_case_ids_from_dir(dir_a) & _case_ids_from_dir(dir_b))
    if not cases:
        raise SystemExit("No cases to fuse (check dirs and --cases).")

    props_dir = dir_a if args.props_from == "a" else dir_b
    lam = float(args.lam)
    oml = 1.0 - lam

    for case_id in cases:
        out_seg = out_dir / f"{case_id}{file_ending}"
        if args.skip_existing and out_seg.is_file():
            print(f"skip {case_id} (exists)")
            continue
        npz_a = dir_a / f"{case_id}.npz"
        npz_b = dir_b / f"{case_id}.npz"
        pkl_path = props_dir / f"{case_id}.pkl"
        for p in (npz_a, npz_b, pkl_path):
            if not p.is_file():
                raise FileNotFoundError(p)
        z_a = np.load(str(npz_a))
        z_b = np.load(str(npz_b))
        pa = z_a["probabilities"].astype(np.float32, copy=False)
        pb = z_b["probabilities"].astype(np.float32, copy=False)
        if pa.shape != pb.shape:
            msg = f"{case_id}: shape mismatch A {pa.shape} vs B {pb.shape}"
            raise ValueError(msg)
        props = load_pickle(str(pkl_path))
        fused = lam * pa + oml * pb
        seg = label_manager.convert_probabilities_to_segmentation(fused)
        if isinstance(seg, np.ndarray):
            seg_np = seg
        else:
            seg_np = seg.cpu().numpy()
        if seg_np.dtype not in (np.uint8, np.uint16):
            seg_np = seg_np.astype(
                np.uint16 if seg_np.max() > 255 else np.uint8
            )
        out_path = str(out_dir / f"{case_id}{file_ending}")
        rw.write_seg(seg_np, out_path, props)
        print(f"saved {case_id}")


if __name__ == "__main__":
    main()
