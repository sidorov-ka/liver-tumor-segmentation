"""Segmentation metrics for binary tumor (nnU-Net–compatible fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class BinaryConfusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def add_tensor(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: (N, 1, H, W) binary 0/1."""
        p = (pred > 0.5).float()
        t = (target > 0.5).float()
        self.tp += float((p * t).sum().item())
        self.fp += float((p * (1 - t)).sum().item())
        self.fn += float(((1 - p) * t).sum().item())
        self.tn += float(((1 - p) * (1 - t)).sum().item())

    def add_other(self, other: "BinaryConfusion") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    def to_metrics_dict(self, label_key: str = "2") -> Dict[str, Any]:
        """Match nnU-Net summary keys for one foreground class (tumor = label 2 in LiTS)."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        denom_dice = 2 * tp + fp + fn
        dice = float((2 * tp) / denom_dice) if denom_dice > 0 else 1.0
        denom_iou = tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else 1.0
        n_pred = tp + fp
        n_ref = tp + fn
        return {
            label_key: {
                "Dice": dice,
                "IoU": iou,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "TP": tp,
                "n_pred": n_pred,
                "n_ref": n_ref,
            }
        }


def parse_case_id_from_npz(path_str: str) -> str:
    """Expects stem like 'case_0000_0042' -> case_id 'case_0000', slice 0042."""
    from pathlib import Path

    stem = Path(path_str).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 4 and parts[1].isdigit():
        return parts[0]
    return stem


def merge_per_case_metrics(
    case_confusions: Dict[str, BinaryConfusion], label_key: str = "2"
) -> List[Dict[str, Any]]:
    """Build nnU-Net–like metric_per_case entries (label key \"2\" = tumor)."""
    out: List[Dict[str, Any]] = []
    for case_id in sorted(case_confusions.keys()):
        c = case_confusions[case_id]
        m = c.to_metrics_dict(label_key=label_key)
        out.append(
            {
                "case_id": case_id,
                "metrics": m,
                "note": "Aggregated over all val .npz slices for this case (2D crops).",
            }
        )
    return out
