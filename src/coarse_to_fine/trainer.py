"""Train / validate coarse-to-fine model; logs under coarse_to_fine_results/.../coarse_to_fine/run_*/."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coarse_to_fine.metrics import BinaryConfusion, merge_per_case_metrics
from coarse_to_fine.model import TinyUNet2d
from coarse_to_fine.utils import bce_dice_loss, dice_coefficient, save_checkpoint

LABEL_TUMOR = "2"  # LiTS tumor label id (matches nnU-Net summary.json)


def coarse_to_fine_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "case_id": [b["case_id"] for b in batch],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_weight: float = 0.5,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce_dice_loss(logits, y, bce_weight=bce_weight)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            d = dice_coefficient(prob, y).mean().item()
        bs = x.size(0)
        loss_sum += loss.item() * bs
        dice_sum += d * bs
        n += bs
    return {"loss": loss_sum / max(n, 1), "dice": dice_sum / max(n, 1)}


@torch.no_grad()
def validate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce_weight: float = 0.5,
) -> Dict[str, Any]:
    """Returns loss, dice (batch-mean), global confusion, per-case confusion."""
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    global_conf = BinaryConfusion()
    per_case: Dict[str, BinaryConfusion] = {}

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        case_ids: List[str] = batch["case_id"]
        logits = model(x)
        loss = bce_dice_loss(logits, y, bce_weight=bce_weight)
        prob = torch.sigmoid(logits)
        d = dice_coefficient(prob, y).mean().item()
        bs = x.size(0)
        loss_sum += loss.item() * bs
        dice_sum += d * bs
        n += bs

        pred_bin = prob
        target_bin = y
        for i in range(bs):
            global_conf.add_tensor(pred_bin[i : i + 1], target_bin[i : i + 1])
            cid = case_ids[i]
            if cid not in per_case:
                per_case[cid] = BinaryConfusion()
            per_case[cid].add_tensor(pred_bin[i : i + 1], target_bin[i : i + 1])

    m = global_conf.to_metrics_dict(label_key=LABEL_TUMOR)[LABEL_TUMOR]
    return {
        "loss": loss_sum / max(n, 1),
        "dice": dice_sum / max(n, 1),
        "dice_from_confusion": m["Dice"],
        "iou_from_confusion": m["IoU"],
        "global_confusion": global_conf,
        "per_case_confusion": per_case,
    }


def run_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
    in_channels: int = 2,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    bce_weight: float = 0.5,
    training_args: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet2d(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_dice = -1.0
    best_path = out_dir / "checkpoint_best.pth"

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = out_dir / f"training_log_{ts}.txt"
    log_lines: List[str] = []

    def log(msg: str, also_print: bool = True) -> None:
        line = f"{datetime.now().isoformat()}: {msg}"
        log_lines.append(line)
        if also_print:
            print(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(
        "#######################################################################\n"
        "Coarse-to-fine stage 2 — binary tumor refinement (prob map + ROI-aligned crops by default; "
        "see Li et al. UTR arXiv:2409.09796 for probability refinement idea). Not nnU-Net; "
        "metrics layout is compatible with nnU-Net validation summary.\n"
        "#######################################################################",
        also_print=True,
    )
    if training_args:
        args_path = out_dir / "training_args.json"
        args_path.write_text(json.dumps(training_args, indent=2), encoding="utf-8")
        log(f"Wrote training_args.json to {args_path}", also_print=False)

    best_summary: Optional[Dict[str, Any]] = None

    for epoch in range(epochs):
        t0 = time.perf_counter()
        log(f"Epoch {epoch}")
        log(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}", also_print=False)

        tr = train_one_epoch(model, train_loader, optimizer, device, bce_weight)
        va = validate_detailed(model, val_loader, device, bce_weight)
        elapsed = time.perf_counter() - t0

        gc = va["global_confusion"]
        gm = gc.to_metrics_dict(LABEL_TUMOR)[LABEL_TUMOR]

        log(f"train_loss {tr['loss']:.4f}", also_print=False)
        log(f"val_loss {va['loss']:.4f}", also_print=False)
        log(
            f"Pseudo dice (batch mean) [tumor]: {va['dice']:.4f}",
            also_print=False,
        )
        log(
            f"Dice (aggregated pixels, label {LABEL_TUMOR}): {gm['Dice']:.6f}",
            also_print=True,
        )
        log(
            f"IoU (aggregated pixels, label {LABEL_TUMOR}): {gm['IoU']:.6f}",
            also_print=True,
        )
        log(
            f"TP {gm['TP']:.0f} FP {gm['FP']:.0f} FN {gm['FN']:.0f} TN {gm['TN']:.0f}",
            also_print=False,
        )
        log(f"Epoch time: {elapsed:.2f} s", also_print=True)

        dice_for_checkpoint = float(gm["Dice"])
        if dice_for_checkpoint > best_dice:
            best_dice = dice_for_checkpoint
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_dice,
                meta={"val_dice_aggregated": best_dice, "val_dice_batch_mean": va["dice"]},
            )
            per_case_list = merge_per_case_metrics(va["per_case_confusion"], LABEL_TUMOR)
            best_summary = {
                "task": "coarse_to_fine_stage2_binary_tumor",
                "label": "tumor maps to nnU-Net label key",
                "label_key": LABEL_TUMOR,
                "foreground_mean": {k: float(v) if isinstance(v, (float, int)) else v for k, v in gm.items()},
                "mean": {LABEL_TUMOR: {k: float(v) if isinstance(v, (float, int)) else v for k, v in gm.items()}},
                "metric_per_case": per_case_list,
                "best_epoch": epoch,
                "note": "Metrics on resized ROI tensors (crop_size, ROI-aligned with infer by default). See meta.json use_coarse_prob, roi_aligned.",
            }
            (val_dir / "summary.json").write_text(
                json.dumps(best_summary, indent=2),
                encoding="utf-8",
            )
            log(f"Yayy! New best aggregated Dice (label {LABEL_TUMOR}): {best_dice:.6f}", also_print=True)
        log("", also_print=False)

    log("Training done.", also_print=True)
    return best_path
