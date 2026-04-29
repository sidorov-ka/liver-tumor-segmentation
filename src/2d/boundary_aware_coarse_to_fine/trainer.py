"""Train / validate boundary-aware coarse-to-fine TinyUNet (5 ch: 3 HU windows + prob + entropy)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from boundary_aware_coarse_to_fine.metrics import BinaryConfusion, merge_per_case_metrics
from boundary_aware_coarse_to_fine.model import BoundaryAwareTinyUNet2d
from boundary_aware_coarse_to_fine.utils import (
    bce_dice_with_optional_ring,
    dice_coefficient,
    load_checkpoint,
    save_checkpoint,
)

LABEL_TUMOR = "2"

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def boundary_aware_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "ring": torch.stack([b["ring"] for b in batch]),
        "case_id": [b["case_id"] for b in batch],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_weight: float = 0.5,
    lambda_boundary: float = 0.0,
    focal_gamma: float = 0.0,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        ring = batch["ring"].to(device) if lambda_boundary > 0.0 else None
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = bce_dice_with_optional_ring(
            logits,
            y,
            ring,
            bce_weight=bce_weight,
            lambda_boundary=lambda_boundary,
            focal_gamma=focal_gamma,
        )
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
    lambda_boundary: float = 0.0,
    focal_gamma: float = 0.0,
) -> Dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    global_conf = BinaryConfusion()
    per_case: Dict[str, BinaryConfusion] = {}

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        ring = batch["ring"].to(device) if lambda_boundary > 0.0 else None
        case_ids: List[str] = batch["case_id"]
        logits = model(x)
        loss = bce_dice_with_optional_ring(
            logits,
            y,
            ring,
            bce_weight=bce_weight,
            lambda_boundary=lambda_boundary,
            focal_gamma=focal_gamma,
        )
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
    in_channels: int = 5,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    bce_weight: float = 0.5,
    lambda_boundary: float = 0.0,
    focal_gamma: float = 0.0,
    training_args: Optional[Dict[str, Any]] = None,
    tensorboard_dir: Optional[Path] = None,
    resume_path: Optional[Path] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryAwareTinyUNet2d(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_dice = -1.0
    start_epoch = 0
    best_path = out_dir / "checkpoint_best.pth"
    last_path = out_dir / "checkpoint_last.pth"

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = (
        out_dir / f"training_log_resume_{ts}.txt"
        if resume_path is not None
        else out_dir / f"training_log_{ts}.txt"
    )

    if resume_path is not None:
        rp = Path(resume_path)
        if not rp.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {rp}")
        ckpt = load_checkpoint(rp, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_dice = float(ckpt.get("best_metric", -1.0))
    log_lines: List[str] = []

    def log(msg: str, also_print: bool = True) -> None:
        line = f"{datetime.now().isoformat()}: {msg}"
        log_lines.append(line)
        if also_print:
            print(msg)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log(
        "Boundary-aware coarse-to-fine stage 2 — binary tumor (5 ch: 3 HU windows + nnU-Net prob + entropy); "
        "optional focal BCE + ring-weighted loss; inference uses ring-only merge.",
        also_print=True,
    )
    if resume_path is not None:
        log(
            f"Resumed from {resume_path} — next epoch index {start_epoch}, "
            f"best aggregated Dice so far {best_dice:.6f}",
            also_print=True,
        )
    if training_args:
        args_path = out_dir / "training_args.json"
        args_path.write_text(json.dumps(training_args, indent=2), encoding="utf-8")
        log(f"Wrote training_args.json to {args_path}", also_print=False)

    tb_writer: Optional["SummaryWriter"] = None
    if tensorboard_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
            log(
                f"TensorBoard logs -> {tensorboard_dir.resolve()} "
                f"(tensorboard --logdir {tensorboard_dir})",
                also_print=True,
            )
        except ImportError:
            log(
                "TensorBoard unavailable (pip install tensorboard); continuing without it.",
                also_print=True,
            )

    best_summary: Optional[Dict[str, Any]] = None

    try:
        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            t0 = time.perf_counter()
            log(f"Epoch {epoch}")
            log(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}", also_print=False)

            tr = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                bce_weight,
                lambda_boundary,
                focal_gamma,
            )
            va = validate_detailed(
                model, val_loader, device, bce_weight, lambda_boundary, focal_gamma
            )
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

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", tr["loss"], epoch)
                tb_writer.add_scalar("train/dice_mean_batch", tr["dice"], epoch)
                tb_writer.add_scalar("val/loss", va["loss"], epoch)
                tb_writer.add_scalar("val/dice_mean_batch", va["dice"], epoch)
                tb_writer.add_scalar("val/dice_aggregated", float(gm["Dice"]), epoch)
                tb_writer.add_scalar("val/iou_aggregated", float(gm["IoU"]), epoch)
                tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                tb_writer.add_scalar("epoch_time_sec", elapsed, epoch)

            dice_for_checkpoint = float(gm["Dice"])
            if dice_for_checkpoint > best_dice:
                best_dice = dice_for_checkpoint
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    best_dice,
                    meta={
                        "val_dice_aggregated": best_dice,
                        "val_dice_batch_mean": va["dice"],
                        "lambda_boundary": lambda_boundary,
                        "focal_gamma": focal_gamma,
                    },
                )
                per_case_list = merge_per_case_metrics(va["per_case_confusion"], LABEL_TUMOR)
                best_summary = {
                    "task": "boundary_aware_coarse_to_fine_stage2_binary_tumor",
                    "label": "tumor maps to nnU-Net label key",
                    "label_key": LABEL_TUMOR,
                    "foreground_mean": {
                        k: float(v) if isinstance(v, (float, int)) else v for k, v in gm.items()
                    },
                    "mean": {
                        LABEL_TUMOR: {
                            k: float(v) if isinstance(v, (float, int)) else v for k, v in gm.items()
                        }
                    },
                    "metric_per_case": per_case_list,
                    "best_epoch": epoch,
                    "note": "Val metrics on ROI crops; see meta.json / training_args.json.",
                }
                (val_dir / "summary.json").write_text(
                    json.dumps(best_summary, indent=2),
                    encoding="utf-8",
                )
                log(
                    f"Yayy! New best aggregated Dice (label {LABEL_TUMOR}): {best_dice:.6f}",
                    also_print=True,
                )
            if tb_writer is not None:
                tb_writer.add_scalar("val/best_dice_aggregated_so_far", best_dice, epoch)
            log("", also_print=False)

            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                best_dice,
                meta={
                    "val_dice_aggregated": float(gm["Dice"]),
                    "val_dice_batch_mean": va["dice"],
                    "lambda_boundary": lambda_boundary,
                    "focal_gamma": focal_gamma,
                    "last_epoch": epoch,
                },
            )

    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    log("Training done.", also_print=True)
    return best_path
