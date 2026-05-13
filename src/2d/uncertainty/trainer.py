"""Train / validate uncertainty stage-2; output under ``results_uncertainty/.../uncertainty/run_*``."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from coarse_to_fine.metrics import BinaryConfusion, merge_per_case_metrics
from coarse_to_fine.utils import bce_dice_boundary_loss, bce_dice_loss, dice_coefficient, save_checkpoint
from uncertainty.model import build_uncertainty_model

LABEL_TUMOR = "2"


def uncertainty_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "case_id": [b["case_id"] for b in batch],
    }
    if "y_error" in batch[0]:
        out["y_error"] = torch.stack([b["y_error"] for b in batch])
    return out


def _forward_tumor(
    model: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(x)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_weight: float = 0.5,
    boundary_weight: float = 0.0,
    lambda_error: float = 0.3,
    use_error_head: bool = False,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_err = batch.get("y_error")
        if use_error_head and y_err is None:
            raise RuntimeError("use_error_head but batch has no y_error")
        optimizer.zero_grad(set_to_none=True)
        logit_t, logit_e = _forward_tumor(model, x)
        loss_t = bce_dice_boundary_loss(
            logit_t, y, bce_weight=bce_weight, boundary_weight=boundary_weight
        )
        if use_error_head and logit_e is not None and y_err is not None:
            ye = y_err.to(device)
            loss_e = bce_dice_loss(logit_e, ye, bce_weight=bce_weight)
            loss = loss_t + float(lambda_error) * loss_e
        else:
            loss = loss_t
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prob = torch.sigmoid(logit_t)
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
    boundary_weight: float = 0.0,
    lambda_error: float = 0.3,
    use_error_head: bool = False,
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
        y_err = batch.get("y_error")
        case_ids: List[str] = batch["case_id"]
        logit_t, logit_e = _forward_tumor(model, x)
        loss_t = bce_dice_boundary_loss(
            logit_t, y, bce_weight=bce_weight, boundary_weight=boundary_weight
        )
        if use_error_head and logit_e is not None and y_err is not None:
            ye = y_err.to(device)
            loss_e = bce_dice_loss(logit_e, ye, bce_weight=bce_weight)
            loss = loss_t + float(lambda_error) * loss_e
        else:
            loss = loss_t
        prob = torch.sigmoid(logit_t)
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
    base: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    bce_weight: float = 0.5,
    boundary_weight: float = 0.0,
    training_args: Optional[Dict[str, Any]] = None,
    use_error_head: bool = False,
    lambda_error: float = 0.3,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_uncertainty_model(base=base, use_error_head=use_error_head).to(device)
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

    arch = "UncertaintyDualHeadUNet2d" if use_error_head else "UncertaintyUNet2d"
    log(
        f"{arch} — 5 ch (3 HU + prob + entropy)"
        + (
            f"; + error head; lambda_error={lambda_error}; "
            if use_error_head
            else "; "
        )
        + f"loss BCE+Dice + boundary_weight={boundary_weight}; aligned with infer_uncertainty.",
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

        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            bce_weight,
            boundary_weight,
            lambda_error=lambda_error,
            use_error_head=use_error_head,
        )
        va = validate_detailed(
            model,
            val_loader,
            device,
            bce_weight,
            boundary_weight,
            lambda_error=lambda_error,
            use_error_head=use_error_head,
        )
        elapsed = time.perf_counter() - t0

        gc = va["global_confusion"]
        gm = gc.to_metrics_dict(LABEL_TUMOR)[LABEL_TUMOR]

        log(f"train_loss {tr['loss']:.4f}", also_print=False)
        log(f"val_loss {va['loss']:.4f}", also_print=False)
        log(f"Pseudo dice (batch mean) [tumor]: {va['dice']:.4f}", also_print=False)
        log(f"Dice (aggregated pixels, label {LABEL_TUMOR}): {gm['Dice']:.6f}", also_print=True)
        log(f"IoU (aggregated pixels, label {LABEL_TUMOR}): {gm['IoU']:.6f}", also_print=True)
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
            gm_float = {k: float(v) if isinstance(v, (float, int)) else v for k, v in gm.items()}
            best_summary = {
                "task": "uncertainty_binary_tumor",
                "label": "tumor maps to nnU-Net label key",
                "label_key": LABEL_TUMOR,
                "foreground_mean": gm_float,
                "mean": {LABEL_TUMOR: dict(gm_float)},
                "metric_per_case": per_case_list,
                "best_epoch": epoch,
                "note": (
                    "Val on 5-channel ROI crops; see meta.json / training_args.json for hyperparameters."
                ),
            }
            (val_dir / "summary.json").write_text(
                json.dumps(best_summary, indent=2),
                encoding="utf-8",
            )
            log(f"New best aggregated Dice (label {LABEL_TUMOR}): {best_dice:.6f}", also_print=True)
        log("", also_print=False)

    log("Training done.", also_print=True)
    return best_path
