"""Configuration for uncertainty-guided (entropy channel) tumor refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Same triple as multiview (train / val / infer) — kept local to avoid importing multiview package side effects.
DEFAULT_HU_WINDOWS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (200.0, 50.0),
    (400.0, 40.0),
    (1500.0, -400.0),
)


@dataclass
class UncertaintyConfig:
    """
    ROI is driven primarily by coarse tumor probability (wide bbox + pad), not only entropy.
    Optional ``uncertainty_threshold`` unions high-uncertainty voxels (normalized U in [0,1])
    with probability above ``prob_min_for_uncertainty_union`` to widen context.

    ``update_mode``: ``replace`` → refined prob replaces nnU-Net inside ROI; ``blend`` mixes with ``alpha``.
    """

    # Coarse tumor seed: voxels with p >= this (typical 0.5).
    roi_positive_threshold: float = 0.5
    # Optional: union seed with (U_norm > threshold) & (p >= prob_min_for_uncertainty_union). None = off.
    uncertainty_threshold: Optional[float] = None
    prob_min_for_uncertainty_union: float = 0.05

    # Lower = more small 3D CCs get refined (stronger coverage of fragmented tumor).
    min_component_voxels: int = 48
    roi_pad: Tuple[int, int, int] = (2, 24, 24)
    min_roi_side: Tuple[int, int, int] = (1, 32, 32)

    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = DEFAULT_HU_WINDOWS
    crop_size: Tuple[int, int] = (256, 256)

    update_mode: str = "blend"
    alpha: float = 0.5

    # Dual-head (error) training / inference — see UncertaintyDualHeadUNet2d (train defaults to on via CLI).
    use_error_head: bool = False
    lambda_error: float = 0.3
    # Optional: floor error gate in [0,1] on inference (None = off).
    error_gate_floor: Optional[float] = None


def uncertainty_config_to_json_dict(cfg: UncertaintyConfig) -> Dict[str, Any]:
    """Serialize for meta.json ``uncertainty_config``."""
    return {
        "roi_positive_threshold": cfg.roi_positive_threshold,
        "uncertainty_threshold": cfg.uncertainty_threshold,
        "prob_min_for_uncertainty_union": cfg.prob_min_for_uncertainty_union,
        "min_component_voxels": cfg.min_component_voxels,
        "roi_pad": list(cfg.roi_pad),
        "min_roi_side": list(cfg.min_roi_side),
        "crop_size": list(cfg.crop_size),
        "update_mode": cfg.update_mode,
        "alpha": cfg.alpha,
        "use_error_head": cfg.use_error_head,
        "lambda_error": cfg.lambda_error,
        "error_gate_floor": cfg.error_gate_floor,
    }


def merge_uncertainty_config_from_meta_dict(cfg: UncertaintyConfig, mc: dict) -> None:
    """Patch cfg from meta.json ``uncertainty_config``."""
    if "roi_positive_threshold" in mc:
        cfg.roi_positive_threshold = float(mc["roi_positive_threshold"])
    if "uncertainty_threshold" in mc:
        v = mc["uncertainty_threshold"]
        cfg.uncertainty_threshold = None if v is None else float(v)
    if "prob_min_for_uncertainty_union" in mc:
        cfg.prob_min_for_uncertainty_union = float(mc["prob_min_for_uncertainty_union"])
    if "min_component_voxels" in mc:
        cfg.min_component_voxels = int(mc["min_component_voxels"])
    if "roi_pad" in mc and len(mc["roi_pad"]) == 3:
        cfg.roi_pad = (int(mc["roi_pad"][0]), int(mc["roi_pad"][1]), int(mc["roi_pad"][2]))
    if "min_roi_side" in mc and len(mc["min_roi_side"]) == 3:
        cfg.min_roi_side = (
            int(mc["min_roi_side"][0]),
            int(mc["min_roi_side"][1]),
            int(mc["min_roi_side"][2]),
        )
    if "crop_size" in mc and len(mc["crop_size"]) == 2:
        cfg.crop_size = (int(mc["crop_size"][0]), int(mc["crop_size"][1]))
    if "update_mode" in mc:
        cfg.update_mode = str(mc["update_mode"])
    if "alpha" in mc:
        cfg.alpha = float(mc["alpha"])
    if "use_error_head" in mc:
        cfg.use_error_head = bool(mc["use_error_head"])
    if "lambda_error" in mc:
        cfg.lambda_error = float(mc["lambda_error"])
    if "error_gate_floor" in mc:
        v = mc["error_gate_floor"]
        cfg.error_gate_floor = None if v is None else float(v)
