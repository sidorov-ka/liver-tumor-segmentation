"""Configuration for multiview (suspicious-ROI multi-window) refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# Fixed triple (width, level) in HU — liver/abdominal CT (same three for train, val, and infer).
# Ranges: narrow liver/lesion → standard soft tissue → wide context.
DEFAULT_HU_WINDOWS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (200.0, 50.0),  # narrow: parenchyma / focal lesions
    (400.0, 40.0),  # soft tissue (typical abdominal window)
    (1500.0, -400.0),  # wide: vessels + global context
)


@dataclass
class MultiviewConfig:
    """Suspicious voxels, connected components, ROI padding, multi-window CT, MultiviewUNet2d (4 ch)."""

    # Primary ambiguous band: prob_lo <= p <= prob_hi (inclusive).
    prob_lo: float = 0.25
    prob_hi: float = 0.75

    # Optional second band (e.g. high-confidence boundary): union with primary when both set.
    prob_high_band_lo: Optional[float] = None
    prob_high_band_hi: Optional[float] = None

    # Skip tiny 3D suspicious blobs before refinement (reduces speckle; raise if large leaks persist).
    min_component_voxels: int = 64
    # Padding around each component bbox (Z, Y, X) — same idea as coarse_to_fine.roi.bbox3d_from_mask
    roi_pad: Tuple[int, int, int] = (2, 16, 16)
    min_roi_side: Tuple[int, int, int] = (1, 32, 32)

    # Three HU windows (width, level) each
    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = DEFAULT_HU_WINDOWS

    # 2D crop to network (H, W)
    crop_size: Tuple[int, int] = (256, 256)

    # "replace": refined prob overwrites nnU-Net tumor prob in ROI (default, backward compatible).
    # "blend": p_out = alpha*p_refine + (1-alpha)*p_nnunet in ROI.
    refine_blend_mode: str = "replace"
    # When blend and component_size_alpha_threshold_voxels == 0, use this alpha for all components.
    refine_alpha: float = 1.0
    # If > 0: suspicious components with nv < threshold use alpha_blend_small, else alpha_blend_large.
    component_size_alpha_threshold_voxels: int = 0
    alpha_blend_small: float = 0.75
    alpha_blend_large: float = 0.35

    # After refinement: remove tumor CCs at p>0.5 with volume below this (0 = off).
    post_remove_tumor_components_below_voxels: int = 0


def multiview_config_to_json_dict(cfg: MultiviewConfig) -> Dict[str, Any]:
    """Serialize for meta.json ``multiview_config`` (infer_multiview / train_multiview)."""
    return {
        "prob_lo": cfg.prob_lo,
        "prob_hi": cfg.prob_hi,
        "prob_high_band_lo": cfg.prob_high_band_lo,
        "prob_high_band_hi": cfg.prob_high_band_hi,
        "min_component_voxels": cfg.min_component_voxels,
        "roi_pad": list(cfg.roi_pad),
        "min_roi_side": list(cfg.min_roi_side),
        "crop_size": list(cfg.crop_size),
        "refine_blend_mode": cfg.refine_blend_mode,
        "refine_alpha": cfg.refine_alpha,
        "component_size_alpha_threshold_voxels": cfg.component_size_alpha_threshold_voxels,
        "alpha_blend_small": cfg.alpha_blend_small,
        "alpha_blend_large": cfg.alpha_blend_large,
        "post_remove_tumor_components_below_voxels": cfg.post_remove_tumor_components_below_voxels,
    }


def merge_multiview_config_from_meta_dict(cfg: MultiviewConfig, mc: dict) -> None:
    """Patch cfg from meta.json ``multiview_config`` (missing keys = keep current / defaults)."""
    if "prob_lo" in mc:
        cfg.prob_lo = float(mc["prob_lo"])
    if "prob_hi" in mc:
        cfg.prob_hi = float(mc["prob_hi"])
    if "prob_high_band_lo" in mc:
        v = mc["prob_high_band_lo"]
        cfg.prob_high_band_lo = None if v is None else float(v)
    if "prob_high_band_hi" in mc:
        v = mc["prob_high_band_hi"]
        cfg.prob_high_band_hi = None if v is None else float(v)
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
    if "refine_blend_mode" in mc:
        cfg.refine_blend_mode = str(mc["refine_blend_mode"])
    if "refine_alpha" in mc:
        cfg.refine_alpha = float(mc["refine_alpha"])
    if "component_size_alpha_threshold_voxels" in mc:
        cfg.component_size_alpha_threshold_voxels = int(mc["component_size_alpha_threshold_voxels"])
    if "alpha_blend_small" in mc:
        cfg.alpha_blend_small = float(mc["alpha_blend_small"])
    if "alpha_blend_large" in mc:
        cfg.alpha_blend_large = float(mc["alpha_blend_large"])
    if "post_remove_tumor_components_below_voxels" in mc:
        cfg.post_remove_tumor_components_below_voxels = int(mc["post_remove_tumor_components_below_voxels"])
