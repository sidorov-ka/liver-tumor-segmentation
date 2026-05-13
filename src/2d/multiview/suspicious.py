"""Suspicious voxels and 3D connected components."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from multiview.config import MultiviewConfig


def suspicious_mask(prob_tumor: np.ndarray, cfg: MultiviewConfig) -> np.ndarray:
    """Binary mask: union of primary [prob_lo, prob_hi] and optional high band."""
    p = prob_tumor.astype(np.float32)
    primary = (p >= cfg.prob_lo) & (p <= cfg.prob_hi)
    hlo, hhi = cfg.prob_high_band_lo, cfg.prob_high_band_hi
    if hlo is None and hhi is None:
        return primary.astype(np.uint8)
    if (hlo is None) ^ (hhi is None):
        raise ValueError("Set both prob_high_band_lo and prob_high_band_hi, or neither.")
    if not (float(hlo) < float(hhi)):
        raise ValueError("prob_high_band_lo must be < prob_high_band_hi.")
    high = (p >= float(hlo)) & (p <= float(hhi))
    return (primary | high).astype(np.uint8)


def connected_components_3d(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """26-connected labeling. Returns (labeled, num_features)."""
    from scipy.ndimage import label

    struct = np.ones((3, 3, 3), dtype=np.int32)
    return label(mask, structure=struct)


def component_indices(labeled: np.ndarray, label_id: int) -> np.ndarray:
    return labeled == label_id
