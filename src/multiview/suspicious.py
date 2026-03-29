"""Suspicious voxels and 3D connected components."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from multiview.config import MultiviewConfig


def suspicious_mask(prob_tumor: np.ndarray, cfg: MultiviewConfig) -> np.ndarray:
    """Binary mask: ambiguous tumor probability band."""
    return ((prob_tumor >= cfg.prob_lo) & (prob_tumor <= cfg.prob_hi)).astype(np.uint8)


def connected_components_3d(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """26-connected labeling. Returns (labeled, num_features)."""
    from scipy.ndimage import label

    struct = np.ones((3, 3, 3), dtype=np.int32)
    return label(mask, structure=struct)


def component_indices(labeled: np.ndarray, label_id: int) -> np.ndarray:
    return labeled == label_id
