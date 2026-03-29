"""Fixed triple HU windowing for multi-window CT (2D slice or 3D volume)."""

from __future__ import annotations

import numpy as np


def apply_hu_window(volume: np.ndarray, width: float, level: float) -> np.ndarray:
    """Clip to [level - W/2, level + W/2] and linearly scale to [0, 1]."""
    low = level - width / 2.0
    high = level + width / 2.0
    x = volume.astype(np.float32)
    x = np.clip(x, low, high)
    return (x - low) / (high - low + 1e-6)


def stack_multi_window(
    ct_hu: np.ndarray, windows: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> np.ndarray:
    """
    ct_hu: (Y, X) or (Z, Y, X) intensity in HU-like units after nnU-Net preprocessing (aligned with prob map).
    Returns (3, Y, X) or (3, Z, Y, X) for three fixed windows.
    """
    if len(windows) != 3:
        raise ValueError("Expected exactly three (width, level) windows.")
    ch = [apply_hu_window(ct_hu, w, l) for w, l in windows]
    return np.stack(ch, axis=0)
