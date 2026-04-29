"""Boundary ring around a coarse binary tumor mask (morphology)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def boundary_ring_from_binary_mask(
    mask: np.ndarray,
    dilate_iters: int = 2,
    erode_iters: int = 2,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Narrow band around the boundary of a binary mask.

    ``dilated = D(mask)``, ``eroded = E(mask)`` with iteration counts; ``ring = dilated XOR eroded``.
    Returns float32 in {0, 1}, same shape as ``mask``.
    """
    m = mask > 0.5
    dil = binary_dilation(m, structure=structure, iterations=int(max(0, dilate_iters)))
    ero = binary_erosion(m, structure=structure, iterations=int(max(0, erode_iters)))
    ring = np.logical_xor(dil, ero)
    return ring.astype(np.float32)
