"""ROI from coarse tumor mask: bbox, crop, paste back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BBox3D:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    def slices(self) -> Tuple[slice, slice, slice]:
        return (
            slice(self.z0, self.z1),
            slice(self.y0, self.y1),
            slice(self.x0, self.x1),
        )


def threshold_coarse_tumor(
    coarse_seg: np.ndarray, tumor_label: int
) -> np.ndarray:
    """coarse_seg: integer labels (Z, Y, X) or (1, Z, Y, X)."""
    if coarse_seg.ndim == 4:
        coarse_seg = coarse_seg[0]
    return (coarse_seg == tumor_label).astype(np.float32)


def bbox2d_from_mask(
    mask: np.ndarray,
    pad: Tuple[int, int] = (16, 16),
    min_side: Tuple[int, int] = (32, 32),
) -> Optional[Tuple[int, int, int, int]]:
    """
    2D bbox for a single axial slice. mask: (Y, X) binary.
    Returns (y0, y1, x0, x1) inclusive-exclusive, or None if empty.
    Padding and minimum side — same spirit as bbox3d_from_mask (Y/X only).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be (Y,X), got {mask.shape}")
    if not np.any(mask > 0.5):
        return None
    yy, xx = np.where(mask > 0.5)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    py, px = pad
    Y, X = mask.shape
    y0 = max(0, y0 - py)
    y1 = min(Y, y1 + py)
    x0 = max(0, x0 - px)
    x1 = min(X, x1 + px)

    def expand_axis(a0: int, a1: int, amin: int, amax: int) -> Tuple[int, int]:
        cur = a1 - a0
        if cur >= amin:
            return a0, a1
        need = amin - cur
        left = need // 2
        right = need - left
        a0n = max(0, a0 - left)
        a1n = min(amax, a1 + right)
        if a1n - a0n < amin:
            if a0n == 0:
                a1n = min(amax, amin)
            else:
                a0n = max(0, amax - amin)
        return a0n, a1n

    y0, y1 = expand_axis(y0, y1, min_side[0], Y)
    x0, x1 = expand_axis(x0, x1, min_side[1], X)
    return y0, y1, x0, x1


def bbox3d_from_mask(
    mask: np.ndarray,
    pad: Tuple[int, int, int] = (2, 16, 16),
    min_side: Tuple[int, int, int] = (1, 32, 32),
    shape_limit: Optional[Tuple[int, int, int]] = None,
) -> Optional[BBox3D]:
    """
    mask: binary (Z, Y, X). Returns None if empty.
    Pads bbox; clips to volume shape. Ensures minimum spatial extent (min_side on Y/X).
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be (Z,Y,X), got {mask.shape}")
    if not np.any(mask):
        return None

    zz, yy, xx = np.where(mask > 0.5)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1

    pz, py, px = pad
    Z, Y, X = mask.shape

    z0 = max(0, z0 - pz)
    z1 = min(Z, z1 + pz)
    y0 = max(0, y0 - py)
    y1 = min(Y, y1 + py)
    x0 = max(0, x0 - px)
    x1 = min(X, x1 + px)

    # Enforce minimum size (centered expansion)
    def expand_axis(a0: int, a1: int, amin: int, amax: int) -> Tuple[int, int]:
        cur = a1 - a0
        if cur >= amin:
            return a0, a1
        need = amin - cur
        left = need // 2
        right = need - left
        a0n = max(0, a0 - left)
        a1n = min(amax, a1 + right)
        if a1n - a0n < amin:
            if a0n == 0:
                a1n = min(amax, amin)
            else:
                a0n = max(0, amax - amin)
        return a0n, a1n

    z0, z1 = expand_axis(z0, z1, max(1, min_side[0]), Z)
    y0, y1 = expand_axis(y0, y1, min_side[1], Y)
    x0, x1 = expand_axis(x0, x1, min_side[2], X)

    if shape_limit is not None:
        lz, ly, lx = shape_limit
        if z1 - z0 > lz:
            c = (z0 + z1) // 2
            z0 = max(0, c - lz // 2)
            z1 = min(Z, z0 + lz)
            z0 = max(0, z1 - lz)
        if y1 - y0 > ly:
            c = (y0 + y1) // 2
            y0 = max(0, c - ly // 2)
            y1 = min(Y, y0 + ly)
            y0 = max(0, y1 - ly)
        if x1 - x0 > lx:
            c = (x0 + x1) // 2
            x0 = max(0, c - lx // 2)
            x1 = min(X, x0 + lx)
            x0 = max(0, x1 - lx)

    return BBox3D(z0, z1, y0, y1, x0, x1)


def crop_volume(
    vol: np.ndarray, bbox: BBox3D
) -> np.ndarray:
    """vol: (Z, Y, X) or (C, Z, Y, X)."""
    sl = bbox.slices()
    if vol.ndim == 3:
        return vol[sl[0], sl[1], sl[2]].copy()
    if vol.ndim == 4:
        return vol[:, sl[0], sl[1], sl[2]].copy()
    raise ValueError(f"Unexpected ndim {vol.ndim}")


def paste_mask_back(
    full_shape: Tuple[int, ...],
    crop_mask: np.ndarray,
    bbox: BBox3D,
    fill: float = 0.0,
) -> np.ndarray:
    """
    crop_mask: (Zc, Yc, Xc) binary or probability.
    Returns array of shape full_shape (Z, Y, X).
    """
    out = np.full(full_shape, fill, dtype=np.float32)
    z0, z1, y0, y1, x0, x1 = bbox.z0, bbox.z1, bbox.y0, bbox.y1, bbox.x0, bbox.x1
    out[z0:z1, y0:y1, x0:x1] = crop_mask.astype(np.float32)
    return out
