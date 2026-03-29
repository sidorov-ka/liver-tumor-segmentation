"""Configuration for multiview (suspicious-ROI multi-window) refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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

    # Pixels with tumor prob in [prob_lo, prob_hi] are "suspicious" (ambiguous / boundary band)
    prob_lo: float = 0.12
    prob_hi: float = 0.88

    min_component_voxels: int = 32
    # Padding around each component bbox (Z, Y, X) — same idea as coarse_to_fine.roi.bbox3d_from_mask
    roi_pad: Tuple[int, int, int] = (2, 16, 16)
    min_roi_side: Tuple[int, int, int] = (1, 32, 32)

    # Three HU windows (width, level) each
    hu_windows: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = DEFAULT_HU_WINDOWS

    # 2D crop to network (H, W)
    crop_size: Tuple[int, int] = (256, 256)
