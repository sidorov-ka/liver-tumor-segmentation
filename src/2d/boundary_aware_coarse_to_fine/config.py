"""HU window triple for multi-channel CT (aligned with multiview / uncertainty).

Three *different* (width, level) pairs so each channel encodes a distinct contrast:
narrow liver/lesion, typical abdominal soft-tissue, wide field for vessels/context.
Each channel is ``apply_hu_window`` → [0,1] then per-slice z-score in the dataset (same as uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Same fixed triple as multiview.config / uncertainty.config — do not collapse to one window.
DEFAULT_HU_WINDOWS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (200.0, 50.0),  # narrow: parenchyma / focal lesions
    (400.0, 40.0),  # soft tissue (typical abdominal)
    (1500.0, -400.0),  # wide: vessels + global context
)


def parse_hu_windows_arg(flat: list[float]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Six floats: W1 L1 W2 L2 W3 L3."""
    if len(flat) != 6:
        raise ValueError("Expected six floats for three HU windows: W1 L1 W2 L2 W3 L3")
    return (
        (float(flat[0]), float(flat[1])),
        (float(flat[2]), float(flat[3])),
        (float(flat[4]), float(flat[5])),
    )


def hu_windows_to_json_list(
    w: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> list[list[float]]:
    return [[float(a), float(b)] for a, b in w]


def hu_windows_from_json_list(data: object) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    if not isinstance(data, list) or len(data) != 3:
        raise ValueError("hu_windows in meta must be [[W,L], [W,L], [W,L]]")
    out: list[Tuple[float, float]] = []
    for pair in data:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each hu_window entry must be [width, level]")
        out.append((float(pair[0]), float(pair[1])))
    return (out[0], out[1], out[2])


# --- Adaptive boundary inference (used by infer_boundary_aware_coarse_to_fine.py) ---


@dataclass
class AdaptiveInferenceConfig:
    enabled: bool = True
    small_voxel_threshold: int = 6000
    large_voxel_threshold: int = 280000
    mean_prob_suspicious: float = 0.22
    refine_tau_default: float = 0.52
    refine_tau_small: float = 0.45
    refine_tau_large: float = 0.56
    refine_tau_suspicious: float = 0.62
    ring_blend_large: float = 0.42
    ring_blend_suspicious: float = 0.28
    ring_blend_default: float = 1.0
    skip_refine_mean_prob_below: Optional[float] = None
    skip_refine_voxels_above: Optional[int] = None


@dataclass(frozen=True)
class AdaptiveResolved:
    regime: str
    dilate_iters: int
    erode_iters: int
    refine_prob_threshold: float
    ring_blend_alpha: float
    skip_refine: bool


def adaptive_config_from_dict(d: Optional[Dict[str, Any]]) -> AdaptiveInferenceConfig:
    if not d:
        return AdaptiveInferenceConfig()
    return AdaptiveInferenceConfig(
        enabled=bool(d.get("enabled", True)),
        small_voxel_threshold=int(d.get("small_voxel_threshold", 6000)),
        large_voxel_threshold=int(d.get("large_voxel_threshold", 280000)),
        mean_prob_suspicious=float(d.get("mean_prob_suspicious", 0.22)),
        refine_tau_default=float(d.get("refine_tau_default", 0.52)),
        refine_tau_small=float(d.get("refine_tau_small", 0.45)),
        refine_tau_large=float(d.get("refine_tau_large", 0.56)),
        refine_tau_suspicious=float(d.get("refine_tau_suspicious", 0.62)),
        ring_blend_large=float(d.get("ring_blend_large", 0.42)),
        ring_blend_suspicious=float(d.get("ring_blend_suspicious", 0.28)),
        ring_blend_default=float(d.get("ring_blend_default", 1.0)),
        skip_refine_mean_prob_below=(
            float(d["skip_refine_mean_prob_below"])
            if d.get("skip_refine_mean_prob_below") is not None
            else None
        ),
        skip_refine_voxels_above=(
            int(d["skip_refine_voxels_above"])
            if d.get("skip_refine_voxels_above") is not None
            else None
        ),
    )


def classify_and_resolve(
    nv: int,
    mean_p: Optional[float],
    base_dilate: int,
    base_erode: int,
    cfg: AdaptiveInferenceConfig,
) -> AdaptiveResolved:
    if cfg.skip_refine_mean_prob_below is not None and mean_p is not None:
        if mean_p < cfg.skip_refine_mean_prob_below:
            return AdaptiveResolved(
                "skip_low_prob", base_dilate, base_erode, cfg.refine_tau_default, cfg.ring_blend_default, True
            )
    if cfg.skip_refine_voxels_above is not None and nv > cfg.skip_refine_voxels_above:
        return AdaptiveResolved(
            "skip_huge_volume", base_dilate, base_erode, cfg.refine_tau_default, cfg.ring_blend_default, True
        )
    if not cfg.enabled:
        return AdaptiveResolved(
            "fixed", base_dilate, base_erode, cfg.refine_tau_default, cfg.ring_blend_default, False
        )

    dil, ero = base_dilate, base_erode
    if mean_p is not None and mean_p < cfg.mean_prob_suspicious:
        return AdaptiveResolved(
            "suspicious", dil + 1, ero, cfg.refine_tau_suspicious, cfg.ring_blend_suspicious, False
        )
    if nv < cfg.small_voxel_threshold:
        return AdaptiveResolved(
            "small", dil + 1, ero, cfg.refine_tau_small, cfg.ring_blend_default, False
        )
    if nv > cfg.large_voxel_threshold:
        return AdaptiveResolved(
            "large", dil, ero, cfg.refine_tau_large, cfg.ring_blend_large, False
        )
    return AdaptiveResolved(
        "medium", dil, ero, cfg.refine_tau_default, cfg.ring_blend_default, False
    )
