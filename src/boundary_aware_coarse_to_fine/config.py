"""Configuration for boundary_aware coarse-to-fine: adaptive inference-time ring / τ / blend (like ``multiview.config``)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional

Regime = Literal["default", "small", "large", "suspicious"]


@dataclass
class AdaptiveInferenceConfig:
    """Override via CLI or ``meta.json`` key ``adaptive_inference``."""

    enabled: bool = True

    small_voxel_threshold: int = 6000
    large_voxel_threshold: int = 280_000

    mean_prob_suspicious: float = 0.22

    skip_refine_mean_prob_below: Optional[float] = None
    skip_refine_voxels_above: Optional[int] = None

    small_dilate_delta: int = 2
    small_erode_delta: int = -1
    large_dilate_delta: int = -1
    large_erode_delta: int = 1
    suspicious_dilate_delta: int = -2
    suspicious_erode_delta: int = 2

    refine_tau_default: float = 0.5
    refine_tau_small: float = 0.45
    refine_tau_large: float = 0.56
    refine_tau_suspicious: float = 0.62

    ring_blend_default: float = 1.0
    ring_blend_small: float = 1.0
    ring_blend_large: float = 0.42
    ring_blend_suspicious: float = 0.28

    dilate_min: int = 1
    dilate_max: int = 8
    erode_min: int = 0
    erode_max: int = 8


@dataclass
class AdaptiveInferenceResult:
    regime: Regime
    n_tumor_voxels: int
    mean_prob_in_tumor: Optional[float]
    dilate_iters: int
    erode_iters: int
    refine_prob_threshold: float
    ring_blend_alpha: float
    skip_refine: bool


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def classify_and_resolve(
    n_tumor_voxels: int,
    mean_prob_in_tumor: Optional[float],
    base_dilate: int,
    base_erode: int,
    cfg: AdaptiveInferenceConfig,
) -> AdaptiveInferenceResult:
    skip = False
    if cfg.skip_refine_mean_prob_below is not None and mean_prob_in_tumor is not None:
        if mean_prob_in_tumor < float(cfg.skip_refine_mean_prob_below):
            skip = True
    if cfg.skip_refine_voxels_above is not None and n_tumor_voxels > int(cfg.skip_refine_voxels_above):
        skip = True

    if not cfg.enabled:
        return AdaptiveInferenceResult(
            regime="default",
            n_tumor_voxels=n_tumor_voxels,
            mean_prob_in_tumor=mean_prob_in_tumor,
            dilate_iters=base_dilate,
            erode_iters=base_erode,
            refine_prob_threshold=cfg.refine_tau_default,
            ring_blend_alpha=cfg.ring_blend_default,
            skip_refine=skip,
        )

    regime: Regime = "default"
    if mean_prob_in_tumor is not None and mean_prob_in_tumor < cfg.mean_prob_suspicious:
        regime = "suspicious"
    elif n_tumor_voxels < cfg.small_voxel_threshold:
        regime = "small"
    elif n_tumor_voxels > cfg.large_voxel_threshold:
        regime = "large"

    if regime == "suspicious":
        dil = _clamp(base_dilate + cfg.suspicious_dilate_delta, cfg.dilate_min, cfg.dilate_max)
        ero = _clamp(base_erode + cfg.suspicious_erode_delta, cfg.erode_min, cfg.erode_max)
        tau = cfg.refine_tau_suspicious
        blend = cfg.ring_blend_suspicious
    elif regime == "small":
        dil = _clamp(base_dilate + cfg.small_dilate_delta, cfg.dilate_min, cfg.dilate_max)
        ero = _clamp(base_erode + cfg.small_erode_delta, cfg.erode_min, cfg.erode_max)
        tau = cfg.refine_tau_small
        blend = cfg.ring_blend_small
    elif regime == "large":
        dil = _clamp(base_dilate + cfg.large_dilate_delta, cfg.dilate_min, cfg.dilate_max)
        ero = _clamp(base_erode + cfg.large_erode_delta, cfg.erode_min, cfg.erode_max)
        tau = cfg.refine_tau_large
        blend = cfg.ring_blend_large
    else:
        dil = base_dilate
        ero = base_erode
        tau = cfg.refine_tau_default
        blend = cfg.ring_blend_default

    return AdaptiveInferenceResult(
        regime=regime,
        n_tumor_voxels=n_tumor_voxels,
        mean_prob_in_tumor=mean_prob_in_tumor,
        dilate_iters=dil,
        erode_iters=ero,
        refine_prob_threshold=tau,
        ring_blend_alpha=blend,
        skip_refine=skip,
    )


def adaptive_config_from_dict(d: Dict[str, Any]) -> AdaptiveInferenceConfig:
    if not d:
        return AdaptiveInferenceConfig()
    kwargs: Dict[str, Any] = {}
    for f in fields(AdaptiveInferenceConfig):
        if f.name in d:
            kwargs[f.name] = d[f.name]
    return AdaptiveInferenceConfig(**kwargs)
