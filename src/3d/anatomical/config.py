from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AnatomicalConfig:
    """Anatomy-constrained additive loss (boundary ring + liver FP + Tversky)."""

    num_epochs: int = 500
    initial_lr: float = 1e-2
    tumor_label: int = 2
    liver_label: int = 1
    boundary_weight: float = 0.10
    overseg_weight: float = 0.05
    outside_liver_fp_weight: float = 4.0
    inside_liver_fp_weight: float = 0.5
    boundary_radius: int = 2
    outside_liver_ignore_radius: int = 2
    inside_liver_ignore_radius: int = 4
    outside_liver_topk_fraction: float = 0.01
    inside_liver_topk_fraction: float = 0.002
    tversky_guard_weight: float = 0.05
    tversky_guard_alpha: float = 0.30
    tversky_guard_beta: float = 0.70

    @classmethod
    def from_env(cls) -> "AnatomicalConfig":
        return cls(
            num_epochs=_env_int("NNUNET_ANATOMICAL_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_ANATOMICAL_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_ANATOMICAL_TUMOR_LABEL", cls.tumor_label),
            liver_label=_env_int("NNUNET_ANATOMICAL_LIVER_LABEL", cls.liver_label),
            boundary_weight=_env_float(
                "NNUNET_ANATOMICAL_BOUNDARY_WEIGHT",
                cls.boundary_weight,
            ),
            overseg_weight=_env_float(
                "NNUNET_ANATOMICAL_OVERSEG_WEIGHT",
                cls.overseg_weight,
            ),
            outside_liver_fp_weight=_env_float(
                "NNUNET_ANATOMICAL_OUTSIDE_LIVER_FP_WEIGHT",
                cls.outside_liver_fp_weight,
            ),
            inside_liver_fp_weight=_env_float(
                "NNUNET_ANATOMICAL_INSIDE_LIVER_FP_WEIGHT",
                cls.inside_liver_fp_weight,
            ),
            boundary_radius=_env_int(
                "NNUNET_ANATOMICAL_BOUNDARY_RADIUS",
                cls.boundary_radius,
            ),
            outside_liver_ignore_radius=_env_int(
                "NNUNET_ANATOMICAL_OUTSIDE_LIVER_IGNORE_RADIUS",
                cls.outside_liver_ignore_radius,
            ),
            inside_liver_ignore_radius=_env_int(
                "NNUNET_ANATOMICAL_INSIDE_LIVER_IGNORE_RADIUS",
                cls.inside_liver_ignore_radius,
            ),
            outside_liver_topk_fraction=_env_float(
                "NNUNET_ANATOMICAL_OUTSIDE_LIVER_TOPK_FRACTION",
                cls.outside_liver_topk_fraction,
            ),
            inside_liver_topk_fraction=_env_float(
                "NNUNET_ANATOMICAL_INSIDE_LIVER_TOPK_FRACTION",
                cls.inside_liver_topk_fraction,
            ),
            tversky_guard_weight=_env_float(
                "NNUNET_ANATOMICAL_TVERSKY_WEIGHT",
                cls.tversky_guard_weight,
            ),
            tversky_guard_alpha=_env_float(
                "NNUNET_ANATOMICAL_TVERSKY_ALPHA",
                cls.tversky_guard_alpha,
            ),
            tversky_guard_beta=_env_float(
                "NNUNET_ANATOMICAL_TVERSKY_BETA",
                cls.tversky_guard_beta,
            ),
        )


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)
