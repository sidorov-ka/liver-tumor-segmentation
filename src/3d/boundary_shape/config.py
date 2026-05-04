from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BoundaryOversegConfig:
    """Environment-configurable knobs for boundary/oversegmentation fine-tuning."""

    num_epochs: int = 50
    initial_lr: float = 1e-3
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
    inside_liver_volume_guard_threshold: float = 0.02
    inside_liver_volume_guard_min_scale: float = 0.10
    tversky_guard_weight: float = 0.05
    tversky_guard_alpha: float = 0.30
    tversky_guard_beta: float = 0.70
    boundary_start_epoch: int = 5
    fp_start_epoch: int = 10
    custom_loss_ramp_epochs: int = 10

    @classmethod
    def from_env(cls) -> "BoundaryOversegConfig":
        return cls(
            num_epochs=_env_int("NNUNET_BOUNDARY_OVERSEG_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_BOUNDARY_OVERSEG_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_BOUNDARY_OVERSEG_TUMOR_LABEL", cls.tumor_label),
            liver_label=_env_int("NNUNET_BOUNDARY_OVERSEG_LIVER_LABEL", cls.liver_label),
            boundary_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT",
                cls.boundary_weight,
            ),
            overseg_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT",
                cls.overseg_weight,
            ),
            outside_liver_fp_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_FP_WEIGHT",
                cls.outside_liver_fp_weight,
            ),
            inside_liver_fp_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_FP_WEIGHT",
                cls.inside_liver_fp_weight,
            ),
            boundary_radius=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS",
                cls.boundary_radius,
            ),
            outside_liver_ignore_radius=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_IGNORE_RADIUS",
                cls.outside_liver_ignore_radius,
            ),
            inside_liver_ignore_radius=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_IGNORE_RADIUS",
                cls.inside_liver_ignore_radius,
            ),
            outside_liver_topk_fraction=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_TOPK_FRACTION",
                cls.outside_liver_topk_fraction,
            ),
            inside_liver_topk_fraction=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_TOPK_FRACTION",
                cls.inside_liver_topk_fraction,
            ),
            inside_liver_volume_guard_threshold=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_THRESHOLD",
                cls.inside_liver_volume_guard_threshold,
            ),
            inside_liver_volume_guard_min_scale=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_MIN_SCALE",
                cls.inside_liver_volume_guard_min_scale,
            ),
            tversky_guard_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_WEIGHT",
                cls.tversky_guard_weight,
            ),
            tversky_guard_alpha=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_ALPHA",
                cls.tversky_guard_alpha,
            ),
            tversky_guard_beta=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_BETA",
                cls.tversky_guard_beta,
            ),
            boundary_start_epoch=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_BOUNDARY_START_EPOCH",
                cls.boundary_start_epoch,
            ),
            fp_start_epoch=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_FP_START_EPOCH",
                cls.fp_start_epoch,
            ),
            custom_loss_ramp_epochs=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_RAMP_EPOCHS",
                cls.custom_loss_ramp_epochs,
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
