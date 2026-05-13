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
    adaptive_large_tumor_threshold: float = 0.02
    adaptive_large_tumor_max_threshold: float = 0.10
    adaptive_fp_min_scale: float = 1.0
    #: If >= 0, `adaptive_fp_min_scale` is reached only after this epoch: before
    #: that, use 1.0 (full FP hard-negative weights). Linear ramp over
    #: `adaptive_fp_min_schedule_ramp_epochs`; if ramp is 0, jump to target at start.
    adaptive_fp_min_schedule_start_epoch: int = -1
    adaptive_fp_min_schedule_ramp_epochs: int = 0
    adaptive_ignore_extra_radius: int = 0
    under_volume_guard_weight: float = 0.0
    under_volume_guard_threshold: float = 0.05
    under_volume_guard_fraction: float = 0.85
    #: If True, under-volume uses weight ``(1 - custom_loss_gate)`` per sample
    #: (strong on large-tumor / low-gate patches); default False keeps ``custom_loss_gate``.
    under_volume_inverse_gate: bool = False
    custom_loss_gate_threshold: float = 0.04
    custom_loss_gate_temperature: float = 0.015
    custom_loss_gate_min_scale: float = 0.0
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
            adaptive_large_tumor_threshold=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_THRESHOLD",
                cls.adaptive_large_tumor_threshold,
            ),
            adaptive_large_tumor_max_threshold=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_MAX_THRESHOLD",
                cls.adaptive_large_tumor_max_threshold,
            ),
            adaptive_fp_min_scale=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_MIN_SCALE",
                cls.adaptive_fp_min_scale,
            ),
            adaptive_fp_min_schedule_start_epoch=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_START_EPOCH",
                cls.adaptive_fp_min_schedule_start_epoch,
            ),
            adaptive_fp_min_schedule_ramp_epochs=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_RAMP_EPOCHS",
                cls.adaptive_fp_min_schedule_ramp_epochs,
            ),
            adaptive_ignore_extra_radius=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_IGNORE_EXTRA_RADIUS",
                cls.adaptive_ignore_extra_radius,
            ),
            under_volume_guard_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_WEIGHT",
                cls.under_volume_guard_weight,
            ),
            under_volume_guard_threshold=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_THRESHOLD",
                cls.under_volume_guard_threshold,
            ),
            under_volume_guard_fraction=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_FRACTION",
                cls.under_volume_guard_fraction,
            ),
            under_volume_inverse_gate=_env_bool(
                "NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_INVERSE_GATE",
                cls.under_volume_inverse_gate,
            ),
            custom_loss_gate_threshold=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_THRESHOLD",
                cls.custom_loss_gate_threshold,
            ),
            custom_loss_gate_temperature=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_TEMPERATURE",
                cls.custom_loss_gate_temperature,
            ),
            custom_loss_gate_min_scale=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_MIN_SCALE",
                cls.custom_loss_gate_min_scale,
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    v = value.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return bool(int(v))


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
