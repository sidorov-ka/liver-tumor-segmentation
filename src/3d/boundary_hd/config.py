from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BoundaryHDConfig:
    """Soft Hausdorff / boundary-distance additive loss."""

    num_epochs: int = 500
    initial_lr: float = 1e-2
    tumor_label: int = 2
    liver_label: int = 1
    hd_weight: float = 0.10
    distance_iterations: int = 8

    @classmethod
    def from_env(cls) -> "BoundaryHDConfig":
        return cls(
            num_epochs=_env_int("NNUNET_BOUNDARY_HD_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_BOUNDARY_HD_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_BOUNDARY_HD_TUMOR_LABEL", cls.tumor_label),
            liver_label=_env_int("NNUNET_BOUNDARY_HD_LIVER_LABEL", cls.liver_label),
            hd_weight=_env_float("NNUNET_BOUNDARY_HD_WEIGHT", cls.hd_weight),
            distance_iterations=_env_int(
                "NNUNET_BOUNDARY_HD_DISTANCE_ITERATIONS",
                cls.distance_iterations,
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
