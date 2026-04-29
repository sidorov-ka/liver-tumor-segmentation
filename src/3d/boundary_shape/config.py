from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BoundaryOversegConfig:
    """Environment-configurable knobs for boundary/oversegmentation fine-tuning."""

    num_epochs: int = 50
    initial_lr: float = 1e-3
    tumor_label: int = 2
    boundary_weight: float = 0.25
    overseg_weight: float = 0.05
    boundary_radius: int = 2

    @classmethod
    def from_env(cls) -> "BoundaryOversegConfig":
        return cls(
            num_epochs=_env_int("NNUNET_BOUNDARY_OVERSEG_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_BOUNDARY_OVERSEG_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_BOUNDARY_OVERSEG_TUMOR_LABEL", cls.tumor_label),
            boundary_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT",
                cls.boundary_weight,
            ),
            overseg_weight=_env_float(
                "NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT",
                cls.overseg_weight,
            ),
            boundary_radius=_env_int(
                "NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS",
                cls.boundary_radius,
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
