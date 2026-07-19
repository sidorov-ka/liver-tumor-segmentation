from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class TopologyConfig:
    """Soft clDice topology-aware additive loss."""

    num_epochs: int = 500
    initial_lr: float = 1e-2
    tumor_label: int = 2
    liver_label: int = 1
    cldice_weight: float = 0.10
    skeleton_iterations: int = 10

    @classmethod
    def from_env(cls) -> "TopologyConfig":
        return cls(
            num_epochs=_env_int("NNUNET_TOPOLOGY_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_TOPOLOGY_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_TOPOLOGY_TUMOR_LABEL", cls.tumor_label),
            liver_label=_env_int("NNUNET_TOPOLOGY_LIVER_LABEL", cls.liver_label),
            cldice_weight=_env_float("NNUNET_TOPOLOGY_CLDICE_WEIGHT", cls.cldice_weight),
            skeleton_iterations=_env_int(
                "NNUNET_TOPOLOGY_SKELETON_ITERATIONS",
                cls.skeleton_iterations,
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
