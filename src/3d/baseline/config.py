from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BaselineConfig:
    """Matched from-scratch baseline (default nnU-Net Dice+CE)."""

    num_epochs: int = 500
    initial_lr: float = 1e-2

    @classmethod
    def from_env(cls) -> "BaselineConfig":
        return cls(
            num_epochs=_env_int("NNUNET_BASELINE_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_BASELINE_LR", cls.initial_lr),
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
