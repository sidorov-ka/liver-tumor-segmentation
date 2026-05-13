from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DefaultFinetuneConfig:
    """Environment-configurable knobs for default-loss fine-tuning."""

    num_epochs: int = 50
    initial_lr: float = 1e-3

    @classmethod
    def from_env(cls) -> "DefaultFinetuneConfig":
        return cls(
            num_epochs=_env_int("NNUNET_DEFAULT_FINETUNE_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_DEFAULT_FINETUNE_LR", cls.initial_lr),
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
