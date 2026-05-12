from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MultiWindowRefineConfig:
    """Env-driven settings for 4-channel (prob + Lim windows) refinement."""

    num_epochs: int = 50
    initial_lr: float = 1e-3
    tumor_label: int = 2
    tversky_weight: float = 0.25
    tversky_alpha: float = 0.30
    tversky_beta: float = 0.70
    fingerprint_json: str = ""

    @classmethod
    def from_env(cls, repo_root: str | None = None) -> "MultiWindowRefineConfig":
        default_fp = ""
        if repo_root:
            default_fp = os.path.join(
                repo_root,
                "nnUNet_preprocessed",
                "Dataset001_LiverTumor",
                "dataset_fingerprint.json",
            )
        fp = os.environ.get("NNUNET_MW_FINGERPRINT_JSON", default_fp or "")
        return cls(
            num_epochs=_env_int("NNUNET_MW_EPOCHS", cls.num_epochs),
            initial_lr=_env_float("NNUNET_MW_LR", cls.initial_lr),
            tumor_label=_env_int("NNUNET_MW_TUMOR_LABEL", cls.tumor_label),
            tversky_weight=_env_float("NNUNET_MW_TVERSKY_WEIGHT", cls.tversky_weight),
            tversky_alpha=_env_float("NNUNET_MW_TVERSKY_ALPHA", cls.tversky_alpha),
            tversky_beta=_env_float("NNUNET_MW_TVERSKY_BETA", cls.tversky_beta),
            fingerprint_json=fp,
        )


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)
