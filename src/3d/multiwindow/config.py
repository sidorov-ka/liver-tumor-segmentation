from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MultiWindowRefineConfig:
    """Env-driven settings for 4-channel (prob + Lim windows) refinement."""

    num_epochs: int = 150
    initial_lr: float = 1e-3
    iterations_per_epoch: int = 100
    val_iterations_per_epoch: int = 15
    # Cap plans batch_size (often 2) to reduce host RAM; NNUNET_MW_MAX_BATCH_SIZE=0 disables.
    max_batch_size: int | None = 1
    # NonDetMultiThreadedAugmenter queue depth (default nnU-Net uses ~6+); lower = less RAM.
    da_num_cached_train: int = 2
    da_num_cached_val: int = 2
    # Pinned CPU memory for CUDA copies; False saves RAM on tight systems.
    pin_memory: bool = False
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
            iterations_per_epoch=_env_int("NNUNET_MW_ITER_PER_EPOCH", cls.iterations_per_epoch),
            val_iterations_per_epoch=_env_int("NNUNET_MW_VAL_ITER", cls.val_iterations_per_epoch),
            max_batch_size=_env_optional_pos_int("NNUNET_MW_MAX_BATCH_SIZE", 1),
            da_num_cached_train=_env_int("NNUNET_MW_DA_NUM_CACHED_TRAIN", cls.da_num_cached_train),
            da_num_cached_val=_env_int("NNUNET_MW_DA_NUM_CACHED_VAL", cls.da_num_cached_val),
            pin_memory=_env_bool("NNUNET_MW_PIN_MEMORY", cls.pin_memory),
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


def _env_optional_pos_int(name: str, default: int | None) -> int | None:
    """Unset -> default; 0 or negative -> None (no cap)."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    i = int(v)
    if i <= 0:
        return None
    return i


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)
