#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_3D_SRC = REPO_ROOT / "src" / "3d"
LOCAL_NNUNET_ROOT = REPO_ROOT / "src" / "3d" / "nnunetv2"
LOCAL_TRAINER_ROOT = LOCAL_NNUNET_ROOT / "training" / "nnUNetTrainer"


def _register_local_trainers() -> None:
    """Make repo-local 3D trainers discoverable by nnU-Net's trainer lookup."""
    import nnunetv2
    import nnunetv2.training.nnUNetTrainer as trainer_pkg
    import nnunetv2.utilities.find_class_by_name as finder

    if not LOCAL_TRAINER_ROOT.is_dir():
        raise RuntimeError(f"Local trainer folder not found: {LOCAL_TRAINER_ROOT}")

    sys.path.insert(0, str(LOCAL_3D_SRC))
    trainer_pkg.__path__.insert(0, str(LOCAL_TRAINER_ROOT))
    _patch_trainer_lookup(nnunetv2, finder)


def _patch_trainer_lookup(nnunetv2_module: ModuleType, finder: ModuleType) -> None:
    installed_nnunet_root = Path(nnunetv2_module.__path__[0])
    installed_trainer_root = installed_nnunet_root / "training" / "nnUNetTrainer"
    original_find = finder.recursive_find_python_class

    def find_with_local_trainers(folder: str, class_name: str, current_module: str):
        if (
            Path(folder).resolve() == installed_trainer_root.resolve()
            and current_module == "nnunetv2.training.nnUNetTrainer"
        ):
            trainer = original_find(str(LOCAL_TRAINER_ROOT), class_name, current_module)
            if trainer is not None:
                return trainer
        return original_find(folder, class_name, current_module)

    finder.recursive_find_python_class = find_with_local_trainers


def _load_full_pretrained_weights(network, fname: str, verbose: bool = False) -> None:
    """Load compatible nnU-Net weights without dropping segmentation heads."""
    import os

    import torch
    import torch.distributed as dist
    from torch._dynamo import OptimizedModule
    from torch.nn.parallel import DistributedDataParallel as DDP

    map_location = (
        torch.device("cuda", dist.get_rank())
        if dist.is_initialized()
        else torch.device("cpu")
    )
    saved_model = torch.load(fname, map_location=map_location, weights_only=False)
    pretrained_dict = saved_model["network_weights"]

    mod = network.module if isinstance(network, DDP) else network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    partial = os.environ.get("NNUNET_PRETRAINED_PARTIAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if partial:
        matched = 0
        skipped_shape = 0
        new_sd = {k: v.clone() for k, v in model_dict.items()}
        for k in new_sd:
            if k in pretrained_dict and pretrained_dict[k].shape == new_sd[k].shape:
                new_sd[k] = pretrained_dict[k].to(dtype=new_sd[k].dtype, device=new_sd[k].device)
                matched += 1
            elif k in pretrained_dict:
                skipped_shape += 1
        print(
            "################### Partial pretrained load from",
            fname,
            f"(matched_tensors={matched}, skipped_shape_mismatch={skipped_shape}) ###################",
        )
        mod.load_state_dict(new_sd, strict=True)
        return

    missing_keys = sorted(set(model_dict) - set(pretrained_dict))
    unexpected_keys = sorted(set(pretrained_dict) - set(model_dict))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint does not exactly match the network. "
            f"Missing keys: {missing_keys[:10]} "
            f"{'...' if len(missing_keys) > 10 else ''}; "
            f"unexpected keys: {unexpected_keys[:10]} "
            f"{'...' if len(unexpected_keys) > 10 else ''}"
        )

    shape_mismatches = [
        (key, pretrained_dict[key].shape, model_dict[key].shape)
        for key in model_dict
        if pretrained_dict[key].shape != model_dict[key].shape
    ]
    if shape_mismatches:
        sample = ", ".join(
            f"{key}: pretrained {src}, model {dst}"
            for key, src, dst in shape_mismatches[:10]
        )
        raise RuntimeError(f"Pretrained checkpoint shape mismatch: {sample}")

    print(
        "################### Loading full pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print("Below is the list of loaded blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")
    mod.load_state_dict(pretrained_dict)


def _patch_pretrained_weight_loading() -> None:
    """Fine-tuning should start from the exact checkpoint, including seg_layers."""
    import nnunetv2.run.run_training as run_training

    run_training.load_pretrained_weights = _load_full_pretrained_weights


def main() -> None:
    _register_local_trainers()
    _patch_pretrained_weight_loading()
    from nnunetv2.run.run_training import run_training_entry

    run_training_entry()


if __name__ == "__main__":
    sys.exit(main())
