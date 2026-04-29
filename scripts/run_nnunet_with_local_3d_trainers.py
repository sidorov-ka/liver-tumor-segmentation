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


def main() -> None:
    _register_local_trainers()
    from nnunetv2.run.run_training import run_training_entry

    run_training_entry()


if __name__ == "__main__":
    sys.exit(main())
