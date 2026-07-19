3D from-scratch loss arms (matched nnU-Net settings, 500 epochs).

| Arm | Package | Trainer |
|-----|---------|---------|
| Baseline | `baseline/` | `nnUNetTrainer_500_Baseline` |
| Anatomical | `anatomical/` | `nnUNetTrainer_500_Anatomical` |
| Boundary / HD-soft | `boundary_hd/` | `nnUNetTrainer_500_BoundaryHD` |
| Topology | `topology/` | `nnUNetTrainer_500_Topology` |

Shared helpers: `common/loss_utils.py`.  
Base epoch class: `nnunet_trainer_500_compat.py`.

Launch via `scripts/3d/train_3d_*.sh` (`FOLD=0..4`).
No fine-tuning and no gradual loss enablement — additive terms are on from epoch 0.
