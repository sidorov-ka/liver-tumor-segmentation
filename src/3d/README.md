3D training experiments live here.

The baseline 3D nnU-Net training currently runs through `scripts/train_3d.sh`.
Future fine-tuning experiments should be organized under this folder and should
keep inference/postprocessing comparable across variants.

The first fine-tuning experiment is implemented as a repo-local nnU-Net trainer:

- Control trainer: `nnUNetTrainer_150_DefaultFinetune_50epochs`
- Control entry point: `scripts/train_3d_default_finetune.sh`
- Control output root: `results_3d_default_finetune/`
- Trainer: `nnUNetTrainer_150_BoundaryOverseg_50epochs`
- Implementation: `boundary_shape/`
- nnU-Net discovery shim: `nnunetv2/training/nnUNetTrainer/`
- Entry point: `scripts/train_3d_boundary_shape.sh`
- Boundary/shape output root: `results_3d_boundary_shape/`
- Starts from: `nnUNetTrainer_150__nnUNetPlans_3d_midres125__3d_fullres/fold_0/checkpoint_final.pth`
- Changes: loss only; architecture, plans, input channels, and inference stay aligned with the 150-epoch baseline.

Both launch scripts read the baseline checkpoint from `BASE_NNUNET_RESULTS`
(default: `nnUNet_results/`) and write their own nnU-Net output tree under the
experiment-specific result root.

Compare the control trainer against `boundary_shape` to separate the effect of
additional training time from the effect of the boundary/over-segmentation loss.

Planned experiment groups:

- `boundary_shape`: boundary-aware and over-segmentation-aware losses.
- `default_finetune`: default-loss fine-tuning control.
- `multiwindow`: multi-HU-window 3D input variants.
- `hard_negative`: hard-negative and component-aware fine-tuning.
