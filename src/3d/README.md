3D training experiments live here.

The baseline 3D nnU-Net training runs through `scripts/3d/train_nnunet_3d.sh`.
Fine-tuning experiments are organized under this folder and keep inference/postprocessing
comparable across variants.

## Implemented experiments

- **Control:** `nnUNetTrainer_150_DefaultFinetune_50epochs`
  - Entry: `scripts/3d/train_3d_default_finetune.sh`
  - Output: `results_3d_default_finetune/`

- **Boundary/shape loss:** `nnUNetTrainer_150_BoundaryOverseg_50epochs`
  - Implementation: `boundary_shape/`
  - Entry: `scripts/3d/train_3d_boundary_shape.sh`
  - Output: `results_3d_boundary_shape_runs/` (saved-good run: `20260504_083549_saved_good_boundary/`)
  - Starts from: `nnUNetTrainer_150__nnUNetPlans_3d_midres125__3d_fullres/fold_0/checkpoint_final.pth`
  - Changes: loss only; architecture, plans, input channels, and inference stay aligned with the 150-epoch baseline.

Both launch scripts read the baseline checkpoint from `BASE_NNUNET_RESULTS`
(default: `nnUNet_results/`) and write their own nnU-Net output tree under the
experiment-specific result root.

After training, full-volume predictions are written to `fold_*/validation/`.
Both scripts pass **`--val_best`** by default (validate with `checkpoint_best.pth`).
Set `NNUNET_VALIDATION_WITH_BEST=0` to validate with final-epoch weights.

Compare the control trainer against `boundary_shape` to separate the effect of
additional training time from the effect of the boundary/over-segmentation loss.

## Planned

- `hard_negative`: hard-negative and component-aware fine-tuning (placeholder: `hard_negative/README.md`).
