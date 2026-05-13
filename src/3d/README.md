3D training experiments live here.

The baseline 3D nnU-Net training currently runs through `scripts/3d/train_nnunet_3d.sh`.
Future fine-tuning experiments should be organized under this folder and should
keep inference/postprocessing comparable across variants.

The first fine-tuning experiment is implemented as a repo-local nnU-Net trainer:

- Control trainer: `nnUNetTrainer_150_DefaultFinetune_50epochs`
- Control entry point: `scripts/3d/train_3d_default_finetune.sh`
- Control output root: `results_3d_default_finetune/`
- Trainer: `nnUNetTrainer_150_BoundaryOverseg_50epochs`
- Implementation: `boundary_shape/`
- nnU-Net discovery shim: `nnunetv2/training/nnUNetTrainer/`
- Entry point: `scripts/3d/train_3d_boundary_shape.sh`
- Boundary/shape output root: `results_3d_boundary_shape_runs/` (saved-good run:
  `20260504_083549_saved_good_boundary/`; новые раны — с timestamp в имени каталога)
- Starts from: `nnUNetTrainer_150__nnUNetPlans_3d_midres125__3d_fullres/fold_0/checkpoint_final.pth`
- Changes: loss only; architecture, plans, input channels, and inference stay aligned with the 150-epoch baseline.

Both launch scripts read the baseline checkpoint from `BASE_NNUNET_RESULTS`
(default: `nnUNet_results/`) and write their own nnU-Net output tree under the
experiment-specific result root.

After training, full-volume predictions are written to `fold_*/validation/`.
Both `train_3d_default_finetune.sh` and `train_3d_boundary_shape.sh` pass
**`--val_best`** by default (validate with `checkpoint_best.pth`). Set
`NNUNET_VALIDATION_WITH_BEST=0` to validate with final-epoch weights.

Compare the control trainer against `boundary_shape` to separate the effect of
additional training time from the effect of the boundary/over-segmentation loss.

Planned experiment groups:

- `boundary_shape`: boundary-aware and over-segmentation-aware losses.
- `default_finetune`: default-loss fine-tuning control.
- `multiwindow`: multi-HU-window 3D input variants.
  - Trainer: `nnUNetTrainer_150_MultiWindowRefine_50epochs` (`multiwindow/`)
  - Scripts: `scripts/3d/cache_tumor_prob_for_multiwindow.py`, `scripts/3d/train_3d_multiwindow_refinement.sh`, `scripts/3d/infer_multiwindow_refinement_3d.py`, `scripts/3d/infer_fuse_softmax_blend.py`
  - Output root: `results_3d_multiwindow_runs/<YYYYMMDD_HHMMSS>_multiwindow_3d_fullres/` (default `RUN_NAME`)
  - See `multiwindow/README.md`.
- `hard_negative`: hard-negative and component-aware fine-tuning.
