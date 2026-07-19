3D training experiments live here.

The baseline 3D nnU-Net training runs through `scripts/3d/train_nnunet_3d.sh`.
Fine-tuning experiments keep architecture, plans, and inference aligned; only the
loss (and training schedule) changes.

## Experiments

### Control

- Trainer: `nnUNetTrainer_150_DefaultFinetune_50epochs`
- Code: `default_finetune/`
- Entry: `scripts/3d/train_3d_default_finetune.sh`
- Output: `results_3d_default_finetune/`

### Boundary / shape loss (three runs)

Trainer: `nnUNetTrainer_150_BoundaryOverseg_50epochs` (`boundary_shape/`).  
Entry: `scripts/3d/train_3d_boundary_shape.sh`.  
Starts from: `nnUNetTrainer_150__nnUNetPlans_3d_midres125__3d_fullres/fold_0/checkpoint_final.pth`.

| Run | Preset | Output folder |
|-----|--------|---------------|
| Saved-good (Tversky guard) | `presets/tversky_guard_2026_05_04.env` | `results_3d_boundary_shape_runs/20260504_083549_saved_good_boundary/` |
| Adaptive large tumor | `presets/adaptive_large_tumor_2026_05_09.env` | `results_3d_boundary_shape_runs/20260509_131406_boundary_adaptive_large_tumor/` |
| Size-gated | `presets/size_gated_boundary_2026_05_09.env` | `results_3d_boundary_shape_runs/20260509_160927_boundary_size_gated/` |

Compare control vs each boundary run to separate extra training time from the
custom loss.

Both fine-tune scripts pass **`--val_best`** by default. Set
`NNUNET_VALIDATION_WITH_BEST=0` to validate with final-epoch weights.
