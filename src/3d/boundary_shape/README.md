Boundary/shape-aware 3D fine-tuning.

Use this folder for experiments that keep the 3D nnU-Net architecture mostly
unchanged and modify the objective to reduce boundary over-segmentation and
excess tumor volume.

Current experiment:

- `nnUNetTrainer_150_BoundaryOverseg_50epochs`
- wraps the default nnU-Net Dice+CE loss
- adds a tumor boundary-ring BCE+Dice term
- adds hard-negative tumor FP penalties outside the GT liver and inside the GT
  liver, excluding a small dilated ring around GT tumors
- applies a smooth tumor-burden gate to all custom terms, so large tumors stay
  close to the default nnU-Net Dice+CE fine-tune

Files:

- `config.py`: environment-driven experiment parameters.
- `losses.py`: additive boundary and over-segmentation loss wrapper.
- `trainer.py`: nnU-Net trainer subclass used for fine-tuning.

Main knobs are environment variables:

- `NNUNET_BOUNDARY_OVERSEG_EPOCHS` default `50`
- `NNUNET_BOUNDARY_OVERSEG_LR` default `1e-3`
- `NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT` default `0.10`
- `NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT` default `0.05`
- `NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_FP_WEIGHT` default `4.0`
- `NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_FP_WEIGHT` default `0.5`
- `NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS` default `2`
- `NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_IGNORE_RADIUS` default `2`
- `NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_IGNORE_RADIUS` default `4`
- `NNUNET_BOUNDARY_OVERSEG_OUTSIDE_LIVER_TOPK_FRACTION` default `0.01`
- `NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_TOPK_FRACTION` default `0.002`
- `NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_THRESHOLD` default `0.02`
- `NNUNET_BOUNDARY_OVERSEG_INSIDE_LIVER_VOLUME_GUARD_MIN_SCALE` default `0.10`
- `NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_WEIGHT` default `0.05`
- `NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_ALPHA` default `0.30`
- `NNUNET_BOUNDARY_OVERSEG_TVERSKY_GUARD_BETA` default `0.70`
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_THRESHOLD` default `0.02`
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_LARGE_TUMOR_MAX_THRESHOLD` default `0.10`
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_MIN_SCALE` default `1.0`
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_START_EPOCH` default `-1` (disabled; if
  `>= 0`, use floor `1.0` before this epoch, then ramp down to `ADAPTIVE_FP_MIN_SCALE`)
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_FP_SCHEDULE_RAMP_EPOCHS` default `0` (0 = jump to target
  at start epoch; `> 0` = linear ramp over that many epochs)
- `NNUNET_BOUNDARY_OVERSEG_ADAPTIVE_IGNORE_EXTRA_RADIUS` default `0`
- `NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_WEIGHT` default `0.0`
- `NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_THRESHOLD` default `0.05`
- `NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_GUARD_FRACTION` default `0.85`
- `NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_INVERSE_GATE` default `0` (if `1`, under-volume uses weight `(1 - custom_loss_gate)` per sample so large-tumor patches get recall pressure; default `0` keeps weight `custom_loss_gate` as before)
- `NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_THRESHOLD` default `0.04`
- `NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_TEMPERATURE` default `0.015`
- `NNUNET_BOUNDARY_OVERSEG_CUSTOM_LOSS_GATE_MIN_SCALE` default `0.0`
- `NNUNET_BOUNDARY_OVERSEG_BOUNDARY_START_EPOCH` default `5`
- `NNUNET_BOUNDARY_OVERSEG_FP_START_EPOCH` default `10`
- `NNUNET_BOUNDARY_OVERSEG_RAMP_EPOCHS` default `10`

The inside-liver FP penalty is volume-aware: when GT tumor occupies a large
share of the foreground (`tumor / (tumor + liver)`), the in-liver hard-negative
loss is scaled by `threshold / tumor_fraction`, clamped to
`[min_scale, 1.0]`. This keeps FP suppression active for small tumors while
reducing the risk of collapsing very large tumors.

A recall-biased Tversky guard is enabled during the FP phase. It uses soft
tumor TP/FP/FN with `beta > alpha`, so false negatives are penalized more than
false positives while the hard-negative terms suppress excess tumor islands.

The size-gated mode starts from the saved-good Tversky defaults. It computes a
per-sample `tumor_fraction = tumor / (tumor + liver)` and uses
`1 - sigmoid((tumor_fraction - threshold) / temperature)` as a multiplier for
all custom additive loss terms. Small tumors keep BoundaryOverseg behavior;
large tumors mostly fall back to default Dice+CE fine-tuning. Optional
**inverse-gated under-volume** (`NNUNET_BOUNDARY_OVERSEG_UNDER_VOLUME_INVERSE_GATE=1`)
reweights the under-volume guard by `(1 - custom_loss_gate)` so anti-undershoot
pressure applies mainly on those large-tumor patches (without turning on the
full adaptive-large FP relaxation path).

Presets:

- `presets/size_gated_boundary_2026_05_09.env`: current size-gated defaults.
- `presets/adaptive_large_tumor_2026_05_09.env`: aggressive adaptive-large-tumor run.
- `presets/tversky_guard_2026_05_04.env`: saved-good Tversky run.

Training is launched via `scripts/train_3d_boundary_shape.sh`, which forwards
`--val_best` to nnU-Net by default so `fold_*/validation/` uses
`checkpoint_best.pth`. Set `NNUNET_VALIDATION_WITH_BEST=0` for final-epoch
validation.

The shell wrapper writes new runs under `results_3d_boundary_shape_runs/<RUN_NAME>`.
Reference result folders in this repo’s workflow: `20260504_083549_saved_good_boundary/`,
`20260509_131406_boundary_adaptive_large_tumor/`, `20260509_160927_boundary_size_gated/`
(saved-good, adaptive-large-tumor, size-gated). Re-validate with
`scripts/revalidate_3d_boundary_shape_runs.sh`.
