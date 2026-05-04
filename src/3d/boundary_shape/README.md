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
