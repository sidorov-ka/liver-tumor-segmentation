# Arm 2 — Anatomical

Wraps Dice+CE with:

- tumor **boundary-ring** BCE+Dice
- **hard-negative FP** outside GT liver and inside GT liver (ignore ring around GT tumor)
- recall-biased **Tversky** term

No size gates, adaptive FP scaling, under-volume term, or epoch curriculum —
all additive weights are active from epoch 0.

- Trainer: `nnUNetTrainer_500_Anatomical`
- Entry: `bash scripts/3d/train.sh anatomical`
- Output: `results_3d_anatomical/`

Main env knobs (`NNUNET_ANATOMICAL_*`): `EPOCHS`, `LR`, `BOUNDARY_WEIGHT`,
`OVERSEG_WEIGHT`, `OUTSIDE_LIVER_FP_WEIGHT`, `INSIDE_LIVER_FP_WEIGHT`,
`TVERSKY_WEIGHT`, `TVERSKY_ALPHA`, `TVERSKY_BETA`.
