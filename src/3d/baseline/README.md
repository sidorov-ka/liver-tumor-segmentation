# Arm 1 — Baseline

Default nnU-Net **Dice + CE**, train from scratch for 500 epochs.

- Trainer: `nnUNetTrainer_500_Baseline`
- Entry: `bash scripts/3d/train.sh baseline`
- Output: `results_3d_baseline/`

Env overrides: `NNUNET_BASELINE_EPOCHS`, `NNUNET_BASELINE_LR` (default `1e-2`).
