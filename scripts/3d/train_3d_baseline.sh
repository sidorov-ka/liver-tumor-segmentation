#!/usr/bin/env bash
# Arm 1 — baseline nnU-Net Dice+CE, 500 epochs from scratch.
#   bash scripts/3d/train_3d_baseline.sh
#   FOLD=0 bash scripts/3d/train_3d_baseline.sh --skip-preprocess
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_3d_arm.sh" \
  "results_3d_baseline" \
  "nnUNetTrainer_500_Baseline" \
  "$@"
