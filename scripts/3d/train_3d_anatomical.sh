#!/usr/bin/env bash
# Arm 2 — anatomical additive loss, 500 epochs from scratch.
#   bash scripts/3d/train_3d_anatomical.sh --skip-preprocess
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_3d_arm.sh" \
  "results_3d_anatomical" \
  "nnUNetTrainer_500_Anatomical" \
  "$@"
