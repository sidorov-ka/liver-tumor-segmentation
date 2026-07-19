#!/usr/bin/env bash
# Arm 3 — soft Hausdorff additive loss, 500 epochs from scratch.
#   bash scripts/3d/train_3d_boundary_hd.sh --skip-preprocess
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_3d_arm.sh" \
  "results_3d_boundary_hd" \
  "nnUNetTrainer_500_BoundaryHD" \
  "$@"
