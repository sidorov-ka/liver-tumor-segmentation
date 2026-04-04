#!/usr/bin/env bash
# Train MultiviewUNet2d (stage 2) on the same .npz export as coarse_to_fine.
# Run nnU-Net training first (scripts/train.sh), then export, then this script.
#
# Default export dir: <repo>/refinement_export/fold0 — override with:
#   EXPORT_DIR=/path/to/slices bash scripts/train_multiview.sh
# Interpreter: PYTHON=python3 (default) or PYTHON=python
# Extra args are passed through: bash scripts/train_multiview.sh --epochs 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/refinement_export/fold0}"

PYTHON="${PYTHON:-python3}"
exec "${PYTHON}" "${REPO_ROOT}/scripts/train_multiview.py" --export-dir "${EXPORT_DIR}" "$@"
