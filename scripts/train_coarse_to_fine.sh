#!/usr/bin/env bash
# Train coarse_to_fine (stage 2) on slices from scripts/export.py.
# Run nnU-Net training first (scripts/train.sh), then export, then this script.
#
# Default export dir: <repo>/refinement_export/fold0 — override with:
#   EXPORT_DIR=/path/to/slices bash scripts/train_coarse_to_fine.sh
# Interpreter: PYTHON=python3 (default) or PYTHON=python
# Extra args are passed through: bash scripts/train_coarse_to_fine.sh --epochs 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/refinement_export/fold0}"

PYTHON="${PYTHON:-python3}"
exec "${PYTHON}" "${REPO_ROOT}/scripts/train_coarse_to_fine.py" --export-dir "${EXPORT_DIR}" "$@"
