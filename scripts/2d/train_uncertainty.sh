#!/usr/bin/env bash
# Train UncertaintyUNet2d (stage 2) on the same .npz export as coarse_to_fine / multiview.
#
# Default export dir: <repo>/refinement_export/fold0 — override:
#   EXPORT_DIR=/path/to/slices bash scripts/2d/train_uncertainty.sh
# Interpreter: PYTHON=python3 (default) or PYTHON=python
# Extra args are passed through: bash scripts/2d/train_uncertainty.sh --epochs 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/refinement_export/fold0}"

PYTHON="${PYTHON:-python3}"
exec "${PYTHON}" "${REPO_ROOT}/scripts/2d/train_uncertainty.py" --export-dir "${EXPORT_DIR}" "$@"
