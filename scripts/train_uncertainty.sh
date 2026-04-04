#!/usr/bin/env bash
# Train UncertaintyUNet2d (stage 2) on the same .npz export as coarse_to_fine / multiview.
#
# Default export dir: <repo>/coarse_to_fine_export/slices — override with:
#   EXPORT_DIR=/path/to/slices bash scripts/train_uncertainty.sh
# Extra args are passed through: bash scripts/train_uncertainty.sh --epochs 50
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

EXPORT_DIR="${EXPORT_DIR:-${REPO_ROOT}/coarse_to_fine_export/slices}"

exec python "${REPO_ROOT}/scripts/train_uncertainty.py" --export-dir "${EXPORT_DIR}" "$@"
