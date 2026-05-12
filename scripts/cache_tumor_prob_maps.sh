#!/usr/bin/env bash
# Full-volume tumour probability maps for 3D multi-window refinement (NNUNET_MW_PROB_DIR).
# Writes one <case_id>.npz per case with float16 array ``prob`` (1, Z, Y, X), nnU-Net preprocessed grid.
# Default model: BoundaryOverseg size-gated 3d_fullres (same run as former export_3d_finetune preset).
#
# Usage:
#   bash scripts/cache_tumor_prob_maps.sh
#   PROB_MODEL_DIR=/path/to/nnUNetTrainer_...__3d_fullres OUTPUT_DIR=/path/to/cache bash scripts/cache_tumor_prob_maps.sh
# Extra args after -- are passed to scripts/cache_tumor_prob_for_multiwindow.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

readonly BO="nnUNetTrainer_150_BoundaryOverseg_50epochs__nnUNetPlans_3d_midres125__3d_fullres"
readonly DEFAULT_MODEL="${REPO_ROOT}/results_3d_boundary_shape_runs/20260509_160927_boundary_size_gated/Dataset001_LiverTumor/${BO}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${NNUNET_MW_PROB_DIR:-${REPO_ROOT}/results_3d_multiwindow_runs/_prob_cache_fold0_size_gated}}"

mkdir -p "${OUTPUT_DIR}"

EXTRA=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA=("$@")
elif [[ "$#" -gt 0 ]]; then
  echo "usage: $0 [-- cache_tumor_prob_for_multiwindow.py args]" >&2
  exit 2
fi

echo "model-dir: ${PROB_MODEL_DIR:-${DEFAULT_MODEL}}"
echo "output-dir: ${OUTPUT_DIR}"

exec "${PYTHON}" "${REPO_ROOT}/scripts/cache_tumor_prob_for_multiwindow.py" \
  --model-dir "${PROB_MODEL_DIR:-${DEFAULT_MODEL}}" \
  --output-dir "${OUTPUT_DIR}" \
  --fold 0 \
  --split all \
  "${EXTRA[@]}"
