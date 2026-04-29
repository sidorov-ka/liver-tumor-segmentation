#!/usr/bin/env bash
# Fine-tune the 150-epoch 3D nnU-Net baseline with boundary/over-segmentation loss.
#
# Optional overrides:
#   NNUNET_BOUNDARY_OVERSEG_EPOCHS=50
#   NNUNET_BOUNDARY_OVERSEG_LR=1e-3
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT=0.25
#   NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT=0.05
#   NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS=2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_3d_boundary_shape}"
export nnUNet_results="${RESULTS_ROOT}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150_BoundaryOverseg_50epochs"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${BASE_NNUNET_RESULTS}/Dataset001_LiverTumor/nnUNetTrainer_150__${PLANS}__${CONFIGURATION}/fold_0/checkpoint_final.pth}"

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "Missing pretrained weights: ${PRETRAINED_WEIGHTS}" >&2
  exit 1
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}"
