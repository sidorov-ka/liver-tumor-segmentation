#!/usr/bin/env bash
# Control fine-tune for the 150-epoch 3D nnU-Net baseline with default loss.
#
# Optional overrides:
#   NNUNET_DEFAULT_FINETUNE_EPOCHS=50
#   NNUNET_DEFAULT_FINETUNE_LR=1e-3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_3d_default_finetune}"
export nnUNet_results="${RESULTS_ROOT}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150_DefaultFinetune_50epochs"
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
