#!/usr/bin/env bash
# Control fine-tune for the 150-epoch 3D nnU-Net baseline with default loss.
#
# Optional overrides:
#   NNUNET_DEFAULT_FINETUNE_EPOCHS=50
#   NNUNET_DEFAULT_FINETUNE_LR=1e-3
# Skip heavy preprocessing if cache is already valid:
#   bash scripts/3d/train_3d_default_finetune.sh --skip-preprocess
#   SKIP_NNUNET_PREPROCESS=1 bash scripts/3d/train_3d_default_finetune.sh
#
# Full-volume fold_*/validation uses checkpoint_best.pth (nnU-Net --val_best) by default.
# For validation from the final epoch instead:
#   NNUNET_VALIDATION_WITH_BEST=0 bash scripts/3d/train_3d_default_finetune.sh --skip-preprocess
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
readonly TARGET_SPACING=(1.25 1.0 1.0)
readonly PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${BASE_NNUNET_RESULTS}/Dataset001_LiverTumor/nnUNetTrainer_150__${PLANS}__${CONFIGURATION}/fold_0/checkpoint_final.pth}"

SKIP_PREPROCESS=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
elif [[ "$#" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi
if [[ "${SKIP_NNUNET_PREPROCESS:-0}" == "1" ]]; then
  SKIP_PREPROCESS=1
fi

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "Missing pretrained weights: ${PRETRAINED_WEIGHTS}" >&2
  exit 1
fi

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}" \
    --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid; use after a full preprocess run)."
fi

VAL_BEST_ARGS=()
if [[ "${NNUNET_VALIDATION_WITH_BEST:-1}" != "0" ]]; then
  VAL_BEST_ARGS=(--val_best)
  echo "nnU-Net post-training validation will use checkpoint_best.pth (fold_${FOLD}/validation/)."
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}" \
  "${VAL_BEST_ARGS[@]}"
