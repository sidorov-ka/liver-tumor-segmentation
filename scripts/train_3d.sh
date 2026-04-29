#!/usr/bin/env bash
# Stage 1 only: nnU-Net v2 for Dataset001_LiverTumor — plan, preprocess (3d_fullres, splits_final.json), train fold 0.
# Same workflow as scripts/train.sh but 3D full resolution instead of 2d.
# Skip heavy preprocessing if cache is already valid: bash scripts/train_3d.sh --skip-preprocess
# or: SKIP_NNUNET_PREPROCESS=1 bash scripts/train_3d.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly TARGET_SPACING=(1.25 1.0 1.0)

SKIP_PREPROCESS=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
fi
if [[ "${SKIP_NNUNET_PREPROCESS:-0}" == "1" ]]; then
  SKIP_PREPROCESS=1
fi

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}"
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid; use after a full preprocess run)."
fi

nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}"
