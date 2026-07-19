#!/usr/bin/env bash
# Plan + preprocess Dataset001 for 3d_fullres (shared by all arms).
#   bash scripts/3d/preprocess_nnunet_3d.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly TARGET_SPACING=(1.25 1.0 1.0)

nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
  -overwrite_target_spacing "${TARGET_SPACING[@]}" \
  -overwrite_plans_name "${PLANS}" \
  --clean

echo "Preprocess done. Plans=${PLANS}, configuration=${CONFIGURATION}"
