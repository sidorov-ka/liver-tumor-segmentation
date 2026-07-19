#!/usr/bin/env bash
# Shared from-scratch trainer launcher for the four loss arms.
#
# Usage (prefer the thin wrappers):
#   bash scripts/3d/train_3d_baseline.sh
#   FOLD=0 bash scripts/3d/train_3d_anatomical.sh --skip-preprocess
#
# Env:
#   FOLD                 default 0 (use 0..4 for 5-fold)
#   NNUNET_VALIDATION_WITH_BEST  default 1 (-> --val_best)
#   SKIP_NNUNET_PREPROCESS=1 or --skip-preprocess
set -euo pipefail

if [[ "${#}" -lt 2 ]]; then
  echo "Usage: $0 <results_subdir> <trainer_class> [--skip-preprocess]" >&2
  exit 2
fi

RESULTS_SUBDIR="$1"
TRAINER="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/${RESULTS_SUBDIR}}"
export nnUNet_results="${RESULTS_ROOT}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD="${FOLD:-0}"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly TARGET_SPACING=(1.25 1.0 1.0)

SKIP_PREPROCESS=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
elif [[ "${#}" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi
if [[ "${SKIP_NNUNET_PREPROCESS:-0}" == "1" ]]; then
  SKIP_PREPROCESS=1
fi

if [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "FOLD must be 0..4, got: ${FOLD}" >&2
  exit 2
fi

mkdir -p "${RESULTS_ROOT}"
echo "Arm trainer=${TRAINER} fold=${FOLD} -> ${RESULTS_ROOT}"

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}" \
    --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid)."
fi

VAL_BEST_ARGS=()
if [[ "${NNUNET_VALIDATION_WITH_BEST:-1}" != "0" ]]; then
  VAL_BEST_ARGS=(--val_best)
  echo "Post-training validation will use checkpoint_best.pth (fold_${FOLD}/validation/)."
fi

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  "${VAL_BEST_ARGS[@]}"
