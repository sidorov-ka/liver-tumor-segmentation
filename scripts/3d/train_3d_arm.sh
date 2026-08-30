#!/usr/bin/env bash
# Internal launcher — use scripts/3d/train.sh instead of calling this directly.
#
#   bash scripts/3d/train_3d_arm.sh <results_subdir> <trainer_class> [--skip-preprocess]
#
# Env: FOLD, PLANS, NNUNET_VALIDATION_WITH_BEST, SKIP_NNUNET_PREPROCESS
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

# shellcheck source=resolve_python.sh
source "${REPO_ROOT}/scripts/3d/resolve_python.sh"
PYTHON_BIN_SET="${PYTHON_BIN:-}"
require_nnunet_python "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/${RESULTS_SUBDIR}}"
export nnUNet_results="${RESULTS_ROOT}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD="${FOLD:-0}"
readonly PLANS="${PLANS:-nnUNetPlans}"
readonly PLANS_JSON="${nnUNet_preprocessed}/Dataset001_LiverTumor/${PLANS}.json"

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
echo "Arm trainer=${TRAINER} fold=${FOLD} plans=${PLANS} -> ${RESULTS_ROOT}"

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  PLAN_ARGS=()
  if [[ "${NNUNET_CLEAN_PREPROCESS:-0}" == "1" ]]; then
    PLAN_ARGS+=(--clean)
  fi
  "${PLAN_BIN}" \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity \
    -c "${CONFIGURATION}" \
    -npfp "${NNUNET_FINGERPRINT_PROCESSES:-4}" \
    -np "${NNUNET_PREPROCESS_PROCESSES:-4}" \
    "${PLAN_ARGS[@]}"
else
  if [[ ! -f "${PLANS_JSON}" ]]; then
    echo "Missing preprocessed plans: ${PLANS_JSON}" >&2
    echo "Run once: bash scripts/3d/train.sh plan" >&2
    exit 1
  fi
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid)."
fi

VAL_BEST_ARGS=()
if [[ "${NNUNET_VALIDATION_WITH_BEST:-1}" != "0" ]]; then
  VAL_BEST_ARGS=(--val_best)
  echo "Post-training validation will use checkpoint_best.pth (fold_${FOLD}/validation/)."
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  "${VAL_BEST_ARGS[@]}"
