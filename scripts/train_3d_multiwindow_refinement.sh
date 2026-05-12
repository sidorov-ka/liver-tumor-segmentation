#!/usr/bin/env bash
# Train 3D multi-window refinement (4 inputs: tumour prob + Lim HU windows).
#
# Prerequisites:
#   1) nnU-Net preprocessing for Dataset001 (same as boundary runs).
#   2) Tumour probability cache: bash scripts/cache_tumor_prob_maps.sh
#      (or scripts/cache_tumor_prob_for_multiwindow.py; set NNUNET_MW_PROB_DIR to that folder).
#
# Optional env (see src/3d/multiwindow/config.py):
#   NNUNET_MW_EPOCHS NNUNET_MW_LR NNUNET_MW_TVERSKY_WEIGHT NNUNET_MW_TVERSKY_ALPHA NNUNET_MW_TVERSKY_BETA
#
# Partial weight init from 1-channel 3D baseline (extra input channels stay at init).
# Default baseline file is checkpoint_best.pth (aligned with prob cache / inference). Override:
#   PRETRAINED_WEIGHTS=/path/to/checkpoint_final.pth bash scripts/train_3d_multiwindow_refinement.sh ...
#   NNUNET_PRETRAINED_PARTIAL=1
#
# Run:
#   RUN_NAME=my_run bash scripts/train_3d_multiwindow_refinement.sh --skip-preprocess
#   RESULTS_ROOT=/abs/path bash scripts/train_3d_multiwindow_refinement.sh --skip-preprocess
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_multiwindow_refine}"
readonly RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results_3d_multiwindow_runs/${RUN_NAME}}"
export nnUNet_results="${RESULTS_ROOT}"

readonly BO="nnUNetTrainer_150_BoundaryOverseg_50epochs__nnUNetPlans_3d_midres125__3d_fullres"
readonly DEFAULT_PROB_MODEL="${REPO_ROOT}/results_3d_boundary_shape_runs/20260509_160927_boundary_size_gated/Dataset001_LiverTumor/${BO}"
export NNUNET_MW_PROB_DIR="${NNUNET_MW_PROB_DIR:-${REPO_ROOT}/results_3d_multiwindow_runs/_prob_cache_fold0_size_gated}"

readonly DATASET_ID=1
readonly CONFIGURATION="3d_fullres"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_150_MultiWindowRefine_50epochs"
readonly PLANS="nnUNetPlans_3d_midres125"
readonly TARGET_SPACING=(1.25 1.0 1.0)
readonly PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${BASE_NNUNET_RESULTS}/Dataset001_LiverTumor/nnUNetTrainer_150__${PLANS}__${CONFIGURATION}/fold_0/checkpoint_best.pth}"

SKIP_PREPROCESS=0
CACHE_PROB=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
  shift || true
fi
if [[ "${1:-}" == "--cache-prob" ]]; then
  CACHE_PROB=1
  shift || true
fi
if [[ "$#" -gt 0 ]]; then
  echo "Unknown arguments: $*" >&2
  exit 2
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

mkdir -p "${RESULTS_ROOT}"
mkdir -p "${NNUNET_MW_PROB_DIR}"

if [[ "${CACHE_PROB}" -eq 1 ]] || [[ "${AUTO_CACHE_PROB:-0}" == "1" ]]; then
  echo "Caching tumour probabilities -> ${NNUNET_MW_PROB_DIR}"
  "${PYTHON}" "${REPO_ROOT}/scripts/cache_tumor_prob_for_multiwindow.py" \
    --model-dir "${PROB_MODEL_DIR:-${DEFAULT_PROB_MODEL}}" \
    --output-dir "${NNUNET_MW_PROB_DIR}" \
    --fold "${FOLD}" \
    --split all
fi

if ! ls "${NNUNET_MW_PROB_DIR}"/*.npz 1>/dev/null 2>&1; then
  echo "ERROR: No *.npz under NNUNET_MW_PROB_DIR=${NNUNET_MW_PROB_DIR}" >&2
  echo "  Run:  ${PYTHON} ${REPO_ROOT}/scripts/cache_tumor_prob_for_multiwindow.py \\" >&2
  echo "          --model-dir <BoundaryOverseg_3d_folder> --output-dir \"\${NNUNET_MW_PROB_DIR}\" --split all" >&2
  echo "  Or:  bash $0 --cache-prob --skip-preprocess" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "Missing pretrained weights: ${PRETRAINED_WEIGHTS}" >&2
  exit 1
fi

echo "Writing MultiWindow refine run to: ${RESULTS_ROOT}"
echo "Using prob cache dir: ${NNUNET_MW_PROB_DIR}"

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" \
    -overwrite_target_spacing "${TARGET_SPACING[@]}" \
    -overwrite_plans_name "${PLANS}" \
    --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess."
fi

export NNUNET_PRETRAINED_PARTIAL="${NNUNET_PRETRAINED_PARTIAL:-1}"

VAL_BEST_ARGS=()
if [[ "${NNUNET_VALIDATION_WITH_BEST:-1}" != "0" ]]; then
  VAL_BEST_ARGS=(--val_best)
fi

"${PYTHON}" "${REPO_ROOT}/scripts/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}" \
  "${VAL_BEST_ARGS[@]}"
