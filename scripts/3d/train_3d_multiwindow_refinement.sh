#!/usr/bin/env bash
# Train 3D multi-window refinement (4 inputs: tumour prob + Lim HU windows).
#
# Results (like boundary runs): each default run is
#   results_3d_multiwindow_runs/<YYYYMMDD_HHMMSS>_multiwindow_3d_fullres/
# Override: RUN_NAME=my_tag bash ...  or  RESULTS_ROOT=/abs/path bash ...
#
# Target spacing (must match nnUNet_preprocessed + prob cache + stage-1 model):
#   Default matches this repo's 3d_fullres plans (nnUNetPlans_3d_midres125.json → configurations.3d_fullres.spacing):
#     1.25 1.0 1.0   (same as scripts/3d/train_nnunet_3d.sh and train_3d_boundary_shape.sh)
#   Native median spacing from the dataset (original_median_spacing_after_transp) if you re-preprocess everything:
#     NNUNET_MW_TARGET_SPACING="1.0 0.76953125 0.76953125" bash scripts/3d/train_3d_multiwindow_refinement.sh --skip-preprocess
#   If you change spacing: re-run plan_and_preprocess (no --skip-preprocess once), then re-cache prob maps.
#
# Prerequisites:
#   1) nnU-Net preprocessing for Dataset001 with the SAME target spacing as above.
#   2) Tumour probability cache aligned with that preprocess (see scripts/3d/cache_tumor_prob_maps.sh).
#
# RAM-friendly defaults (override as needed):
#   export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"   # fewer DA workers than nnU-Net default 12
# MultiWindow-only (see src/3d/multiwindow/config.py):
#   NNUNET_MW_MAX_BATCH_SIZE (default 1; set 0 to use full plan batch_size)
#   NNUNET_MW_DA_NUM_CACHED_TRAIN / NNUNET_MW_DA_NUM_CACHED_VAL (default 2)
#   NNUNET_MW_PIN_MEMORY (default 0/false)
# Optional env:
#   NNUNET_MW_EPOCHS (default 150 via script export; config default 150)
#   NNUNET_MW_VAL_PERFORM_ON_DEVICE=0 to run sliding-window math on CPU (saves VRAM; slower)
#   NNUNET_MW_LR NNUNET_MW_ITER_PER_EPOCH NNUNET_MW_VAL_ITER
#   NNUNET_MW_TVERSKY_WEIGHT NNUNET_MW_TVERSKY_ALPHA NNUNET_MW_TVERSKY_BETA
#
# Partial weight init from 1-channel 3D baseline (extra input channels stay at init).
# Default baseline file is checkpoint_best.pth. Override:
#   PRETRAINED_WEIGHTS=/path/to/checkpoint_final.pth bash scripts/3d/train_3d_multiwindow_refinement.sh ...
#   NNUNET_PRETRAINED_PARTIAL=1
#
# Run:
#   bash scripts/3d/train_3d_multiwindow_refinement.sh --skip-preprocess
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Fewer background DA processes than nnU-Net default (saves host RAM on 4-channel batches).
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"

# Default 150 epochs (override: NNUNET_MW_EPOCHS=50 …).
export NNUNET_MW_EPOCHS="${NNUNET_MW_EPOCHS:-150}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
readonly BASE_NNUNET_RESULTS="${BASE_NNUNET_RESULTS:-${REPO_ROOT}/nnUNet_results}"
readonly RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_multiwindow_3d_fullres}"
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
NNUNET_MW_TARGET_SPACING="${NNUNET_MW_TARGET_SPACING:-1.25 1.0 1.0}"
IFS=' ' read -ra TARGET_SPACING <<< "${NNUNET_MW_TARGET_SPACING}"
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
  "${PYTHON}" "${REPO_ROOT}/scripts/3d/cache_tumor_prob_for_multiwindow.py" \
    --model-dir "${PROB_MODEL_DIR:-${DEFAULT_PROB_MODEL}}" \
    --output-dir "${NNUNET_MW_PROB_DIR}" \
    --fold "${FOLD}" \
    --split all
fi

if ! ls "${NNUNET_MW_PROB_DIR}"/*.npz 1>/dev/null 2>&1; then
  echo "ERROR: No *.npz under NNUNET_MW_PROB_DIR=${NNUNET_MW_PROB_DIR}" >&2
  echo "  Run:  ${PYTHON} ${REPO_ROOT}/scripts/3d/cache_tumor_prob_for_multiwindow.py \\" >&2
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
echo "Target spacing (NNUNET_MW_TARGET_SPACING): ${TARGET_SPACING[*]}"

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

"${PYTHON}" "${REPO_ROOT}/scripts/3d/run_nnunet_with_local_3d_trainers.py" \
  "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  -pretrained_weights "${PRETRAINED_WEIGHTS}" \
  "${VAL_BEST_ARGS[@]}"
