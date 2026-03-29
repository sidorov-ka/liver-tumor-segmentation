#!/usr/bin/env bash
# Stage 1 only: nnU-Net v2 for Dataset001_LiverTumor — plan, preprocess (2d, splits_final.json), train fold 0.
# Second stages (coarse_to_fine / multiview) use scripts/export.py then scripts/train_coarse_to_fine.sh or train_multiview.sh.
# Skip heavy preprocessing if cache is already valid: bash scripts/train.sh --skip-preprocess
# or: SKIP_NNUNET_PREPROCESS=1 bash scripts/train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/nnUNet_results}"

readonly DATASET_ID=1
readonly CONFIGURATION="2d"
readonly FOLD=0
readonly TRAINER="nnUNetTrainer_100epochs"

SKIP_PREPROCESS=0
if [[ "${1:-}" == "--skip-preprocess" ]]; then
  SKIP_PREPROCESS=1
fi
if [[ "${SKIP_NNUNET_PREPROCESS:-0}" == "1" ]]; then
  SKIP_PREPROCESS=1
fi

if [[ "${SKIP_PREPROCESS}" -eq 0 ]]; then
  # Single-process preprocessing to limit RAM; --clean rebuilds after raw data changes.
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" --clean
else
  echo "Skipping nnUNetv2_plan_and_preprocess (preprocessed data assumed valid; use after a full preprocess run)."
fi

# -p is plans identifier, not dataloader workers (see nnUNetv2_train --help).
nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}"
