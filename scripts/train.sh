#!/usr/bin/env bash
# Train Dataset001_LiverTumor with nnU-Net v2: plan, preprocess (2d), then train fold 0.
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

# Single-process preprocessing to limit RAM; --clean rebuilds after raw data changes.
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -npfp 1 -np 1 -c "${CONFIGURATION}" --clean

# -p is plans identifier, not dataloader workers (see nnUNetv2_train --help).
nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}"
