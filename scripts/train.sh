#!/bin/bash
# nnU-Net v2 training for Dataset001_LiverTumor.
# Sets nnUNet_* env vars from repo if not already set.
cd "$(dirname "$0")/.."
export nnUNet_raw="${nnUNet_raw:-$PWD/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PWD/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PWD/nnUNet_results}"

DATASET_ID=1

# 1) Plan and preprocess (use -np 1 for less load)
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -np 1

# 2) Train 3D full resolution, fold 0
nnUNetv2_train "$DATASET_ID" 3d_fullres 0
