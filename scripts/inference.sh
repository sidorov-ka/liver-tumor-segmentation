#!/bin/bash
# nnU-Net v2 inference. Set env vars (nnUNet_raw, nnUNet_results). Args: input_dir output_dir.
cd "$(dirname "$0")/.."
export nnUNet_raw="${nnUNet_raw:-$PWD/nnUNet_raw}"
export nnUNet_results="${nnUNet_results:-$PWD/nnUNet_results}"

DATASET_ID=1
INPUT_FOLDER="${1:-/path/to/test/images}"
OUTPUT_FOLDER="${2:-/path/to/predictions}"

nnUNetv2_predict -i "$INPUT_FOLDER" -o "$OUTPUT_FOLDER" -d "$DATASET_ID" -c 3d_fullres -f 0
