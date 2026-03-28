# Scripts (liver-tumor-segmentation)

| Script | Role |
|--------|------|
| `train.sh` | nnU-Net v2: plan + preprocess + train fold 0. Use `--skip-preprocess` if preprocessed cache is already valid. |
| `infer.py` | **Main** full-volume inference: `--stage1-only` (baseline) or two-stage with `--refinement-dir`. Same nnU-Net stack as refinement export. |
| `infer_two_stage.py` | Thin wrapper → calls `infer.py` (same flags). |
| `compare_nnunet_vs_refined.sh` | Val-only staging + stage1 + two-stage + `evaluate_segmentations.py`. Env: `FOLD`, `TILE_STEP`, `NO_MIRROR`. |
| `predict_val_fold_lowmem.sh` | `nnUNetv2_predict` on val only (larger step, no TTA); not identical defaults to `infer.py`. |
| `export_stage1_preds.py` | Export `.npz` slices for `train_refiner.py`. |
| `train_refiner.py` | Train stage-2 U-Net. |
| `evaluate_segmentations.py` | Tumor Dice/IoU vs `labelsTr` for prediction folders. |

Environment for nnU-Net tools: `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` (defaults: under repo root).
