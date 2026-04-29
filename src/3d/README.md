3D training experiments live here.

The baseline 3D nnU-Net training currently runs through `scripts/train_3d.sh`.
Future fine-tuning experiments should be organized under this folder and should
keep inference/postprocessing comparable across variants.

Planned experiment groups:

- `boundary_shape`: boundary-aware and over-segmentation-aware losses.
- `multiwindow`: multi-HU-window 3D input variants.
- `hard_negative`: hard-negative and component-aware fine-tuning.
