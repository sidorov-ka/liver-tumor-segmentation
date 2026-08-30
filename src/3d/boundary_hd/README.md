# Arm 3 — Boundary / soft Hausdorff

Wraps Dice+CE with a **differentiable Hausdorff-inspired** term: soft
morphological boundaries + banded distance maps (proxy for surface distance;
report true HD95 at evaluation).

- Trainer: `nnUNetTrainer_500_BoundaryHD`
- Entry: `bash scripts/3d/train.sh boundary-hd`
- Output: `results_3d_boundary_hd/`

Env: `NNUNET_BOUNDARY_HD_EPOCHS`, `NNUNET_BOUNDARY_HD_LR`,
`NNUNET_BOUNDARY_HD_WEIGHT`, `NNUNET_BOUNDARY_HD_DISTANCE_ITERATIONS`.
