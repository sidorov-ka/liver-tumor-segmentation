3D from-scratch loss arms (standard `nnUNetPlans`, 500 epochs).

| Arm | Package | Trainer |
|-----|---------|---------|
| Baseline | `baseline/` | `nnUNetTrainer_500_Baseline` |
| Anatomical | `anatomical/` | `nnUNetTrainer_500_Anatomical` |
| Boundary-HD | `boundary_hd/` | `nnUNetTrainer_500_BoundaryHD` |
| Topology | `topology/` | `nnUNetTrainer_500_Topology` |

Launch: `bash scripts/3d/train.sh <baseline|anatomical|boundary-hd|topology>`
