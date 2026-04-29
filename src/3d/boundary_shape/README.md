Boundary/shape-aware 3D fine-tuning.

Use this folder for experiments that keep the 3D nnU-Net architecture mostly
unchanged and modify the objective to reduce boundary over-segmentation and
excess tumor volume.

Current experiment:

- `nnUNetTrainer_150_BoundaryOverseg_50epochs`
- wraps the default nnU-Net Dice+CE loss
- adds a tumor boundary-ring BCE+Dice term
- adds a soft penalty only when tumor probability volume exceeds GT tumor volume

Files:

- `config.py`: environment-driven experiment parameters.
- `losses.py`: additive boundary and over-segmentation loss wrapper.
- `trainer.py`: nnU-Net trainer subclass used for fine-tuning.

Main knobs are environment variables:

- `NNUNET_BOUNDARY_OVERSEG_EPOCHS` default `50`
- `NNUNET_BOUNDARY_OVERSEG_LR` default `1e-3`
- `NNUNET_BOUNDARY_OVERSEG_BOUNDARY_WEIGHT` default `0.25`
- `NNUNET_BOUNDARY_OVERSEG_OVERSEG_WEIGHT` default `0.05`
- `NNUNET_BOUNDARY_OVERSEG_BOUNDARY_RADIUS` default `2`
