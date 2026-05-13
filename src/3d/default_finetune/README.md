Default-loss 3D fine-tuning control.

This experiment starts from the same 150-epoch 3D nnU-Net checkpoint as the
boundary/shape experiment, but keeps the original nnU-Net Dice+CE loss.

Use it to separate plain additional training time from the effect of the
boundary/over-segmentation loss.

Main knobs are environment variables:

- `NNUNET_DEFAULT_FINETUNE_EPOCHS` default `50`
- `NNUNET_DEFAULT_FINETUNE_LR` default `1e-3`
