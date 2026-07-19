# Arm 4 — Topology (soft clDice)

Wraps Dice+CE with **soft center-line Dice** (Shit et al.) on the tumor class
to encourage topological agreement of soft skeletons.

- Trainer: `nnUNetTrainer_500_Topology`
- Entry: `bash scripts/3d/train_3d_topology.sh`
- Output: `results_3d_topology/`

Env: `NNUNET_TOPOLOGY_EPOCHS`, `NNUNET_TOPOLOGY_LR`,
`NNUNET_TOPOLOGY_CLDICE_WEIGHT`, `NNUNET_TOPOLOGY_SKELETON_ITERATIONS`.
