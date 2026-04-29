2D training packages live here.

This folder contains the existing second-stage 2D experiments:

- `coarse_to_fine`
- `multiview`
- `uncertainty`
- `boundary_aware_coarse_to_fine`

Scripts that use these packages add `src/2d` to `sys.path`, so the package
imports stay unchanged, for example `from coarse_to_fine.model import ...`.
