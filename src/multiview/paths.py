"""Default directory names for multiview runs (separate from coarse_to_fine_results)."""

# Training checkpoints / logs / meta.json for MultiviewUNet2d — not coarse_to_fine.
DEFAULT_MULTIVIEW_RESULTS_ROOT = "multiview_results"

# Subdirectory under .../<dataset>/fold_*/ — same pattern as coarse_to_fine.paths.COARSE_TO_FINE_TASK_DIR.
MULTIVIEW_TASK_DIR = "multiview"
