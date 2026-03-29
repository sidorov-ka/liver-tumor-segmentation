"""Default roots for coarse_to_fine training runs (parallel layout to multiview.paths)."""

# Top-level folder in repo (separate from multiview_results and nnUNet_results).
DEFAULT_COARSE_TO_FINE_RESULTS_ROOT = "coarse_to_fine_results"

# Subdirectory under .../<dataset>/fold_*/ — keeps one consistent layout with multiview (…/multiview/run_*).
COARSE_TO_FINE_TASK_DIR = "coarse_to_fine"
