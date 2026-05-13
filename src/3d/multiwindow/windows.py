"""Multi-channel HU windows for liver–tumor refinement (Lim et al., Diagnostics 2025).

Maps nnU-Net–normalized intensities back to approximate HU using dataset fingerprint
percentiles, then builds three soft windows (scaled to [0, 1]):

- **full**: HU in [-1000, 1000] (wide abdominal / “all values” window).
- **soft_tissue**: [0, 1000] — emphasizes liver and soft tissue while suppressing very low HU.
- **dense_contrast**: [400, 1000] — highlights hyperdense / contrast-rich structures.

These are used as channels 1–3 after the coarse tumor probability channel.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def load_fingerprint_stats(fingerprint_path: str) -> dict[str, float]:
    import json
    from pathlib import Path

    p = Path(fingerprint_path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    ch0: Mapping[str, Any] = raw["foreground_intensity_properties_per_channel"]["0"]
    return {
        "percentile_00_5": float(ch0["percentile_00_5"]),
        "percentile_99_5": float(ch0["percentile_99_5"]),
        "mean": float(ch0["mean"]),
        "std": float(ch0["std"]),
    }


def denorm_approx_hu(norm: np.ndarray, stats: Mapping[str, float]) -> np.ndarray:
    """Invert z-score used after clipping to fingerprint percentiles (nnU-Net CT default)."""
    mean = float(stats["mean"])
    std = max(float(stats["std"]), 1e-6)
    hu = norm.astype(np.float64) * std + mean
    p0 = float(stats["percentile_00_5"])
    p1 = float(stats["percentile_99_5"])
    return np.clip(hu, p0 - 500.0, p1 + 2000.0)


def _window01(hu: np.ndarray, low: float, high: float) -> np.ndarray:
    denom = max(high - low, 1e-6)
    return np.clip((hu - low) / denom, 0.0, 1.0).astype(np.float32)


def lim_three_windows_from_norm(
    norm_patch_zyx: np.ndarray,
    stats: Mapping[str, float],
) -> np.ndarray:
    """Return stack (3, *spatial) from a single normalized channel patch (Z, Y, X)."""
    hu = denorm_approx_hu(norm_patch_zyx, stats)
    w0 = _window01(hu, -1000.0, 1000.0)
    w1 = _window01(hu, 0.0, 1000.0)
    w2 = _window01(hu, 400.0, 1000.0)
    return np.stack([w0, w1, w2], axis=0)
