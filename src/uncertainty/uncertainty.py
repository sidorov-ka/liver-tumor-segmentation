"""Entropy-based uncertainty for tumor probability maps (binary / Bernoulli)."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

# Max entropy for binary case: -0.5*log(0.5)*2 = log(2)
LOG2 = float(np.log(2.0))


def binary_entropy_probability(
    p: np.ndarray,
    eps: float = 1e-8,
    *,
    clamp: bool = True,
) -> np.ndarray:
    """
    U(p) = -p * log(p) - (1 - p) * log(1 - p) (natural log), element-wise.

    ``p`` is tumor probability in [0, 1]. For numerical stability, probabilities are
    clamped to ``[eps, 1 - eps]`` when ``clamp`` is True.

    Returns the same shape as ``p``, float32. Extend later with MC-dropout / ensemble
    by adding sibling functions that map logits/samples to U without changing this API.
    """
    x = p.astype(np.float32)
    if clamp:
        x = np.clip(x, eps, 1.0 - eps)
    else:
        x = np.clip(x, 0.0, 1.0)
        x = np.where(x <= 0.0, eps, x)
        x = np.where(x >= 1.0, 1.0 - eps, x)
    log1p = np.log(x)
    log1m = np.log(1.0 - x)
    u = -(x * log1p + (1.0 - x) * log1m)
    return u.astype(np.float32)


def binary_entropy_probability_torch(
    p: torch.Tensor,
    eps: float = 1e-8,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """PyTorch variant of :func:`binary_entropy_probability` (same formula)."""
    x = p.float()
    if clamp:
        x = torch.clamp(x, eps, 1.0 - eps)
    else:
        x = torch.clamp(x, 0.0, 1.0)
        x = torch.where(x <= 0, torch.full_like(x, eps), x)
        x = torch.where(x >= 1, torch.full_like(x, 1.0 - eps), x)
    return -(x * torch.log(x) + (1.0 - x) * torch.log(1.0 - x))


def normalize_entropy_01(u: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Map entropy to approximately [0, 1] by dividing by log(2)."""
    if isinstance(u, torch.Tensor):
        return u / (u.new_tensor(LOG2) + 1e-8)
    return (u / (LOG2 + 1e-8)).astype(np.float32)
