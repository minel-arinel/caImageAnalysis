"""Small utilities for functional analysis notebooks (extend as needed)."""

from __future__ import annotations

import numpy as np

__all__: list[str] = []


def zscore_rows(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Z-score each row (or column) of a 2d array; useful before clustering."""
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / (s + eps)
