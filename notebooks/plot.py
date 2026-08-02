"""Plot helpers for functional analysis notebooks (extend as needed)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["style_axis", "mean_sem", "unique_clusters_in_order", "bin_time_observations"]


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise observations, column-wise timepoints. Returns (mean, SEM)."""
    mean = np.mean(mat, axis=0)
    n = mat.shape[0]
    if n > 1:
        sem = np.std(mat, axis=0, ddof=1) / np.sqrt(n)
    else:
        sem = np.zeros(mat.shape[1])
    return mean, sem


def bin_time_observations(mat: np.ndarray, bin_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-overlapping temporal mean along time. ``mat`` shape ``(n_obs, n_frames)``.
    Trailing frames are dropped if ``n_frames`` is not divisible by ``bin_size``.

    Returns ``(binned_mat, t_bin_centers)`` where ``t_bin_centers`` are original frame
    indices at the center of each bin (for aligning with ``pulse_frames``).
    """
    n, T = mat.shape
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    if bin_size == 1:
        return np.asarray(mat).copy(), np.arange(T, dtype=float)
    n_bins = T // bin_size
    if n_bins == 0:
        return mat[:, :0], np.array([])
    trimmed = mat[:, : n_bins * bin_size].reshape(n, n_bins, bin_size).mean(axis=2)
    t_centers = np.arange(n_bins, dtype=float) * bin_size + (bin_size - 1) / 2
    return trimmed, t_centers


def unique_clusters_in_order(T_sorted: np.ndarray) -> np.ndarray:
    """
    Cluster ids in dendrogram order (first row to last), one entry per block.
    Use with X_sorted, T_sorted after the same row reordering.
    """
    if T_sorted.size == 0:
        return T_sorted
    out = [T_sorted[0]]
    for i in range(1, len(T_sorted)):
        if T_sorted[i] != T_sorted[i - 1]:
            out.append(T_sorted[i])
    return np.array(out)
