"""MATLAB-style parameter bundle for hierarchical clustering + heatmap plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Control:
    """
    Mirrors a typical MATLAB ``c`` struct: time windows (seconds), linkage settings,
    cluster cutoff (fraction of max merge height), heatmap limits, and optional time axis ``t``.

    Adjust attributes to match your experiment (CS = conditioned stimulus epoch, etc.).
    """

    # Trajectory windows along the stitched epoch (seconds)
    preCSlength_traj: float = 5.0
    postCSlength_traj: float = 10.0
    CSlength_traj: float = 2.0
    fps: float = 5.0

    # scipy.cluster.hierarchy.linkage: method name; metric used if linkage gets condensed distances elsewhere
    distance_method: str = "ward"
    distance_metric: str = "euclidean"

    # fcluster: cutoff = cutoff * max(Z[:, 2]) when criterion="distance" (MATLAB-style)
    cutoff: float = 0.7

    heatmapColorScale: Tuple[float, float] = (-1.0, 1.0)

    # Filled in the notebook, e.g. np.linspace(-pre, post, n_frames_per_epoch_half * 2) — optional
    t: Optional[np.ndarray] = None

    # Catch-all for extra MATLAB fields without editing this file
    extra: dict = field(default_factory=dict)

    def set_time_axis(self, n_cols: int) -> np.ndarray:
        """
        If ``clusteringMat`` has shape (n_rows, n_cols) with n_cols == 2 * epoch_frames,
        MATLAB used half-width epochs; here we only need a 1d axis with ``n_cols`` points.

        Sets ``self.t`` to linspace from ``-preCSlength_traj`` to ``postCSlength_traj``.
        """
        self.t = np.linspace(-self.preCSlength_traj, self.postCSlength_traj, n_cols)
        return self.t
