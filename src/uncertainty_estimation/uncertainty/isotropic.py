"""Isotropic uncertainty estimator — baseline covariance model.

Assigns the same sigma^2 * I covariance to every observation regardless of track
quality, depth, or baseline. This is the standard assumption in most VO
pipelines and serves as the baseline to beat.
"""

import numpy as np
from uncertainty_estimation.uncertainty.base import UncertaintyEstimator


class IsotropicUncertainty(UncertaintyEstimator):
    """Return sigma^2 * I for every observation.

    Args:
        sigma: standard deviation in pixels (default 1.0).
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def estimate(
        self,
        tracks: dict[int, dict[int, np.ndarray]],
        poses: list[tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        images: list[np.ndarray] | None = None,  # noqa: ARG002
    ) -> dict[tuple[int, int], np.ndarray]:
        cov = (self.sigma ** 2) * np.eye(2)
        return {
            (tid, fid): cov
            for tid, obs in tracks.items()
            for fid in obs
        }
