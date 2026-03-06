"""Abstract interface for uncertainty estimators.

Every uncertainty method in this project implements UncertaintyEstimator.
This lets the pipeline swap methods with one line and compare them fairly.

The output is a 2×2 covariance per 2D observation — one per (track, frame) pair.
This is the noise model that goes directly into GTSAM's GenericProjectionFactor.

Why 2D (image-plane) rather than 3D (world-space)?
    GTSAM's reprojection factor models observation noise in image space: the
    difference between the projected 3D point and the detected 2D keypoint.
    All uncertainty methods ultimately need to produce this 2D noise model,
    even if they compute it via an intermediate 3D representation.

    Isotropic baseline:  cov = σ²I  (same for every observation)
    Empirical method:    triangulate → 3D scatter → project via Jacobian J → J Σ₃ Jᵀ
    Learned method:      network directly outputs 2×2 per keypoint

Usage in pipeline:
    estimator: UncertaintyEstimator = IsotropicUncertainty(sigma=1.0)
    obs_covs = estimator.estimate(tracks, poses, K, images)

    for (track_id, frame_idx), cov_2x2 in obs_covs.items():
        noise = gtsam.noiseModel.Gaussian.Covariance(cov_2x2)
        graph.add(GenericProjectionFactor(pixel, noise, pose_key, landmark_key, cal))

To add a new method:
    1. Create a new file in this directory (e.g. analytical.py, learned.py)
    2. Subclass UncertaintyEstimator and implement estimate()
    3. That's it — the pipeline and backend accept any estimator transparently
"""

from abc import ABC, abstractmethod
import numpy as np


class UncertaintyEstimator(ABC):
    """Interface for producing 2D observation covariances for a set of feature tracks.

    Given all tracks in a window, the camera poses, intrinsics, and optionally
    the raw images, returns a 2×2 covariance matrix for every (track, frame)
    observation. These covariances are the noise models for GTSAM's
    GenericProjectionFactor — larger covariance = noisier observation = less
    influence on the optimisation.
    """

    @abstractmethod
    def estimate(
        self,
        tracks: dict[int, dict[int, np.ndarray]],
        poses: list[tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        images: list[np.ndarray] | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Estimate a 2×2 observation covariance for every (track, frame) pair.

        Args:
            tracks:  {track_id: {frame_idx: np.ndarray (u, v)}}
                     2D pixel positions. Only frames where the feature is
                     visible are included.
            poses:   [(R, t), ...] length n_frames. R is (3,3), t is (3,).
                     Convention: camera-to-world (X_world = R @ X_cam + t).
            K:       (3, 3) shared camera intrinsic matrix.
            images:  list of (H, W, 3) or (H, W) images, indexed by frame_idx.
                     Required by image-based methods (e.g. learned); may be
                     ignored by geometry-only methods.

        Returns:
            {(track_id, frame_idx): np.ndarray (2, 2)}
            One symmetric positive-definite covariance matrix per observation.
            Every (track_id, frame_idx) key present in `tracks` must appear
            in the output.
        """
        ...
