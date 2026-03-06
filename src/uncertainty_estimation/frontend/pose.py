"""Geometric pose estimation from 2D correspondences.

Given matched keypoints across two frames, estimates the relative camera pose
and triangulates 3D points. These are the classical multi-view geometry steps.

Inputs:  matched 2D keypoints, camera intrinsics K
Outputs: relative pose (R, t), triangulated 3D points
"""

import numpy as np


def estimate_relative_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate relative pose (R, t) between two frames via F -> E -> pose.

    Args:
        pts1: (N, 2) matched keypoints in frame 1.
        pts2: (N, 2) matched keypoints in frame 2.
        K:    (3, 3) shared intrinsic matrix.

    Returns:
        R:            (3, 3) rotation.
        t:            (3,)   unit translation.
        inlier_mask:  (N,)   bool, True for RANSAC inliers.
    """
    raise NotImplementedError


def triangulate(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from two projection matrices and 2D correspondences.

    Args:
        P1:   (3, 4) projection matrix of frame 1.
        P2:   (3, 4) projection matrix of frame 2.
        pts1: (N, 2) points in frame 1.
        pts2: (N, 2) points in frame 2.

    Returns:
        points_3d: (N, 3) triangulated 3D points in world coordinates.
    """
    raise NotImplementedError
