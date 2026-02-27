"""Evaluation metrics for validating your multi-view geometry pipeline."""

import numpy as np


def reprojection_error(
    points_3d: np.ndarray,
    observations: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Compute per-point reprojection error.

    Args:
        points_3d: (N, 3) triangulated 3D points.
        observations: (N, 2) observed 2D keypoints.
        P: (3, 4) projection matrix.

    Returns:
        (N,) array of reprojection errors in pixels.
    """
    N = len(points_3d)
    pts_h = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    projected = (P @ pts_h.T).T  # (N, 3)
    projected_2d = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(observations - projected_2d, axis=1)
    return errors


def mean_reprojection_error(
    points_3d: np.ndarray,
    observations: np.ndarray,
    P: np.ndarray,
) -> float:
    """Mean reprojection error across all points."""
    return float(np.mean(reprojection_error(points_3d, observations, P)))


def triangulation_error(
    points_3d_estimated: np.ndarray,
    points_3d_gt: np.ndarray,
) -> np.ndarray:
    """Euclidean distance between estimated and ground truth 3D points.

    Args:
        points_3d_estimated: (N, 3) your triangulated points.
        points_3d_gt: (N, 3) ground truth 3D positions.

    Returns:
        (N,) array of 3D distances.
    """
    return np.linalg.norm(points_3d_estimated - points_3d_gt, axis=1)


def uncertainty_calibration(
    points_3d_estimated: np.ndarray,
    points_3d_gt: np.ndarray,
    covariances: np.ndarray,
) -> dict[str, float]:
    """Check how well your uncertainty estimates are calibrated.

    For well-calibrated Gaussian uncertainty, ~68% of ground truth points should
    fall within the 1-sigma ellipsoid, ~95% within 2-sigma, ~99.7% within 3-sigma.

    Computes the Mahalanobis distance of each GT point from the estimate, then
    checks what fraction fall within various sigma thresholds.

    Args:
        points_3d_estimated: (N, 3) your estimated points.
        points_3d_gt: (N, 3) ground truth points.
        covariances: (N, 3, 3) your estimated covariance matrices.

    Returns:
        Dict with keys "within_1sigma", "within_2sigma", "within_3sigma"
        giving the fraction of points within each threshold.
    """
    N = len(points_3d_estimated)
    mahal_sq = np.zeros(N)

    for i in range(N):
        diff = points_3d_gt[i] - points_3d_estimated[i]
        try:
            cov_inv = np.linalg.inv(covariances[i])
            mahal_sq[i] = diff @ cov_inv @ diff
        except np.linalg.LinAlgError:
            mahal_sq[i] = np.inf

    mahal = np.sqrt(mahal_sq)

    return {
        "within_1sigma": float(np.mean(mahal <= 1.0)),
        "within_2sigma": float(np.mean(mahal <= 2.0)),
        "within_3sigma": float(np.mean(mahal <= 3.0)),
        "mean_mahalanobis": float(np.mean(mahal[np.isfinite(mahal)])),
    }
