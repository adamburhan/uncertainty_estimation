"""Evaluation metrics for validating your multi-view geometry pipeline.

Organised into three groups:

  Geometry:   reprojection_error, triangulation_error
  Trajectory: ate, rpe                        ← pose-level accuracy vs GT
  Consistency: nees, uncertainty_calibration  ← are your covariances well-calibrated?
"""

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


# =============================================================================
# Trajectory metrics (pose-level, requires ground truth poses)
# =============================================================================

def ate(
    poses_est: list[np.ndarray],
    poses_gt: list[np.ndarray],
) -> dict[str, float]:
    """Absolute Trajectory Error (ATE).

    Aligns the estimated trajectory to ground truth (Umeyama / Procrustes,
    no scale) then computes per-frame translation error after alignment.

    ATE tells you the global consistency of the trajectory — how far each
    estimated pose is from its GT counterpart after best-fitting alignment.

    Args:
        poses_est: list of (4, 4) SE3 matrices, estimated poses.
        poses_gt:  list of (4, 4) SE3 matrices, ground truth poses.
                   Same length and correspondence as poses_est.

    Returns:
        {
            "rmse":   root-mean-square translation error in metres,
            "mean":   mean translation error,
            "median": median translation error,
            "max":    maximum translation error,
        }
    """
    raise NotImplementedError


def rpe(
    poses_est: list[np.ndarray],
    poses_gt: list[np.ndarray],
    delta: int = 1,
) -> dict[str, float]:
    """Relative Pose Error (RPE).

    Computes the error of relative transformations between pairs of frames
    separated by `delta` steps. Unlike ATE, RPE is not affected by drift
    accumulation — it measures local consistency.

    For each pair (i, i+delta):
        Q_est = inv(T_est[i]) @ T_est[i+delta]
        Q_gt  = inv(T_gt[i])  @ T_gt[i+delta]
        error = inv(Q_gt) @ Q_est   (relative error in SE3)

    Args:
        poses_est: list of (4, 4) SE3 matrices, estimated poses.
        poses_gt:  list of (4, 4) SE3 matrices, ground truth poses.
        delta:     frame separation for relative pairs (default 1 = consecutive).

    Returns:
        {
            "rmse_trans":  RMSE of translation component of relative errors,
            "rmse_rot":    RMSE of rotation component in degrees,
            "mean_trans":  mean translation error,
            "mean_rot":    mean rotation error in degrees,
        }
    """
    raise NotImplementedError


# =============================================================================
# Consistency metrics (do your covariances match your actual errors?)
# =============================================================================

def nees(
    errors: np.ndarray,
    covariances: np.ndarray,
) -> dict[str, float]:
    """Normalized Estimation Error Squared (NEES).

    For a well-calibrated estimator with d-dimensional Gaussian errors,
    the NEES score ε = eᵀ Σ⁻¹ e should follow a chi-squared distribution
    with d degrees of freedom, giving an expected value of d.

        NEES > d  →  covariances are too small (overconfident)
        NEES < d  →  covariances are too large (underconfident)
        NEES ≈ d  →  well-calibrated

    Use this on reprojection residuals (d=2) to check whether your 2×2
    observation covariances are consistent with the actual residuals after
    optimisation, or on pose errors (d=3 or d=6) to check pose uncertainty.

    Args:
        errors:      (N, d) array of estimation errors.
        covariances: (N, d, d) corresponding covariance matrices.

    Returns:
        {
            "mean_nees":     mean NEES score (should ≈ d for calibration),
            "dof":           d (degrees of freedom, for reference),
            "normalised":    mean_nees / d  (should ≈ 1.0),
        }
    """
    raise NotImplementedError
