"""Empirical uncertainty estimator via temporal triangulation scatter.

For each track, triangulates the landmark from every pair of frames that
observes it, then fits a Gaussian to the 3D scatter. This 3D covariance Σ₃
is then projected to image space for each frame using the projection Jacobian J:

    Σ₂ = J Σ₃ Jᵀ,  where J = ∂π/∂X  (2×3 Jacobian of the projection function)

The result is a per-observation 2×2 covariance ready for GenericProjectionFactor.

Intuition:
    - Consistent triangulations across many frame pairs → small 3D scatter → small Σ₂
    - Scattered triangulations (short baseline, degenerate geometry, noisy tracks)
      → large 3D scatter → large Σ₂

This is a geometry-only method: no image appearance, no assumed noise model.
"""

import numpy as np
import cv2
from uncertainty_estimation.uncertainty.base import UncertaintyEstimator


def _projection_matrix(R_cw: np.ndarray, t_cw: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Build a 3×4 projection matrix from a camera-to-world pose and intrinsics.

    Inverts the camera-to-world (R_cw, t_cw) to world-to-camera, then
    computes P = K @ [R_wc | t_wc].

    Args:
        R_cw: (3, 3) rotation, camera-to-world.
        t_cw: (3,) translation, camera-to-world.
        K:    (3, 3) intrinsic matrix.

    Returns:
        (3, 4) projection matrix.
    """
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw
    return K @ np.hstack([R_wc, t_wc.reshape(3, 1)])


def _triangulate_all(
    obs: dict[int, np.ndarray],
    poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    max_depth: float = 100.0,
    min_frame_gap: int = 5,
) -> np.ndarray:
    """Triangulate a track from every pair of frames that observes it.

    Frame pairs are skipped if they are too close in time (likely insufficient
    parallax). Each triangulated point is also validated for positive depth in
    both cameras and below max_depth.

    Args:
        obs:           {frame_idx: (u, v)} observations for one track.
        poses:         [(R_cw, t_cw), ...] camera-to-world poses, indexed by frame.
        K:             (3, 3) intrinsic matrix.
        max_depth:     discard triangulations deeper than this (metres).
        min_frame_gap: discard pairs separated by fewer than this many frames.

    Returns:
        (N, 3) array of triangulated 3D points, one per valid frame pair.
        Returns empty (0, 3) array if no valid pairs exist.
    """
    points = []
    for frame_i, pt_i in obs.items():
        for frame_j, pt_j in obs.items():
            if frame_i >= frame_j:
                continue
            if frame_j - frame_i < min_frame_gap:
                continue
            R_i, t_i = poses[frame_i]
            R_j, t_j = poses[frame_j]
            P_i = _projection_matrix(R_i, t_i, K)
            P_j = _projection_matrix(R_j, t_j, K)
            X = cv2.triangulatePoints(P_i, P_j, pt_i.reshape(2, 1), pt_j.reshape(2, 1))
            X3 = (X[:3] / X[3]).ravel()
            # Depth in each camera (Z component of the world-to-camera transform)
            depth_i = (R_i.T @ (X3 - t_i))[2]
            depth_j = (R_j.T @ (X3 - t_j))[2]
            if 0 < depth_i < max_depth and 0 < depth_j < max_depth:
                points.append(X3)
    return np.array(points) if points else np.empty((0, 3))


def _projection_jacobian(X: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute the 2×3 Jacobian of the projection function ∂π/∂X at a 3D point.

    Projects X into camera coordinates, then differentiates the pinhole
    projection π(X) = [fx * Xc/Zc + cx, fy * Yc/Zc + cy] with respect to X_world.

    Args:
        X:    (3,) 3D point in world coordinates.
        R_cw: (3, 3) rotation, camera-to-world.
        t_cw: (3,) translation, camera-to-world.
        K:    (3, 3) intrinsic matrix.

    Returns:
        (2, 3) Jacobian matrix J such that Σ₂ ≈ J @ Σ₃ @ J.T.
    """
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw
    Xc = R_wc @ X + t_wc  # (3,) point in camera frame

    Xc_, Yc_, Zc_ = Xc
    fx, fy = K[0, 0], K[1, 1]

    # ∂π/∂Xc (2×3), then chain rule: ∂π/∂X = ∂π/∂Xc @ R_wc
    dpi_dXc = np.array([
        [fx / Zc_,       0., -fx * Xc_ / Zc_**2],
        [      0., fy / Zc_, -fy * Yc_ / Zc_**2],
    ])
    return dpi_dXc @ R_wc


class TemporalUncertainty(UncertaintyEstimator):
    """Fit a Gaussian to multiple triangulations, project covariance to 2D.

    Steps per track:
        1. Triangulate from every (frame_i, frame_j) pair that observes the track
        2. Fit sample Gaussian: mean + 3×3 covariance to the N triangulations
        3. For each frame f that observes the track, compute the 2×3 projection
           Jacobian J at (mean, pose_f, K), then Σ₂ = J Σ₃ Jᵀ

    Args:
        max_depth:          discard triangulations deeper than this (metres). Filters
                            out sky/background points (TartanAir sky pixels have
                            depths of 10 000 m+).
        min_frame_gap:      only triangulate frame pairs separated by at least this
                            many frames. Filters out near-zero-parallax pairs from
                            consecutive frames.
        target_median_trace: if set, rescale all covariances after estimation so that
                            their median trace equals this value. The relative structure
                            (which observations are more/less certain) is preserved;
                            only the absolute scale is adjusted. Useful because raw
                            triangulation scatter often produces covariances that are
                            orders of magnitude too large for use as pixel noise models.
                            A value of ~10–25 px² corresponds to σ ≈ 2–5 px, which is
                            a reasonable pixel noise model. Set to None to disable.
        regularisation:     isotropic floor added to every 2×2 covariance before
                            returning: cov += regularisation * I. Prevents
                            near-singular matrices that cause GTSAM numerical issues
                            when the Jacobian is nearly degenerate. Default 1.0 px².
        max_condition:      discard observations whose covariance condition number
                            exceeds this. Very ill-conditioned matrices (e.g. 1e8)
                            indicate a degenerate projection direction and can cause
                            optimizer divergence. Set to None to disable.
    """

    def __init__(
        self,
        max_depth: float = 100.0,
        min_frame_gap: int = 5,
        target_median_trace: float | None = 16.0,
        regularisation: float = 1.0,
        max_condition: float | None = 1e6,
    ):
        self.max_depth = max_depth
        self.min_frame_gap = min_frame_gap
        self.target_median_trace = target_median_trace
        self.regularisation = regularisation
        self.max_condition = max_condition

    def estimate(
        self,
        tracks: dict[int, dict[int, np.ndarray]],
        poses: list[tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        images: list[np.ndarray] | None = None,  # noqa: ARG002  (geometry-only method)
    ) -> dict[tuple[int, int], np.ndarray]:
        result: dict[tuple[int, int], np.ndarray] = {}

        for tid, obs in tracks.items():
            points_3d = _triangulate_all(obs, poses, K, max_depth=self.max_depth, min_frame_gap=self.min_frame_gap)

            if len(points_3d) < 2:
                continue

            mean_3d = points_3d.mean(axis=0)   # (3,)
            cov_3d = np.cov(points_3d.T)        # (3, 3)

            for frame_f, _ in obs.items():
                R_f, t_f = poses[frame_f]
                J = _projection_jacobian(mean_3d, R_f, t_f, K)  # (2, 3)
                cov_2d = J @ cov_3d @ J.T                        # (2, 2)
                cov_2d += self.regularisation * np.eye(2)
                if self.max_condition is not None:
                    eigvals = np.linalg.eigvalsh(cov_2d)
                    if eigvals[0] <= 0 or eigvals[-1] / eigvals[0] > self.max_condition:
                        continue
                result[(tid, frame_f)] = cov_2d

        if self.target_median_trace is not None and result:
            median_trace = float(np.median([np.trace(c) for c in result.values()]))
            if median_trace > 0:
                scale = self.target_median_trace / median_trace
                result = {k: scale * v for k, v in result.items()}

        return result


def _draw_ellipse(ax, mean: np.ndarray, cov: np.ndarray, n_std: float = 2.0, **kwargs):
    """Draw a 2D confidence ellipse on a matplotlib axes.

    Args:
        ax:    matplotlib axes.
        mean:  (2,) centre of the ellipse (u, v).
        cov:   (2, 2) covariance matrix.
        n_std: number of standard deviations to scale the ellipse.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    from matplotlib.patches import Ellipse
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ellipse)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from uncertainty_estimation.data.tartanair import TartanAirSequence, TARTANAIR_K
    from uncertainty_estimation.frontend.lk import LKTracker

    if len(sys.argv) not in (2, 3):
        print("Usage: uv run python -m uncertainty_estimation.uncertainty.temporal <path/to/P000> [n_frames]")
        sys.exit(1)

    seq_path = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) == 3 else 10

    seq = TartanAirSequence(seq_path)
    n_frames = min(n_frames, len(seq))
    frames = [seq[i] for i in range(n_frames)]
    images = [f.image for f in frames]
    poses = [(f.pose[:3, :3], f.pose[:3, 3]) for f in frames]
    K = TARTANAIR_K
    print(f"Loaded {n_frames} frames from {seq_path}")

    tracker = LKTracker(max_features=300, min_tracks=80)
    tracks = tracker.track(images)
    print(f"Tracked {len(tracks)} features across {n_frames} frames.")

    estimator = TemporalUncertainty(max_depth=100.0, min_frame_gap=3)
    obs_covs = estimator.estimate(tracks, poses, K)
    print(f"Estimated covariances for {len(obs_covs)} observations.")

    # --- Plot 1: 2D ellipses overlaid on the middle frame ---
    last_frame = n_frames // 2
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(images[last_frame])

    # Colour ellipses by trace(Σ₂): blue=low uncertainty, red=high
    # Use all covariances (not just last frame) to set a stable colour scale
    all_traces_for_norm = [np.trace(c) for c in obs_covs.values()]
    if not all_traces_for_norm:
        print("No covariances estimated — try increasing n_frames or decreasing min_frame_gap.")
        exit(1)
    norm = Normalize(vmin=np.percentile(all_traces_for_norm, 5), vmax=np.percentile(all_traces_for_norm, 95))
    cmap = plt.colormaps["coolwarm"]

    for tid, obs in tracks.items():
        if last_frame not in obs or (tid, last_frame) not in obs_covs:
            continue
        pt = obs[last_frame]
        cov_2d = obs_covs[(tid, last_frame)]
        tr = np.trace(cov_2d)
        colour = cmap(norm(tr))
        ax.plot(pt[0], pt[1], ".", color=colour, markersize=4)
        _draw_ellipse(ax, pt, cov_2d, n_std=2.0,
                      facecolor="none", edgecolor=colour, linewidth=0.8, alpha=0.8)

    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="trace(Σ₂)")
    ax.set_title(f"2σ uncertainty ellipses — frame {last_frame} (middle)  (blue=low, red=high)")
    ax.axis("off")
    plt.tight_layout()

    # --- Plot 2: trace(Σ₂) vs track length ---
    track_lengths, mean_traces = [], []
    for tid, obs in tracks.items():
        covs = [obs_covs[(tid, f)] for f in obs if (tid, f) in obs_covs]
        if not covs:
            continue
        track_lengths.append(len(obs))
        mean_traces.append(np.mean([np.trace(c) for c in covs]))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(track_lengths, mean_traces, s=10, alpha=0.5)
    ax2.set_xlabel("Track length (frames)")
    ax2.set_ylabel("Mean trace(Σ₂)")
    ax2.set_title("Uncertainty vs track length  (shorter tracks → higher uncertainty expected)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # --- Plot 3: histogram of trace(Σ₂) ---
    all_traces = [np.trace(c) for c in obs_covs.values()]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(all_traces, bins=60, edgecolor="none")
    ax3.set_xlabel("trace(Σ₂)")
    ax3.set_ylabel("Count")
    ax3.set_yscale("log")
    ax3.set_title("Distribution of observation uncertainties (log scale)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.show()
