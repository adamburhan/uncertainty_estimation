"""Temporal uncertainty estimation pipeline for 3D reconstruction.

Tracks features over N consecutive frames (monocular left camera). For each
feature observed in multiple frames, every pair of frames triangulates an
independent 3D estimate. The spread of these estimates gives an empirical 3D
distribution —> this is the core uncertainty model.

Rather than propagating assumed 2D noise analytically, we observe actual
geometric disagreement across multiple views and fit a Gaussian to it.

Run directly to test on KITTI or ETH3D:
    uv run python -m uncertainty_estimation.pipeline --sequence path/to/sequences/00
    uv run python -m uncertainty_estimation.pipeline --sequence path/to/cables_1 --dataset eth3d
"""

import argparse
import numpy as np
import cv2

from uncertainty_estimation.data.kitti import KITTISequence
from uncertainty_estimation.data.eth3d import ETH3DSequence
from uncertainty_estimation.visualization.matches import draw_features_by_depth
from uncertainty_estimation.visualization.point_cloud import visualize_reconstruction


# =============================================================================
# STEP 1: Fundamental / Essential matrix estimation
# =============================================================================

def estimate_fundamental(
    kp1: np.ndarray,
    kp2: np.ndarray,
    matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the fundamental matrix from matched keypoints.

    TODO: Implement this. Key concepts to understand:
        - The epipolar constraint: x2^T F x1 = 0
        - The 8-point algorithm (and why normalization matters)
        - RANSAC for robust estimation — why do we need it?
        - Relationship between F and E: E = K2^T F K1

    Args:
        kp1: (N, 2) keypoints from image 1.
        kp2: (M, 2) keypoints from image 2.
        matches: (K, 2) match index pairs.

    Returns:
        F: (3, 3) fundamental matrix.
        inlier_mask: (K,) boolean array marking RANSAC inliers.
    """
    F, inlier_mask = cv2.findFundamentalMat(
        kp1[matches[:, 0]], kp2[matches[:, 1]], method=cv2.FM_RANSAC, ransacReprojThreshold=1.0
    )
    return F, inlier_mask.ravel().astype(bool)


def fundamental_to_essential(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convert fundamental matrix to essential matrix: E = K2^T @ F @ K1.

    TODO: Implement this.

    Args:
        F: (3, 3) fundamental matrix.
        K1: (3, 3) intrinsic matrix of camera 1.
        K2: (3, 3) intrinsic matrix of camera 2.

    Returns:
        E: (3, 3) essential matrix.
    """
    return K2.T @ F @ K1


# =============================================================================
# STEP 2: Pose recovery
# =============================================================================

def recover_pose(
    E: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    matches: np.ndarray,
    inlier_mask: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover relative camera pose (R, t) from the essential matrix.

    TODO: Implement this. Key concepts:
        - SVD decomposition of E gives 4 possible (R, t) solutions
        - Cheirality check: triangulated points must be in front of both cameras
        - Only ONE of the 4 solutions satisfies this for all inliers
        - t is only known up to scale — why? What does this mean for your reconstruction?

    Args:
        E: (3, 3) essential matrix.
        kp1: (N, 2) keypoints from image 1.
        kp2: (M, 2) keypoints from image 2.
        matches: (K, 2) match index pairs.
        inlier_mask: (K,) boolean array of RANSAC inliers.
        K: (3, 3) intrinsic matrix (assuming same for both cameras).

    Returns:
        R: (3, 3) rotation matrix (camera 2 relative to camera 1).
        t: (3,) translation vector (unit norm, camera 2 relative to camera 1).
    """
    inlier_matches = matches[inlier_mask]
    pts1_inliers = kp1[inlier_matches[:, 0]]
    pts2_inliers = kp2[inlier_matches[:, 1]]

    _, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    return R, t.flatten()


# =============================================================================
# STEP 3: Triangulation
# =============================================================================

def triangulate_points(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from corresponding 2D observations in two views.

    TODO: Implement this using the DLT (Direct Linear Transform) method.
    For each point correspondence, you set up Ax = 0 and solve via SVD.
    Think about:
        - Why does each 2D point give 2 equations (not 3)?
        - Why do we need at least 2 views?
        - How does the baseline affect triangulation accuracy?
        - What happens when rays are nearly parallel?

    Args:
        P1: (3, 4) projection matrix of camera 1.
        P2: (3, 4) projection matrix of camera 2.
        pts1: (N, 2) matched keypoints in image 1.
        pts2: (N, 2) matched keypoints in image 2.

    Returns:
        points_3d: (N, 3) triangulated 3D points in world coordinates.
    """
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T  # (N, 4) homogeneous coordinates
    points_3d = points_4d[:, :3] / points_4d[:, 3][:, np.newaxis]  # Convert to non-homogeneous (divide by w)
    return points_3d




# =============================================================================
# TEMPORAL: Feature tracking across frames
# =============================================================================

def track_features_across_frames(
    images: list[np.ndarray],
    max_features: int = 500,
) -> dict[int, dict[int, np.ndarray]]:
    """Track features across a sequence of frames using Lucas-Kanade optical flow.

    Detects good features to track in the first frame, then propagates them
    forward using sparse optical flow. A feature's track ends when the tracker
    loses it (low confidence).


    Args:
        images: list of grayscale images (H, W) uint8.
        max_features: maximum number of features to initialize in frame 0.

    Returns:
        tracks: dict mapping track_id -> {frame_idx: pixel_coords (u, v)}.
                Only includes features visible in at least 2 frames.
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Detect features in the first frame
    pts0 = cv2.goodFeaturesToTrack(
        images[0], maxCorners=max_features, qualityLevel=0.01, minDistance=10
    )
    if pts0 is None or len(pts0) == 0:
        return {}

    # tracks[track_id] = {frame_idx: (u, v)}
    tracks: dict[int, dict[int, np.ndarray]] = {}
    for i, pt in enumerate(pts0):
        tracks[i] = {0: pt[0]}  # pt[0] is shape (2,)

    active_pts = pts0.copy()        # (N, 1, 2) — currently tracked points
    active_ids = list(range(len(pts0)))  # which track IDs are still alive

    for frame_idx in range(1, len(images)):
        if len(active_pts) == 0:
            break

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            images[frame_idx - 1], images[frame_idx], active_pts, None, **lk_params
        )

        # Also run backwards to check consistency (forward-backward error)
        prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            images[frame_idx], images[frame_idx - 1], next_pts, None, **lk_params
        )
        fb_error = np.linalg.norm(
            active_pts.reshape(-1, 2) - prev_pts_back.reshape(-1, 2), axis=1
        )
        valid = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 1.0)

        new_active_pts = []
        new_active_ids = []
        for i, (ok, pt, track_id) in enumerate(zip(valid, next_pts, active_ids)):
            if ok:
                tracks[track_id][frame_idx] = pt[0]
                new_active_pts.append(pt)
                new_active_ids.append(track_id)

        active_pts = np.array(new_active_pts) if new_active_pts else np.empty((0, 1, 2))
        active_ids = new_active_ids

    # Only return tracks visible in at least 2 frames
    return {tid: obs for tid, obs in tracks.items() if len(obs) >= 2}


# =============================================================================
# TEMPORAL: Uncertainty from multiple triangulations of the same feature
# =============================================================================

def estimate_temporal_uncertainty(
    observations_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 3D position and uncertainty from multiple triangulations of the same feature.

    When a feature is observed in N frames, we can triangulate it from C(N,2) frame
    pairs. Each triangulation yields a slightly different 3D estimate due to different
    baselines, viewing angles, and accumulated pose drift. Rather than assuming a
    noise model and propagating it analytically, we treat these triangulations as
    samples and fit a Gaussian directly.

    Think about:
        - When will the samples be tightly clustered vs. spread out?
        - How does the number of frame pairs affect estimate quality?
        - What if some triangulations are outliers (bad pose, degenerate geometry)?
        - How does this compare to the Jacobian-based covariance in estimate_uncertainty?

    Args:
        observations_3d: (K, 3) array of K triangulated 3D estimates of the
                         same feature, each from a different frame pair.

    Returns:
        mean: (3,) estimated 3D position (sample mean).
        covariance: (3, 3) sample covariance matrix encoding 3D uncertainty.
                    Returns a large diagonal matrix if only 1 observation.
    """
    mean = observations_3d.mean(axis=0)
    if len(observations_3d) < 2:
        return mean, np.eye(3) * 1e6

    centered = observations_3d - mean
    covariance = (centered.T @ centered) / (len(observations_3d) - 1)
    return mean, covariance


# =============================================================================
# Pipeline runners
# =============================================================================

def _load_sequence(sequence_path: str, dataset: str):
    """Load a dataset sequence by name.  Supported: 'kitti', 'eth3d'."""
    if dataset == "kitti":
        return KITTISequence(sequence_path)
    elif dataset == "eth3d":
        return ETH3DSequence(sequence_path)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose 'kitti' or 'eth3d'.")


def run_temporal_pipeline(sequence_path: str, start_frame: int = 0, n_frames: int = 5,
                          dataset: str = "kitti"):
    """Run the temporal uncertainty estimation pipeline on a KITTI or ETH3D sequence.

    Core idea:
        1. Track features across N consecutive left-camera frames.
        2. For each consecutive pair (i, i+1), estimate relative pose (R, t) using
          estimate_fundamental / recover_pose functions.
        3. Accumulate poses so every frame's projection matrix is expressed in the
           world frame (frame 0 coordinates).
        4. For each tracked feature, triangulate from every pair of frames that
           sees it —> giving multiple independent 3D estimates.
        5. Fit a Gaussian to those estimates: mean = best 3D position,
           covariance = spread of triangulations = empirical 3D uncertainty.

    Why is step 4 interesting?
        - A feature seen in frames 0, 2, 4 has baseline 0→2, 0→4, and 2→4.
        - Long baselines → low depth uncertainty but may lose the feature.
        - Short baselines → nearly parallel rays → high depth uncertainty.
        - The covariance captures this geometry automatically.

    Args:
        sequence_path: path to KITTI sequence directory.
        start_frame: index of the first frame to use.
        n_frames: number of consecutive frames to process.
        dataset: dataset type — 'kitti' or 'eth3d'.
    """
    seq = _load_sequence(sequence_path, dataset)
    calib = seq.calibration
    K = calib.K_left

    end_frame = min(start_frame + n_frames, len(seq))
    actual_n = end_frame - start_frame
    print(f"Loading {actual_n} frames [{start_frame}, {end_frame})...")
    images = [seq[i].left for i in range(start_frame, end_frame)]

    # --- Step 1: Track features across all frames ---
    print("\n[Step 1] Tracking features across frames...")
    tracks = track_features_across_frames(images, max_features=500)
    print(f"  {len(tracks)} tracks visible in ≥2 frames")

    # --- Step 2: Estimate relative pose for each consecutive pair ---
    # poses[i] = (R_i, t_i) expressing frame i's camera in world (frame 0) coords.
    # Projection matrix: P_i = K @ [R_i | t_i]
    print("\n[Step 2] Estimating relative poses between consecutive frames...")
    R_world = np.eye(3)
    t_world = np.zeros(3)
    poses = [(R_world.copy(), t_world.copy())]

    for i in range(actual_n - 1):
        # Collect matched points from tracks spanning both frames
        pts_a, pts_b = [], []
        for obs in tracks.values():
            if i in obs and (i + 1) in obs:
                pts_a.append(obs[i])
                pts_b.append(obs[i + 1])

        if len(pts_a) < 8:
            print(f"  Warning: only {len(pts_a)} matches for pair ({i},{i+1}) — duplicating last pose")
            poses.append(poses[-1])
            continue

        pts_a = np.array(pts_a, dtype=np.float32)  # (M, 2)
        pts_b = np.array(pts_b, dtype=np.float32)

        # Build identity-index matches: point k in pts_a <-> point k in pts_b
        M = len(pts_a)
        identity_matches = np.column_stack([np.arange(M), np.arange(M)])

        F, inlier_mask = estimate_fundamental(pts_a, pts_b, identity_matches)
        E = fundamental_to_essential(F, K, K)

       
        R_rel, t_rel = recover_pose(E, pts_a, pts_b, identity_matches, inlier_mask, K)

        # Accumulate: R_new @ X_world + t_new = X_cam_{i+1}
        R_prev, t_prev = poses[-1]
        R_new = R_rel @ R_prev
        t_new = R_rel @ t_prev + t_rel
        poses.append((R_new, t_new))

        n_inliers = inlier_mask.sum()
        print(f"  Pair ({i},{i+1}): {n_inliers}/{M} inliers")

    # --- Step 3: Triangulate each feature from all frame pairs ---
    print("\n[Step 3] Triangulating features from all frame pairs and fitting distributions...")

    # Precompute projection matrices for all frames
    proj_matrices = [K @ np.hstack([R, t[:, None]]) for R, t in poses]

    all_means = []
    all_covariances = []
    all_pts_frame0 = []   # frame-0 pixel coord for each feature (for 2D overlay)

    for track_id, obs in tracks.items():
        frame_ids = sorted(obs.keys())
        if len(frame_ids) < 2:
            continue

        # Triangulate from every pair of frames (i, j) with i < j
        triangulations = []
        for a in range(len(frame_ids)):
            for b in range(a + 1, len(frame_ids)):
                fi, fj = frame_ids[a], frame_ids[b]
                P_i = proj_matrices[fi]
                P_j = proj_matrices[fj]
                pt_i = obs[fi][np.newaxis]  # (1, 2)
                pt_j = obs[fj][np.newaxis]
                pts_3d = triangulate_points(P_i, P_j, pt_i, pt_j)  # (1, 3)
                triangulations.append(pts_3d[0])

        if not triangulations:
            continue

        triangulations = np.array(triangulations)  # (K, 3)
        mean, cov = estimate_temporal_uncertainty(triangulations)
        all_means.append(mean)
        all_covariances.append(cov)
        all_pts_frame0.append(obs[0])  # (u, v) in frame 0; all tracks start there

    all_means = np.array(all_means)
    all_covariances = np.array(all_covariances)
    all_pts_frame0 = np.array(all_pts_frame0)  # (N, 2)

    print(f"  3D positions + uncertainty estimated for {len(all_means)} features")

    # Filter to points in front of camera (positive depth, reasonable range)
    depth_mask = (all_means[:, 2] > 0.5) & (all_means[:, 2] < 200.0)
    print(f"  {depth_mask.sum()} points with positive depth in [0.5, 200] m")
    all_means = all_means[depth_mask]
    all_covariances = all_covariances[depth_mask]
    all_pts_frame0 = all_pts_frame0[depth_mask]

    # Filter outlier covariances: keep points within 50x the median trace
    traces = np.trace(all_covariances, axis1=1, axis2=2)
    median_trace = np.median(traces)
    print(f"  Covariance trace: min={traces.min():.3f}, median={median_trace:.3f}, "
          f"p95={np.percentile(traces, 95):.3f}, max={traces.max():.3f}")
    cov_mask = traces < median_trace * 50
    print(f"  Keeping {cov_mask.sum()} points (trace < 50 × median = {median_trace * 50:.2f})")
    all_means = all_means[cov_mask]
    all_covariances = all_covariances[cov_mask]
    all_pts_frame0 = all_pts_frame0[cov_mask]

    # --- Visualize: 2D overlay + 3D reconstruction side by side ---
    print("\n[Viz] Showing frame-0 features by depth alongside 3D reconstruction...")
    fig = draw_features_by_depth(
        images[0], all_pts_frame0, all_means[:, 2],
        title=f"Frame 0 Features by Depth ({actual_n} frames)",
    )
    img_size = images[0].shape[:2]  # (H, W)
    visualize_reconstruction(
        all_means, all_covariances, poses, K, img_size,
        alongside_fig=fig,
        title=f"Temporal Triangulation ({actual_n} frames)",
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D uncertainty estimation pipeline on KITTI or ETH3D sequences"
    )
    parser.add_argument("--sequence", type=str, required=True,
                        help="Path to sequence directory "
                             "(KITTI: dataset/sequences/00  |  ETH3D: path/to/cables_1)")
    parser.add_argument("--dataset", type=str, default="kitti", choices=["kitti", "eth3d"],
                        help="Dataset type (default: kitti)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Starting frame index (default: 0)")
    parser.add_argument("--n-frames", type=int, default=5,
                        help="Number of consecutive frames to use in temporal mode (default: 5)")
    args = parser.parse_args()

    
    run_temporal_pipeline(args.sequence, args.frame, args.n_frames, dataset=args.dataset)
