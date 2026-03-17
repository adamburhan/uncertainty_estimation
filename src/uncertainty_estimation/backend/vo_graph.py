"""GTSAM factor graph for Visual Odometry pose optimisation."""

import numpy as np
import gtsam


def _pose3_from_Rt(R_cw: np.ndarray, t_cw: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(R_cw), t_cw)


def build_and_optimise(
    initial_poses: list[tuple[np.ndarray, np.ndarray]],
    initial_landmarks: dict[int, np.ndarray],
    observations: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    fix_landmarks: bool = True,
    fix_scale: bool = False,
) -> tuple[list[gtsam.Pose3], float, float]:
    """Build the VO factor graph and return optimised poses.

    Args:
        initial_poses:     [(R_cw, t_cw), ...] camera-to-world, one per frame.
        initial_landmarks: {track_id: (3,)} world-space 3D points (held fixed).
        observations:      {(track_id, frame_id): (z_2d (2,), Sigma_2x2 (2,2))}.
        K:                 (3,3) camera intrinsics.
        fix_landmarks:     if True, add tight priors to freeze landmarks.

    Returns:
        (optimised_poses, initial_error, final_error)
    """
    P = gtsam.symbol_shorthand.P
    L = gtsam.symbol_shorthand.L

    cal = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])
    graph = gtsam.NonlinearFactorGraph()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))
    graph.add(gtsam.PriorFactorPose3(P(0), _pose3_from_Rt(*initial_poses[0]), prior_noise))
    if fix_scale and len(initial_poses) > 1:
        graph.add(gtsam.PriorFactorPose3(P(1), _pose3_from_Rt(*initial_poses[1]), prior_noise))

    for (tid, fid), (z_2d, cov_2x2) in observations.items():
        if tid not in initial_landmarks:
            continue
        noise = gtsam.noiseModel.Gaussian.Covariance(cov_2x2)
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(z_2d[0], z_2d[1]), noise, P(fid), L(tid), cal,
        ))

    initial = gtsam.Values()
    for i, (R, t) in enumerate(initial_poses):
        initial.insert(P(i), _pose3_from_Rt(R, t))
    for tid, X in initial_landmarks.items():
        initial.insert(L(tid), gtsam.Point3(X))

    if fix_landmarks:
        lm_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 3))
        for tid, X in initial_landmarks.items():
            graph.add(gtsam.PriorFactorPoint3(L(tid), gtsam.Point3(X), lm_noise))

    initial_error = graph.error(initial)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    final_error = graph.error(result)

    return [result.atPose3(P(i)) for i in range(len(initial_poses))], initial_error, final_error


def _perturb_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
    rot_std_deg: float,
    trans_std_m: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from scipy.spatial.transform import Rotation
    noisy = []
    for R, t in poses:
        rot_noise = Rotation.from_rotvec(rng.normal(0, np.radians(rot_std_deg), 3)).as_matrix()
        noisy.append((rot_noise @ R, t + rng.normal(0, trans_std_m, 3)))
    return noisy


def _ate(gt_poses: list[tuple[np.ndarray, np.ndarray]], opt_poses: list[gtsam.Pose3]) -> float:
    errors = [np.linalg.norm(t - p.translation()) for (_, t), p in zip(gt_poses, opt_poses)]
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def _backproject_landmarks(
    tracks: dict[int, dict[int, np.ndarray]],
    frames,
    gt_poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    max_depth: float = 10.0,
) -> dict[int, np.ndarray]:
    """Backproject GT depth to world-space landmarks for each track."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = frames[0].depth.shape
    landmarks = {}
    for tid, obs in tracks.items():
        for fid in sorted(obs):
            u, v = obs[fid]
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            d = float(frames[fid].depth[vi, ui])
            if not (0 < d < max_depth):
                continue
            X_cam = np.array([(ui - cx) / fx * d, (vi - cy) / fy * d, d])
            R_cw, t_cw = gt_poses[fid]
            landmarks[tid] = R_cw @ X_cam + t_cw
            break
    return landmarks


def _filter_landmarks(
    landmarks: dict[int, np.ndarray],
    tracks: dict[int, dict[int, np.ndarray]],
    gt_poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    max_median_reproj_px: float = 5.0,
) -> dict[int, np.ndarray]:
    """Keep landmarks whose median reprojection error across all frames is low."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    kept = {}
    for tid, X_world in landmarks.items():
        errors = []
        for fid, z_2d in tracks[tid].items():
            R_cw, t_cw = gt_poses[fid]
            X_cam = R_cw.T @ X_world - R_cw.T @ t_cw
            if X_cam[2] <= 0:
                continue
            u = fx * X_cam[0] / X_cam[2] + cx
            v = fy * X_cam[1] / X_cam[2] + cy
            errors.append(np.linalg.norm(z_2d - np.array([u, v])))
        if errors and float(np.median(errors)) <= max_median_reproj_px:
            kept[tid] = X_world
    return kept


def _triangulate_landmarks_from_poses(
    tracks: dict[int, dict[int, np.ndarray]],
    poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    min_frame_gap: int = 3,
    max_reproj_px: float = 5.0,
    min_depth: float = 1e-3,
) -> dict[int, np.ndarray]:
    """Triangulate world-space landmarks using the given poses (not GT depth).

    For each track, picks the pair (f0, f1) with the largest frame gap >= min_frame_gap
    to maximise baseline. Applies cheirality + reprojection error filtering.
    """
    from uncertainty_estimation.frontend.pose import triangulate

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def proj(X, R, t):
        Xc = R.T @ (X - t)
        return np.array([fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy])

    landmarks = {}
    for tid, obs in tracks.items():
        fids = [f for f in sorted(obs) if f < len(poses)]
        if len(fids) < 2:
            continue

        # Pick pair with largest gap >= min_frame_gap; fall back to largest available gap
        best = None
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                gap = fids[j] - fids[i]
                if best is None or gap > best[0]:
                    best = (gap, fids[i], fids[j])
        _, f0, f1 = best
        if f1 - f0 < min_frame_gap:
            continue

        R0, t0 = poses[f0]
        R1, t1 = poses[f1]
        Rt0 = np.hstack([R0.T, (-R0.T @ t0)[:, None]])
        Rt1 = np.hstack([R1.T, (-R1.T @ t1)[:, None]])
        pts = triangulate(K @ Rt0, K @ Rt1, obs[f0][None], obs[f1][None])
        X = pts[0]

        # Cheirality
        if (R0.T @ (X - t0))[2] <= min_depth or (R1.T @ (X - t1))[2] <= min_depth:
            continue

        # Reprojection filter on both frames
        e0 = np.linalg.norm(proj(X, R0, t0) - obs[f0])
        e1 = np.linalg.norm(proj(X, R1, t1) - obs[f1])
        if e0 > max_reproj_px or e1 > max_reproj_px:
            continue

        landmarks[tid] = X
    return landmarks


def _estimate_poses_from_tracks(
    tracks: dict[int, dict[int, np.ndarray]],
    n_frames: int,
    K: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chain pairwise relative poses into absolute poses (unit / scale-ambiguous).

    For each consecutive frame pair (fid, fid+1), collects matched points from
    tracks visible in both frames, estimates R and unit-t via E decomposition,
    then accumulates into global poses.
    """
    from uncertainty_estimation.frontend.pose import estimate_relative_pose

    R_global, t_global = np.eye(3), np.zeros(3)
    poses = [(R_global.copy(), t_global.copy())]

    for fid in range(n_frames - 1):
        tids = [tid for tid in tracks if fid in tracks[tid] and fid + 1 in tracks[tid]]
        if len(tids) < 8:
            # Not enough points — carry forward previous pose
            poses.append(poses[-1])
            continue
        pts1 = np.array([tracks[tid][fid]     for tid in tids])
        pts2 = np.array([tracks[tid][fid + 1] for tid in tids])
        R_rel, t_rel, _ = estimate_relative_pose(pts1, pts2, K)
        # Accumulate: new_global = R_rel @ old_global, t_rel is in the previous frame
        t_global = R_rel @ t_global + t_rel
        R_global = R_rel @ R_global
        poses.append((R_global.copy(), t_global.copy()))

    return poses


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from uncertainty_estimation.frontend.lk import LKTracker
    from uncertainty_estimation.uncertainty.temporal import TemporalUncertainty
    from uncertainty_estimation.uncertainty.isotropic import IsotropicUncertainty

    if len(sys.argv) not in (3, 4):
        print("Usage: uv run python -m uncertainty_estimation.backend.vo_graph <tartanair|thales> <path> [n_frames]")
        sys.exit(1)

    dataset_type = sys.argv[1]
    seq_path = sys.argv[2]
    n_frames = int(sys.argv[3]) if len(sys.argv) == 4 else 10

    if dataset_type == "tartanair":
        from uncertainty_estimation.data.tartanair import TartanAirSequence, TARTANAIR_K
        seq = TartanAirSequence(seq_path)
        frames = [seq[i] for i in range(min(n_frames, len(seq)))]
        images = [f.image for f in frames]
        gt_poses = [(f.pose[:3, :3], f.pose[:3, 3]) for f in frames]
        K = TARTANAIR_K
        has_gt = True
    elif dataset_type == "thales":
        from uncertainty_estimation.data.thales import ThalesSequence
        seq = ThalesSequence(seq_path)
        frames = [seq[i] for i in range(min(n_frames, len(seq)))]
        images = [f.image for f in frames]
        gt_poses = None
        K = seq.K
        has_gt = False
    else:
        print(f"Unknown dataset type '{dataset_type}'. Choose 'tartanair' or 'thales'.")
        sys.exit(1)

    print(f"Loaded {len(frames)} frames")

    tracker = LKTracker(max_features=600, min_tracks=150)
    tracks = tracker.track(images)
    print(f"Tracked {len(tracks)} features")

    estimated_poses = _estimate_poses_from_tracks(tracks, n_frames, K)
    estimated_landmarks = _triangulate_landmarks_from_poses(tracks, estimated_poses, K)
    print(f"Triangulated landmarks (estimated poses): {len(estimated_landmarks)}")

    estimators = [
        ("isotropic sigma=1",   IsotropicUncertainty(sigma=1.0),   "tab:orange"),
        ("isotropic sigma=5",   IsotropicUncertainty(sigma=5.0),   "tab:red"),
        ("isotropic sigma=30",  IsotropicUncertainty(sigma=30.0),  "tab:purple"),
        ("isotropic sigma=50",  IsotropicUncertainty(sigma=50.0),  "tab:brown"),
        ("temporal",            TemporalUncertainty(max_depth=100.0, min_frame_gap=3, target_median_trace=16.0), "tab:cyan"),
    ]

    if has_gt:
        landmarks = _backproject_landmarks(tracks, frames, gt_poses, K, max_depth=1000.0)
        landmarks = _filter_landmarks(landmarks, tracks, gt_poses, K, max_median_reproj_px=10.0)
        print(f"Landmarks after filter (≤10px median reproj): {len(landmarks)}")

        def run_suite(initial_poses, label_prefix, lms=None):
            lms = lms if lms is not None else landmarks
            ate_before = _ate(gt_poses, [gtsam.Pose3(gtsam.Rot3(R), t) for R, t in initial_poses])
            print(f"\n{'='*55}\n{label_prefix}  ATE before: {ate_before:.4f} m\n{'='*55}")
            results = []
            for label, estimator, colour in estimators:
                obs_covs = estimator.estimate(tracks, gt_poses, K)
                observations = {(tid, fid): (tracks[tid][fid], cov)
                                for (tid, fid), cov in obs_covs.items() if tid in lms}
                optimised, e0, ef = build_and_optimise(initial_poses, lms, observations, K,
                                                        fix_scale=(label_prefix == "estimated"))
                ate = _ate(gt_poses, optimised)
                print(f"  {label:22s}  ATE: {ate:.4f} m   (err {e0:.1f}→{ef:.1f})")
                results.append((optimised, colour, label))
            return results

        rng = np.random.default_rng(42)
        small_perturbed = _perturb_poses(gt_poses, rot_std_deg=0.5, trans_std_m=0.02, rng=rng)
        rng = np.random.default_rng(42)
        large_perturbed = _perturb_poses(gt_poses, rot_std_deg=1.5, trans_std_m=0.05, rng=rng)

        suites = [
            (run_suite(small_perturbed,  "small (0.5 degrees,2cm)"),  small_perturbed,  "small perturbation"),
            (run_suite(large_perturbed,  "large (1.5 degrees,5cm)"),  large_perturbed,  "large perturbation"),
            (run_suite(estimated_poses,  "estimated", lms=estimated_landmarks), estimated_poses, "estimated poses"),
        ]

        gt_xyz = np.array([t for _, t in gt_poses])
        for results, initial, title in suites:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], "b-o", markersize=3, lw=2, label="GT")
            ax.plot([t[0] for _, t in initial], [t[2] for _, t in initial],
                    "k--", lw=1, alpha=0.4, label="initial")
            for optimised, colour, label in results:
                opt_xyz = np.array([p.translation() for p in optimised])
                ax.plot(opt_xyz[:, 0], opt_xyz[:, 2], "--o", color=colour, markersize=3, lw=1.5, label=label)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
            ax.set_title(f"Trajectory — {title}")
            ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.4); ax.set_aspect("equal")
            plt.tight_layout()

    else:
        # No GT — estimated poses only, visualise trajectory shape
        def run_no_gt(initial_poses, lms):
            print(f"\n{'='*55}\nestimated  graph error before BA\n{'='*55}")
            results = []
            for label, estimator, colour in estimators:
                obs_covs = estimator.estimate(tracks, initial_poses, K)
                observations = {(tid, fid): (tracks[tid][fid], cov)
                                for (tid, fid), cov in obs_covs.items() if tid in lms}
                optimised, e0, ef = build_and_optimise(initial_poses, lms, observations, K, fix_scale=True)
                print(f"  {label:22s}  graph err {e0:.1f}→{ef:.1f}")
                results.append((optimised, colour, label))
            return results

        results = run_no_gt(estimated_poses, estimated_landmarks)

        fig, ax = plt.subplots(figsize=(9, 6))
        est_xyz = np.array([t for _, t in estimated_poses])
        ax.plot(est_xyz[:, 1], est_xyz[:, 2], "k--o", markersize=3, lw=1, alpha=0.5, label="initial estimate")
        for optimised, colour, label in results:
            opt_xyz = np.array([p.translation() for p in optimised])
            ax.plot(opt_xyz[:, 1], opt_xyz[:, 2], "--o", color=colour, markersize=3, lw=1.5, label=label)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.set_title("Trajectory — estimated poses (scale-ambiguous, no GT)")
        ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.4); ax.set_aspect("equal")
        plt.tight_layout()

    plt.show()
