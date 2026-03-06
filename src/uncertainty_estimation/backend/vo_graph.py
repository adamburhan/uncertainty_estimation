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


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from uncertainty_estimation.data.tartanair import TartanAirSequence, TARTANAIR_K
    from uncertainty_estimation.frontend.lk import LKTracker
    from uncertainty_estimation.uncertainty.temporal import TemporalUncertainty
    from uncertainty_estimation.uncertainty.isotropic import IsotropicUncertainty

    if len(sys.argv) not in (2, 3):
        print("Usage: uv run python -m uncertainty_estimation.backend.vo_graph <path/to/P000> [n_frames]")
        sys.exit(1)

    seq = TartanAirSequence(sys.argv[1])
    n_frames = int(sys.argv[2]) if len(sys.argv) == 3 else 10
    frames = [seq[i] for i in range(min(n_frames, len(seq)))]
    images = [f.image for f in frames]
    gt_poses = [(f.pose[:3, :3], f.pose[:3, 3]) for f in frames]
    K = TARTANAIR_K
    print(f"Loaded {len(frames)} frames")

    tracker = LKTracker(max_features=600, min_tracks=150)
    tracks = tracker.track(images)
    print(f"Tracked {len(tracks)} features")

    landmarks = _backproject_landmarks(tracks, frames, gt_poses, K, max_depth=1000.0)
    landmarks = _filter_landmarks(landmarks, tracks, gt_poses, K, max_median_reproj_px=10.0)
    print(f"Landmarks after filter (≤10px median reproj): {len(landmarks)}")

    estimators = [
        ("isotropic sigma=1",  IsotropicUncertainty(sigma=1.0),  "tab:orange"),
        ("isotropic sigma=5",  IsotropicUncertainty(sigma=5.0),  "tab:red"),
        ("isotropic sigma=30",  IsotropicUncertainty(sigma=30.0),  "tab:purple"),
        ("isotropic sigma=50",  IsotropicUncertainty(sigma=50.0),  "tab:brown"),
        ("temporal t=4",   TemporalUncertainty(max_depth=10.0, min_frame_gap=3, target_median_trace=4.0),  "tab:green"),
        ("temporal t=16",  TemporalUncertainty(max_depth=10.0, min_frame_gap=3, target_median_trace=16.0), "tab:olive"),
        ("temporal t=64",  TemporalUncertainty(max_depth=10.0, min_frame_gap=3, target_median_trace=64.0), "tab:cyan"),
    ]

    def run_suite(perturbed_poses, label_prefix):
        ate_before = _ate(gt_poses, [gtsam.Pose3(gtsam.Rot3(R), t) for R, t in perturbed_poses])
        print(f"\n{'='*55}\n{label_prefix}  ATE before: {ate_before:.4f} m\n{'='*55}")
        results = []
        for label, estimator, colour in estimators:
            obs_covs = estimator.estimate(tracks, gt_poses, K)
            observations = {
                (tid, fid): (tracks[tid][fid], cov)
                for (tid, fid), cov in obs_covs.items()
                if tid in landmarks
            }
            optimised, e0, ef = build_and_optimise(perturbed_poses, landmarks, observations, K)
            ate = _ate(gt_poses, optimised)
            print(f"  {label:22s}  ATE: {ate:.4f} m   (err {e0:.1f}→{ef:.1f})")
            results.append((optimised, colour, f"{label_prefix} {label}"))
        return results

    rng = np.random.default_rng(42)
    small_perturbed = _perturb_poses(gt_poses, rot_std_deg=0.5, trans_std_m=0.02, rng=rng)
    rng = np.random.default_rng(42)
    large_perturbed = _perturb_poses(gt_poses, rot_std_deg=1.5, trans_std_m=0.05, rng=rng)

    small_results = run_suite(small_perturbed, "small (0.5 degrees,2cm)")
    large_results = run_suite(large_perturbed, "large (1.5 degrees,5cm)")

    gt_xyz = np.array([t for _, t in gt_poses])
    for results, perturbed, title in [
        (small_results, small_perturbed, "small perturbation (0.5 degrees, 2cm)"),
        (large_results, large_perturbed, "large perturbation (1.5 degrees, 5cm)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], "b-o", markersize=3, lw=2, label="GT")
        ax.plot([t[0] for _, t in perturbed], [t[2] for _, t in perturbed],
                "k--", lw=1, alpha=0.4, label="perturbed")
        for optimised, colour, label in results:
            opt_xyz = np.array([p.translation() for p in optimised])
            ax.plot(opt_xyz[:, 0], opt_xyz[:, 2], "--o", color=colour, markersize=3, lw=1.5, label=label)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
        ax.set_title(f"Trajectory — {title}")
        ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.4); ax.set_aspect("equal")
        plt.tight_layout()

    plt.show()
