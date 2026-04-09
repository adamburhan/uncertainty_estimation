"""Empirical residual analysis using TartanAir stereo pairs with GT depth.

For each frame in each sequence:
  - Load left image + GT depth, right image
  - Match keypoints with ORB + BFMatcher (no stereo geometric constraints)
  - Look up GT depth at matched left keypoint positions
  - Reproject: kp_left + depth → 3D in cam_left → T_lr → cam_right → kp_right_reproj
  - Residual: r = kp_right_observed - kp_right_reproj

T_lr is constant for the rigid stereo rig — computed once from frame-0 poses.

Outputs:
  outputs/residual_analysis.png  — 6-subplot figure
  stdout                         — full summary statistics
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from uncertainty_estimation.data.tartanair import (
    TartanAirSequence, TARTANAIR_K, pose_vec_to_matrix,
)
from uncertainty_estimation.geometry.stereo import reproject

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/Volumes/T9/datasets/tartanair-v2/ArchVizTinyHouseDay/Data_easy/ArchVizTinyHouseDay/Data_easy")
SEQUENCES = ["P000", "P001", "P002", "P003", "P004", "P005", "P006"]

ORB_MAX_KEYPOINTS = 500
ORB_MAX_HAMMING   = 64
MAX_DEPTH         = 200.0   # metres — skip sky/background
MAX_FRAMES        = None    # None = all frames; set e.g. 200 for a quick run

SEED = 42


# ---------------------------------------------------------------------------
# Minimal right-camera loader (left camera already handled by TartanAirSequence)
# ---------------------------------------------------------------------------
def load_right_image(seq_path: Path, idx: int) -> np.ndarray:
    """Load right RGB image for frame idx."""
    right_dir = seq_path / "image_rcam_front"
    files = sorted(p for p in right_dir.glob("*.png") if not p.name.startswith("._"))
    img = cv2.imread(str(files[idx]), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_right_poses(seq_path: Path) -> list[np.ndarray]:
    """Load all right-camera poses as (4,4) SE3 matrices."""
    lines = (seq_path / "pose_rcam_front.txt").read_text().splitlines()
    return [pose_vec_to_matrix(np.fromstring(l, sep=" ")) for l in lines if l.strip()]


def compute_T_lr(pose_left: np.ndarray, pose_right: np.ndarray) -> np.ndarray:
    """T_lr maps 3D points from left cam frame to right cam frame.

    pose_left/right are cam-to-world (SE3), so:
        P_world = pose_left  @ P_left
        P_right = inv(pose_right) @ P_world
        ⟹ T_lr  = inv(pose_right) @ pose_left
    """
    return np.linalg.inv(pose_right) @ pose_left


# ---------------------------------------------------------------------------
# Matching + outlier filtering
# ---------------------------------------------------------------------------
RESIDUAL_THRESHOLD = 10.0   # px — hard cap after essential-matrix RANSAC


def essential_matrix(T_lr: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute the fundamental matrix F from the known T_lr and K.

    E = [t]_× R,  F = K^{-T} E K^{-1}
    (same K assumed for both cameras, as in TartanAir.)
    """
    R = T_lr[:3, :3]
    t = T_lr[:3, 3]
    t_skew = np.array([
        [ 0,    -t[2],  t[1]],
        [ t[2],  0,    -t[0]],
        [-t[1],  t[0],  0   ],
    ], dtype=np.float64)
    E = t_skew @ R
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


def sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson distance for each correspondence under fundamental matrix F.

    Args:
        F:    (3, 3) fundamental matrix
        pts1: (N, 2) points in image 1
        pts2: (N, 2) points in image 2

    Returns:
        (N,) Sampson distances (in pixels²)
    """
    p1 = np.column_stack([pts1, np.ones(len(pts1))])   # (N, 3)
    p2 = np.column_stack([pts2, np.ones(len(pts2))])   # (N, 3)
    Fp1  = (F @ p1.T).T    # (N, 3)
    FTp2 = (F.T @ p2.T).T  # (N, 3)
    num   = (np.sum(p2 * Fp1, axis=1)) ** 2
    denom = Fp1[:, 0]**2 + Fp1[:, 1]**2 + FTp2[:, 0]**2 + FTp2[:, 1]**2
    return num / (denom + 1e-10)


def match_orb(img1: np.ndarray, img2: np.ndarray, F: np.ndarray):
    """ORB + BFMatcher, filtered by Sampson distance then residual threshold.

    Args:
        img1, img2: RGB images
        F: (3, 3) fundamental matrix for geometric inlier filtering

    Returns:
        kp1: (M, 2) float32 (u, v) in img1  — inliers only
        kp2: (M, 2) float32 (u, v) in img2  — inliers only
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(ORB_MAX_KEYPOINTS)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1_cv, des1 = orb.detectAndCompute(g1, None)
    kp2_cv, des2 = orb.detectAndCompute(g2, None)

    if des1 is None or des2 is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    matches = [m for m in bf.match(des1, des2) if m.distance <= ORB_MAX_HAMMING]
    if not matches:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    pts1 = np.array([kp1_cv[m.queryIdx].pt for m in matches], np.float32)
    pts2 = np.array([kp2_cv[m.trainIdx].pt for m in matches], np.float32)

    # Essential-matrix RANSAC: filter geometric outliers (mismatches)
    sampson = sampson_distance(F.astype(np.float64),
                               pts1.astype(np.float64),
                               pts2.astype(np.float64))
    # Threshold: Sampson distance in px² — 1px² is tight for known geometry
    inliers = sampson < 9.0
    return pts1[inliers], pts2[inliers]


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------
def accumulate_residuals(max_frames: int | None = None) -> np.ndarray:
    """Return residuals (N, 2): r = kp_right_observed - kp_right_reproj."""
    K_t = torch.from_numpy(TARTANAIR_K.astype(np.float32))
    all_residuals = []
    frame_count = 0

    for seq_name in SEQUENCES:
        seq_path = DATA_ROOT / seq_name
        try:
            seq = TartanAirSequence(seq_path)
        except FileNotFoundError as e:
            print(f"  Skipping {seq_name}: {e}")
            continue

        right_poses = load_right_poses(seq_path)

        # T_lr and F are constant for the rigid rig — compute from frame 0
        T_lr = compute_T_lr(seq[0].pose, right_poses[0]).astype(np.float32)
        print(f"  {seq_name}: T_lr =\n{T_lr}")
        T_lr_t = torch.from_numpy(T_lr)
        F = essential_matrix(T_lr.astype(np.float64), TARTANAIR_K)

        n_frames = min(len(seq), max_frames - frame_count) if max_frames else len(seq)
        print(f"  {seq_name}: processing {n_frames} frames", flush=True)

        for i in range(n_frames):
            frame_left  = seq[i]
            img_right   = load_right_image(seq_path, i)

            kp_left, kp_right = match_orb(frame_left.image, img_right, F)
            if len(kp_left) == 0:
                continue

            # GT depth lookup at left keypoint positions
            H, W = frame_left.depth.shape
            ix = np.clip(np.round(kp_left[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.round(kp_left[:, 1]).astype(int), 0, H - 1)
            depth_vals = frame_left.depth[iy, ix]

            valid = np.isfinite(depth_vals) & (depth_vals > 0) & (depth_vals < MAX_DEPTH)
            if valid.sum() == 0:
                continue

            kp_l_v = kp_left[valid]
            kp_r_v = kp_right[valid]
            depth_v = depth_vals[valid]

            kp_right_reproj = reproject(
                torch.from_numpy(kp_l_v),
                torch.from_numpy(depth_v),
                K_t, T_lr_t,
            ).numpy()  # (M, 2)

            in_bounds = (
                (kp_right_reproj[:, 0] >= 0) & (kp_right_reproj[:, 0] < W) &
                (kp_right_reproj[:, 1] >= 0) & (kp_right_reproj[:, 1] < H)
            )
            if in_bounds.sum() == 0:
                continue

            r = kp_r_v[in_bounds] - kp_right_reproj[in_bounds]

            # Mild residual cap — removes any remaining outliers after RANSAC
            r = r[np.linalg.norm(r, axis=1) < RESIDUAL_THRESHOLD]
            all_residuals.append(r)
            frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    print(f"\n  Total frames processed: {frame_count}")
    return np.concatenate(all_residuals, axis=0)


# ---------------------------------------------------------------------------
# Statistics + plotting
# ---------------------------------------------------------------------------
def empirical_cov_and_eig(res: np.ndarray):
    cov = np.cov(res.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return cov, eigvals, eigvecs


def draw_cov_ellipses(ax, mean, cov, n_sigma=(1, 2, 3), color="C0"):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    for s in n_sigma:
        w = 2 * s * math.sqrt(eigvals[-1])
        h = 2 * s * math.sqrt(eigvals[0])
        ell = patches.Ellipse(
            mean, width=w, height=h, angle=angle,
            edgecolor=color, facecolor="none", linewidth=1.5, alpha=0.9,
            label=f"{s}σ" if s == n_sigma[0] else None,
        )
        ax.add_patch(ell)


def plot_residuals(res: np.ndarray, out_path: Path):
    cov, eigvals, eigvecs = empirical_cov_and_eig(res)
    mags   = np.linalg.norm(res, axis=1)
    angles = np.degrees(np.arctan2(res[:, 1], res[:, 0]))
    ZOOM   = 5.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Empirical reprojection residuals — TartanAir stereo (GT depth)\n"
        f"N = {len(res):,} valid residuals",
        fontsize=13,
    )

    # 1. Full scatter (99th-pct clip)
    ax = axes[0, 0]
    lim = np.percentile(np.abs(res), 99)
    ax.scatter(res[:, 0], res[:, 1], s=0.3, alpha=0.15, c="C0", rasterized=True)
    ax.axhline(0, color="k", lw=0.7, ls="--"); ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title("Scatter (99th-pct clip)")
    ax.set_xlabel("Δu (px)"); ax.set_ylabel("Δv (px)")

    # 2. Zoomed scatter ±ZOOM px + cov ellipses
    ax = axes[0, 1]
    mask = (np.abs(res[:, 0]) < ZOOM) & (np.abs(res[:, 1]) < ZOOM)
    ax.scatter(res[mask, 0], res[mask, 1], s=0.5, alpha=0.2, c="C0", rasterized=True)
    draw_cov_ellipses(ax, (0, 0), cov, color="red")
    ax.axhline(0, color="k", lw=0.7, ls="--"); ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_aspect("equal"); ax.set_xlim(-ZOOM, ZOOM); ax.set_ylim(-ZOOM, ZOOM)
    ax.set_title(f"Zoomed ±{ZOOM}px + empirical cov ellipses")
    ax.set_xlabel("Δu (px)"); ax.set_ylabel("Δv (px)"); ax.legend(fontsize=8)

    # 3. Marginal histograms
    ax = axes[0, 2]
    bins = np.linspace(-ZOOM, ZOOM, 120)
    ax.hist(res[:, 0], bins=bins, alpha=0.6, color="C0", density=True, label="Δu")
    ax.hist(res[:, 1], bins=bins, alpha=0.6, color="C1", density=True, label="Δv")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title("Marginal distributions (clipped ±5px)")
    ax.set_xlabel("residual (px)"); ax.set_ylabel("density"); ax.legend()

    # 4. Empirical cov ellipses + eigenvalue annotation
    ax = axes[1, 0]
    draw_cov_ellipses(ax, (0, 0), cov, color="C0")
    ax.axhline(0, color="k", lw=0.5, ls="--"); ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_aspect("equal")
    r_ax = max(3 * math.sqrt(eigvals[-1]), 1.0)
    ax.set_xlim(-r_ax, r_ax); ax.set_ylim(-r_ax, r_ax)
    λ1, λ2 = eigvals[0], eigvals[1]
    cond = λ2 / max(λ1, 1e-12)
    ax.set_title(
        f"Empirical cov ellipses (1σ, 2σ, 3σ)\n"
        f"λ1={λ1:.4f}  λ2={λ2:.4f}  cond={cond:.2f}\n"
        f"major axis: [{eigvecs[0,1]:.3f}, {eigvecs[1,1]:.3f}]",
        fontsize=9,
    )
    ax.set_xlabel("Δu (px)"); ax.set_ylabel("Δv (px)"); ax.legend(fontsize=8)

    # 5. Magnitude histogram
    ax = axes[1, 1]
    clip_m = np.percentile(mags, 98)
    ax.hist(mags, bins=np.linspace(0, clip_m, 100), alpha=0.7, density=True, color="C0")
    ax.axvline(np.median(mags), color="red", lw=1.5, ls="--",
               label=f"median={np.median(mags):.2f}px")
    ax.set_title("Residual magnitude ‖r‖ (98th-pct clip)")
    ax.set_xlabel("‖r‖ (px)"); ax.set_ylabel("density"); ax.legend(fontsize=8)

    # 6. Direction histogram
    ax = axes[1, 2]
    ax.hist(angles, bins=np.linspace(-180, 180, 73), alpha=0.7, density=True, color="C0")
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_title("Residual direction (°)\n0°=right, 90°=down")
    ax.set_xlabel("atan2(Δv, Δu) (°)"); ax.set_ylabel("density")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")


def print_stats(res: np.ndarray):
    cov, eigvals, eigvecs = empirical_cov_and_eig(res)
    mags = np.linalg.norm(res, axis=1)
    λ1, λ2 = eigvals[0], eigvals[1]
    cond = λ2 / max(λ1, 1e-12)

    print(f"\n{'='*60}")
    print(f"  N = {len(res):,} residuals")
    print(f"{'='*60}")
    print(f"  Mean:    Δu={res[:,0].mean():.4f}  Δv={res[:,1].mean():.4f}  px")
    print(f"  Std:     Δu={res[:,0].std():.4f}   Δv={res[:,1].std():.4f}   px")
    print(f"  Empirical covariance:")
    print(f"    [[{cov[0,0]:.5f}, {cov[0,1]:.5f}],")
    print(f"     [{cov[1,0]:.5f}, {cov[1,1]:.5f}]]")
    print(f"  Eigenvalues (ascending):  λ1={λ1:.5f}  λ2={λ2:.5f}")
    print(f"  Eigenvectors:")
    print(f"    v1 (minor) = [{eigvecs[0,0]:.4f}, {eigvecs[1,0]:.4f}]")
    print(f"    v2 (major) = [{eigvecs[0,1]:.4f}, {eigvecs[1,1]:.4f}]")
    print(f"  Condition number (λ2/λ1): {cond:.2f}")
    print(f"  Median ‖r‖:               {np.median(mags):.4f} px")
    print(f"  Mean   ‖r‖:               {mags.mean():.4f} px")
    print(f"  90th-pct ‖r‖:             {np.percentile(mags, 90):.4f} px")
    print(f"  Frac with |Δv| > |Δu|:   {np.mean(np.abs(res[:,1]) > np.abs(res[:,0])):.4f}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Data:      {DATA_ROOT}")
    print(f"Sequences: {SEQUENCES}")
    print(f"ORB:       max_keypoints={ORB_MAX_KEYPOINTS}  hamming≤{ORB_MAX_HAMMING}")
    print(f"Max depth: {MAX_DEPTH}m   Max frames: {MAX_FRAMES if MAX_FRAMES else 'all'}")
    print()

    res = accumulate_residuals(max_frames=MAX_FRAMES)
    print_stats(res)
    plot_residuals(res, Path("outputs/residual_analysis.png"))


if __name__ == "__main__":
    main()
