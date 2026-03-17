"""Stereo reprojection geometry.

Provides two operations:

    depth_from_disparity  — compute per-keypoint depth from L/R u-coordinates.
    reproject             — backproject from source to 3D and reproject into dest.

Depth is privileged information: it enters only the loss computation, never
the network input. The stereo feature matches supply the disparity without
requiring a separate dense stereo matcher.

Also provides sample_at_kps for sampling a dense covariance map at
(possibly sub-pixel) keypoint locations.
"""

import torch
import torch.nn.functional as F


def depth_from_disparity(
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    focal: float,
    baseline: float,
    min_depth: float = 0.1,
    max_depth: float = 200.0,
) -> torch.Tensor:
    """Compute depth from stereo disparity.

        depth = focal * baseline / (u_left − u_right)

    Args:
        u_left:    (P,) u-coordinates of keypoints in the left image.
        u_right:   (P,) u-coordinates of the matched keypoints in the right image.
        focal:     focal length in pixels (assumed equal for both cameras after
                   rectification).
        baseline:  stereo baseline in metres.
        min_depth: depth values below this are clamped (removes degenerate matches).
        max_depth: depth values above this are clamped (removes sky / far points).

    Returns:
        (P,) depth in metres, clamped to [min_depth, max_depth].
    """
    disparity = (u_left - u_right).clamp(min=1e-3)
    return (focal * baseline / disparity).clamp(min_depth, max_depth)


def reproject(
    kp_src: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    T_src_dst: torch.Tensor,
) -> torch.Tensor:
    """Backproject keypoints to 3D and reproject into the destination image.

        p_3d      = depth * K^{-1} @ [u, v, 1]^T        (left camera frame)
        p_dst_3d  = T_src_dst @ p_3d_h                   (destination frame)
        kp_reproj = K @ p_dst_3d[:3] / p_dst_3d[2]       (pixel coords)

    Args:
        kp_src:    (P, 2) keypoints in the source image (u, v).
        depth:     (P,) depth of each keypoint in the source camera frame.
        K:         (3, 3) shared camera intrinsics.
        T_src_dst: (4, 4) SE3 transform from source to destination frame.

    Returns:
        (P, 2) reprojected pixel coordinates in the destination image.
    """
    K_inv = torch.linalg.inv(K)
    homo = F.pad(kp_src, (0, 1), value=1.0)            # (P, 3)
    pts_src = depth[:, None] * (K_inv @ homo.T).T       # (P, 3) in source frame
    pts_src_h = F.pad(pts_src, (0, 1), value=1.0)       # (P, 4)
    pts_dst = (T_src_dst @ pts_src_h.T).T               # (P, 4)
    px = (K @ pts_dst[:, :3].T).T                       # (P, 3)
    return px[:, :2] / px[:, 2:3]                       # (P, 2)


def sample_at_kps(
    cov_map: torch.Tensor,
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sample a dense covariance map at keypoint locations.

    Args:
        cov_map:   (H, W, 2, 2) per-pixel covariance map (single image).
        keypoints: (P, 2) pixel coordinates (u, v) — may be sub-pixel.

    Returns:
        (P, 2, 2) covariances at the keypoint locations.
    """
    H, W = cov_map.shape[:2]

    # Flatten 2x2 → 4 channels, reshape to (1, 4, H, W) for grid_sample
    feat = cov_map.reshape(H, W, 4).permute(2, 0, 1).unsqueeze(0).float()  # (1, 4, H, W)

    # Normalise keypoints to [-1, 1].  grid_sample expects (x, y) = (col, row).
    grid_x = keypoints[:, 0] * 2.0 / (W - 1) - 1.0   # u → x
    grid_y = keypoints[:, 1] * 2.0 / (H - 1) - 1.0   # v → y
    grid = torch.stack([grid_x, grid_y], dim=-1)        # (P, 2)
    grid = grid[None, None, :, :].float()               # (1, 1, P, 2)

    sampled = F.grid_sample(
        feat, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )  # (1, 4, 1, P)

    return sampled[0, :, 0, :].T.reshape(-1, 2, 2)     # (P, 2, 2)
