"""Stereo reprojection utilities.

These helpers are shared by the privileged-information stereo training setup
and evaluation code. Depth enters through disparity during training but is
never a model input at inference time.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def _as_broadcastable(param: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Convert a scalar or tensor parameter to a shape broadcastable with ref."""
    tensor = torch.as_tensor(param, dtype=ref.dtype, device=ref.device)
    while tensor.ndim < ref.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


def depth_from_disparity(
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    focal: float | torch.Tensor,
    baseline: float | torch.Tensor,
    min_depth: float = 0.1,
    max_depth: float = 200.0,
) -> torch.Tensor:
    """Compute depth from rectified stereo disparity.

    Supports either unbatched inputs ``(P,)`` or batched inputs ``(B, P)``.
    ``focal`` and ``baseline`` may be scalars or tensors broadcastable to the
    disparity shape, such as ``(B,)``.
    """
    disparity = (u_left - u_right).clamp(min=1e-3)
    focal = _as_broadcastable(focal, disparity)
    baseline = _as_broadcastable(baseline, disparity)
    return (focal * baseline / disparity).clamp(min_depth, max_depth)


def reproject(
    kp_src: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    T_src_dst: torch.Tensor,
) -> torch.Tensor:
    """Backproject source keypoints to 3D and reproject them into the target image.

    Supports either:
        kp_src: ``(P, 2)``, depth: ``(P,)``, K: ``(3, 3)``, T_src_dst: ``(4, 4)``
    or batched inputs:
        kp_src: ``(B, P, 2)``, depth: ``(B, P)``, K: ``(B, 3, 3)`` or ``(3, 3)``,
        T_src_dst: ``(B, 4, 4)`` or ``(4, 4)``.
    """
    squeeze_batch = kp_src.ndim == 2
    if squeeze_batch:
        kp_src = kp_src.unsqueeze(0)
        depth = depth.unsqueeze(0)

    batch_size = kp_src.shape[0]

    if K.ndim == 2:
        K = K.unsqueeze(0).expand(batch_size, -1, -1)
    if T_src_dst.ndim == 2:
        T_src_dst = T_src_dst.unsqueeze(0).expand(batch_size, -1, -1)

    K_inv = torch.linalg.inv(K)                                        # (B, 3, 3)
    homo = F.pad(kp_src, (0, 1), value=1.0)                            # (B, P, 3)
    rays = torch.einsum("bij,bpj->bpi", K_inv, homo)                   # (B, P, 3)
    pts_src = depth.unsqueeze(-1) * rays                               # (B, P, 3)
    pts_src_h = F.pad(pts_src, (0, 1), value=1.0)                      # (B, P, 4)
    pts_dst = torch.einsum("bij,bpj->bpi", T_src_dst, pts_src_h)       # (B, P, 4)
    px = torch.einsum("bij,bpj->bpi", K, pts_dst[..., :3])             # (B, P, 3)
    kp_reproj = px[..., :2] / px[..., 2:3]                             # (B, P, 2)

    if squeeze_batch:
        return kp_reproj.squeeze(0)
    return kp_reproj


def extract_covs(
    img_covs: torch.Tensor,  # B*2, H, W, 2, 2  (interleaved: left=0, right=1)
    left_kps: torch.Tensor,  # B, P, 2  (pixel coords, xy)
    right_kps: torch.Tensor, # B, P, 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = left_kps.shape[0]
    left_covs = img_covs[0::2]   # B, H, W, 2, 2
    right_covs = img_covs[1::2]  # B, H, W, 2, 2

    H, W = img_covs.shape[1], img_covs.shape[2]

    def sample(covs, kps):
        # grid_sample requires coordinates in [-1, +1], not pixels -> normalize kps accordingly
        # covs: B, H, W, 2, 2  →  grid_sample expects B, C, H, W
        # kps: B, P, 2 (xy pixel coords)
        norm = kps.clone().float()
        norm[..., 0] = (kps[..., 0] * 2.0 / (W - 1)) - 1.0  # x
        norm[..., 1] = (kps[..., 1] * 2.0 / (H - 1)) - 1.0  # y
        flat = covs.flatten(-2, -1).permute(0, 3, 1, 2)  # B, 4, H, W
        out = torch.nn.functional.grid_sample(flat, norm[:, None, :, :], mode="nearest", align_corners=True)
        # out: B, 4, 1, P  →  B, P, 2, 2
        return out[:, :, 0, :].permute(0, 2, 1).reshape(B, -1, 2, 2)

    return sample(left_covs, left_kps), sample(right_covs, right_kps)
