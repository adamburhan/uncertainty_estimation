import torch
from typing import Tuple

from uncertainty_estimation.geometry.stereo import depth_from_disparity, reproject, extract_covs


def _lookup_depth(depth_map: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
    """Sample a dense depth map at keypoint locations (nearest-neighbour).

    Args:
        depth_map: (B, H, W) depth in metres.
        kps:       (B, P, 2) keypoints in pixel coords (x=col, y=row).

    Returns:
        (B, P) depth values at each keypoint.
    """
    B, H, W = depth_map.shape
    col = kps[..., 0].round().long().clamp(0, W - 1)  # B, P
    row = kps[..., 1].round().long().clamp(0, H - 1)  # B, P
    b_idx = torch.arange(B, device=kps.device).unsqueeze(1).expand_as(col)
    return depth_map[b_idx, row, col]  # B, P


def _get_depth(
    depth_source: str,
    batch: dict,
    left_kps: torch.Tensor,
    right_kps: torch.Tensor,
    focal: torch.Tensor,
    baseline: torch.Tensor,
    device: torch.device,
    max_depth: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-keypoint depth according to the configured depth source.

    Returns:
        (B, P) depth tensor on `device`.
    """
    if depth_source == "orb_disparity":
            depth_left = depth_from_disparity(left_kps[..., 0], right_kps[..., 0], focal, baseline)
            depth_right = depth_from_disparity(left_kps[..., 0], right_kps[..., 0], focal, baseline)
            return depth_left.clamp(0.1, max_depth), depth_right.clamp(0.1, max_depth)
    if depth_source == "gt":
        depth_left  = _lookup_depth(batch["depth_left"].to(device), left_kps)
        depth_right = _lookup_depth(batch["depth_right"].to(device), right_kps)
        return depth_left.clamp(0.1, max_depth), depth_right.clamp(0.1, max_depth)


def train_step(
    model,
    batch,
    optimizer,
    loss_fn,
    matching_fn,
    device,
    depth_source: str = "orb_disparity",
    max_depth: float = 200.0,
):
    model.train()
    optimizer.zero_grad()

    images   = batch['images'].to(device)    # B, 2, C, H, W
    K_inv    = batch['K_inv'].to(device)     # B, 3, 3
    T_lr     = batch['T_lr'].to(device)      # B, 4, 4
    T_rl     = torch.linalg.inv(T_lr)        # B, 4, 4
    baseline = batch['baseline'].to(device)  # B,

    left_kps, right_kps, masks = matching_fn(images)  # B, P, 2 each; B, P bool
    left_kps  = left_kps.to(device)
    right_kps = right_kps.to(device)
    masks     = masks.to(device)

    K     = torch.linalg.inv(K_inv)  # B, 3, 3
    focal = K[:, 0, 0]               # B,  (f_x; assumes rectified stereo)

    depth_left, depth_right = _get_depth(depth_source, batch, left_kps, right_kps, focal, baseline, device, max_depth)

    cov_preds = model(images)  # B*2, H, W, 2, 2

    left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)  # B, P, 2, 2

    # Reproject both directions
    right_kps_reproj = reproject(left_kps,  depth_left,  K, T_lr)  # B, P, 2
    left_kps_reproj  = reproject(right_kps, depth_right, K, T_rl)  # B, P, 2

    # Mask out reprojected points that land outside the image
    H, W = images.shape[-2], images.shape[-1]
    def _in_bounds(kps):  # kps: B, P, 2  →  B, P bool
        return (kps[..., 0] >= 0) & (kps[..., 0] < W) & (kps[..., 1] >= 0) & (kps[..., 1] < H)
    masks = masks * _in_bounds(right_kps_reproj) * _in_bounds(left_kps_reproj)

    # --- Flatten batch+point dims and apply mask ---
    # loss_fn operates on (N, 2), (N, 2, 2) where N = valid keypoints across the batch.
    # K_inv is shared across all keypoints (assumes uniform intrinsics in the batch).
    flat_masks = masks.flatten().bool()  # B*P,

    def _flat_valid(t):
        # t: B, P, ...  →  B*P, ...  →  filter by mask
        return t.flatten(0, 1)[flat_masks]

    K_inv_single = K_inv[0]  # (3, 3) — assumes uniform intrinsics within a batch
    # TODO: support mixed-dataset batches — bearing_nll/linear need per-point K_inv

    loss_lr = loss_fn(
        _flat_valid(right_kps),         # observed in target (right)
        _flat_valid(right_kps_reproj),  # reprojected from source (left)
        _flat_valid(right_covs),        # covariance at observation
        K_inv_single,
    )
    loss_rl = loss_fn(
        _flat_valid(left_kps),
        _flat_valid(left_kps_reproj),
        _flat_valid(left_covs),
        K_inv_single,
    )

    loss = loss_lr + loss_rl
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {"loss": loss.item(), "n_valid_kps": int(flat_masks.sum().item())}


def eval_step(
    model,
    batch,
    loss_fn,
    matching_fn,
    device,
    depth_source: str = "orb_disparity",
    max_depth: float = 200.0,
):
    model.eval()

    with torch.no_grad():
        images   = batch['images'].to(device)    # B, 2, C, H, W
        K_inv    = batch['K_inv'].to(device)     # B, 3, 3
        T_lr     = batch['T_lr'].to(device)      # B, 4, 4
        T_rl     = torch.linalg.inv(T_lr)        # B, 4, 4
        baseline = batch['baseline'].to(device)  # B,

        left_kps, right_kps, masks = matching_fn(images)  # B, P, 2 each; B, P bool
        left_kps  = left_kps.to(device)
        right_kps = right_kps.to(device)
        masks     = masks.to(device)

        K     = torch.linalg.inv(K_inv)  # B, 3, 3
        focal = K[:, 0, 0]               # B,

        depth_left, depth_right = _get_depth(depth_source, batch, left_kps, right_kps, focal, baseline, device, max_depth)

        cov_preds = model(images)  # B*2, H, W, 2, 2

        left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)  # B, P, 2, 2

        # Reproject both directions
        right_kps_reproj = reproject(left_kps,  depth_left, K, T_lr)  # B, P, 2
        left_kps_reproj  = reproject(right_kps, depth_right, K, T_rl)  # B, P, 2

        # Mask out reprojected points that land outside the image
        H, W = images.shape[-2], images.shape[-1]
        def _in_bounds(kps):  # kps: B, P, 2  →  B, P bool
            return (kps[..., 0] >= 0) & (kps[..., 0] < W) & (kps[..., 1] >= 0) & (kps[..., 1] < H)
        masks = masks * _in_bounds(right_kps_reproj) * _in_bounds(left_kps_reproj)

        flat_masks = masks.flatten().bool()  # B*P,

        def _flat_valid(t):
            # t: B, P, ...  →  B*P, ...  →  filter by mask
            return t.flatten(0, 1)[flat_masks]

        K_inv_single = K_inv[0]  # (3, 3) — assumes uniform intrinsics within a batch

        loss_lr = loss_fn(
            _flat_valid(right_kps),         # observed in target (right)
            _flat_valid(right_kps_reproj),  # reprojected from source (left)
            _flat_valid(right_covs),        # covariance at observation
            K_inv_single,
        )
        loss_rl = loss_fn(
            _flat_valid(left_kps),
            _flat_valid(left_kps_reproj),
            _flat_valid(left_covs),
            K_inv_single,
        )

        loss = loss_lr + loss_rl

        return {"loss": loss.item(), "n_valid_kps": int(flat_masks.sum().item())}
