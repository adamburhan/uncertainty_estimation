import torch

from uncertainty_estimation.geometry.stereo import depth_from_disparity, reproject, extract_covs


def train_step(
    model,
    batch,
    optimizer,
    loss_fn,
    matching_fn,
    device,
):
    model.train()
    optimizer.zero_grad()

    images   = batch['images'].to(device)    # B, 2, C, H, W
    K        = batch['K'].to(device)         # B, 3, 3
    T_lr     = batch['T_lr'].to(device)      # B, 4, 4
    T_rl     = torch.linalg.inv(T_lr)        # B, 4, 4
    baseline = batch['baseline'].to(device)  # B,

    K_inv = torch.linalg.inv(K)  # B, 3, 3

    left_kps, right_kps, masks = matching_fn(images)  # B, P, 2 each; B, P bool
    left_kps  = left_kps.to(device)
    right_kps = right_kps.to(device)
    masks     = masks.to(device)

    focal = K[:, 0, 0]  # B,  (f_x; assumes rectified stereo)
    depth_left  = depth_from_disparity(left_kps[..., 0], right_kps[..., 0], focal, baseline)
    depth_right = depth_from_disparity(right_kps[..., 0], left_kps[..., 0], focal, baseline)

    cov_preds = model(images)  # B*2, H, W, 2, 2

    left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)  # B, P, 2, 2

    # Reproject both directions 
    right_kps_reproj = reproject(left_kps,  depth_left,  K, T_lr)  # B, P, 2
    left_kps_reproj  = reproject(right_kps, depth_right, K, T_rl)  # B, P, 2

    # --- Flatten batch+point dims and apply mask ---
    # loss_fn operates on (N, 2), (N, 2, 2) where N = valid keypoints across the batch.
    # K_inv is shared across all keypoints (assumes uniform intrinsics in the batch).
    flat_masks = masks.flatten()  # B*P,

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
    optimizer.step()

    return loss.item()
