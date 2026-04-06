"""Covariance visualization: dense maps and keypoint ellipse overlays."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from uncertainty_estimation.models.transforms import covs_to_image


def visualize_covariances(
    model,
    batch,
    matching_fn,
    device,
    scale_limit=50.0,
    ellipse_scale=15.0,
    min_ellipse_size=4.0,
    max_kps=40,
    random_subset=False,
):
    """Render dense covariance map + keypoint ellipse overlay for one batch.

    Returns dict of matplotlib Figures:
        {"vis/cov_map": fig_dense, "vis/kp_ellipses": fig_kps}

    The caller is responsible for wrapping in wandb.Image or saving to disk.
    """
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)
        cov_preds = model(images)

        left_cov = cov_preds[0].cpu()
        cov_rgb = covs_to_image(left_cov, (None, scale_limit))

        # Dense covariance map
        fig_dense, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        # Get matches for first sample only
        K = torch.linalg.inv(batch["K_inv"].to(device))
        left_kps, _, masks = matching_fn(images[:1], K[:1])
        left_kps = left_kps.to(device)
        masks = masks[0].bool()
        kps = left_kps[0][masks].cpu().numpy()

        H, W = left_cov.shape[:2]

        if len(kps) == 0:
            left_img = images[0, 0, 0].cpu().numpy()
            fig_kps, ax2 = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
            ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)
            ax2.axis("off")
            ax2.set_title("Keypoint covariance ellipses (0 matches)")
            fig_kps.tight_layout()
            return {"vis/cov_map": fig_dense, "vis/kp_ellipses": fig_kps}

        # Subset to reduce clutter
        if len(kps) > max_kps:
            if random_subset:
                idx = np.random.choice(len(kps), size=max_kps, replace=False)
            else:
                idx = np.linspace(0, len(kps) - 1, max_kps).astype(int)
            kps = kps[idx]

        # Sample covariance at keypoint locations
        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()

        # Ellipse overlay on left image
        left_img = images[0, 0, 0].cpu().numpy()
        fig_kps, ax2 = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)

        for (x, y), cov in zip(kps, covs_at_kps):
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0.0)
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            axis_lengths = 2.0 * ellipse_scale * np.sqrt(vals[::-1])
            width = max(axis_lengths[0], min_ellipse_size)
            height = max(axis_lengths[1], min_ellipse_size)

            ellipse = patches.Ellipse(
                (x, y), width=width, height=height, angle=angle,
                edgecolor="lime", facecolor="none", linewidth=1.5, alpha=0.9,
            )
            ax2.add_patch(ellipse)

        ax2.scatter(kps[:, 0], kps[:, 1], s=14, c="red", linewidths=0)
        ax2.axis("off")
        ax2.set_title(f"Keypoint covariance ellipses ({len(kps)} shown)")
        fig_kps.tight_layout()

    return {"vis/cov_map": fig_dense, "vis/kp_ellipses": fig_kps}
