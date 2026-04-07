"""Covariance visualization: dense maps and keypoint confidence ellipses."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2

from uncertainty_estimation.models.transforms import covs_to_image


def visualize_covariances(
    model,
    batch,
    matching_fn,
    device,
    scale_limit: float = 50.0,
    max_kps: int = 25,
    random_subset: bool = True,
    confidence: float = 0.95,
    display_scale: float = 1.0,
    min_eigval: float = 1e-6,
    point_size: float = 14.0,
    ellipse_linewidth: float = 1.2,
    ellipse_alpha: float = 0.9,
):
    """Render dense covariance map + keypoint confidence ellipse overlay.

    Returns:
        {
            "vis/cov_map": fig_dense,
            "vis/kp_ellipses": fig_kps,
        }

    Notes:
    - Ellipses are drawn at a statistically meaningful confidence level
      (chi-square quantile, 2 dof) instead of an arbitrary visual scale.
    - No minimum ellipse size is enforced, so small covariances stay small.
    - `display_scale` is a purely cosmetic multiplier; leave at 1.0 unless
      ellipses are too small to see in the rendered figure.
    - The dense covariance map is unchanged: hue=angle, sat=anisotropy, val=scale.
    """
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)          # (B, 2, C, H, W)
        cov_preds = model(images)                    # expected (2B, H, W, 2, 2) or similar

        # Dense covariance map for the first left image prediction
        left_cov = cov_preds[0].detach().cpu()       # (H, W, 2, 2)
        cov_rgb = covs_to_image(left_cov, (None, scale_limit))

        fig_dense, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        # Get first-sample matches only
        K = torch.linalg.inv(batch["K_inv"].to(device))
        left_kps, _, masks = matching_fn(images[:1], K[:1])
        left_kps = left_kps.to(device)

        valid = masks[0].bool()
        kps = left_kps[0][valid].detach().cpu().numpy()  # (P, 2)

        left_img = images[0, 0, 0].detach().cpu().numpy()
        H, W = left_cov.shape[:2]

        fig_kps, ax2 = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)
        ax2.axis("off")

        if len(kps) == 0:
            ax2.set_title("Keypoint covariance ellipses (0 matches)")
            fig_kps.tight_layout()
            return {"vis/cov_map": fig_dense, "vis/kp_ellipses": fig_kps}

        # Subsample keypoints for readability
        if len(kps) > max_kps:
            if random_subset:
                idx = np.random.choice(len(kps), size=max_kps, replace=False)
            else:
                idx = np.linspace(0, len(kps) - 1, max_kps).astype(int)
            kps = kps[idx]

        # Sample covariance at rounded keypoint pixels
        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()  # (K, 2, 2)

        # Chi-square quantile for 2D Gaussian confidence ellipse (2 dof)
        conf_scale = float(np.sqrt(chi2.ppf(confidence, df=2)))

        for (x, y), cov in zip(kps, covs_at_kps):
            # Ensure symmetric numeric form
            cov = 0.5 * (cov + cov.T)

            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, min_eigval)

            # Largest eigenvalue/eigenvector last from eigh
            major_vec = vecs[:, -1]
            angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

            # Full axis lengths for matplotlib Ellipse
            axis_lengths = 2.0 * display_scale * conf_scale * np.sqrt(vals[::-1])
            width = float(axis_lengths[0])
            height = float(axis_lengths[1])

            ellipse = patches.Ellipse(
                (float(x), float(y)),
                width=width,
                height=height,
                angle=float(angle),
                edgecolor="lime",
                facecolor="none",
                linewidth=ellipse_linewidth,
                alpha=ellipse_alpha,
            )
            ax2.add_patch(ellipse)

        ax2.scatter(kps[:, 0], kps[:, 1], s=point_size, c="red", linewidths=0)
        ax2.set_title(
            f"Keypoint covariance ellipses ({len(kps)} shown, "
            f"{int(confidence * 100)}% confidence)"
        )
        fig_kps.tight_layout()

    return {"vis/cov_map": fig_dense, "vis/kp_ellipses": fig_kps}
