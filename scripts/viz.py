"""Visualization utilities for stereo covariance models.

visualize_covariances()        — returns wandb.Image dict (used as training callback)
visualize_covariances_local()  — returns raw matplotlib Figures (standalone use)
visualize_matches()            — ORB match visualization
visualize_reprojection()       — reprojected vs actual keypoints
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import wandb

from uncertainty_estimation.geometry.stereo import reproject
from uncertainty_estimation.models.transforms import covs_to_image
from uncertainty_estimation.visualization.matches import draw_matches


########################################################################################
# Covariance visualization (wandb callback version)
########################################################################################

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

    Returns dict of {"vis/cov_map": wandb.Image, "vis/kp_ellipses": wandb.Image}.
    Intended as the vis_fn callback for train_model().
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
        left_kps, _, masks = matching_fn(images[:1])
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

            result = {
                "vis/cov_map": wandb.Image(fig_dense),
                "vis/kp_ellipses": wandb.Image(fig_kps),
            }
            plt.close(fig_dense)
            plt.close(fig_kps)
            return result

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

    result = {
        "vis/cov_map": wandb.Image(fig_dense),
        "vis/kp_ellipses": wandb.Image(fig_kps),
    }
    plt.close(fig_dense)
    plt.close(fig_kps)
    return result


########################################################################################
# Covariance visualization (local / standalone version)
########################################################################################

def visualize_covariances_local(
    model, batch, matching_fn, device,
    scale_limit=50.0, ellipse_scale=5.0, min_ellipse_size=1.0, max_kps=500,
):
    """Same as visualize_covariances but returns raw Figure objects (no wandb)."""
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)
        cov_preds = model(images)

        left_cov = cov_preds[0].cpu()
        cov_rgb = covs_to_image(left_cov, (None, scale_limit))

        fig_dense, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=120)
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        left_kps, _, masks = matching_fn(images[:1])
        left_kps = left_kps.to(device)
        masks = masks[0].bool()
        kps = left_kps[0][masks].cpu().numpy()

        H, W = left_cov.shape[:2]

        if len(kps) == 0:
            print("No matches found.")
            return fig_dense, None, None

        if len(kps) > max_kps:
            idx = np.linspace(0, len(kps) - 1, max_kps).astype(int)
            kps = kps[idx]

        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()

        # Ellipse overlay
        left_img = images[0, 0, 0].cpu().numpy()
        fig_kps, ax2 = plt.subplots(1, 1, figsize=(12, 8), dpi=120)
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)
        for (x, y), cov in zip(kps, covs_at_kps):
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0.0)
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            axis_lengths = 2.0 * ellipse_scale * np.sqrt(vals[::-1])
            ellipse = patches.Ellipse(
                (x, y),
                width=max(axis_lengths[0], min_ellipse_size),
                height=max(axis_lengths[1], min_ellipse_size),
                angle=angle, edgecolor="lime", facecolor="none", linewidth=1.5, alpha=0.9,
            )
            ax2.add_patch(ellipse)
        ax2.scatter(kps[:, 0], kps[:, 1], s=14, c="red", linewidths=0)
        ax2.axis("off")
        ax2.set_title(f"Keypoint covariance ellipses ({len(kps)} shown)")
        fig_kps.tight_layout()

        # Eigenvalue histogram
        all_kps = left_kps[0][masks].cpu().numpy()
        col_all = torch.from_numpy(all_kps[:, 0]).round().long().clamp(0, W - 1)
        row_all = torch.from_numpy(all_kps[:, 1]).round().long().clamp(0, H - 1)
        covs_all = left_cov[row_all, col_all].numpy()
        eigvals = np.linalg.eigvalsh(covs_all)

        fig_hist, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=120)
        for ax_h, vals, label in zip(axes, eigvals.T, ["Minor eigenvalue (λ₁)", "Major eigenvalue (λ₂)"]):
            ax_h.hist(vals, bins=50, color="steelblue", edgecolor="none")
            ax_h.set_xlabel(label)
            ax_h.set_ylabel("Count")
            ax_h.set_title(f"{label}\nmedian={np.median(vals):.3f}, max={vals.max():.3f}")
        fig_hist.suptitle("Covariance eigenvalue distribution at keypoints")
        fig_hist.tight_layout()

    return fig_dense, fig_kps, fig_hist


########################################################################################
# Match visualization
########################################################################################

def visualize_matches(
    image_1: np.ndarray,
    image_2: np.ndarray,
    matching_fn,
    device: torch.device = None,
    max_display: int = 100,
    title: str = "Feature Matches",
):
    """Run matching_fn on two images and draw the resulting matches.

    Returns:
        (figure, kp1, kp2) where kp1/kp2 are (K, 2) numpy arrays.
    """
    if device is None:
        device = torch.device("cpu")

    t1 = torch.from_numpy(image_1)[None, None]
    t2 = torch.from_numpy(image_2)[None, None]
    images = torch.stack([t1, t2], dim=1).to(device)

    left_kps, right_kps, masks = matching_fn(images, device)

    valid = masks[0].bool()
    kp1 = left_kps[0][valid].cpu().numpy()
    kp2 = right_kps[0][valid].cpu().numpy()

    n = len(kp1)
    match_indices = np.stack([np.arange(n), np.arange(n)], axis=1)

    fig = draw_matches(
        image_1, kp1, image_2, kp2, match_indices,
        max_display=max_display, title=title,
    )
    return fig, kp1, kp2


########################################################################################
# Reprojection visualization
########################################################################################

def visualize_reprojection(
    image_2: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_1_2: np.ndarray,
    max_display: int = 200,
    title: str = "Reprojection",
):
    """Reproject kp1 into image 2 and visualize against actual kp2."""
    if image_2.ndim == 2:
        canvas = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)
    else:
        canvas = image_2.copy()

    kp1_reproj = reproject(
        torch.as_tensor(kp1).float(),
        torch.as_tensor(depth).float(),
        torch.as_tensor(K).float(),
        torch.as_tensor(T_1_2).float(),
    ).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")

    for (u_r, v_r), (u2, v2) in zip(kp1_reproj, kp2):
        ax.plot([u_r, u2], [v_r, v2], color="white", linewidth=0.5, alpha=0.5)

    ax.scatter(kp2[:, 0], kp2[:, 1], c="lime", s=12, linewidths=0.3,
               edgecolors="k", zorder=3, label="kp2 (actual)")
    ax.scatter(kp1_reproj[:, 0], kp1_reproj[:, 1], c="red", s=12, linewidths=0.3,
               edgecolors="k", zorder=3, label="kp1 reprojected")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    return fig


########################################################################################
# KITTI calibration helper
########################################################################################

def load_stereo_calibration_kitti(sequence_dir):
    data = {}
    with open(sequence_dir / "calib.txt") as f:
        for line in f:
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])

    P0 = data["P0"].reshape(3, 4)
    P1 = data["P1"].reshape(3, 4)

    K = torch.from_numpy(P0[:, :3]).float()
    tx = P1[0, 3] / P0[0, 0]
    baseline = float(-tx)

    T_lr = torch.eye(4)
    T_lr[0, 3] = tx

    return K, T_lr, baseline


########################################################################################
# Standalone entry point
########################################################################################

if __name__ == "__main__":
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

    import hydra

    from uncertainty_estimation.matching.orb import ORB
    from uncertainty_estimation.models.factory import build_model
    from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset

    def build_dataset(cfg: DictConfig, split: str):
        if cfg.dataset.name == "tartanair":
            return TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split)
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'")

    @hydra.main(version_base=None, config_path="../configs", config_name="base")
    def main(cfg):
        device = torch.device(cfg.training.device)

        val_dataset = build_dataset(cfg, split="val")
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        model = build_model(cfg.model).to(device)
        if cfg.model.checkpoint is None:
            print("WARNING: no checkpoint — model is randomly initialized.")
        else:
            print(f"Loaded checkpoint: {cfg.model.checkpoint}")

        matching_fn = lambda images: ORB(
            images, device, max_keypoints=cfg.matching.max_keypoints,
        )

        batch = next(iter(val_loader))
        visualize_covariances_local(model, batch, matching_fn, device)
        plt.show()

    main()
