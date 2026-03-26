"""Visualization script: visualize matches between two images using a matching function."""

import numpy as np
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt
import cv2
import wandb
import matplotlib.patches as patches

from uncertainty_estimation.matching.base import MatchingFunction
from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset
from uncertainty_estimation.visualization.matches import draw_matches
from uncertainty_estimation.geometry.stereo import reproject
from uncertainty_estimation.models.transforms import covs_to_image
import hydra
from torch.utils.data import DataLoader
from uncertainty_estimation.matching.orb import ORB
from uncertainty_estimation.models.factory import build_model


def visualize_matches(
    image_1: np.ndarray,
    image_2: np.ndarray,
    matching_fn: MatchingFunction,
    device: torch.device | None = None,
    max_display: int = 100,
    title: str = "Feature Matches",
) -> plt.Figure:
    """Run matching_fn on two images and draw the resulting matches.

    Args:
        image_1: (H, W) or (H, W, 3) uint8 image.
        image_2: (H, W) or (H, W, 3) uint8 image.
        matching_fn: callable with signature
            (images: Tensor[1, 2, 1, H, W], device) -> (left_kps, right_kps, masks)
            where left_kps/right_kps are (1, P, 2) and masks are (1, P).
        device: torch device (defaults to cpu).
        max_display: max number of matches to draw.
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    if device is None:
        device = torch.device("cpu")

    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    gray1 = image_1
    gray2 = image_2

    # Pack into (1, 2, 1, H, W) as expected by matching functions
    t1 = torch.from_numpy(gray1)[None, None]  # (1, 1, H, W)
    t2 = torch.from_numpy(gray2)[None, None]
    images = torch.stack([t1, t2], dim=1).to(device)  # (1, 2, 1, H, W)

    left_kps, right_kps, masks = matching_fn(images, device)

    # Extract valid matches for the single batch entry
    valid = masks[0].bool()  # (P,)
    kp1 = left_kps[0][valid].cpu().numpy()   # (K, 2)
    kp2 = right_kps[0][valid].cpu().numpy()  # (K, 2)

    # Matches are already aligned: match i pairs kp1[i] <-> kp2[i]
    n = len(kp1)
    match_indices = np.stack([np.arange(n), np.arange(n)], axis=1)  # (K, 2)

    return draw_matches(
        image_1, kp1,
        image_2, kp2,
        match_indices,
        max_display=max_display,
        title=title,
    ), kp1, kp2


def visualize_reprojection(
    image_2: np.ndarray,
    kp1: np.ndarray,
    kp2: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_1_2: np.ndarray,
    max_display: int = 200,
    title: str = "Reprojection",
) -> plt.Figure:
    """Reproject kp1 into image 2 and visualize against the actual kp2.

    On image 2, draws:
      - kp2 (actual matches) in green
      - kp1 reprojected into cam2 frame in red
      - a line connecting each reprojected point to its corresponding kp2

    Args:
        image_2: (H, W) or (H, W, 3) uint8 image to draw on.
        kp1: (P, 2) keypoint pixel coords (u, v) in image 1.
        kp2: (P, 2) corresponding keypoint pixel coords in image 2.
        depth: (P,) depth of each kp1 point in the camera 1 frame (metres).
        K: (3, 3) camera intrinsic matrix (shared for both views).
        T_1_2: (4, 4) transform that maps 3D points from cam1 to cam2
                      (i.e. T_src_dst in reproject convention).
        max_display: max number of correspondences to draw.
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    if image_2.ndim == 2:
        canvas = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)
    else:
        canvas = image_2.copy()

    kp1_t = torch.as_tensor(kp1).float()
    depth_t = torch.as_tensor(depth).float()
    K_t = torch.as_tensor(K).float()
    T_t = torch.as_tensor(T_1_2).float()

    kp1_reproj = reproject(kp1_t, depth_t, K_t, T_t).numpy()  # (P, 2)

    n = len(kp1_reproj)
    # if n > max_display:
    #     idx = np.random.choice(n, max_display, replace=False)
    #     kp1_reproj = kp1_reproj[idx]
    #     kp2 = kp2[idx]

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

def load_stereo_calibration_kitti(sequence_dir):
    data = {}
    with open(sequence_dir / "calib.txt") as f:
        for line in f:
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])

    P0 = data["P0"].reshape(3, 4)
    P1 = data["P1"].reshape(3, 4)

    K  = torch.from_numpy(P0[:, :3]).float()
    tx = P1[0, 3] / P0[0, 0]          # negative (~-0.537)
    baseline = float(-tx)              # positive

    T_lr = torch.eye(4)
    T_lr[0, 3] = tx                    # negative

    return K, T_lr, baseline


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
    """Render dense covariance map + more visible keypoint ellipse overlay for one batch."""
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)      # B, 2, C, H, W
        cov_preds = model(images)                # e.g. B*2, H, W, 2, 2

        left_cov = cov_preds[0].cpu()            # H, W, 2, 2
        cov_rgb = covs_to_image(left_cov, (None, scale_limit))  # H, W, 3

        # Dense covariance visualization
        fig_dense, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        # Get matches for first sample only
        left_kps, _, masks = matching_fn(images[:1])
        left_kps = left_kps.to(device)
        masks = masks[0].bool()
        kps = left_kps[0][masks].cpu().numpy()   # N, 2

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
            # plt.close(fig_dense)
            # plt.close(fig_kps)
            return result

        # Optional subset to reduce clutter
        if len(kps) > max_kps:
            if random_subset:
                idx = np.random.choice(len(kps), size=max_kps, replace=False)
            else:
                idx = np.linspace(0, len(kps) - 1, max_kps).astype(int)
            kps = kps[idx]

        # Sample covariance at keypoint locations
        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()   # N, 2, 2

        # Base image
        left_img = images[0, 0, 0].cpu().numpy()
        fig_kps, ax2 = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)

        for (x, y), cov in zip(kps, covs_at_kps):
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0.0)

            # Major-axis orientation from eigenvector of largest eigenvalue
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))

            # Reverse so width corresponds to largest eigenvalue
            axis_lengths = 2.0 * ellipse_scale * np.sqrt(vals[::-1])
            width = max(axis_lengths[0], min_ellipse_size)
            height = max(axis_lengths[1], min_ellipse_size)

            ellipse = patches.Ellipse(
                (x, y),
                width=width,
                height=height,
                angle=angle,
                edgecolor="lime",
                facecolor="none",
                linewidth=1.5,
                alpha=0.9,
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
    # plt.close(fig_dense)
    # plt.close(fig_kps)
    return result

# dataset factory
def build_dataset(cfg: DictConfig, split: str):
    if cfg.dataset.name == "tartanair":
        return TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split)

    raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Choose from: tartanair, kitti")

def _visualize_covariances_local(model, batch, matching_fn, device, scale_limit=50.0,
                                  ellipse_scale=5.0, min_ellipse_size=1.0, max_kps=500):
    """Same as visualize_covariances but returns raw Figure objects (no wandb wrapping)."""
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

        # -- Ellipse overlay --
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

        # -- Eigenvalue histogram (catches degenerate / collapsed covariances) --
        all_kps = left_kps[0][masks].cpu().numpy()
        col_all = torch.from_numpy(all_kps[:, 0]).round().long().clamp(0, W - 1)
        row_all = torch.from_numpy(all_kps[:, 1]).round().long().clamp(0, H - 1)
        covs_all = left_cov[row_all, col_all].numpy()
        eigvals = np.linalg.eigvalsh(covs_all)  # N, 2 — sorted ascending

        fig_hist, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=120)
        for ax_h, vals, label in zip(axes, eigvals.T, ["Minor eigenvalue (λ₁)", "Major eigenvalue (λ₂)"]):
            ax_h.hist(vals, bins=50, color="steelblue", edgecolor="none")
            ax_h.set_xlabel(label)
            ax_h.set_ylabel("Count")
            ax_h.set_title(f"{label}\nmedian={np.median(vals):.3f}, max={vals.max():.3f}")
        fig_hist.suptitle("Covariance eigenvalue distribution at keypoints")
        fig_hist.tight_layout()

    return fig_dense, fig_kps, fig_hist


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg):
    device = torch.device(cfg.training.device)

    val_dataset = build_dataset(cfg, split="val")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = build_model(cfg.model, device)

    if cfg.model.checkpoint is None:
        print("WARNING: no checkpoint specified — model is randomly initialized. "
              "Pass model.checkpoint=<path> on the command line.")
    else:
        print(f"Loaded checkpoint: {cfg.model.checkpoint}")

    matching_fn = lambda images: ORB(
        images, device,
        max_keypoints=cfg.matching.max_keypoints,
        #max_hamming_distance=cfg.matching.max_hamming,
        #max_epipolar_error=cfg.matching.max_epipolar_error,
    )

    batch = next(iter(val_loader))
    _visualize_covariances_local(model, batch, matching_fn, device)

    plt.show()



if __name__ == "__main__":
    # import cv2
    # from pathlib import Path
    # from uncertainty_estimation.matching.orb import ORB

    # path = Path("/Volumes/T9/datasets/KITTI/odometry_gray/sequences")

    # img1 = cv2.imread(str(path / "00" / "image_0" / "000117.png"), cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(str(path / "00" / "image_1" / "000117.png"), cv2.IMREAD_GRAYSCALE)

    # fig, kp1, kp2 = visualize_matches(img1, img2, ORB)

    # stereo_calib = load_stereo_calibration_kitti(path / "00")
    # K, T_lr, baseline = stereo_calib

    # depth = baseline * K[0, 0] / (kp1[:, 0] - kp2[:, 0])  # simple disparity-to-depth
    # fig2 = visualize_reprojection(img2, kp1, kp2, depth, K.numpy(), T_lr.numpy())

    # plt.show()

    # stereo = cv2.StereoBM_create(numDisparities=192, blockSize=5)

    # stereo = cv2.StereoSGBM_create(numDisparities=192, blockSize=5)
    # disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0  # OpenCV scales disparity by 16 for subpixel precision
    # plt.imshow(disparity, cmap="plasma")
    # plt.colorbar(label="Disparity (pixels)")
    # plt.title("Stereo Disparity")
    # plt.axis("off")
    # plt.show()
    main()



