"""Visualization utilities for feature matches between image pairs."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from uncertainty_estimation.visualization.point_cloud import _depth_colormap


def draw_matches(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    matches: np.ndarray,
    max_display: int = 100,
    title: str = "Feature Matches",
) -> plt.Figure:
    """Draw feature matches between two images side by side.

    Args:
        img1: first image (H, W) or (H, W, 3).
        kp1: (M, 2) keypoint coordinates in image 1 (u, v).
        img2: second image.
        kp2: (M, 2) keypoint coordinates in image 2 (u, v).
        matches: (K, 2) array of index pairs — matches[i] = [idx_in_kp1, idx_in_kp2].
        max_display: max number of matches to draw (subsampled randomly).
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")

    n_matches = len(matches)
    if n_matches > max_display:
        indices = np.random.choice(n_matches, max_display, replace=False)
        matches = matches[indices]

    colors = plt.cm.viridis(np.linspace(0, 1, len(matches)))

    for (i1, i2), color in zip(matches, colors):
        u1, v1 = kp1[int(i1)]
        u2, v2 = kp2[int(i2)]
        ax.plot(
            [u1, u2 + w1], [v1, v2],
            color=color, linewidth=0.7, alpha=0.7,
        )
        ax.plot(u1, v1, "o", color=color, markersize=3)
        ax.plot(u2 + w1, v2, "o", color=color, markersize=3)

    fig.tight_layout()
    return fig


def draw_features_by_depth(
    img: np.ndarray,
    pts_2d: np.ndarray,
    depths: np.ndarray,
    title: str = "Features Colored by Depth",
) -> plt.Figure:
    """Overlay tracked features on an image, colored by 3D depth.

    Uses the same blue→green→yellow colormap as the 3D point cloud viewer,
    so features can be cross-referenced visually between the 2D and 3D views.

    Args:
        img: image (H, W) or (H, W, 3).
        pts_2d: (N, 2) pixel coordinates (u, v).
        depths: (N,) depth values in camera coordinates (meters).
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    colors = _depth_colormap(depths)  # (N, 3) RGB in [0, 1], same as Open3D view

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.imshow(img)
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=colors, s=25, linewidths=0.4, edgecolors='k')
    ax.set_title(
        f"{title}\n"
        f"blue=near, yellow=far  |  depth range [{depths.min():.1f}, {depths.max():.1f}] m  |  "
        f"{len(pts_2d)} features"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_epipolar_lines(
    img: np.ndarray,
    lines: np.ndarray,
    points: np.ndarray | None = None,
    title: str = "Epipolar Lines",
) -> plt.Figure:
    """Draw epipolar lines on an image.

    Args:
        img: image (H, W) or (H, W, 3).
        lines: (N, 3) array of epipolar lines in homogeneous form (ax + by + c = 0).
        points: optional (N, 2) corresponding points to mark.
        title: plot title.

    Returns:
        matplotlib Figure.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = img.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(title)

    colors = plt.cm.viridis(np.linspace(0, 1, len(lines)))

    for i, (line, color) in enumerate(zip(lines, colors)):
        a, b, c = line
        if abs(b) > 1e-8:
            x0, x1 = 0, w
            y0 = -(a * x0 + c) / b
            y1 = -(a * x1 + c) / b
        else:
            y0, y1 = 0, h
            x0 = -(b * y0 + c) / a
            x1 = -(b * y1 + c) / a

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.8, alpha=0.6)

        if points is not None:
            ax.plot(points[i, 0], points[i, 1], "o", color=color, markersize=4)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig
