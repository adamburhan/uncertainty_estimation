"""One-cell plotting prototype: image + predicted cov ellipses + residual arrows.

Loads one .npz dump, renders one figure. Iterate on this until it looks right,
then promote to the 2x5 grid.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

NPZ_PATH       = Path("outputs/eval/exp_A/A_real_horizontal_50cm_seed0_sample10.npz")
OUT_PATH       = Path("outputs/eval/one_cell.png")

MAX_KPS        = 25      # subsample for readability
ELLIPSE_SCALE  = 1.0     # cosmetic — 1.0 = true scale, ellipses tiny
ARROW_SCALE    = 10.0     # cosmetic — same idea for residuals
CONFIDENCE     = 0.95    # chi^2 quantile for ellipse size (2 dof)
SEED           = 0


def farthest_point_sample(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling in 2D. Returns indices into pts."""
    n = len(pts)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    d2 = np.sum((pts - pts[idx[0]]) ** 2, axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d2))
        idx.append(nxt)
        d2 = np.minimum(d2, np.sum((pts - pts[nxt]) ** 2, axis=1))
    return np.array(idx)


def draw_ellipse(ax, xy, cov, scale, **kwargs):
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    width  = 2.0 * scale * np.sqrt(vals[-1])
    height = 2.0 * scale * np.sqrt(vals[0])
    ax.add_patch(patches.Ellipse(xy, width=width, height=height, angle=angle, **kwargs))


def main():
    d = np.load(NPZ_PATH, allow_pickle=True)
    image     = d["image_left"]          # (H, W)
    kps       = d["left_kps"]            # (K, 2)
    covs      = d["left_covs"]           # (K, 2, 2)
    residuals = d["residuals"]           # (K, 2) = kp_obs - kp_reproj
    stereo    = str(d["stereo"])
    epoch     = int(d["epoch"])

    # In-bounds filter: drop kps (or their reprojections) that fall outside the
    # image — keeps arrows on canvas and ellipses from clipping at the border.
    H, W = image.shape
    margin = 8
    reproj = kps - residuals  # obs - residual = reproj
    in_obs    = ((kps[:, 0]    >= margin) & (kps[:, 0]    < W - margin) &
                 (kps[:, 1]    >= margin) & (kps[:, 1]    < H - margin))
    in_reproj = ((reproj[:, 0] >= margin) & (reproj[:, 0] < W - margin) &
                 (reproj[:, 1] >= margin) & (reproj[:, 1] < H - margin))
    keep = in_obs & in_reproj
    kps, covs, residuals = kps[keep], covs[keep], residuals[keep]

    # Spread-out subsample via farthest-point sampling on observed kp positions
    idx = farthest_point_sample(kps, MAX_KPS, seed=SEED)
    kps, covs, residuals = kps[idx], covs[idx], residuals[idx]

    # Confidence-ellipse scale factor (chi^2 with 2 dof)
    conf_scale = float(np.sqrt(chi2.ppf(CONFIDENCE, df=2))) * ELLIPSE_SCALE
    reproj = kps - residuals  # obs - residual = reproj

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    # ---- Left panel: predicted ellipses at observed kps ----
    ax = axes[0]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    for (x, y), cov in zip(kps, covs):
        draw_ellipse(
            ax, (x, y), cov, scale=conf_scale,
            edgecolor="lime", facecolor="none", linewidth=1.0, alpha=0.9,
        )
    ax.scatter(kps[:, 0], kps[:, 1], s=6, c="yellow", edgecolors="black", linewidths=0.3)
    ax.set_title(
        f"Predicted {int(CONFIDENCE*100)}% covariance ellipses  (×{ELLIPSE_SCALE:g})",
        fontsize=10,
    )
    ax.axis("off")

    # ---- Right panel: obs -> reproj arrows, both points shown ----
    ax = axes[1]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.quiver(
        kps[:, 0], kps[:, 1],
        -residuals[:, 0] * ARROW_SCALE, -residuals[:, 1] * ARROW_SCALE,
        angles="xy", scale_units="xy", scale=1.0,
        color="red", width=0.003, alpha=0.85, minlength=0.1,
    )
    ax.scatter(kps[:, 0],    kps[:, 1],    s=10, c="yellow", edgecolors="black",
               linewidths=0.3, label="observed")
    ax.scatter(reproj[:, 0], reproj[:, 1], s=10, c="cyan",   edgecolors="black",
               linewidths=0.3, label="reprojected")
    ax.set_title(f"Residuals: obs → reproj  (arrow ×{ARROW_SCALE:g})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axis("off")

    fig.suptitle(
        f"{stereo}  |  epoch {epoch}  |  {len(kps)} kps shown",
        fontsize=11,
    )
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
