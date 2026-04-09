"""Stereo ablation grids — residual arrow version, for Experiments A / B / D.

Same layout as plot_stereo_ablation.py (rows = orientation, cols = baseline)
but each cell shows obs (yellow) + reproj (cyan) keypoints with red arrows
from obs to reproj, instead of predicted covariance ellipses.

Pair with plot_stereo_ablation.py: ellipses figure shows what the model
*predicts*, this figure shows what the residuals *actually look like*.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR     = Path("outputs/eval/figs")
SAMPLE_IDX  = 10
SEED        = 0

MAX_KPS     = 25
ARROW_SCALE = 8.0
MARGIN      = 8

ROW_LABELS  = ["horizontal", "vertical"]

EXPERIMENTS = [
    {
        "name":      "A",
        "title":     "Experiment A — real ORB matcher",
        "dump_dir":  Path("outputs/eval/exp_A"),
        "prefix":    "A_real",
        "baselines": [5, 10, 20, 50, 100],
    },
    {
        "name":      "B",
        "title":     "Experiment B — synthetic 2D iso noise (sigma=2 px)",
        "dump_dir":  Path("outputs/eval/exp_B"),
        "prefix":    "B_synth",
        "baselines": [5, 100],
    },
    {
        "name":      "D",
        "title":     "Experiment D — synthetic 3D iso noise (sigma=0.05 m)",
        "dump_dir":  Path("outputs/eval/exp_D"),
        "prefix":    "D_synth3d",
        "baselines": [5, 100],
    },
]


def farthest_point_sample(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
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


def load_cell(dump_dir: Path, prefix: str, orient: str, baseline_cm: int):
    fp = dump_dir / f"{prefix}_{orient}_{baseline_cm}cm_seed{SEED}_sample{SAMPLE_IDX:02d}.npz"
    d = np.load(fp, allow_pickle=True)
    image = d["image_left"]
    kps   = d["left_kps"]
    res   = d["residuals"]
    epoch = int(d["epoch"])

    # In-bounds filter (drop kps near border + reproj off-image)
    H, W = image.shape
    reproj = kps - res
    keep = ((kps[:, 0]    >= MARGIN) & (kps[:, 0]    < W - MARGIN) &
            (kps[:, 1]    >= MARGIN) & (kps[:, 1]    < H - MARGIN) &
            (reproj[:, 0] >= MARGIN) & (reproj[:, 0] < W - MARGIN) &
            (reproj[:, 1] >= MARGIN) & (reproj[:, 1] < H - MARGIN))
    kps, res = kps[keep], res[keep]

    # Spatially-spread subsample
    if len(kps) > MAX_KPS:
        idx = farthest_point_sample(kps, MAX_KPS, seed=SEED)
        kps, res = kps[idx], res[idx]

    return image, kps, res, epoch


def plot_experiment(exp: dict) -> None:
    n_cols = len(exp["baselines"])
    n_rows = len(ROW_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=120)
    if n_cols == 1:
        axes = axes[:, None]

    for r, orient in enumerate(ROW_LABELS):
        for c, baseline in enumerate(exp["baselines"]):
            image, kps, res, epoch = load_cell(
                exp["dump_dir"], exp["prefix"], orient, baseline
            )
            ax = axes[r, c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)

            reproj = kps - res
            ax.quiver(
                kps[:, 0], kps[:, 1],
                -res[:, 0] * ARROW_SCALE, -res[:, 1] * ARROW_SCALE,
                angles="xy", scale_units="xy", scale=1.0,
                color="red", width=0.004, alpha=0.85, minlength=0.1,
            )
            ax.scatter(kps[:, 0],    kps[:, 1],    s=8, c="yellow",
                       edgecolors="black", linewidths=0.2, label="observed")
            ax.scatter(reproj[:, 0], reproj[:, 1], s=8, c="cyan",
                       edgecolors="black", linewidths=0.2, label="reprojected")

            if r == 0:
                ax.set_title(f"{baseline} cm", fontsize=12)
            if c == 0:
                ax.set_ylabel(orient, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])

            mean_r = float(np.linalg.norm(res, axis=1).mean()) if len(res) else 0.0
            ax.text(
                0.02, 0.98,
                f"ep {epoch}  n={len(kps)}  ||r||={mean_r:.1f}px",
                transform=ax.transAxes, color="white", fontsize=8, va="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"),
            )

    fig.suptitle(
        f"{exp['title']}  |  sample {SAMPLE_IDX}  |  arrows ×{ARROW_SCALE:g}",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUT_DIR / f"stereo_ablation_{exp['name']}_arrows_sample{SAMPLE_IDX:02d}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for exp in EXPERIMENTS:
        plot_experiment(exp)


if __name__ == "__main__":
    main()
