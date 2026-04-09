"""Stereo ablation grids for Experiments A / B / D.

Rows: orientation (horizontal, vertical)
Cols: baseline (cm) — varies per experiment
Each cell: left image + predicted covariance ellipses at the same scene/sample.

A is the headline (10 configs, 2x5 grid). B and D are the falsification
controls — same layout, fewer columns (only the extreme baselines were trained).
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

OUT_DIR       = Path("outputs/eval/figs")
SAMPLE_IDX    = 10
SEED          = 0

MAX_KPS       = 25
ELLIPSE_SCALE = 1.0
CONFIDENCE    = 0.95
MARGIN        = 8

ROW_LABELS    = ["horizontal", "vertical"]

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


def draw_ellipse(ax, xy, cov, scale, **kwargs):
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    width  = 2.0 * scale * np.sqrt(vals[-1])
    height = 2.0 * scale * np.sqrt(vals[0])
    ax.add_patch(patches.Ellipse(xy, width=width, height=height, angle=angle, **kwargs))


def load_cell(dump_dir: Path, prefix: str, orient: str, baseline_cm: int):
    fp = dump_dir / f"{prefix}_{orient}_{baseline_cm}cm_seed{SEED}_sample{SAMPLE_IDX:02d}.npz"
    d = np.load(fp, allow_pickle=True)
    image = d["image_left"]
    kps   = d["left_kps"]
    covs  = d["left_covs"]
    res   = d["residuals"]
    epoch = int(d["epoch"])

    # In-bounds filter (drop kps near border + reproj off-image)
    H, W = image.shape
    reproj = kps - res
    keep = ((kps[:, 0]    >= MARGIN) & (kps[:, 0]    < W - MARGIN) &
            (kps[:, 1]    >= MARGIN) & (kps[:, 1]    < H - MARGIN) &
            (reproj[:, 0] >= MARGIN) & (reproj[:, 0] < W - MARGIN) &
            (reproj[:, 1] >= MARGIN) & (reproj[:, 1] < H - MARGIN))
    kps, covs, res = kps[keep], covs[keep], res[keep]

    # Spatially-spread subsample
    if len(kps) > MAX_KPS:
        idx = farthest_point_sample(kps, MAX_KPS, seed=SEED)
        kps, covs, res = kps[idx], covs[idx], res[idx]

    return image, kps, covs, res, epoch


def plot_experiment(exp: dict, conf_scale: float) -> None:
    n_cols = len(exp["baselines"])
    n_rows = len(ROW_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=120)
    if n_cols == 1:
        axes = axes[:, None]

    for r, orient in enumerate(ROW_LABELS):
        for c, baseline in enumerate(exp["baselines"]):
            image, kps, covs, res, epoch = load_cell(
                exp["dump_dir"], exp["prefix"], orient, baseline
            )
            ax = axes[r, c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            for (x, y), cov in zip(kps, covs):
                draw_ellipse(
                    ax, (x, y), cov, scale=conf_scale,
                    edgecolor="lime", facecolor="none", linewidth=0.9, alpha=0.9,
                )
            ax.scatter(kps[:, 0], kps[:, 1],
                       s=5, c="yellow", edgecolors="black", linewidths=0.2)

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
        f"{exp['title']}  |  sample {SAMPLE_IDX}  |  "
        f"{int(CONFIDENCE*100)}% ellipses ×{ELLIPSE_SCALE:g}",
        fontsize=13,
    )
    fig.tight_layout()
    out = OUT_DIR / f"stereo_ablation_{exp['name']}_ellipses_sample{SAMPLE_IDX:02d}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conf_scale = float(np.sqrt(chi2.ppf(CONFIDENCE, df=2))) * ELLIPSE_SCALE
    for exp in EXPERIMENTS:
        plot_experiment(exp, conf_scale)


if __name__ == "__main__":
    main()
