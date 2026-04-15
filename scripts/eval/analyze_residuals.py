"""Load cached residuals, apply an outlier-rejection method, plot distributions.

Add a new rejector by writing a function `residuals -> bool mask` and
registering it in REJECTORS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IN_DIR = Path("outputs/eval/residuals")
OUT_DIR = Path("outputs/eval/figs")

ORIENTATIONS = ["horizontal", "vertical"]
BASELINES = [5, 10, 20, 50, 100]

METHOD = "abs_norm"   # one of REJECTORS below
PARAMS = {"thresh": 5.0}


# ---------------------------------------------------------------------------
# Outlier rejectors: residuals (N, 2) -> bool mask (N,)
# ---------------------------------------------------------------------------

def none(r):
    return np.ones(len(r), dtype=bool)


def abs_norm(r, thresh=5.0):
    return np.linalg.norm(r, axis=1) < thresh


def pct_norm(r, q=99.0):
    n = np.linalg.norm(r, axis=1)
    return n < np.percentile(n, q)


REJECTORS = {
    "none": none,
    "abs_norm": abs_norm,
    "pct_norm": pct_norm,
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(residuals_by_cfg: dict, title: str, out_path: Path):
    fig, axes = plt.subplots(
        len(ORIENTATIONS), len(BASELINES),
        figsize=(3.0 * len(BASELINES), 3.0 * len(ORIENTATIONS)),
        sharex=True, sharey=True,
    )
    for i, orient in enumerate(ORIENTATIONS):
        for j, b in enumerate(BASELINES):
            ax = axes[i, j]
            r = residuals_by_cfg[f"{orient}_{b}cm"]
            if len(r) == 0:
                ax.set_title(f"{orient} {b}cm\n(empty)")
                continue
            ax.scatter(r[:, 0], r[:, 1], s=2, alpha=0.2, rasterized=True)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_aspect("equal")
            sx, sy = r[:, 0].std(), r[:, 1].std()
            ax.set_title(f"{orient} {b}cm\nN={len(r)}  σx={sx:.2f} σy={sy:.2f}",
                         fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"-> {out_path}")


def main():
    rejector = REJECTORS[METHOD]
    tag = METHOD + "_" + "_".join(f"{k}{v}" for k, v in PARAMS.items())

    residuals_by_cfg = {}
    for orient in ORIENTATIONS:
        for b in BASELINES:
            key = f"{orient}_{b}cm"
            data = np.load(IN_DIR / f"{key}.npz")
            r = data["residuals"]
            mask = rejector(r, **PARAMS)
            residuals_by_cfg[key] = r[mask]
            print(f"{key}: {mask.sum()}/{len(r)} kept")

    plot_grid(residuals_by_cfg, title=f"Residuals | {tag}",
              out_path=OUT_DIR / f"residuals_{tag}.png")


if __name__ == "__main__":
    main()
