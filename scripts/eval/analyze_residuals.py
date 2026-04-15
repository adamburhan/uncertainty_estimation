"""Load cached residuals, apply an outlier-rejection method, plot distributions.

Rejectors take the full column dict and return a bool mask. Add one by
writing `f(data, **kwargs) -> np.ndarray[bool]` and registering it in
REJECTORS.

Every rejector combines with the "geometric validity" mask
(`depth_valid & in_bounds_left & in_bounds_right`), which is always
applied — those residuals are degenerate by construction, not outliers.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


IN_DIR  = Path("outputs/eval/residuals")
OUT_DIR = Path("outputs/eval/figs")

ORIENTATIONS = ["horizontal", "vertical"]
BASELINES    = [5, 10, 20, 50, 100]

RESIDUAL = "left"     # "left" or "right" — which image residual lives in
CONFIDENCE = 0.95

METHOD = "none"
PARAMS: dict = {}


# ---------------------------------------------------------------------------
# Rejectors: data (dict of arrays) -> bool mask (N,)
# ---------------------------------------------------------------------------

def none(data):
    return np.ones(len(data["r_left_x"]), dtype=bool)


def abs_norm(data, thresh=5.0, side=RESIDUAL):
    r = _residual(data, side)
    return np.linalg.norm(r, axis=1) < thresh


def pct_clip(data, q=99.0, side=RESIDUAL):
    """Relative diagnostic — keeps the bottom q% by ||r||. Not a cleaner."""
    r = _residual(data, side)
    n = np.linalg.norm(r, axis=1)
    return n < np.percentile(n, q)


REJECTORS = {
    "none": none,
    "abs_norm": abs_norm,
    "pct_clip": pct_clip,
}


# ---------------------------------------------------------------------------
# Loading / concatenation
# ---------------------------------------------------------------------------

def _residual(data, side):
    return np.stack([data[f"r_{side}_x"], data[f"r_{side}_y"]], axis=1)


def load_all() -> dict:
    """Return a dict of per-column concatenated arrays across all configs,
    with an added `config` column (e.g. 'horizontal_5cm')."""
    per_cfg = {}
    for orient in ORIENTATIONS:
        for b in BASELINES:
            key = f"{orient}_{b}cm"
            z = np.load(IN_DIR / f"{key}.npz", allow_pickle=False)
            per_cfg[key] = {k: z[k] for k in z.files if not k.startswith("_stat_")}

    keys = list(next(iter(per_cfg.values())).keys())
    out = {k: np.concatenate([per_cfg[c][k] for c in per_cfg]) for k in keys}
    out["config"] = np.concatenate([
        np.full(len(per_cfg[c]["r_left_x"]), c) for c in per_cfg
    ])
    return out


def geometric_valid(data) -> np.ndarray:
    return data["depth_valid"] & data["in_bounds_left"] & data["in_bounds_right"]


# ---------------------------------------------------------------------------
# Covariance / ellipse helpers
# ---------------------------------------------------------------------------

def cov_summary(r: np.ndarray) -> dict:
    if len(r) < 2:
        return {"n": len(r), "cov": np.zeros((2, 2)), "eigvals": np.zeros(2),
                "angle_deg": 0.0, "rms": 0.0}
    cov = np.cov(r.T)
    vals, vecs = np.linalg.eigh(cov)
    return {
        "n": len(r),
        "cov": cov,
        "eigvals": vals,
        "angle_deg": float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))),
        "rms": float(np.sqrt((r ** 2).sum(axis=1).mean())),
    }


def draw_ellipse(ax, cov, scale, **kw):
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w = 2.0 * scale * np.sqrt(vals[-1])
    h = 2.0 * scale * np.sqrt(vals[0])
    ax.add_patch(patches.Ellipse((0, 0), width=w, height=h, angle=angle, **kw))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(per_cfg: dict, title: str, out_path: Path):
    conf_scale = float(np.sqrt(chi2.ppf(CONFIDENCE, df=2)))
    max_axis = 0.0
    for s in per_cfg.values():
        if s["n"] >= 2:
            max_axis = max(max_axis, conf_scale * np.sqrt(s["eigvals"][-1]))
    half = 1.2 * max_axis if max_axis > 0 else 10.0

    fig, axes = plt.subplots(
        len(ORIENTATIONS), len(BASELINES),
        figsize=(3.2 * len(BASELINES), 3.2 * len(ORIENTATIONS)),
        dpi=120,
    )
    for i, orient in enumerate(ORIENTATIONS):
        for j, b in enumerate(BASELINES):
            ax = axes[i, j]
            key = f"{orient}_{b}cm"
            s = per_cfg[key]
            r = s["r"]
            if len(r):
                n_show = min(len(r), 3000)
                idx = np.random.default_rng(0).choice(len(r), size=n_show, replace=False)
                ax.scatter(r[idx, 0], r[idx, 1], s=2, alpha=0.15,
                           rasterized=True, color="steelblue")
            ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
            ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
            if s["n"] >= 2:
                draw_ellipse(ax, s["cov"], scale=conf_scale,
                             edgecolor="crimson", facecolor="none", linewidth=1.6)
                ax.text(
                    0.02, 0.98,
                    f"n={s['n']}\nrms={s['rms']:.2f}px\n"
                    f"eig=({s['eigvals'][0]:.2f},{s['eigvals'][-1]:.2f})\n"
                    f"angle={s['angle_deg']:+.0f}°",
                    transform=ax.transAxes, fontsize=8, va="top", ha="left",
                    family="monospace",
                    bbox=dict(facecolor="white", alpha=0.8, pad=2, edgecolor="none"),
                )
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(f"{b} cm", fontsize=12)
            if j == 0:
                ax.set_ylabel(orient, fontsize=12)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


# ---------------------------------------------------------------------------

def main():
    data = load_all()
    geo = geometric_valid(data)
    rej = REJECTORS[METHOD](data, **PARAMS)
    keep = geo & rej

    params_tag = "_".join(f"{k}{v}" for k, v in PARAMS.items())
    tag = f"{METHOD}_{params_tag}_{RESIDUAL}" if params_tag else f"{METHOD}_{RESIDUAL}"

    r_all = _residual(data, RESIDUAL)
    per_cfg = {}
    for orient in ORIENTATIONS:
        for b in BASELINES:
            key = f"{orient}_{b}cm"
            m = (data["config"] == key) & keep
            s = cov_summary(r_all[m])
            s["r"] = r_all[m]
            per_cfg[key] = s

            m_geo = (data["config"] == key) & geo
            print(f"{key:18s}  kept {m.sum():7d}/{m_geo.sum():7d}  "
                  f"rms={s['rms']:6.3f}  eig=({s['eigvals'][0]:6.3f},{s['eigvals'][-1]:7.3f})  "
                  f"angle={s['angle_deg']:+6.1f}°")

    title = f"Residuals (r_{RESIDUAL}) | rejector={METHOD} {PARAMS}"
    plot_grid(per_cfg, title=title, out_path=OUT_DIR / f"residuals_{tag}.png")


if __name__ == "__main__":
    main()
