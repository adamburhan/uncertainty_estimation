"""Empirical residual covariance per stereo config (training set, no model).

For each of the 10 stereo configs:
  1. Iterate over training samples (capped by MAX_SAMPLES).
  2. Run match -> GT depth -> reproject -> residual using the SAME masking
     pipeline as trainer._forward_step (matcher mask, depth validity, in-bounds).
  3. Pool all valid LEFT residuals (left_kps - reproject(right_kps -> left)).
  4. Compute one 2x2 empirical covariance, eigvals, principal-axis angle.

Output:
  - outputs/eval/empirical_cov.npz   (per-config: cov, mean, n, eigvals, angle_deg)
  - outputs/eval/figs/empirical_cov_ellipses.png
        2x5 grid, each cell shows the empirical 95% ellipse centered at origin.
        Direct ground-truth analogue of the model's predicted ellipses.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import chi2
from torch.utils.data import DataLoader

from uncertainty_estimation.geometry.stereo import reproject
from uncertainty_estimation.training.data.semistaticsim import (
    SemiStaticSimStereoDataset,
    stereo_collate,
)
from uncertainty_estimation.training.trainer import (
    _get_depth,
    _in_bounds,
)


OUT_NPZ      = Path("outputs/eval/empirical_cov.npz")
OUT_FIG      = Path("outputs/eval/figs/empirical_cov_ellipses.png")

MAX_SAMPLES  = 300        # cap per config — plenty for stable empirical cov
BATCH_SIZE   = 8
NUM_WORKERS  = 4
DEVICE       = "cuda"
SPLIT        = "train"
CONFIDENCE   = 0.95
CLIP_PCT     = 99.0       # drop top 1% ||r|| outliers before computing cov;
                          # set to None to disable

ROW_LABELS    = ["horizontal", "vertical"]
COL_BASELINES = [5, 10, 20, 50, 100]


def collect_residuals(stereo: str) -> tuple[np.ndarray, dict]:
    """Pool valid left-image residuals across (up to) MAX_SAMPLES training samples."""
    base    = OmegaConf.load("configs/base.yaml")
    dataset = OmegaConf.load("configs/dataset/sss.yaml")
    cfg     = OmegaConf.merge(base, {"dataset": dataset})
    cfg.dataset.stereo_config = stereo

    ds = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, SPLIT, cfg.matching)
    n_take = min(MAX_SAMPLES, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n_take)))

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=stereo_collate,
        pin_memory=False,
    )

    residuals_chunks = []
    n_total = 0
    n_valid = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(DEVICE)
            K_inv  = batch["K_inv"].to(DEVICE)
            T_lr   = batch["T_lr"].to(DEVICE)
            T_rl   = torch.linalg.inv(T_lr)
            K      = torch.linalg.inv(K_inv)
            baseline = batch["baseline"].to(DEVICE)
            focal    = K[:, 0, 0]

            left_kps  = batch["left_kps"].to(DEVICE)
            right_kps = batch["right_kps"].to(DEVICE)
            mask      = batch["match_mask"].to(DEVICE)

            depth_left, depth_right, depth_valid = _get_depth(
                "gt", batch, left_kps, right_kps, focal, baseline,
                DEVICE, cfg.dataset.max_depth,
            )
            mask = mask * depth_valid

            left_kps_reproj  = reproject(right_kps, depth_right, K, T_rl)
            right_kps_reproj = reproject(left_kps,  depth_left,  K, T_lr)

            H, W = images.shape[-2], images.shape[-1]
            mask = (
                mask
                * _in_bounds(left_kps_reproj,  H, W)
                * _in_bounds(right_kps_reproj, H, W)
            )

            res = (left_kps - left_kps_reproj)  # (B, P, 2)
            n_total += int(mask.numel())
            valid = mask.bool()
            n_valid += int(valid.sum())
            residuals_chunks.append(res[valid].cpu().numpy())

    residuals = np.concatenate(residuals_chunks, axis=0) if residuals_chunks else np.zeros((0, 2))
    stats = {
        "n_samples_used": n_take,
        "n_residuals_total": n_total,
        "n_residuals_valid": n_valid,
    }
    return residuals, stats


def _cov_summary(residuals: np.ndarray) -> dict:
    """2x2 cov + eigendecomposition + rms for a (N, 2) residual array."""
    cov  = np.cov(residuals.T)
    vals, vecs = np.linalg.eigh(cov)
    return {
        "cov":       cov,
        "eigvals":   vals,
        "angle_deg": float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))),
        "rms":       float(np.sqrt((residuals ** 2).sum(axis=1).mean())),
    }


def summarize(residuals: np.ndarray, clip_pct: float | None = None) -> dict:
    """Compute UNCLIPPED and CLIPPED empirical cov + percentile stats.

    Both summaries are kept so we can directly compare "what the data really
    is" against "what survives outlier removal" — the diagnostic the meeting
    discussion is built around.
    """
    if len(residuals) < 2:
        return {"n": len(residuals), "n_raw": len(residuals)}

    norms = np.linalg.norm(residuals, axis=1)
    pct = np.percentile(norms, [50, 75, 90, 95, 99, 99.9])

    unclipped = _cov_summary(residuals)
    if clip_pct is not None:
        cutoff = float(np.percentile(norms, clip_pct))
        residuals_used = residuals[norms <= cutoff]
        clipped = _cov_summary(residuals_used)
    else:
        cutoff = float("inf")
        residuals_used = residuals
        clipped = unclipped

    return {
        # Sample sizes
        "n":           int(len(residuals_used)),
        "n_raw":       int(len(residuals)),
        "clip_value":  cutoff,
        # Unclipped (what the data really is)
        "cov_unclipped":     unclipped["cov"],
        "eigvals_unclipped": unclipped["eigvals"],
        "angle_unclipped":   unclipped["angle_deg"],
        "rms_unclipped":     unclipped["rms"],
        # Clipped (what's left after dropping the worst outliers)
        "cov":       clipped["cov"],
        "eigvals":   clipped["eigvals"],
        "angle_deg": clipped["angle_deg"],
        "rms":       clipped["rms"],
        # Distribution shape
        "pct_norm":  pct,            # [p50, p75, p90, p95, p99, p99.9]
    }


def draw_ellipse(ax, cov, scale, **kwargs):
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    width  = 2.0 * scale * np.sqrt(vals[-1])
    height = 2.0 * scale * np.sqrt(vals[0])
    ax.add_patch(patches.Ellipse((0, 0), width=width, height=height, angle=angle, **kwargs))


def main():
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    print(f"Aggregating left residuals across {SPLIT} set "
          f"(<= {MAX_SAMPLES} samples / config)")
    for orient in ROW_LABELS:
        for b in COL_BASELINES:
            stereo = f"{orient}_{b}cm"
            residuals, stats = collect_residuals(stereo)
            summary = summarize(residuals, clip_pct=CLIP_PCT)
            results[stereo] = {**summary, **stats}
            pct      = summary.get("pct_norm", np.zeros(6))
            ev_u     = summary.get("eigvals_unclipped", np.zeros(2))
            ev_c     = summary.get("eigvals", np.zeros(2))
            ang_u    = summary.get("angle_unclipped", 0.0)
            ang_c    = summary.get("angle_deg", 0.0)
            rms_u    = summary.get("rms_unclipped", 0.0)
            rms_c    = summary.get("rms", 0.0)
            keep_pct = 100.0 * stats["n_residuals_valid"] / max(stats["n_residuals_total"], 1)

            print(f"  {stereo:18s}  "
                  f"n_valid={stats['n_residuals_valid']:7d}/{stats['n_residuals_total']:7d} "
                  f"({keep_pct:5.1f}%)")
            print(f"    unclipped: rms={rms_u:7.3f}  "
                  f"eig=({ev_u[0]:7.3f},{ev_u[-1]:8.3f})  angle={ang_u:+6.1f}°")
            print(f"    clipped @{CLIP_PCT}%: rms={rms_c:7.3f}  "
                  f"eig=({ev_c[0]:7.3f},{ev_c[-1]:8.3f})  angle={ang_c:+6.1f}°  "
                  f"cutoff={summary.get('clip_value', 0):.2f}px")
            print(f"    ||r|| percentiles  p50={pct[0]:6.3f}  p90={pct[2]:6.3f}  "
                  f"p99={pct[4]:7.3f}  p99.9={pct[5]:8.3f}")

    # Save raw stats
    save_fields = (
        "n", "n_raw", "clip_value",
        "cov", "eigvals", "angle_deg", "rms",
        "cov_unclipped", "eigvals_unclipped", "angle_unclipped", "rms_unclipped",
        "pct_norm",
    )
    np.savez(
        OUT_NPZ,
        configs=np.array(list(results.keys())),
        **{f"{k}__{f}": np.asarray(v[f])
           for k, v in results.items()
           for f in save_fields
           if f in v},
    )
    print(f"Saved {OUT_NPZ}")

    # 2x5 grid of empirical ellipses centered at origin
    conf_scale = float(np.sqrt(chi2.ppf(CONFIDENCE, df=2)))

    # Find a global half-extent so all cells share axis limits
    max_axis = 0.0
    for stereo, summary in results.items():
        if "eigvals" not in summary:
            continue
        max_axis = max(max_axis, conf_scale * np.sqrt(summary["eigvals"][-1]))
    half = 1.2 * max_axis if max_axis > 0 else 10.0

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=120)
    for r, orient in enumerate(ROW_LABELS):
        for c, b in enumerate(COL_BASELINES):
            stereo = f"{orient}_{b}cm"
            ax = axes[r, c]
            summary = results[stereo]

            if "cov" in summary:
                # Scatter a subsample of residuals as light dots in the background
                # for context (purely cosmetic, not used for the cov).
                ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
                ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
                draw_ellipse(
                    ax, summary["cov"], scale=conf_scale,
                    edgecolor="crimson", facecolor="none", linewidth=1.6,
                )
                ax.text(
                    0.02, 0.98,
                    f"n={summary['n']}\nrms={summary['rms']:.2f}px\n"
                    f"eig=({summary['eigvals'][0]:.2f},{summary['eigvals'][-1]:.2f})\n"
                    f"angle={summary['angle_deg']:+.0f}°",
                    transform=ax.transAxes, fontsize=8, va="top", ha="left",
                    family="monospace",
                    bbox=dict(facecolor="white", alpha=0.8, pad=2, edgecolor="none"),
                )

            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(f"{b} cm", fontsize=12)
            if c == 0:
                ax.set_ylabel(orient, fontsize=12)

    fig.suptitle(
        f"Empirical residual covariance — {int(CONFIDENCE*100)}% ellipses  "
        f"|  train split  |  <= {MAX_SAMPLES} samples / config",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
