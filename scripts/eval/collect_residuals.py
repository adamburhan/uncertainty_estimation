"""Pool raw left-image residuals across stereo configs and dump to disk.

Run once; downstream analysis (outlier rejection, plots) loads from the
cached .npz files rather than re-running the dataloader.
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from uncertainty_estimation.geometry.stereo import reproject
from uncertainty_estimation.training.data.semistaticsim import (
    SemiStaticSimStereoDataset,
    stereo_collate,
)
from uncertainty_estimation.training.trainer import _get_depth, _in_bounds


MAX_SAMPLES = 300
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda"
SPLIT = "train"

ORIENTATIONS = ["horizontal", "vertical"]
BASELINES = [5, 10, 20, 50, 100]

OUT_DIR = Path("outputs/eval/residuals")


def collect(stereo: str) -> tuple[np.ndarray, dict]:
    base = OmegaConf.load("configs/base.yaml")
    dataset = OmegaConf.load("configs/dataset/sss.yaml")
    cfg = OmegaConf.merge(base, {"dataset": dataset})
    cfg.dataset.stereo_config = stereo

    ds = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, SPLIT, None)
    n_take = min(MAX_SAMPLES, len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n_take)))
    loader = DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=stereo_collate, pin_memory=False,
    )

    chunks = []
    n_total = 0
    n_valid = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(DEVICE)
            K_inv = batch["K_inv"].to(DEVICE)
            T_lr = batch["T_lr"].to(DEVICE)
            T_rl = torch.linalg.inv(T_lr)
            K = torch.linalg.inv(K_inv)
            baseline = batch["baseline"].to(DEVICE)
            focal = K[:, 0, 0]

            left_kps = batch["left_kps"].to(DEVICE)
            right_kps = batch["right_kps"].to(DEVICE)
            mask = batch["match_mask"].to(DEVICE)

            depth_left, depth_right, depth_valid = _get_depth(
                "gt", batch, left_kps, right_kps, focal, baseline,
                DEVICE, cfg.dataset.max_depth,
            )
            mask = mask * depth_valid

            left_kps_reproj = reproject(right_kps, depth_right, K, T_rl)
            right_kps_reproj = reproject(left_kps, depth_left, K, T_lr)

            H, W = images.shape[-2], images.shape[-1]
            mask = (
                mask
                * _in_bounds(left_kps_reproj, H, W)
                * _in_bounds(right_kps_reproj, H, W)
            )

            res = left_kps - left_kps_reproj
            n_total += int(mask.numel())
            valid = mask.bool()
            n_valid += int(valid.sum())
            chunks.append(res[valid].cpu().numpy())

    residuals = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2))
    stats = {
        "n_samples_used": n_take,
        "n_residuals_total": n_total,
        "n_residuals_valid": n_valid,
    }
    return residuals, stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for orient in ORIENTATIONS:
        for b in BASELINES:
            stereo = f"{orient}_{b}cm"
            print(f"[{stereo}] collecting ...", flush=True)
            residuals, stats = collect(stereo)
            out = OUT_DIR / f"{stereo}.npz"
            np.savez(out, residuals=residuals, **stats)
            print(f"  -> {out}  |  N={len(residuals)}  "
                  f"(valid {stats['n_residuals_valid']}/{stats['n_residuals_total']})")


if __name__ == "__main__":
    main()
