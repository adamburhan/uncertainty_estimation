"""Dump per-keypoint predictions for one checkpoint to a .npz file.

Right-image-anchored: residuals = right_kps - reproject(left_kps, depth_left -> right)
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from uncertainty_estimation.geometry.stereo import extract_covs, reproject
from uncertainty_estimation.models.factory import build_model
from uncertainty_estimation.training.data.semistaticsim import (
    SemiStaticSimStereoDataset,
    stereo_collate,
)
from uncertainty_estimation.training.trainer import _lookup_depth


# ---- edit these for the one checkpoint you want to dump ----
CKPT       = Path("/home/mila/a/adam.burhan/scratch/stereo-UQ/checkpoints/A_stereo__semistaticsim_horizontal_5cm_bearing_nll_real__seed0/'A_stereo__semistaticsim_horizontal_5cm_bearing_nll_real__seed0_best_epoch=10_loss=-32.8917.pth'")
STEREO     = "horizontal_5cm"
EXPERIMENT = "A_real"
SEED       = 0
DEVICE     = "cuda"     # cuda if you're on a GPU node
OUT_DIR    = Path("outputs/eval")
# -------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build config (same defaults as training)
    base    = OmegaConf.load("configs/base.yaml")
    dataset = OmegaConf.load("configs/dataset/sss.yaml")
    cfg     = OmegaConf.merge(base, {"dataset": dataset})
    cfg.dataset.stereo_config = STEREO

    # 2. Build val dataset and grab the first sample (one batch of size 1)
    ds = SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, "val", cfg.matching)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=stereo_collate)
    batch = next(iter(loader))

    # 3. Load model
    model = build_model(cfg.model).to(DEVICE)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {CKPT.name}  (epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')})")

    # 4. Forward pass + per-kp quantities
    with torch.no_grad():
        images = batch["images"].to(DEVICE)              # (1, 2, C, H, W)
        K_inv  = batch["K_inv"].to(DEVICE)               # (1, 3, 3)
        T_lr   = batch["T_lr"].to(DEVICE)                # (1, 4, 4)
        K      = torch.linalg.inv(K_inv)                 # (1, 3, 3)

        left_kps  = batch["left_kps"].to(DEVICE)         # (1, P, 2)
        right_kps = batch["right_kps"].to(DEVICE)        # (1, P, 2)
        mask      = batch["match_mask"].to(DEVICE).bool()  # (1, P)

        depth_left  = batch["depth_left"].to(DEVICE)     # (1, H, W)
        depth_right = batch["depth_right"].to(DEVICE)    # (1, H, W)

        # Predict cov field, sample at the matched kp locations
        cov_preds = model(images)                        # (2, H, W, 2, 2)
        _, right_covs = extract_covs(cov_preds, left_kps, right_kps)  # (1, P, 2, 2)

        # Right-anchored reprojection: left -> right
        depth_at_left_kps = _lookup_depth(depth_left, left_kps)       # (1, P)
        right_kps_reproj  = reproject(left_kps, depth_at_left_kps, K, T_lr)  # (1, P, 2)
        residuals = right_kps - right_kps_reproj                      # (1, P, 2)

        # Depth at the right keypoints — useful for binning/coloring later
        depth_at_right_kps = _lookup_depth(depth_right, right_kps)    # (1, P)

    # 5. Slice batch index 0, apply mask, dump
    m = mask[0].cpu().numpy()
    npz_path = OUT_DIR / f"{EXPERIMENT}_{STEREO}_seed{SEED}.npz"
    np.savez(
        npz_path,
        # background panels
        image_right = images[0, 1, 0].cpu().numpy(),       # (H, W) grayscale
        depth_right = depth_right[0].cpu().numpy(),        # (H, W)
        # per-keypoint quantities (valid only)
        right_kps        = right_kps[0].cpu().numpy()[m],
        right_kps_reproj = right_kps_reproj[0].cpu().numpy()[m],
        right_covs       = right_covs[0].cpu().numpy()[m],
        residuals        = residuals[0].cpu().numpy()[m],
        kp_depth         = depth_at_right_kps[0].cpu().numpy()[m],
        # metadata
        stereo     = STEREO,
        experiment = EXPERIMENT,
        seed       = SEED,
        epoch      = int(ckpt.get("epoch", -1)),
    )
    print(f"Wrote {npz_path}")
    print(f"  shapes: kps={right_kps[0][m].shape}  covs={right_covs[0][m].shape}  residuals={residuals[0][m].shape}")
    print(f"  mean ||r|| = {np.linalg.norm(residuals[0].cpu().numpy()[m], axis=1).mean():.3f} px")


if __name__ == "__main__":
    main()
