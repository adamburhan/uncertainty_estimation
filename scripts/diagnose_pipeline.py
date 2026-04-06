"""Diagnostic script for the stereo covariance pipeline.

Runs one batch through matching -> depth lookup -> reprojection -> covariance
and prints stats + saves visualization plots.

Launch:
    python -m scripts.diagnose_pipeline dataset=sss experiment.name=diag
    python -m scripts.diagnose_pipeline dataset=tartanair experiment.name=diag training.device=mps
"""

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import hydra

from uncertainty_estimation.geometry.stereo import reproject, extract_covs
from uncertainty_estimation.matching.orb import ORB
from uncertainty_estimation.models.factory import build_model
from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset
from uncertainty_estimation.training.data.semistaticsim import SemiStaticSimStereoDataset
from uncertainty_estimation.training.trainer import _lookup_depth


def seed_experiment(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: DictConfig, split: str):
    if cfg.dataset.name == "tartanair":
        return TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split)
    if cfg.dataset.name == "semistaticsim":
        return SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, split)
    raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Available: tartanair, semistaticsim")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_experiment(cfg.training.seed)

    device = torch.device(cfg.training.device)

    # Data
    val_dataset = build_dataset(cfg, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # Model (untrained — just checking output validity)
    model = build_model(cfg.model).to(device)

    # Matching
    matching_fn = lambda images, K: ORB(
        images, device, K,
        max_keypoints=cfg.matching.max_keypoints,
        max_hamming_distance=cfg.matching.max_hamming,
        lowe_ratio=cfg.matching.lowe_ratio,
        ransac_reproj_threshold=cfg.matching.ransac_reproj_threshold,
    )

    # Grab one batch
    for batch in val_loader:
        images      = batch["images"].to(device)        # B, 2, C, H, W
        K_inv       = batch["K_inv"].to(device)          # B, 3, 3
        T_lr        = batch["T_lr"].to(device)           # B, 4, 4
        baseline    = batch["baseline"].to(device)       # B,
        depth_left  = batch["depth_left"].to(device)     # B, H, W
        depth_right = batch["depth_right"].to(device)    # B, H, W
        break

    B = images.shape[0]
    K = torch.linalg.inv(K_inv)
    T_rl = torch.linalg.inv(T_lr)
    max_depth = cfg.dataset.max_depth

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC — batch shapes")
    print(f"  images:      {images.shape}")
    print(f"  depth_left:  {depth_left.shape}")
    print(f"  depth_right: {depth_right.shape}")
    print(f"  K_inv:       {K_inv.shape}")
    print(f"  T_lr:        {T_lr.shape}")
    print(f"  baseline:    {baseline.shape} = {baseline[0].item():.4f}m")
    print(f"{'='*60}\n")

    # 1. Matching 
    print("--- 1. ORB Matching ---")
    left_kps, right_kps, masks = matching_fn(images, K)
    left_kps  = left_kps.to(device)
    right_kps = right_kps.to(device)
    masks     = masks.to(device)

    for b in range(B):
        n_valid = masks[b].sum().item()
        if n_valid == 0:
            print(f"  Sample {b}: 0 matches!")
            continue
        valid = masks[b].bool()
        lk = left_kps[b, valid]
        rk = right_kps[b, valid]

        print(f"  Sample {b}: {int(n_valid)} matches ")

    # 2. Depth at keypoints
    print("\n--- 2. Depth Lookup at Keypoints ---")
    d_left_kp  = _lookup_depth(depth_left, left_kps)
    d_right_kp = _lookup_depth(depth_right, right_kps)

    for b in range(B):
        valid = masks[b].bool()
        if valid.sum() == 0:
            continue
        dl = d_left_kp[b, valid]
        dr = d_right_kp[b, valid]
        n_zero_l = (dl <= 0.01).sum().item()
        n_zero_r = (dr <= 0.01).sum().item()
        print(f"  Sample {b}: "
              f"depth_left  min={dl.min():.3f} max={dl.max():.3f} mean={dl.mean():.3f}m (zero/invalid={n_zero_l}) | "
              f"depth_right min={dr.min():.3f} max={dr.max():.3f} mean={dr.mean():.3f}m (zero/invalid={n_zero_r})")

    # 3. Reprojection 
    print("\n--- 3. Reprojection Error (GT depth + GT extrinsics) ---")
    d_left_clamped  = d_left_kp.clamp(0.1, max_depth)
    d_right_clamped = d_right_kp.clamp(0.1, max_depth)
    right_kps_reproj = reproject(left_kps, d_left_clamped, K, T_lr)
    left_kps_reproj  = reproject(right_kps, d_right_clamped, K, T_rl)

    for b in range(B):
        valid = masks[b].bool()
        if valid.sum() == 0:
            continue
        err_lr = (right_kps_reproj[b, valid] - right_kps[b, valid]).norm(dim=-1)
        err_rl = (left_kps_reproj[b, valid] - left_kps[b, valid]).norm(dim=-1)
        print(f"  Sample {b}: "
              f"L->R reproj err: mean={err_lr.mean():.2f} median={err_lr.median():.2f} max={err_lr.max():.2f}px | "
              f"R->L reproj err: mean={err_rl.mean():.2f} median={err_rl.median():.2f} max={err_rl.max():.2f}px")

    # 4. Covariance model output 
    print("\n--- 4. Covariance Prediction ---")
    model.eval()
    with torch.no_grad():
        cov_preds = model(images)
    print(f"  cov_preds shape: {cov_preds.shape}")
    left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)

    for b in range(B):
        valid = masks[b].bool()
        if valid.sum() == 0:
            continue
        lc = left_covs[b, valid]
        eigvals = torch.linalg.eigvalsh(lc.cpu())
        n_pd = (eigvals > 0).all(dim=-1).sum().item()
        n_total = int(valid.sum().item())
        print(f"  Sample {b}: "
              f"pos-def: {n_pd}/{n_total} | "
              f"eigenvalue range: [{eigvals.min():.6f}, {eigvals.max():.6f}] | "
              f"det range: [{torch.det(lc.cpu()).min():.6f}, {torch.det(lc.cpu()).max():.6f}]")

    # 5. Visualization (first 4 samples) 
    print("\n--- 5. Saving Diagnostic Plots ---")
    n_vis = min(4, B)
    for i in range(n_vis):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        valid = masks[i].bool()
        lk = left_kps[i].cpu().numpy()
        rk = right_kps[i].cpu().numpy()
        rk_reproj = right_kps_reproj[i].cpu().numpy()
        m = valid.cpu().numpy()

        left_img  = images[i, 0].cpu().squeeze().numpy()
        right_img = images[i, 1].cpu().squeeze().numpy()
        dl = depth_left[i].cpu().numpy()
        dr = depth_right[i].cpu().numpy()

        # (0,0) Left image + keypoints
        axes[0, 0].imshow(left_img, cmap="gray")
        axes[0, 0].scatter(lk[m, 0], lk[m, 1], s=4, c="lime", alpha=0.7)
        axes[0, 0].set_title(f"Left + {m.sum()} keypoints")

        # (0,1) Right image + matched keypoints + reprojected
        axes[0, 1].imshow(right_img, cmap="gray")
        axes[0, 1].scatter(rk[m, 0], rk[m, 1], s=4, c="lime", alpha=0.7, label="matched")
        axes[0, 1].scatter(rk_reproj[m, 0], rk_reproj[m, 1], s=4, c="red", alpha=0.5, label="reprojected")
        axes[0, 1].legend(fontsize=8, loc="upper right")
        axes[0, 1].set_title("Right: matched (green) vs reprojected (red)")

        # (0,2) Correspondence lines (subsample for clarity)
        canvas = np.concatenate([left_img, right_img], axis=1)
        axes[0, 2].imshow(canvas, cmap="gray")
        n_lines = min(50, m.sum())
        indices = np.where(m)[0]
        np.random.shuffle(indices)
        for j in indices[:n_lines]:
            axes[0, 2].plot(
                [lk[j, 0], rk[j, 0] + left_img.shape[1]],
                [lk[j, 1], rk[j, 1]],
                linewidth=0.5, alpha=0.6,
            )
        axes[0, 2].set_title(f"Correspondences (showing {n_lines})")

        # (1,0) Depth left
        im_dl = axes[1, 0].imshow(dl, cmap="turbo")
        axes[1, 0].set_title(f"Depth Left [{dl.min():.2f}, {dl.max():.2f}]m")
        plt.colorbar(im_dl, ax=axes[1, 0], fraction=0.046)

        # (1,1) Depth right
        im_dr = axes[1, 1].imshow(dr, cmap="turbo")
        axes[1, 1].set_title(f"Depth Right [{dr.min():.2f}, {dr.max():.2f}]m")
        plt.colorbar(im_dr, ax=axes[1, 1], fraction=0.046)

        # (1,2) Reprojection error histogram
        if valid.sum() > 0:
            err = (right_kps_reproj[i, valid] - right_kps[i, valid]).norm(dim=-1).cpu().numpy()
            axes[1, 2].hist(err, bins=50, edgecolor="black", alpha=0.7)
            axes[1, 2].axvline(np.median(err), color="red", linestyle="--", label=f"median={np.median(err):.2f}px")
            axes[1, 2].set_xlabel("Reprojection error (px)")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].legend()
        axes[1, 2].set_title("L->R Reprojection Error Distribution")

        b_val = baseline[i].item()
        t = T_lr[i].cpu().numpy()
        tx, ty, tz = t[0, 3], t[1, 3], t[2, 3]
        K_i = torch.linalg.inv(K_inv[i]).cpu().numpy()
        fig.suptitle(
            f"Sample {i} | baseline={b_val:.3f}m | T_lr=[{tx:.3f},{ty:.3f},{tz:.3f}] | "
            f"fx={K_i[0,0]:.1f} cy={K_i[1,2]:.1f} | matches={int(valid.sum())}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(f"debug_sample_{i}.png", dpi=150)
        plt.close()
        print(f"  Saved debug_sample_{i}.png")

    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
