"""Stereo covariance training script — Hydra-configured.

Launch:
    python scripts/train_stereo_cov.py experiment.name=001_bearing_gt

Override any field:
    python scripts/train_stereo_cov.py \\
        experiment.name=002_pixel_nll \\
        loss.name=pixel_nll \\
        model.architecture=UNetSmall \\
        dataset.depth_source=gt \\
        training.lr=5e-4

Use KITTI config:
    python scripts/train_stereo_cov.py --config-name base_kitti experiment.name=001_kitti
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import hydra

from uncertainty_estimation.matching.orb import ORB
from uncertainty_estimation.models.factory import build_model
from uncertainty_estimation.models.transforms import covs_to_image
# from uncertainty_estimation.training.data.kitti import KITTILiveDataset
from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset
from uncertainty_estimation.training.losses import build_loss
from uncertainty_estimation.training.trainer import eval_step, train_step


# dataset factory
def build_dataset(cfg: DictConfig, split: str):
    if cfg.dataset.name == "tartanair":
        return TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split)

    if cfg.dataset.name == "kitti":
        raise NotImplementedError("KITTI dataset support is not implemented yet. Please use TartanAir configs for now.")

    raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Choose from: tartanair, kitti")


# Visualisation 

def visualize_covariances(model, batch, matching_fn, device, scale_limit=50.0):
    """Render dense covariance map + keypoint ellipse overlay for one batch."""
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)      # B, 2, C, H, W
        cov_preds = model(images)                # B*2, H, W, 2, 2

        left_cov = cov_preds[0].cpu()            # H, W, 2, 2
        cov_rgb  = covs_to_image(left_cov, (None, scale_limit))  # H, W, 3

        fig_dense, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        left_kps, right_kps, masks = matching_fn(images[:1])
        left_kps = left_kps.to(device)
        masks    = masks[0].bool()
        kps      = left_kps[0][masks].cpu().numpy()  # N, 2

        H, W = left_cov.shape[:2]
        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()  # N, 2, 2

        left_img = images[0, 0, 0].cpu().numpy()
        fig_kps, ax2 = plt.subplots(1, 1, figsize=(10, 3))
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)

        for (x, y), cov in zip(kps, covs_at_kps):
            vals, vecs = np.linalg.eigh(cov)
            vals  = np.maximum(vals, 0)
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            w2, h2 = 2.0 * np.sqrt(vals[::-1])
            ellipse = patches.Ellipse(
                (x, y), width=w2, height=h2, angle=angle,
                edgecolor="lime", facecolor="none", linewidth=0.8, alpha=0.8,
            )
            ax2.add_patch(ellipse)

        ax2.scatter(kps[:, 0], kps[:, 1], s=4, c="red", linewidths=0)
        ax2.axis("off")
        ax2.set_title(f"Keypoint covariance ellipses ({len(kps)} matches)")
        fig_kps.tight_layout()

    result = {"vis/cov_map": wandb.Image(fig_dense), "vis/kp_ellipses": wandb.Image(fig_kps)}
    plt.close(fig_dense)
    plt.close(fig_kps)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # seed everything for reproducibility
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Output dir (Hydra has already changed CWD here) 
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, "config.yaml")  # permanent record alongside checkpoints

    device = torch.device(cfg.training.device)

    # Datasets & loaders
    train_dataset = build_dataset(cfg, split="train")
    val_dataset   = build_dataset(cfg, split="val")

    # Sanity check on first sample
    sample = train_dataset[0]
    print("images:", sample["images"].shape)
    print("K_inv:", sample["K_inv"].shape)
    print("T_lr:", sample["T_lr"].shape)
    print("baseline:", sample["baseline"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # Model, loss, optimizer 
    model = build_model(cfg.model, device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_gamma)

    matching_fn = lambda images: ORB(
        images, device,
        max_keypoints=cfg.matching.max_keypoints,
        max_hamming_distance=cfg.matching.max_hamming,
        max_epipolar_error=cfg.matching.max_epipolar_error,
    )

    depth_source = cfg.dataset.depth_source
    max_depth    = cfg.dataset.max_depth

    # WandB 
    tags = list(cfg.logging.wandb_tags) + [
        cfg.loss.name,
        cfg.dataset.depth_source,
        cfg.dataset.name,
    ]
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment.name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline" if cfg.logging.wandb_offline else "online",
    )

    # Training loop 
    global_step  = 0
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        # Train 
        epoch_loss = 0.0
        epoch_kps  = 0
        for batch in train_loader:
            result = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=loss_fn,
                matching_fn=matching_fn,
                device=device,
                depth_source=depth_source,
                max_depth=max_depth,
            )
            wandb.log({
                "train/batch_loss": result["loss"],
                "train/n_valid_kps": result["n_valid_kps"],
                "train/lr": scheduler.get_last_lr()[0],
                "step": global_step,
            })
            epoch_loss += result["loss"]
            epoch_kps  += result["n_valid_kps"]
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.training.epochs}  train_loss={avg_train_loss:.4f}")
        wandb.log({
            "train/loss": avg_train_loss,
            "train/avg_kps": epoch_kps / len(train_loader),
            "epoch": epoch + 1,
        })

        # Validation 
        if (epoch + 1) % cfg.training.validation_interval == 0:
            val_loss = 0.0
            val_kps  = 0
            for batch in val_loader:
                result = eval_step(
                    model=model,
                    batch=batch,
                    loss_fn=loss_fn,
                    matching_fn=matching_fn,
                    device=device,
                    depth_source=depth_source,
                    max_depth=max_depth,
                )
                val_loss += result["loss"]
                val_kps  += result["n_valid_kps"]

            avg_val_loss = val_loss / len(val_loader)
            print(f"             val_loss={avg_val_loss:.4f}")
            wandb.log({
                "val/loss": avg_val_loss,
                "val/avg_kps": val_kps / len(val_loader),
                "epoch": epoch + 1,
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    checkpoint_dir / "best.pth",
                )

        # Periodic checkpoint 
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir / f"epoch_{epoch+1:04d}.pth",
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

        # Visualisation 
        if (epoch + 1) % cfg.training.vis_interval == 0:
            vis_batch = next(iter(val_loader))
            wandb.log({
                **visualize_covariances(model, vis_batch, matching_fn, device),
                "epoch": epoch + 1,
            })

    wandb.finish()


if __name__ == "__main__":
    main()
