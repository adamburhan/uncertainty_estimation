import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from uncertainty_estimation.matching.orb import ORB
from uncertainty_estimation.models.full_model import UnetCovarianceModel
from uncertainty_estimation.models.output_filter import Filters, OutputFilter, get_filter
from uncertainty_estimation.models.parameterization import CovarianceParameterization, get_parametrization
from uncertainty_estimation.models.transforms import covs_to_image
from uncertainty_estimation.models.unet.unet_model import UNetXS
from uncertainty_estimation.training.data.config import DatasetConfig, DatasetDirectoriesConfig, ImageAugmentationConfig
from uncertainty_estimation.training.data.kitti import KITTILiveDataset
from uncertainty_estimation.training.losses import bearing_nll
from uncertainty_estimation.training.trainer import eval_step, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stereo covariance predictor")
    parser.add_argument("--config", type=str, default="configs/training/stereo_ssl.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict, device: torch.device) -> UnetCovarianceModel:
    unet = UNetXS(n_channels=1, n_classes=3)

    # SAB output filter: scale→lukas, angle→identity, beta→sigmoid(0,1)
    output_filter = OutputFilter(
        filter1=get_filter(Filters.lukas),
        filter2=get_filter(Filters.no),
        filter3=get_filter(Filters.sigmoid, min=0.0, max=1.0),
    )

    parameterization = get_parametrization(CovarianceParameterization.sab)

    model = UnetCovarianceModel(
        unet=unet,
        output_filter=output_filter,
        parameterization=parameterization,
        model_cfg=type("Cfg", (), {"isotropic_covariances": model_cfg["isotropic_covariances"]})(),
    ).to(device)

    if model_cfg.get("checkpoint"):
        ckpt = torch.load(model_cfg["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {model_cfg['checkpoint']}")

    return model


def build_dataset_config(config: dict) -> DatasetConfig:
    return DatasetConfig(
        directories=DatasetDirectoriesConfig(
            dataset=config["dataset"]["directories"]["dataset"],
            sequences=config["dataset"]["directories"]["sequences"],
            left_images=config["dataset"]["directories"]["left_images"],
            right_images=config["dataset"]["directories"]["right_images"],
        ),
        image_augmentations=ImageAugmentationConfig(
            noise=config["image_augmentations"]["noise"],
            noise_var=config["image_augmentations"]["noise_var"],
            random_crop=config["image_augmentations"]["random_crop"],
            crop_size=tuple(config["image_augmentations"]["crop_size"]),
        ),
    )


def visualize_covariances(
    model: UnetCovarianceModel,
    batch: dict,
    matching_fn,
    device: torch.device,
    scale_limit: float = 50.0,
) -> dict:
    """Render dense covariance map + keypoint ellipse overlay for one batch.

    Returns a dict of wandb.Image objects ready to log.
    """
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)          # B, 2, C, H, W
        cov_preds = model(images)                    # B*2, H, W, 2, 2

        # --- Dense covariance map (left image, first batch item) ---
        left_cov = cov_preds[0].cpu()                # H, W, 2, 2
        cov_rgb = covs_to_image(left_cov, (None, scale_limit))  # H, W, 3  in [0,1]

        fig_dense, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.imshow(cov_rgb)
        ax.axis("off")
        ax.set_title("Predicted covariance (HSV: hue=angle, sat=anisotropy, val=scale)")
        fig_dense.tight_layout()

        # --- Keypoint ellipse overlay ---
        left_kps, right_kps, masks = matching_fn(images[:1])  # single image pair
        left_kps = left_kps.to(device)
        masks = masks[0].bool()                      # P,
        kps = left_kps[0][masks].cpu().numpy()       # N, 2  (x, y)

        # Sample covariances at keypoint locations (nearest neighbour)
        H, W = left_cov.shape[:2]
        col = torch.from_numpy(kps[:, 0]).round().long().clamp(0, W - 1)
        row = torch.from_numpy(kps[:, 1]).round().long().clamp(0, H - 1)
        covs_at_kps = left_cov[row, col].numpy()    # N, 2, 2

        left_img = images[0, 0, 0].cpu().numpy()    # H, W  grayscale in [0,1]
        fig_kps, ax2 = plt.subplots(1, 1, figsize=(10, 3))
        ax2.imshow(left_img, cmap="gray", vmin=0, vmax=1)

        for (x, y), cov in zip(kps, covs_at_kps):
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0)               # numerical safety
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            w, h = 2.0 * np.sqrt(vals[::-1])        # 1-sigma axes
            ellipse = patches.Ellipse(
                (x, y), width=w, height=h, angle=angle,
                edgecolor="lime", facecolor="none", linewidth=0.8, alpha=0.8,
            )
            ax2.add_patch(ellipse)

        ax2.scatter(kps[:, 0], kps[:, 1], s=4, c="red", linewidths=0)
        ax2.axis("off")
        ax2.set_title(f"Keypoint covariance ellipses ({len(kps)} matches)")
        fig_kps.tight_layout()

    result = {
        "vis/cov_map":      wandb.Image(fig_dense),
        "vis/kp_ellipses":  wandb.Image(fig_kps),
    }
    plt.close(fig_dense)
    plt.close(fig_kps)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(config["training"]["device"])
    log_cfg = config["logging"]
    train_cfg = config["training"]
    match_cfg = config["matching"]

    wandb.init(
        project=log_cfg["project_name"],
        name=log_cfg["run_name"],
        config=config,
    )

    dataset_config = build_dataset_config(config)

    # Sanity check on first sample
    training_dataset = KITTILiveDataset(dataset_config, split="train")
    sample = training_dataset[0]
    print("images:", sample["images"].shape)
    print("K_inv:", sample["K_inv"].shape)
    print("T_lr:", sample["T_lr"].shape)
    print("baseline:", sample["baseline"])

    val_dataset = KITTILiveDataset(dataset_config, split="val")

    train_loader = DataLoader(
        training_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    model = build_model(config["model"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_gamma"])

    matching_fn = lambda images: ORB(
        images, device,
        max_keypoints=match_cfg["max_points"],
        max_hamming_distance=match_cfg.get("max_hamming_distance", 64),
        max_epipolar_error=match_cfg.get("max_epipolar_error", 2.0),
    )

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(train_cfg["num_epochs"]):
        # ---- Training ----
        epoch_loss = 0.0
        epoch_kps = 0
        for batch in train_loader:
            result = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                loss_fn=bearing_nll,
                matching_fn=matching_fn,
                device=device,
            )
            wandb.log({
                "train/batch_loss": result["loss"],
                "train/n_valid_kps": result["n_valid_kps"],
                "train/lr": scheduler.get_last_lr()[0],
                "step": global_step,
            })
            epoch_loss += result["loss"]
            epoch_kps += result["n_valid_kps"]
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']}  train_loss={avg_train_loss:.4f}")
        wandb.log({"train/loss": avg_train_loss, "train/avg_kps": epoch_kps / len(train_loader), "epoch": epoch + 1})

        # ---- Validation ----
        if (epoch + 1) % train_cfg["validation_interval"] == 0:
            val_loss = 0.0
            val_kps = 0
            for batch in val_loader:
                result = eval_step(
                    model=model,
                    batch=batch,
                    loss_fn=bearing_nll,
                    matching_fn=matching_fn,
                    device=device,
                )
                val_loss += result["loss"]
                val_kps += result["n_valid_kps"]

            avg_val_loss = val_loss / len(val_loader)
            print(f"             val_loss={avg_val_loss:.4f}")
            wandb.log({"val/loss": avg_val_loss, "val/avg_kps": val_kps / len(val_loader), "epoch": epoch + 1})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    checkpoint_dir / "best.pth",
                )

        # ---- Periodic checkpoint ----
        if (epoch + 1) % train_cfg["checkpoint_interval"] == 0:
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint_dir / f"epoch_{epoch+1:04d}.pth",
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

        # ---- Visualisation ----
        if (epoch + 1) % train_cfg["vis_interval"] == 0:
            vis_batch = next(iter(val_loader))
            wandb.log({**visualize_covariances(model, vis_batch, matching_fn, device), "epoch": epoch + 1})

    wandb.finish()


if __name__ == "__main__":
    main()
