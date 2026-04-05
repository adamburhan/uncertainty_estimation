"""Stereo covariance training script — Hydra-configured.

Launch:
    python scripts/train_stereo_cov.py dataset=tartanair experiment.name=001_bearing_gt
    python scripts/train_stereo_cov.py dataset=sss experiment.name=002_sss_h20cm

Override any field:
    python scripts/train_stereo_cov.py \\
        dataset=sss \\
        dataset.stereo_config=horizontal_50cm \\
        experiment.name=003_sss_h50cm \\
        loss.name=pixel_nll \\
        training.lr=5e-4
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

import hydra


# Utilities

# def seed_experiment(seed: int):



class DummyScheduler:
    """LR scheduler that does nothing. Drop-in for the scheduler interface."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, state_dict): pass


# Main

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import wandb

    from uncertainty_estimation.matching.orb import ORB
    from uncertainty_estimation.models.factory import build_model
    from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset
    from uncertainty_estimation.training.data.semistaticsim import SemiStaticSimStereoDataset
    from uncertainty_estimation.training.losses import build_loss
    from uncertainty_estimation.training.trainer import train_model

    """Seed all RNGs for reproducibility."""
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def build_dataset(cfg: DictConfig, split: str):
        if cfg.dataset.name == "tartanair":
            return TartanAirLiveDataset(cfg.dataset, cfg.augmentation, split)
        if cfg.dataset.name == "semistaticsim":
            return SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, split)
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Available: tartanair, semistaticsim")

    # Auto-derive experiment name if not explicitly set
    if OmegaConf.is_missing(cfg.experiment, "name"):
        with open_dict(cfg):
            cfg.experiment.name = f"{cfg.dataset.name}_{cfg.dataset.get('stereo_config', 'default')}"

    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.training.device)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, "config.yaml")

    # Data 
    train_dataset = build_dataset(cfg, split="train")
    val_dataset = build_dataset(cfg, split="val")

    sample = train_dataset[0]
    print(f"images: {sample['images'].shape}, K_inv: {sample['K_inv'].shape}, "
          f"T_lr: {sample['T_lr'].shape}, baseline: {sample['baseline']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    train_loader_for_eval = DataLoader(
        train_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # Model, loss, optimizer 
    model = build_model(cfg.model).to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_gamma)

    # Matching
    matching_fn = lambda images: ORB(
        images, device,
        max_keypoints=cfg.matching.max_keypoints,
        max_hamming_distance=cfg.matching.max_hamming,
        max_epipolar_error=cfg.matching.max_epipolar_error,
    )

    # WandB 
    tags = list(cfg.logging.wandb_tags) + [
        cfg.loss.name, cfg.dataset.depth_source, cfg.dataset.name,
    ]
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment.name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline" if cfg.logging.wandb_offline else "online",
    )

    # Callbacks
    log_fn = lambda metrics: wandb.log(metrics)

    from uncertainty_estimation.visualization.covariance import visualize_covariances
    def vis_fn(model, batch):
        figs = visualize_covariances(model, batch, matching_fn, device)
        result = {k: wandb.Image(v) for k, v in figs.items()}
        for fig in figs.values():
            plt.close(fig)
        return result

    # Train 
    all_metrics = train_model(
        model=model,
        train_loader=train_loader,
        train_loader_for_eval=train_loader_for_eval,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        matching_fn=matching_fn,
        device=device,
        depth_source=cfg.dataset.depth_source,
        max_depth=cfg.dataset.max_depth,
        num_epochs=cfg.training.epochs,
        grad_clip=cfg.training.grad_clip,
        exp_name=cfg.experiment.name,
        checkpoint_dir=str(checkpoint_dir),
        eval_period=cfg.training.validation_interval,
        checkpoint_period=cfg.training.checkpoint_interval,
        vis_period=cfg.training.vis_interval,
        log_fn=log_fn,
        vis_fn=vis_fn,
    )

    wandb.finish()

    # Save final metrics alongside checkpoints
    torch.save(all_metrics, checkpoint_dir / f"{cfg.experiment.name}_metrics.pth")

 
    
if __name__ == "__main__":
    main()
