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

import hashlib
import os
import random
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

import hydra

# All heavy imports (torch, matplotlib, wandb, uncertainty_estimation) are deferred
# to inside main() so that cloudpickle / submitit can serialize the task function
# without hitting unpicklable objects like torch.backends.cudnn.CudnnModule.


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import wandb

    from uncertainty_estimation.models.factory import build_model
    from uncertainty_estimation.training.data.tartanair import TartanAirLiveDataset
    from uncertainty_estimation.training.data.semistaticsim import (
        SemiStaticSimStereoDataset,
        stereo_collate,
    )
    from uncertainty_estimation.training.losses import build_loss
    from uncertainty_estimation.training.trainer import train_model

    # Seed all RNGs for reproducibility
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
            return SemiStaticSimStereoDataset(cfg.dataset, cfg.augmentation, split, cfg.matching)
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Available: tartanair, semistaticsim")

    # Auto-derive experiment name and group from the experiment axes.
    # Group aggregates runs that should be averaged together (same axes, different seeds).
    # Name additionally pins the seed so each run is uniquely identifiable.
    stereo_cfg = cfg.dataset.get("stereo_config", "default")
    corr_mode = cfg.correspondence.mode
    loss_name = cfg.loss.name
    seed = cfg.training.seed
    cell_id = f"{cfg.dataset.name}_{stereo_cfg}_{loss_name}_{corr_mode}"

    with open_dict(cfg):
        if cfg.experiment.group is None:
            cfg.experiment.group = cell_id
        if cfg.experiment.name is None:
            cfg.experiment.name = f"{cfg.experiment.label}__{cell_id}__seed{seed}"

    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.training.device)

    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)  # absolute path, works for both run and multirun

    # Stable checkpoint dir keyed by experiment identity (NOT job num) so that
    # preempted-and-requeued submitit jobs (which land in a new sweep dir) can
    # find the previous epoch's checkpoint and resume from it.
    scratch = os.environ.get("SCRATCH", str(output_dir.parent))
    checkpoint_dir = Path(scratch) / "stereo-UQ" / "checkpoints" / cfg.experiment.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

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
        collate_fn=stereo_collate,
        persistent_workers=cfg.training.num_workers > 0,
    )
    train_loader_for_eval = DataLoader(
        train_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=stereo_collate,
        persistent_workers=cfg.training.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=stereo_collate,
        persistent_workers=cfg.training.num_workers > 0,
    )

    # Model, loss, optimizer 
    model = build_model(cfg.model).to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.lr_gamma)

    # Matching is done per-sample in the dataloader workers (see
    # SemiStaticSimStereoDataset.__getitem__). The matching_fn passed below is
    # only used as a fallback if a batch lacks precomputed kps — kept for the
    # tartanair dataset which hasn't been migrated yet.
    matching_fn = None

    # WandB — every experiment axis becomes a tag for free-form slicing.
    # Parse "horizontal_10cm" / "vertical_50cm" into direction + magnitude tags.
    baseline_dir = "unknown"
    baseline_mag = "unknown"
    if "_" in stereo_cfg:
        baseline_dir, baseline_mag = stereo_cfg.split("_", 1)

    tags = list(cfg.logging.wandb_tags) + [
        f"exp:{cfg.experiment.label}",
        f"dataset:{cfg.dataset.name}",
        f"stereo:{stereo_cfg}",
        f"baseline_dir:{baseline_dir}",
        f"baseline_mag:{baseline_mag}",
        f"loss:{loss_name}",
        f"corr:{corr_mode}",
        f"seed:{seed}",
        f"depth:{cfg.dataset.depth_source}",
    ]
    # Deterministic wandb id derived from experiment identity. A preempted job
    # that gets requeued (or a script restart) will reattach to the SAME wandb
    # run instead of creating a duplicate.
    wandb_id = hashlib.md5(cfg.experiment.name.encode()).hexdigest()[:16]
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.experiment.name,
        group=cfg.experiment.group,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline" if cfg.logging.wandb_offline else "online",
        id=wandb_id,
        resume="allow",
    )

    # Callbacks
    log_fn = lambda metrics: wandb.log(metrics)

    from uncertainty_estimation.visualization.covariance import visualize_covariances
    def vis_fn(model, batch):
        figs = visualize_covariances(model, batch, device)
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
        correspondence_mode=cfg.correspondence.mode,
        correspondence_sigma=cfg.correspondence.sigma,
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
