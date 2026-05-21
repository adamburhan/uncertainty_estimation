import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    import random
    import numpy as np
    import torch
    import wandb
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from uncertainty_estimation.training.data.offline_stereo import OfflineStereoDataset
    from uncertainty_estimation.models.error_regressor import ErrorRegressor

    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    train_set = OfflineStereoDataset(cfg.dataset.train_path)
    test_set  = OfflineStereoDataset(cfg.dataset.test_path, stats=train_set.stats)
    print(f"Loaded {len(train_set)} train / {len(test_set)} test samples.")

    train_loader = DataLoader(
        train_set, batch_size=cfg.training.train_batch_size,
        shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.training.eval_batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True,
    )

    modality = cfg.training.modality  # "image" | "depth" | "both"
    backbone = cfg.training.backbone  # "simple" | "resnet"

    model = ErrorRegressor(modality=modality, backbone=backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"modality={modality}  backbone={backbone}  "
          f"params={sum(p.numel() for p in model.parameters())}")

    wandb.init(
        project=cfg.logging.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.logging.wandb_tags) + [f"modality={modality}"],
        mode="offline" if cfg.logging.wandb_offline else "online",
    )

    checkpoint_dir = cfg.logging.checkpoint_dir
    exp_name = cfg.experiment.name

    best_test_loss = float("inf")
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss, n = 0.0, 0
        for img, depth, target in train_loader:
            img = img.to(device)
            depth = depth.to(device)
            target = target.to(device)
            pred = model(img, depth)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * target.size(0)
            n += target.size(0)
        train_loss /= n

        model.eval()
        test_loss, n = 0.0, 0
        with torch.no_grad():
            for img, depth, target in test_loader:
                img = img.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(img, depth)
                test_loss += loss_fn(pred, target).item() * target.size(0)
                n += target.size(0)
        test_loss /= n

        if test_loss < best_test_loss:
            best_test_loss = test_loss        
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "test_loss": test_loss,
        }, os.path.join(checkpoint_dir, f"{exp_name}_best_epoch={epoch}_loss={test_loss:.4f}.pth"))

        print(f"epoch {epoch+1:3d}  train {train_loss:.4f}  test {test_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})

    wandb.finish()


if __name__ == "__main__":
    main()
