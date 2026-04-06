"""Training and evaluation logic for stereo covariance prediction.

This module is framework-agnostic: no wandb, no hydra, no config objects.
Logging and visualization are handled via callbacks passed to train_model().
"""

import os
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertainty_estimation.geometry.stereo import depth_from_disparity, reproject, extract_covs

# helpers

def _lookup_depth(depth_map: Tensor, kps: Tensor) -> Tensor:
    """Sample a dense depth map at keypoint locations (nearest-neighbour).

    Args:
        depth_map: (B, H, W) depth in metres.
        kps:       (B, P, 2) keypoints in pixel coords (x=col, y=row).

    Returns:
        (B, P) depth values at each keypoint.
    """
    B, H, W = depth_map.shape
    col = kps[..., 0].round().long().clamp(0, W - 1) # (B, P)
    row = kps[..., 1].round().long().clamp(0, H - 1) # (B, P)
    b_idx = torch.arange(B, device=kps.device).unsqueeze(1).expand_as(col) # (B, P)
    return depth_map[b_idx, row, col] # now we can have depth_map[b_idx[i, j], row[i, j], col[i, j]] for each keypoint


def _get_depth(
    depth_source: str,
    batch: dict,
    left_kps: Tensor,
    right_kps: Tensor,
    focal: Tensor,
    baseline: Tensor,
    device: torch.device,
    max_depth: float,
) -> Tuple[Tensor, Tensor]:
    """Compute per-keypoint depth according to the configured depth source."""
    if depth_source == "gt":
        depth_left = _lookup_depth(batch["depth_left"].to(device), left_kps)
        depth_right = _lookup_depth(batch["depth_right"].to(device), right_kps)
        return depth_left.clamp(0.1, max_depth), depth_right.clamp(0.1, max_depth)
    raise ValueError(f"Unknown depth_source '{depth_source}'. Available: gt, orb_disparity")


def _in_bounds(kps: Tensor, H: int, W: int) -> Tensor:
    """Check which keypoints fall within image bounds. Returns (B, P) bool."""
    return (kps[..., 0] >= 0) & (kps[..., 0] < W) & (kps[..., 1] >= 0) & (kps[..., 1] < H)



# Forward step (shared by train and eval)

def _forward_step(
    model: torch.nn.Module,
    batch: dict,
    loss_fn: Callable,
    matching_fn: Callable,
    device: torch.device,
    depth_source: str,
    max_depth: float,
) -> Dict[str, Tensor]:
    """Shared forward pass: matching -> depth -> covariance -> loss.

    Returns dict with 'loss' (scalar tensor with grad) and 'n_valid_kps' (int).
    """
    images = batch["images"].to(device)       # B, 2, C, H, W
    K_inv = batch["K_inv"].to(device)         # B, 3, 3
    T_lr = batch["T_lr"].to(device)           # B, 4, 4
    T_rl = torch.linalg.inv(T_lr)            # B, 4, 4
    baseline = batch["baseline"].to(device)   # B,

    # correspondences and masks for valid keypoints (after matching and outlier rejection)
    K = torch.linalg.inv(K_inv)
    focal = K[:, 0, 0]

    left_kps, right_kps, masks = matching_fn(images, K)
    left_kps = left_kps.to(device)
    right_kps = right_kps.to(device)
    masks = masks.to(device)


    depth_left, depth_right = _get_depth(
        depth_source, batch, left_kps, right_kps, focal, baseline, device, max_depth
    )

    cov_preds = model(images)  # B*2, H, W, 2, 2
    left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)

    # Reproject both directions
    right_kps_reproj = reproject(left_kps, depth_left, K, T_lr)
    left_kps_reproj = reproject(right_kps, depth_right, K, T_rl)

    # Mask out reprojected points that land outside the image
    H, W = images.shape[-2], images.shape[-1]
    masks = masks * _in_bounds(right_kps_reproj, H, W) * _in_bounds(left_kps_reproj, H, W)

    # Flatten batch+point dims and apply mask
    flat_masks = masks.flatten().bool()

    def _flat_valid(t: Tensor) -> Tensor:
        return t.flatten(0, 1)[flat_masks]

    K_inv_single = K_inv[0]  # assumes uniform intrinsics within a batch

    loss_lr = loss_fn(
        _flat_valid(right_kps),
        _flat_valid(right_kps_reproj),
        _flat_valid(right_covs),
        K_inv_single,
    )
    loss_rl = loss_fn(
        _flat_valid(left_kps),
        _flat_valid(left_kps_reproj),
        _flat_valid(left_covs),
        K_inv_single,
    )

    loss = loss_lr + loss_rl
    n_valid = int(flat_masks.sum().item())

    return {"loss": loss, "n_valid_kps": n_valid}


# Train / eval steps

def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    matching_fn: Callable,
    device: torch.device,
    depth_source: str = "gt",
    max_depth: float = 200.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Single training step: forward + backward + optimizer step."""
    model.train()
    optimizer.zero_grad()

    result = _forward_step(model, batch, loss_fn, matching_fn, device, depth_source, max_depth)

    result["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {"loss": result["loss"].item(), "n_valid_kps": result["n_valid_kps"]}


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    batch: dict,
    loss_fn: Callable,
    matching_fn: Callable,
    device: torch.device,
    depth_source: str = "gt",
    max_depth: float = 200.0,
) -> Dict[str, float]:
    """Single evaluation step (no gradients)."""
    model.eval()
    result = _forward_step(model, batch, loss_fn, matching_fn, device, depth_source, max_depth)
    return {"loss": result["loss"].item(), "n_valid_kps": result["n_valid_kps"]}


# Full-loader evaluation

@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    matching_fn: Callable,
    device: torch.device,
    depth_source: str = "orb_disparity",
    max_depth: float = 200.0,
) -> Dict[str, float]:
    """Evaluate model over an entire dataloader. Returns averaged metrics."""
    model.eval()
    total_loss = 0.0
    total_kps = 0
    n_batches = 0

    for batch in loader:
        result = eval_step(model, batch, loss_fn, matching_fn, device, depth_source, max_depth)
        total_loss += result["loss"]
        total_kps += result["n_valid_kps"]
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "avg_kps": 0.0}

    return {
        "loss": total_loss / n_batches,
        "avg_kps": total_kps / n_batches,
    }


# Full training loop

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    train_loader_for_eval: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: Callable,
    matching_fn: Callable,
    device: torch.device,
    # training hyperparams (primitives only)
    depth_source: str,
    max_depth: float,
    num_epochs: int,
    grad_clip: float = 1.0,
    exp_name: str = "experiment",
    checkpoint_dir: str = "checkpoints",
    # schedule
    eval_period: int = 1,
    checkpoint_period: int = 5,
    vis_period: int = 5,
    # callbacks (framework-agnostic)
    log_fn: Optional[Callable[[Dict], None]] = None,
    vis_fn: Optional[Callable[[torch.nn.Module, dict], Dict]] = None,
    # misc
    verbose: bool = True,
) -> Dict:
    """Full training loop with internal metrics tracking and optional callbacks.

    Args:
        train_loader: Training dataloader (shuffled, smaller batch size).
        train_loader_for_eval: Same training data but unshuffled with larger
            batch size, used for deterministic full-pass train metrics.
        val_loader: Validation dataloader.
        log_fn: Called with a dict of metrics to log (e.g. wandb.log).
        vis_fn: Called with (model, batch) -> dict of visualization artifacts.
                The returned dict is passed to log_fn.

    Returns:
        Dictionary of accumulated metrics: {
            "train": {"loss": [...], "avg_kps": [...]},
            "val":   {"loss": [...], "avg_kps": [...]},
            "epochs": [epoch_numbers where eval happened],
        }
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Internal metrics accumulation
    all_metrics = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "epochs": [],
    }

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Train 
        epoch_loss = 0.0
        epoch_kps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not verbose)
        for batch in pbar:
            result = train_step(
                model, batch, optimizer, loss_fn, matching_fn, device,
                depth_source=depth_source, max_depth=max_depth, grad_clip=grad_clip,
            )
            epoch_loss += result["loss"]
            epoch_kps += result["n_valid_kps"]
            global_step += 1

            if log_fn is not None:
                log_fn({
                    "train/batch_loss": result["loss"],
                    "train/n_valid_kps": result["n_valid_kps"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                })

            pbar.set_postfix(loss=f"{result['loss']:.4f}", kps=result["n_valid_kps"])

        scheduler.step()

        # Full-pass train eval (deterministic, no grad) 
        if epoch % eval_period == 0:
            train_metrics = eval_model(
                model, train_loader_for_eval, loss_fn, matching_fn, device,
                depth_source=depth_source, max_depth=max_depth,
            )
            val_metrics = eval_model(
                model, val_loader, loss_fn, matching_fn, device,
                depth_source=depth_source, max_depth=max_depth,
            )

            all_metrics["train"]["loss"].append(train_metrics["loss"])
            all_metrics["train"]["avg_kps"].append(train_metrics["avg_kps"])
            all_metrics["val"]["loss"].append(val_metrics["loss"])
            all_metrics["val"]["avg_kps"].append(val_metrics["avg_kps"])
            all_metrics["epochs"].append(epoch)

            if verbose:
                print(f"  train_loss={train_metrics['loss']:.4f}  "
                      f"val_loss={val_metrics['loss']:.4f}  "
                      f"val_kps={val_metrics['avg_kps']:.0f}")

            if log_fn is not None:
                log_fn({
                    "train/loss": train_metrics["loss"],
                    "train/avg_kps": train_metrics["avg_kps"],
                    "val/loss": val_metrics["loss"],
                    "val/avg_kps": val_metrics["avg_kps"],
                    "epoch": epoch,
                })

            # Best model checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss": val_metrics["loss"],
                }, os.path.join(checkpoint_dir, f"{exp_name}_best_epoch={epoch}_loss={val_metrics['loss']:.4f}.pth"))

        # Periodic checkpoint 
        if epoch % checkpoint_period == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, f"{exp_name}_epoch_{epoch:04d}.pth"))
            if verbose:
                print(f"  Saved checkpoint at epoch {epoch}")

        # Visualization 
        if vis_fn is not None and epoch % vis_period == 0:
            vis_batch = next(iter(val_loader))
            vis_artifacts = vis_fn(model, vis_batch)
            if log_fn is not None and vis_artifacts:
                log_fn({**vis_artifacts, "epoch": epoch})

    # Convert defaultdicts to regular dicts for serialization
    return {
        "train": dict(all_metrics["train"]),
        "val": dict(all_metrics["val"]),
        "epochs": all_metrics["epochs"],
    }
