"""Training and evaluation logic for stereo covariance prediction.

This module is framework-agnostic: no wandb, no hydra, no config objects.
Logging and visualization are handled via callbacks passed to train_model().
"""

import os
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute per-keypoint depth and a per-keypoint depth-validity mask.

    The validity mask flags keypoints whose *raw* depth (before clamping) was
    inside (0.1, max_depth) in BOTH views. Without this, missing/out-of-range
    GT depths get silently clamped to 0.1 or max_depth, producing garbage
    reprojections that the matcher mask doesn't catch.
    """
    if depth_source == "gt":
        raw_left = _lookup_depth(batch["depth_left"].to(device), left_kps)
        raw_right = _lookup_depth(batch["depth_right"].to(device), right_kps)
        return raw_left, raw_right
    raise ValueError(f"Unknown depth_source '{depth_source}'. Available: gt, orb_disparity")


def _in_bounds(kps: Tensor, H: int, W: int) -> Tensor:
    """Check which keypoints fall within image bounds. Returns (B, P) bool."""
    return (kps[..., 0] >= 0) & (kps[..., 0] < W) & (kps[..., 1] >= 0) & (kps[..., 1] < H)


def _project_perturbed_3d(
    kp_src: Tensor,
    depth_src: Tensor,
    K: Tensor,
    K_inv: Tensor,
    T_src_dst: Tensor,
    sigma_3d: float,
    min_z_dst: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """Unproject src kps to 3D, add isotropic Gaussian noise in 3D, project to dst camera.

    Used by the synthetic_3d correspondence mode: the 3D-isotropic noise gets
    pushed through the projection Jacobian, producing pixel-space residuals whose
    anisotropy is determined by the depth + camera geometry.

    Returns a per-keypoint validity mask flagging points whose perturbed 3D
    location landed in front of the destination camera (z_dst > min_z_dst).
    Without this guard, large sigmas push points behind the camera, the
    projection divides by ~0, and pixel coords blow up to ±inf / NaN — feeding
    NaN into the loss.

    Args:
        kp_src:    (B, P, 2) source-image keypoints (pixels, x=col, y=row)
        depth_src: (B, P) depth at those keypoints (metres)
        K:         (B, 3, 3) camera intrinsics
        K_inv:     (B, 3, 3) inverse intrinsics
        T_src_dst: (B, 4, 4) source-to-destination camera transform
        sigma_3d:  std of isotropic 3D Gaussian noise (metres)
        min_z_dst: minimum allowed depth in dst frame; points below are masked.

    Returns:
        kp_dst: (B, P, 2) noisy projected keypoints in the destination image.
        valid:  (B, P) bool — True iff perturbed point is in front of dst camera.
    """
    homo = F.pad(kp_src, (0, 1), value=1.0)                       # (B, P, 3)
    rays = torch.einsum("bij,bpj->bpi", K_inv, homo)              # (B, P, 3)
    pts_src = depth_src.unsqueeze(-1) * rays                      # (B, P, 3) in src frame
    pts_src = pts_src + sigma_3d * torch.randn_like(pts_src)      # perturb in 3D
    pts_src_h = F.pad(pts_src, (0, 1), value=1.0)                 # (B, P, 4)
    pts_dst = torch.einsum("bij,bpj->bpi", T_src_dst, pts_src_h)  # (B, P, 4)
    z_dst = pts_dst[..., 2]                                       # (B, P)
    valid = z_dst > min_z_dst
    # Clamp z for the division so the kp_dst tensor itself stays finite even
    # for invalid points (which will be masked out downstream). This avoids
    # NaN/inf propagating into autograd graphs from masked entries.
    z_safe = z_dst.clamp(min=min_z_dst).unsqueeze(-1)
    px_xy = torch.einsum("bij,bpj->bpi", K, pts_dst[..., :3])[..., :2]
    kp_dst = px_xy / z_safe
    return kp_dst, valid



# Forward step (shared by train and eval)

def _forward_step(
    model: torch.nn.Module,
    batch: dict,
    loss_fn: Callable,
    matching_fn: Callable,
    correspondence_mode: str,
    correspondence_sigma: Optional[float],
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

    if "left_kps" in batch:
        # Matches were precomputed in the dataloader workers (CPU-parallel ORB).
        left_kps = batch["left_kps"].to(device)
        right_kps = batch["right_kps"].to(device)
        masks = batch["match_mask"].to(device)
    else:
        left_kps, right_kps, masks = matching_fn(images, K)
        left_kps = left_kps.to(device)
        right_kps = right_kps.to(device)
        masks = masks.to(device)


    depth_left, depth_right = _get_depth(
        depth_source, batch, left_kps, right_kps, focal, baseline, device, max_depth
    )
  
    cov_preds = model(images)  # B*2, H, W, 2, 2

    # Reproject both directions (clean GT-based projection when depth_source="gt")
    right_kps_reproj = reproject(left_kps, depth_left, K, T_lr)
    left_kps_reproj = reproject(right_kps, depth_right, K, T_rl)

    if correspondence_mode == "synthetic":
        # 2D-isotropic pixel noise: residual seen by the loss = injected noise.
        # Matcher contribution is removed — falsification test for the matcher hypothesis.
        # Drop the matcher's confidence mask (no longer meaningful) but KEEP
        # depth_valid since reproject still uses GT depth.
        if correspondence_sigma is None:
            raise ValueError("correspondence_sigma must be set when correspondence_mode='synthetic'")
        right_kps = right_kps_reproj + torch.randn_like(right_kps_reproj) * correspondence_sigma
        left_kps  = left_kps_reproj  + torch.randn_like(left_kps_reproj)  * correspondence_sigma

    elif correspondence_mode == "synthetic_3d":
        # 3D-isotropic noise pushed through the projection Jacobian: residual is
        # geometry-shaped pixel noise. Positive control — should produce
        # baseline/depth-dependent ellipses, demonstrating model capacity.
        if correspondence_sigma is None:
            raise ValueError("correspondence_sigma must be set when correspondence_mode='synthetic_3d'")
        right_kps_orig = right_kps  # save before overwriting
        right_kps, valid_lr = _project_perturbed_3d(
            left_kps,       depth_left,  K, K_inv, T_lr, correspondence_sigma
        )
        left_kps,  valid_rl = _project_perturbed_3d(
            right_kps_orig, depth_right, K, K_inv, T_rl, correspondence_sigma
        )
        # Mask out points where the perturbation pushed the 3D point behind
        # the dst camera (either direction) AND require valid GT depth.
        masks = valid_lr & valid_rl 
    elif correspondence_mode != "real":
        raise ValueError(
            f"Unknown correspondence_mode '{correspondence_mode}'. "
            f"Available: real, synthetic, synthetic_3d"
        )

    # Sample covariance at the observation locations (matcher in real mode,
    # synthetic-noised projection in synthetic mode). Must come AFTER the
    # synthetic override so cov-sample location matches the loss residual.
    left_covs, right_covs = extract_covs(cov_preds, left_kps, right_kps)

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
    correspondence_mode: str,
    correspondence_sigma: Optional[float],
    device: torch.device,
    depth_source: str = "gt",
    max_depth: float = 200.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Single training step: forward + backward + optimizer step."""
    model.train()
    optimizer.zero_grad()

    result = _forward_step(model, batch, loss_fn, matching_fn, correspondence_mode, correspondence_sigma, device, depth_source, max_depth)

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
    correspondence_mode: str,
    correspondence_sigma: Optional[float],
    device: torch.device,
    depth_source: str = "gt",
    max_depth: float = 200.0,
) -> Dict[str, float]:
    """Single evaluation step (no gradients)."""
    model.eval()
    result = _forward_step(model, batch, loss_fn, matching_fn, correspondence_mode, correspondence_sigma, device, depth_source, max_depth)
    return {"loss": result["loss"].item(), "n_valid_kps": result["n_valid_kps"]}


# Full-loader evaluation

@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    matching_fn: Callable,
    correspondence_mode: str,
    correspondence_sigma: Optional[float],
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
        result = eval_step(model, batch, loss_fn, matching_fn, correspondence_mode, correspondence_sigma, device, depth_source, max_depth)
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
    correspondence_mode: str,
    correspondence_sigma: Optional[float],
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
    start_epoch = 1

    # Resume from latest.pt if it exists (preemption recovery).
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        prev_metrics = ckpt.get("all_metrics")
        if prev_metrics is not None:
            for k in ("train", "val"):
                for metric, values in prev_metrics[k].items():
                    all_metrics[k][metric] = list(values)
            all_metrics["epochs"] = list(prev_metrics["epochs"])
        if verbose:
            print(f"  Resumed from {latest_path} at epoch {start_epoch} (global_step={global_step})")

    for epoch in range(start_epoch, num_epochs + 1):
        # Train 
        epoch_loss = 0.0
        epoch_kps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not verbose)
        for batch in pbar:
            result = train_step(
                model, batch, optimizer, loss_fn, matching_fn,
                correspondence_mode, correspondence_sigma, device,
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
                model, train_loader_for_eval, loss_fn, matching_fn,
                correspondence_mode, correspondence_sigma, device,
                depth_source=depth_source, max_depth=max_depth,
            )
            val_metrics = eval_model(
                model, val_loader, loss_fn, matching_fn,
                correspondence_mode, correspondence_sigma, device,
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

        # Resume checkpoint (overwritten every epoch). Atomic write so a
        # preemption mid-save can't corrupt the file the next attempt depends on.
        latest_tmp = os.path.join(checkpoint_dir, "latest.pt.tmp")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "all_metrics": {
                "train": dict(all_metrics["train"]),
                "val": dict(all_metrics["val"]),
                "epochs": all_metrics["epochs"],
            },
        }, latest_tmp)
        os.replace(latest_tmp, os.path.join(checkpoint_dir, "latest.pt"))

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
