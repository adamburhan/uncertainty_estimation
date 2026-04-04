"""TartanAir training dataset for stereo covariance prediction.

Expected directory structure:
    <root>/<environment>/Data_easy/<sequence>/
        image_lcam_front/   ← left RGB images  (*.png)
        image_rcam_front/   ← right RGB images (*.png)
        depth_lcam_front/   ← left depth maps  (*.png, RGBA-encoded float32)

Camera parameters (fixed across all TartanAir sequences):
    fx = fy = 320.0,  cx = 320.0,  cy = 320.0  (640×640 images)
    baseline = 0.025 m (left→right, pure horizontal translation)

Batch output (same contract as KITTILiveDataset):
    images:      (2, C, H, W) float32 in [0, 1]  — [left, right]
    K_inv:       (3, 3) inverse intrinsics
    T_lr:        (4, 4) left→right extrinsic
    baseline:    scalar tensor (metres)
    depth_left:  (H, W) float32 depth map in metres (after crop), only when
                 depth_source == "gt"; omitted otherwise.
    depth_right: (H, W) float32 depth map in metres (after crop), also only when gt.
"""

from math import sqrt
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from uncertainty_estimation.training.data.augmentations.random_crop import RandomCropWithIntrinsics


# ── Fixed calibration ────────────────────────────────────────────────────────

_BASELINE = 0.025  # metres

_K = torch.tensor([
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 320.0],
    [  0.0,   0.0,   1.0],
], dtype=torch.float32)

_T_LR = torch.tensor([
    [1.0, 0.0, 0.0, -_BASELINE],
    [0.0, 1.0, 0.0,        0.0],
    [0.0, 0.0, 1.0,        0.0],
    [0.0, 0.0, 0.0,        1.0],
], dtype=torch.float32)

# ── Path resolution ──────────────────────────────────────────────────────────

def _find_seq_dir(root: Path, env: str, seq: str) -> Path:
    """Locate a sequence directory, tolerating TartanAir's doubled extraction layout.

    Canonical layout:   <root>/<env>/Data_easy/<seq>/
    Doubled layout:     <root>/<env>/Data_easy/<env>/Data_easy/<seq>/
                        (produced when unzipping into a pre-existing folder)
    """
    candidates = [
        root / env / "Data_easy" / seq,
        root / env / "Data_easy" / env / "Data_easy" / seq,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find sequence '{seq}' for environment '{env}' under '{root}'. "
        f"Tried:\n" + "\n".join(f"  {c}" for c in candidates)
    )


# ── Depth decoding ───────────────────────────────────────────────────────────

def _read_depth(path: Path) -> np.ndarray:
    """Read a TartanAir depth PNG (RGBA-encoded float32) → (H, W) float32."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Failed to read depth image: {path}")
    return raw.view("<f4").squeeze(-1)  # (H, W) float32


# ── Dataset ──────────────────────────────────────────────────────────────────

class TartanAirLiveDataset(Dataset):
    """PyTorch Dataset for TartanAir stereo covariance training."""

    def __init__(self, dataset_cfg, aug_cfg, split: str) -> None:
        """
        Args:
            dataset_cfg: DatasetConfig node with fields:
                         root, environments, sequences_train, sequences_val,
                         left_images, right_images, depth_source, max_depth
            aug_cfg:     AugmentationConfig node with fields:
                         random_crop, crop_size, noise, noise_var
            split:       "train" or "val"
        """
        self.dataset_cfg = dataset_cfg
        self.aug_cfg = aug_cfg
        self.depth_source = dataset_cfg.depth_source
        self.max_depth = dataset_cfg.max_depth

        sequences = (
            list(dataset_cfg.sequences_train)
            if split == "train"
            else list(dataset_cfg.sequences_val)
        )

        # Build index: list of (left_path, right_path, depth_left_path, depth_right_path)
        self.frames: List[Tuple[Path, Path, Path, Path]] = []
        root = Path(dataset_cfg.root)
        for env in dataset_cfg.environments:
            for seq in sequences:
                seq_dir = _find_seq_dir(root, env, seq)
                left_dir       = seq_dir / dataset_cfg.left_images
                right_dir      = seq_dir / dataset_cfg.right_images
                depth_left_dir = seq_dir / "depth_lcam_front"
                depth_right_dir = seq_dir / "depth_rcam_front"

                left_files  = sorted(p for p in left_dir.glob("*.png")  if not p.name.startswith("._"))
                right_files = sorted(p for p in right_dir.glob("*.png") if not p.name.startswith("._"))

                if self.depth_source == "gt":
                    depth_left_files  = sorted(p for p in depth_left_dir.glob("*.png")  if not p.name.startswith("._"))
                    depth_right_files = sorted(p for p in depth_right_dir.glob("*.png") if not p.name.startswith("._"))
                    if len(depth_left_files) != len(left_files):
                        raise RuntimeError(
                            f"Frame count mismatch in {seq_dir}: "
                            f"{len(left_files)} images vs {len(depth_left_files)} left depth maps"
                        )
                    if len(depth_right_files) != len(left_files):
                        raise RuntimeError(
                            f"Frame count mismatch in {seq_dir}: "
                            f"{len(left_files)} images vs {len(depth_right_files)} right depth maps"
                        )
                else:
                    depth_left_files  = [None] * len(left_files)
                    depth_right_files = [None] * len(left_files)

                if len(left_files) != len(right_files):
                    raise RuntimeError(
                        f"Left/right frame count mismatch in {seq_dir}: "
                        f"{len(left_files)} left vs {len(right_files)} right"
                    )

                for lf, rf, dlf, drf in zip(left_files, right_files, depth_left_files, depth_right_files):
                    self.frames.append((lf, rf, dlf, drf))

        crop_size = tuple(aug_cfg.crop_size)
        self.random_crop = (
            RandomCropWithIntrinsics(size=crop_size) if aug_cfg.random_crop else None
        )
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        left_path, right_path, depth_left_path, depth_right_path = self.frames[idx]

        left  = _load_image(left_path)   # 1 or 3, H, W  float32 [0,1]
        right = _load_image(right_path)

        depth_left = depth_right = None
        if self.depth_source == "gt":
            depth_left  = _read_depth(depth_left_path)   # (H, W) float32, metres
            depth_right = _read_depth(depth_right_path)  # (H, W) float32, metres

        # Gaussian noise (applied before crop so crop doesn't see edge effects)
        if self.aug_cfg.noise:
            noise_std = sqrt(self.aug_cfg.noise_var)
            left  = torch.clamp(left  + noise_std * torch.randn_like(left),  0.0, 1.0)
            right = torch.clamp(right + noise_std * torch.randn_like(right), 0.0, 1.0)

        K = _K.clone()

        if self.random_crop is not None:
            # Get shared crop parameters, then apply to images, K, and depth.
            i, j, h, w = self.random_crop.get_params(left, self.random_crop.size)
            left,  K = self.random_crop.single_forward(left,  K,       i, j, h, w)
            right, _ = self.random_crop.single_forward(right, K.clone(), i, j, h, w)
            if depth_left is not None:
                depth_left  = depth_left[i:i + h, j:j + w]
                depth_right = depth_right[i:i + h, j:j + w]
        else:
            h, w = self.crop_size
            left  = left[...,  :h, :w]
            right = right[..., :h, :w]
            if depth_left is not None:
                depth_left  = depth_left[:h, :w]
                depth_right = depth_right[:h, :w]

        batch = {
            "images":   torch.stack([left, right]),   # 2, C, H, W
            "K_inv":    torch.linalg.inv(K),          # 3, 3
            "T_lr":     _T_LR.clone(),                # 4, 4
            "baseline": torch.tensor(_BASELINE),      # scalar
        }
        if depth_left is not None:
            batch["depth_left"]  = torch.from_numpy(depth_left.copy())   # H, W
            batch["depth_right"] = torch.from_numpy(depth_right.copy())  # H, W

        return batch


# ── Image loading (grayscale) ────────────────────────────────────────────────

def _load_image(path: Path) -> torch.Tensor:
    """Load an image as a (1, H, W) float32 tensor in [0, 1] (grayscale)."""
    from skimage import io
    img = io.imread(str(path.resolve()))
    if img.ndim == 3:
        # RGB → grayscale
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # 3, H, W
        return TF.rgb_to_grayscale(t)  # 1, H, W
    return (torch.from_numpy(img).float() / 255.0).unsqueeze(0)      # 1, H, W
