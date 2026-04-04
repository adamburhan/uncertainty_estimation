"""SemiStaticSim dataset for stereo covariance prediction.

Expected directory structure:
    <root>/<split>/<scene_id>/<seed>/privileged/run_<run_id>/
        images/
            rgb_first_view/   <- left RGB  (*.jpg)
            depth/            <- left depth (*.png, uint16 millimetres)
            semantics/
        stereo/
            <stereo_config>/          e.g. horizontal_20cm
                images/rgb/           <- right RGB  (*.jpg)
                images/depth/         <- right depth (*.png, uint16 millimetres)
                poses/

Camera parameters (fixed across all SemiStaticSim scenes):
    fx = fy = 300.0,  cx = 300.0,  cy = 300.0  (600x600 images)

Stereo configs (pure translational offset, parallel stereo):
    horizontal_{5,10,20,50,100}cm  -> X-axis offset
    vertical_{5,10,20,50,100}cm    -> Y-axis offset

Batch output (same contract as TartanAirLiveDataset):
    images:      (2, C, H, W) float32 in [0, 1]  — [left, right]
    K_inv:       (3, 3) inverse intrinsics
    T_lr:        (4, 4) left->right extrinsic
    baseline:    scalar tensor (metres)
    depth_left:  (H, W) float32 depth map in metres (after crop)
    depth_right: (H, W) float32 depth map in metres (after crop)
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


# Fixed calibration 

_K = torch.tensor([
    [300.0,   0.0, 300.0],
    [  0.0, 300.0, 300.0],
    [  0.0,   0.0,   1.0],
], dtype=torch.float32)


# Stereo config parsing 

def _parse_stereo_config(config_name: str) -> Tuple[torch.Tensor, float]:
    """Parse a stereo config name into (T_lr, baseline).

    E.g. "horizontal_20cm" -> tx=0.20, ty=0, baseline=0.20
         "vertical_50cm"   -> tx=0, ty=0.50, baseline=0.50

    Returns:
        T_lr:     (4, 4) left->right extrinsic matrix
        baseline: float, in metres
    """
    parts = config_name.split("_")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse stereo config '{config_name}'. "
                         f"Expected format: '{{horizontal,vertical}}_{{N}}cm'")

    direction = parts[0]
    distance_cm = float(parts[1].replace("cm", ""))
    baseline = distance_cm / 100.0

    T_lr = torch.eye(4, dtype=torch.float32)
    if direction == "horizontal":
        T_lr[0, 3] = baseline   # X offset
    elif direction == "vertical":
        T_lr[1, 3] = baseline   # Y offset
    else:
        raise ValueError(f"Unknown direction '{direction}'. Expected 'horizontal' or 'vertical'.")

    return T_lr, baseline


# Depth decoding 

def _read_depth(path: Path) -> np.ndarray:
    """Read a SemiStaticSim depth PNG (uint16, millimetres) -> (H, W) float32 metres."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Failed to read depth image: {path}")
    return raw.astype(np.float32) / 1000.0


# Image loading (grayscale) 

def _load_image(path: Path) -> torch.Tensor:
    """Load an image as a (1, H, W) float32 tensor in [0, 1] (grayscale)."""
    from skimage import io
    img = io.imread(str(path.resolve()))
    if img.ndim == 3:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return TF.rgb_to_grayscale(t)
    return (torch.from_numpy(img).float() / 255.0).unsqueeze(0)


# Dataset

class SemiStaticSimStereoDataset(Dataset):
    """PyTorch Dataset for SemiStaticSim stereo covariance training."""

    def __init__(self, dataset_cfg, aug_cfg, split: str) -> None:
        """
        Args:
            dataset_cfg: DatasetConfig node with fields:
                         root, scenes_train, scenes_val, stereo_config,
                         seed, run_id, depth_source, max_depth
            aug_cfg:     AugmentationConfig node with fields:
                         random_crop, crop_size, noise, noise_var
            split:       "train" or "val"
        """
        self.dataset_cfg = dataset_cfg
        self.aug_cfg = aug_cfg
        self.depth_source = dataset_cfg.depth_source
        self.max_depth = dataset_cfg.max_depth

        # Parse stereo configuration
        self.T_lr, self.baseline = _parse_stereo_config(dataset_cfg.stereo_config)

        scenes = (
            list(dataset_cfg.scenes_train)
            if split == "train"
            else list(dataset_cfg.scenes_val)
        )

        root = Path(dataset_cfg.root)
        split_dir = dataset_cfg.split_dir
        seed = dataset_cfg.seed
        run_id = dataset_cfg.run_id
        stereo_config = dataset_cfg.stereo_config

        # Build index: list of (left_rgb, right_rgb, depth_left, depth_right)
        self.frames: List[Tuple[Path, Path, Path, Path]] = []
        for scene_id in scenes:
            run_dir = root / split_dir / str(scene_id) / str(seed) / "privileged" / f"run_{run_id}"
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")

            left_rgb_dir   = run_dir / "images" / "rgb_first_view"
            left_depth_dir = run_dir / "images" / "depth"
            right_rgb_dir   = run_dir / "stereo" / stereo_config / "images" / "rgb"
            right_depth_dir = run_dir / "stereo" / stereo_config / "images" / "depth"

            left_files = sorted(
                p for p in left_rgb_dir.glob("*.jpg") if not p.name.startswith("._")
            )
            right_files = sorted(
                p for p in right_rgb_dir.glob("*.jpg") if not p.name.startswith("._")
            )

            if len(left_files) != len(right_files):
                raise RuntimeError(
                    f"Left/right frame count mismatch in scene {scene_id}: "
                    f"{len(left_files)} left vs {len(right_files)} right"
                )

            if self.depth_source == "gt":
                depth_left_files = sorted(
                    p for p in left_depth_dir.glob("*.png") if not p.name.startswith("._")
                )
                depth_right_files = sorted(
                    p for p in right_depth_dir.glob("*.png") if not p.name.startswith("._")
                )
                if len(depth_left_files) != len(left_files):
                    raise RuntimeError(
                        f"Frame/depth count mismatch in scene {scene_id}: "
                        f"{len(left_files)} images vs {len(depth_left_files)} left depth maps"
                    )
                if len(depth_right_files) != len(left_files):
                    raise RuntimeError(
                        f"Frame/depth count mismatch in scene {scene_id}: "
                        f"{len(left_files)} images vs {len(depth_right_files)} right depth maps"
                    )
            else:
                depth_left_files = [None] * len(left_files)
                depth_right_files = [None] * len(left_files)

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

        left  = _load_image(left_path)
        right = _load_image(right_path)

        depth_left = depth_right = None
        if self.depth_source == "gt":
            depth_left  = _read_depth(depth_left_path)
            depth_right = _read_depth(depth_right_path)

        if self.aug_cfg.noise:
            noise_std = sqrt(self.aug_cfg.noise_var)
            left  = torch.clamp(left  + noise_std * torch.randn_like(left),  0.0, 1.0)
            right = torch.clamp(right + noise_std * torch.randn_like(right), 0.0, 1.0)

        K = _K.clone()

        if self.random_crop is not None:
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
            "images":   torch.stack([left, right]),
            "K_inv":    torch.linalg.inv(K),
            "T_lr":     self.T_lr.clone(),
            "baseline": torch.tensor(self.baseline),
        }
        if depth_left is not None:
            batch["depth_left"]  = torch.from_numpy(depth_left.copy())
            batch["depth_right"] = torch.from_numpy(depth_right.copy())

        return batch
