from abc import abstractmethod
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from uncertainty_estimation.training.data.batch import StereoBatch
from uncertainty_estimation.training.data.common import StereoFrameMetadata
from uncertainty_estimation.training.data.config import DatasetConfig
from uncertainty_estimation.training.data.augmentations.random_crop import RandomCropWithIntrinsics


class LiveStereoDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, split: str) -> None:
        if split not in dataset_config.directories.sequences:
            raise ValueError(f"Split '{split}' not in available splits: {list(dataset_config.directories.sequences)}")
        self.cfg = dataset_config
        self.split = split
        self.frames: List[StereoFrameMetadata] = self._build_index()
        self.random_crop = RandomCropWithIntrinsics(size=self.cfg.image_augmentations.crop_size) if self.cfg.image_augmentations.random_crop else None

    def _build_index(self) -> List[StereoFrameMetadata]:
        frames = []
        root = Path(self.cfg.directories.dataset)
        for sequence in self.cfg.directories.sequences[self.split]:
            seq_dir = root / sequence
            K, T_lr, baseline, indices = self.load_sequence(seq_dir)
            for idx in indices:
                frames.append(StereoFrameMetadata(
                    left_path=self.get_img_path(sequence, idx, "left"),
                    right_path=self.get_img_path(sequence, idx, "right"),
                    K=K,
                    T_left_right=T_lr,
                    baseline=baseline,
                ))
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> StereoBatch:
        meta = self.frames[idx]

        left  = self._load_image(meta.left_path)
        right = self._load_image(meta.right_path)

        if self.cfg.image_augmentations.noise:
            noise_std = sqrt(self.cfg.image_augmentations.noise_var)
            left  = torch.clamp(left  + noise_std * torch.randn_like(left),  0.0, 1.0)
            right = torch.clamp(right + noise_std * torch.randn_like(right), 0.0, 1.0)

        K = meta.K.clone()
        if self.random_crop is not None:
            [left, right], [K, _] = self.random_crop([left, right], [K, K.clone()])
        else:
            h, w = self.cfg.image_augmentations.crop_size
            left  = left[..., :h, :w]
            right = right[..., :h, :w]

        return {
            "images":   torch.stack([left, right]),
            "K_inv":        torch.linalg.inv(K),
            "T_lr":     meta.T_left_right,
            "baseline": torch.tensor(meta.baseline),
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        def img2tensor(img: np.ndarray, precision: torch.dtype) -> torch.Tensor:
            if img.ndim == 3:
                return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            return (torch.from_numpy(img).float() / 255.0).unsqueeze(0)

        from skimage import io
        img = io.imread(path.resolve())
        return img2tensor(img, torch.float32)

    @abstractmethod
    def get_img_path(self, sequence: str, idx: int, side: str) -> Path:
        pass

    @abstractmethod
    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, float, List[int]]:
        # returns: K (3,3), T_lr (4,4), baseline (float), frame_indices
        pass
