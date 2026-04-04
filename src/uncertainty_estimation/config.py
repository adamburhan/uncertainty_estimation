"""Unified structured config for stereo covariance training experiments.

This is the single source of truth for all configurable parameters.
The YAML files in configs/ mirror this schema with concrete defaults.
These dataclasses are used for documentation and type-checking; Hydra
loads the YAML directly and returns a DictConfig.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING       # e.g. "001_bearing_gt_no_aug"
    hypothesis: str = ""      # free-text note saved alongside config


@dataclass
class DatasetConfig:
    name: str = "tartanair"                    # "kitti" | "tartanair"
    root: str = MISSING                        # path to dataset root
    # TartanAir: list of environment folders
    environments: List[str] = field(default_factory=lambda: ["ArchVizTinyHouseDay"])
    sequences_train: List[str] = field(default_factory=lambda: ["P000", "P002", "P003", "P004", "P005"])
    sequences_val: List[str] = field(default_factory=lambda: ["P001", "P006"])
    left_images: str = "image_lcam_front"      # subfolder for left images
    right_images: str = "image_rcam_front"     # subfolder for right images
    depth_source: str = "gt"                   # "gt" | "orb_disparity" | "sgbm"
    max_depth: float = 200.0                   # clamp depth to this range (metres)


@dataclass
class AugmentationConfig:
    rotation: bool = False                     # not yet implemented; flag reserved
    rotation_range: List[float] = field(default_factory=lambda: [0.0, 360.0])
    random_crop: bool = True
    crop_size: List[int] = field(default_factory=lambda: [480, 640])
    noise: bool = False
    noise_var: float = 0.0064


@dataclass
class MatchingConfig:
    algorithm: str = "orb"
    max_keypoints: int = 2000
    max_hamming: int = 64
    max_epipolar_error: float = 3.0     # for rectified stereo
    sampson_threshold: float = 4.0      # reserved for non-rectified matching


@dataclass
class LossConfig:
    name: str = "bearing_nll"           # "bearing_nll" | "pixel_nll"


@dataclass
class ModelConfig:
    architecture: str = "UNetXS"        # "UNetXS" | "UNetSmall" | "UNetM" | "UNet"
    parameterization: str = "sab"       # "sab" | "entries" | "inv_entries"
    isotropic: bool = False
    checkpoint: Optional[str] = None    # path to resume from


@dataclass
class TrainingConfig:
    train_batch_size: int = 8
    eval_batch_size: int = 8
    lr: float = 1e-4
    lr_gamma: float = 0.99995
    epochs: int = 50
    grad_clip: float = 1.0
    checkpoint_interval: int = 5
    validation_interval: int = 1
    vis_interval: int = 5
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


@dataclass
class LoggingConfig:
    log_dir: str = "checkpoints"
    wandb_project: str = "stereo_covariance"
    wandb_tags: List[str] = field(default_factory=list)
    wandb_offline: bool = False


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
