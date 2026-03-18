from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import MISSING


@dataclass
class DatasetDirectoriesConfig:
    dataset: str = MISSING
    sequences: Dict[str, List[str]] = MISSING
    left_images: str = MISSING
    right_images: str = MISSING


@dataclass
class ImageAugmentationConfig:
    crop_size: List[int] = MISSING
    random_crop: bool = False
    noise: bool = False
    noise_var: float = 0.01


@dataclass
class MatchingConfig:
    algorithm: str = "orb"
    max_points: int = 500


@dataclass
class DatasetConfig:
    name: str = MISSING
    directories: DatasetDirectoriesConfig = field(default_factory=DatasetDirectoriesConfig)
    image_augmentations: ImageAugmentationConfig = field(default_factory=ImageAugmentationConfig)
