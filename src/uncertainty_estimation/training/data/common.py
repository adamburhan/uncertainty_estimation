from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class StereoFrameMetadata:
    """Lightweight metadata for one stereo pair used during training."""

    left_path: Path
    right_path: Path
    K: torch.Tensor
    T_left_right: torch.Tensor
    baseline: float
