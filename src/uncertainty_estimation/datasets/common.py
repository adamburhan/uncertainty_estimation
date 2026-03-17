from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch


@dataclass
class StereoFrameMetadata:
    """Lightweight metadata for one stereo pair."""
    left_path: Path
    right_path: Path
    K: torch.Tensor             # 3,3 intrinsics
    T_left_right: torch.Tensor  # 4,4 stereo extrinsic
    baseline: float