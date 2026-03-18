from typing import List, Tuple
from pathlib import Path
import numpy as np

import torch

from uncertainty_estimation.training.data.live_matches import LiveStereoDataset

def load_stereo_calibration_kitti(sequence_dir: Path):
    data = {}
    with open(sequence_dir / "calib.txt") as f:
        for line in f:
            key, val = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])

    P0 = data["P0"].reshape(3, 4)
    P1 = data["P1"].reshape(3, 4)

    K  = torch.from_numpy(P0[:, :3]).float()
    tx = P1[0, 3] / P0[0, 0]          # negative (~-0.537)
    baseline = float(-tx)              # positive

    T_lr = torch.eye(4)
    T_lr[0, 3] = tx                    # negative

    return K, T_lr, baseline


def image_indices(sequence_dir: Path, images_dir: str) -> List[int]:
    indices = []
    for file in sorted(sequence_dir.joinpath(images_dir).iterdir()):
        if not file.is_file():
            continue
        stem = file.stem
        try:
            idx = int(stem)
        except:
            continue
        indices.append(idx)
    return indices

def get_img_path(dataset_directory: Path, sequence: str, image_folder: str, idx: int) -> Path:
    image_name = f"{str(idx).zfill(6)}.png"
    return dataset_directory.joinpath(sequence, image_folder, image_name)

def load_sequence(
    sequence_dir: Path, image_folder: str
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    K, T_lr, baseline = load_stereo_calibration_kitti(sequence_dir)
    indices = image_indices(sequence_dir, image_folder)
    return K, T_lr, baseline, indices

class KITTILiveDataset(LiveStereoDataset):
    def __init__(
            self, 
            dataset_config, 
            split: str
    ) -> None:
        super().__init__(dataset_config, split)

    def get_img_path(self, sequence: str, idx: int, side: str) -> Path:
        return get_img_path(
            Path(self.cfg.directories.dataset),
            sequence,
            "image_0" if side == "left" else "image_1",
            idx
        )
    
    def load_sequence(self, sequence_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, float, List[int]]:
        return load_sequence(sequence_dir, image_folder="image_0")