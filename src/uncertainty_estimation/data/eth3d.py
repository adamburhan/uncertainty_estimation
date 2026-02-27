"""ETH3D SLAM dataset loader.

Expected directory structure:
    sequence_name/
        calibration.txt     # single line: "fx fy cx cy"
        rgb.txt             # "timestamp  rgb/filename.png" per line  (# = comment)
        rgb/
            *.png           # images named by their timestamp
        groundtruth.txt     # optional: TUM-format poses (not used by the pipeline)

Download from: https://www.eth3d.net/slam_datasets
You need the "Monocular" or "Stereo" image sequences + calibration archives.
"""

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from uncertainty_estimation.data.kitti import StereoFrame


@dataclass
class ETH3DCalibration:
    """Parsed ETH3D calibration.

    Provides the same interface as KITTICalibration so the pipeline
    can work with either dataset without modification.

    Attributes:
        K: (3, 3) intrinsic matrix (fx, fy, cx, cy).
    """
    K: np.ndarray  # (3, 3)

    @property
    def K_left(self) -> np.ndarray:
        return self.K

    @property
    def K_right(self) -> np.ndarray:
        return self.K

    @property
    def baseline(self) -> float:
        # ETH3D SLAM sequences on disk are monocular.
        return 0.0

    @property
    def P0(self) -> np.ndarray:
        """(3, 4) projection matrix for the left (only) camera."""
        return np.hstack([self.K, np.zeros((3, 1))])

    @property
    def P1(self) -> np.ndarray:
        """(3, 4) projection matrix for the right camera (same as P0 — monocular)."""
        return self.P0


class ETH3DSequence:
    """Loader for an ETH3D SLAM sequence.

    Provides the same interface as KITTISequence so the temporal pipeline
    works unchanged:
        seq = ETH3DSequence("path/to/cables_1")
        calib = seq.calibration         # .K_left, .P0, .P1
        frame = seq[0]                  # .left  (grayscale)
        for frame in seq: ...

    Usage:
        uv run python -m uncertainty_estimation.pipeline \\
            --dataset eth3d --sequence path/to/cables_1
    """

    def __init__(self, sequence_path: str | Path):
        self.path = Path(sequence_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.path}")

        self._calib: ETH3DCalibration | None = None
        self._frame_paths: list[Path] = []
        self._timestamps: list[float] = []

        self._load_frame_list()

    def _load_frame_list(self) -> None:
        """Build ordered list of image paths from rgb.txt."""
        rgb_txt = self.path / "rgb.txt"
        if not rgb_txt.exists():
            raise FileNotFoundError(
                f"rgb.txt not found in {self.path}. "
                "Is this an ETH3D SLAM sequence directory?"
            )

        with open(rgb_txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                img_path = self.path / parts[1]
                if img_path.exists():
                    self._timestamps.append(ts)
                    self._frame_paths.append(img_path)

        if not self._frame_paths:
            raise FileNotFoundError(
                f"No valid image paths found in {rgb_txt}. "
                "Check that the rgb/ folder is present and filenames match."
            )

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, idx: int) -> StereoFrame:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)})")

        img = cv2.imread(str(self._frame_paths[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {self._frame_paths[idx]}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Monocular: right is a duplicate (the stereo pipeline is not applicable,
        # but StereoFrame.right must be an ndarray per the dataclass definition).
        return StereoFrame(
            left=img,
            right=img,
            frame_id=idx,
            timestamp=self._timestamps[idx],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def calibration(self) -> ETH3DCalibration:
        if self._calib is None:
            self._calib = _parse_calibration(self.path)
        return self._calib

    def get_stereo_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: return (left_image, right_image) — identical for monocular."""
        frame = self[idx]
        return frame.left, frame.right


def _parse_calibration(sequence_path: Path) -> ETH3DCalibration:
    """Parse calibration.txt: single line 'fx fy cx cy [ignored...]'."""
    calib_file = sequence_path / "calibration.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"calibration.txt not found in {sequence_path}")

    with open(calib_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:
                continue
            if len(vals) >= 4:
                fx, fy, cx, cy = vals[:4]
                K = np.array([[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]])
                return ETH3DCalibration(K=K)

    raise ValueError(f"Could not parse 'fx fy cx cy' from {calib_file}")
