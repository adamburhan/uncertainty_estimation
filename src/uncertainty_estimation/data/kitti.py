"""KITTI odometry dataset loader.

Expected directory structure (KITTI odometry format):
    dataset/
        sequences/
            00/
                image_0/    (left grayscale)
                    000000.png
                    000001.png
                    ...
                image_1/    (right grayscale)
                    000000.png
                    ...
                calib.txt
                times.txt
            01/
            ...

Download from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
You need "grayscale" or "color" image sequences + calibration files.
"""

import re
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoFrame:
    """A single stereo frame from the dataset.

    Attributes:
        left: left image as (H, W) grayscale or (H, W, 3) BGR.
        right: right image, same format as left.
        frame_id: integer frame index.
        timestamp: timestamp in seconds (if available).
    """
    left: np.ndarray
    right: np.ndarray
    frame_id: int
    timestamp: float | None = None


@dataclass
class KITTICalibration:
    """Parsed KITTI calibration data.

    Attributes:
        P0: (3, 4) projection matrix for camera 0 (left grayscale).
        P1: (3, 4) projection matrix for camera 1 (right grayscale).
        P2: (3, 4) projection matrix for camera 2 (left color).
        P3: (3, 4) projection matrix for camera 3 (right color).
        K_left: (3, 3) intrinsic matrix extracted from P0.
        K_right: (3, 3) intrinsic matrix extracted from P1.
        baseline: stereo baseline in meters (distance between left and right cameras).
    """
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray

    @property
    def K_left(self) -> np.ndarray:
        return self.P0[:, :3]

    @property
    def K_right(self) -> np.ndarray:
        return self.P1[:, :3]

    @property
    def baseline(self) -> float:
        # P1 = K @ [R | t], for rectified stereo R=I so t_x = P1[0,3] / P1[0,0]
        # baseline = -t_x (positive distance)
        fx = self.P1[0, 0]
        tx = self.P1[0, 3]
        return -tx / fx


class KITTISequence:
    """Loader for a single KITTI odometry sequence.

    Usage:
        seq = KITTISequence("path/to/dataset/sequences/00")
        calib = seq.calibration
        frame = seq[0]  # first stereo pair
        for frame in seq:  # iterate all frames
            ...
    """

    def __init__(self, sequence_path: str | Path):
        self.path = Path(sequence_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence path not found: {self.path}")

        self._calib = None
        self._timestamps = None

        # Detect available image directories
        self.left_dir = self.path / "image_0"
        self.right_dir = self.path / "image_1"

        # Fall back to color if grayscale not available
        if not self.left_dir.exists():
            self.left_dir = self.path / "image_2"
            self.right_dir = self.path / "image_3"

        if not self.left_dir.exists():
            raise FileNotFoundError(
                f"No image directories found in {self.path}. "
                "Expected image_0/image_1 or image_2/image_3."
            )

        # Filter out macOS resource fork files (._*) 
        self._frame_files = sorted(
            f for f in self.left_dir.glob("*.png") if not f.name.startswith("._")
        )
        if not self._frame_files:
            self._frame_files = sorted(
                f for f in self.left_dir.glob("*.jpg") if not f.name.startswith("._")
            )

    def __len__(self) -> int:
        return len(self._frame_files)

    def __getitem__(self, idx: int) -> StereoFrame:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)})")

        filename = self._frame_files[idx].name
        left = cv2.imread(str(self.left_dir / filename), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(self.right_dir / filename), cv2.IMREAD_UNCHANGED)

        if left is None:
            raise IOError(f"Failed to read left image: {self.left_dir / filename}")
        if right is None:
            raise IOError(f"Failed to read right image: {self.right_dir / filename}")

        ts = self.timestamps[idx] if self.timestamps is not None else None
        return StereoFrame(left=left, right=right, frame_id=idx, timestamp=ts)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def calibration(self) -> KITTICalibration:
        if self._calib is None:
            self._calib = _parse_calibration(self.path / "calib.txt")
        return self._calib

    @property
    def timestamps(self) -> np.ndarray | None:
        if self._timestamps is None:
            times_file = self.path / "times.txt"
            if times_file.exists():
                self._timestamps = np.loadtxt(str(times_file))
        return self._timestamps

    def get_stereo_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: return (left_image, right_image) tuple."""
        frame = self[idx]
        return frame.left, frame.right

    def get_consecutive_pair(self, idx: int) -> tuple[StereoFrame, StereoFrame]:
        """Return two consecutive stereo frames for temporal matching."""
        return self[idx], self[idx + 1]


def _parse_calibration(calib_path: Path) -> KITTICalibration:
    """Parse a KITTI calib.txt file.

    Each line has the format: "P0: v1 v2 v3 ... v12"
    where v1..v12 are the 12 entries of the 3x4 projection matrix (row-major).
    """
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\w+):\s+(.*)", line)
            if match:
                key = match.group(1)
                values = np.array([float(x) for x in match.group(2).split()])
                if len(values) == 12:
                    data[key] = values.reshape(3, 4)

    required = ["P0", "P1", "P2", "P3"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing {key} in calibration file: {calib_path}")

    return KITTICalibration(P0=data["P0"], P1=data["P1"], P2=data["P2"], P3=data["P3"])
