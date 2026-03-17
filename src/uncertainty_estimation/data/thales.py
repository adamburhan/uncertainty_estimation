"""Thales monocular dataset loader.

Expected directory structure:
    <sequence>/
        rgb_0/
            000000.png
            000001.png
            ...
        rgb.csv   (columns: ts_rgb_0 (ns), path_rgb_0)

No ground-truth poses or depth — monocular only.
Images are undistorted on load using the known calibration.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Camera intrinsics (pixels) and distortion for Thales camera
THALES_K = np.array([
    [1.39638898e+03, 0.0,            6.15989309e+02],
    [0.0,            1.39654317e+03, 3.94241763e+02],
    [0.0,            0.0,            1.0           ],
], dtype=np.float64)

THALES_DIST = np.array(
    [-4.01668881e-01, 2.48067172e-01, -2.77075958e-03, 9.46080835e-05, -1.59405648e-01],
    dtype=np.float64,
)  # [k1, k2, p1, p2, k3]


@dataclass
class ThalesFrame:
    image: np.ndarray    # (H, W) uint8 grayscale, already undistorted
    timestamp_ns: int
    frame_id: int


class ThalesSequence:
    """Loader for a single Thales monocular sequence.

    Undistorts every frame on load using THALES_K and THALES_DIST.
    The undistorted intrinsic matrix is available as `self.K`.

    Usage:
        seq = ThalesSequence("path/to/sequence")
        frame = seq[0]
        for frame in seq:
            ...
    """

    def __init__(self, sequence_path: str | Path):
        self.path = Path(sequence_path)
        csv_path = self.path / "rgb.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"rgb.csv not found in {self.path}")

        # Parse csv: skip header, columns are (timestamp_ns, relative_path)
        self._entries: list[tuple[int, Path]] = []
        with open(csv_path) as f:
            next(f)  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ts_str, rel_path = line.split(",", 1)
                self._entries.append((int(ts_str), self.path / rel_path.strip()))

        if not self._entries:
            raise ValueError(f"No frames found in {csv_path}")

        # Precompute undistortion maps once
        h, w = cv2.imread(str(self._entries[0][1]), cv2.IMREAD_GRAYSCALE).shape[:2]
        self.K, _ = cv2.getOptimalNewCameraMatrix(THALES_K, THALES_DIST, (w, h), alpha=0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            THALES_K, THALES_DIST, None, self.K, (w, h), cv2.CV_32FC1
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> ThalesFrame:
        ts, img_path = self._entries[idx]
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise IOError(f"Failed to read {img_path}")
        undistorted = cv2.remap(raw, self._map1, self._map2, cv2.INTER_LINEAR)
        return ThalesFrame(image=undistorted, timestamp_ns=ts, frame_id=idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
