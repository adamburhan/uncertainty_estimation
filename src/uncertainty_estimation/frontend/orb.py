"""ORB detect-and-match tracker.

Detects ORB keypoints independently in each frame, matches consecutive pairs
with a brute-force Hamming matcher + ratio test, then chains pairwise matches
into multi-frame tracks.

No extra dependencies beyond OpenCV.
"""

import numpy as np
import cv2

from .tracking import Tracks


class ORBTracker:
    """ORB detect-and-match tracker.

    Args:
        max_features:  maximum ORB keypoints per frame.
        ratio_thresh:  Lowe's ratio test threshold for match filtering.
    """

    def __init__(
        self,
        max_features: int = 500,
        ratio_thresh: float = 0.75,
    ):
        self.max_features = max_features
        self.ratio_thresh = ratio_thresh

    def track(self, images: list[np.ndarray]) -> Tracks:
        raise NotImplementedError
