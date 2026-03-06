"""SuperPoint + LightGlue detect-and-match tracker.

Detects SuperPoint keypoints independently in each frame, matches consecutive
pairs with LightGlue, then chains pairwise matches into multi-frame tracks.

Requires:
    pip install lightglue   (bundles both SuperPoint and LightGlue)

Recommended device on Apple Silicon: "mps"
"""

import numpy as np

from .tracking import Tracks


class SuperPointLGTracker:
    """SuperPoint + LightGlue tracker.

    Args:
        max_features:  maximum SuperPoint keypoints per frame.
        device:        torch device string — "mps", "cuda", or "cpu".
    """

    def __init__(
        self,
        max_features: int = 512,
        device: str = "mps",
    ):
        self.max_features = max_features
        self.device = device

    def track(self, images: list[np.ndarray]) -> Tracks:
        raise NotImplementedError
