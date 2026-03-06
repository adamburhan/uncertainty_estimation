"""Feature tracking interface and type definitions.

All frontend trackers implement the Tracker protocol:
    tracker = SomeTracker(...)
    tracks = tracker.track(images)

Tracks format:
    {track_id: {frame_idx: np.ndarray([u, v])}}
    Only tracks visible in ≥2 frames are included.

Available implementations:
    lk.py            — Lucas-Kanade optical flow (CPU, no dependencies)
    orb.py           — ORB detect-and-match (CPU, no dependencies)
    superpoint_lg.py — SuperPoint + LightGlue (GPU via MPS/CUDA)
"""

from typing import Protocol
import numpy as np

# {track_id: {frame_idx: (u, v) pixel coordinates}}
Tracks = dict[int, dict[int, np.ndarray]]


class Tracker(Protocol):
    """Protocol for all feature trackers.

    Implementations must accept a sequence of RGB or grayscale images and
    return feature tracks across the sequence.
    """

    def track(self, images: list[np.ndarray]) -> Tracks:
        """Track features across a sequence of frames.

        Args:
            images: list of (H, W, 3) RGB uint8 or (H, W) grayscale uint8 images.

        Returns:
            Tracks: dict mapping track_id -> {frame_idx -> (u, v) pixel coords}.
                    Only tracks visible in ≥2 frames are included.
        """
        ...
