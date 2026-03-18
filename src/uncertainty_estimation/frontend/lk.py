"""Lucas-Kanade optical flow tracker.

Detects Shi-Tomasi corners in frame 0, then tracks them forward with
cv2.calcOpticalFlowPyrLK. Re-detects in any frame where active tracks
drop below a threshold.

No extra dependencies beyond OpenCV.
"""

import numpy as np
import cv2

from .tracking import Tracks


class LKTracker:
    """Lucas-Kanade optical flow tracker.

    Args:
        max_features:    maximum keypoints to detect per re-detection.
        min_tracks:      re-detect when active tracks fall below this count.
        quality_level:   Shi-Tomasi corner quality (passed to goodFeaturesToTrack).
        min_distance:    minimum pixel distance between detected corners.
    """

    def __init__(
        self,
        max_features: int = 500,
        min_tracks: int = 100,
        quality_level: float = 0.01,
        min_distance: int = 10,
    ):
        self.max_features = max_features
        self.min_tracks = min_tracks
        self.quality_level = quality_level
        self.min_distance = min_distance

    def track(self, images: list[np.ndarray]) -> Tracks:
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        def to_gray(img: np.ndarray) -> np.ndarray:
            if img.ndim == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img

        gray = [to_gray(img) for img in images]

        pts0 = cv2.goodFeaturesToTrack(
            gray[0],
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if pts0 is None or len(pts0) == 0:
            return {}
        
        tracks: Tracks = {}
        for i, pt in enumerate(pts0):
            tracks[i] = {0: pt[0]}

        active_pts = pts0.copy()
        active_ids = list(range(len(pts0)))

        for frame_idx in range(1, len(images)):
            if len(active_pts) == 0:
                break

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame_idx - 1], gray[frame_idx], active_pts, None, **lk_params
            )

            # Also run backwards to check consistency (forward-backward error)
            prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame_idx], gray[frame_idx - 1], next_pts, None, **lk_params
            )
            fb_error = np.linalg.norm(
                active_pts.reshape(-1, 2) - prev_pts_back.reshape(-1, 2), axis=1
            )
            valid = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < 1.0)

            new_active_pts = []
            new_active_ids = []
            for i, (ok, pt, st) in enumerate(zip(valid, next_pts, status)):
                track_id = active_ids[i]
                if ok:  # successfully tracked
                    tracks[track_id][frame_idx] = pt[0]
                    new_active_pts.append(pt)
                    new_active_ids.append(track_id)

            active_pts = np.array(new_active_pts) if new_active_pts else np.empty((0, 1, 2))
            active_ids = new_active_ids

            if len(active_pts) < self.min_tracks:
                pts_new = cv2.goodFeaturesToTrack(
                    gray[frame_idx],
                    maxCorners=self.max_features,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                )
                if pts_new is not None:
                    for pt in pts_new:
                        track_id = max(tracks.keys(), default=-1) + 1
                        tracks[track_id] = {frame_idx: pt[0]}
                        active_pts = np.vstack([active_pts, pt[np.newaxis]])
                        active_ids.append(track_id)
        return {tid: obs for tid, obs in tracks.items() if len(obs) >= 2}
    

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from uncertainty_estimation.data.tartanair import TartanAirSequence

    if len(sys.argv) not in (2, 3):
        print("Usage: uv run python -m uncertainty_estimation.frontend.lk <path/to/P000> [n_frames]")
        sys.exit(1)

    seq_path = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) == 3 else 20

    seq = TartanAirSequence(seq_path)
    n_frames = min(n_frames, len(seq))
    images = [seq[i].image for i in range(n_frames)]
    print(f"Loaded {n_frames} frames from {seq_path}")

    tracker = LKTracker(max_features=300, min_tracks=80)
    tracks = tracker.track(images)
    print(f"Tracked {len(tracks)} features across {n_frames} frames.")

    # Sanity-check the Tracks data structure
    H, W = images[0].shape[:2]
    track_lengths = []
    for tid, obs in tracks.items():
        assert len(obs) >= 2, f"Track {tid} has only {len(obs)} observation(s)"
        for frame_idx, pt in obs.items():
            assert 0 <= frame_idx < n_frames, \
                f"Track {tid}: frame_idx {frame_idx} out of range [0, {n_frames})"
            assert pt.shape == (2,), \
                f"Track {tid} frame {frame_idx}: expected (2,) point, got {pt.shape}"
            assert 0.0 <= pt[0] < W and 0.0 <= pt[1] < H, \
                f"Track {tid} frame {frame_idx}: point {pt} outside image bounds ({W}x{H})"
        track_lengths.append(len(obs))

    lengths = np.array(track_lengths)
    full_tracks = int((lengths == n_frames).sum())
    print(f"[tracks] all {len(tracks)} tracks pass structural checks")
    print(f"[tracks] length  min={lengths.min()}  mean={lengths.mean():.1f}  max={lengths.max()}")
    print(f"[tracks] {full_tracks} tracks survive all {n_frames} frames  "
          f"({100 * full_tracks / len(tracks):.1f}%)")

    # Visualize: last frame with full track trails overlaid 
    # Only show tracks visible in the last frame so trails end where the point currently is
    last_frame_idx = n_frames - 1
    long_tracks = {tid: obs for tid, obs in tracks.items() if last_frame_idx in obs}

    img_vis = images[last_frame_idx].copy()

    # Assign a unique colour per track using a colormap
    cmap = plt.colormaps["hsv"].resampled(len(long_tracks))
    colours_255 = [(int(r * 255), int(g * 255), int(b * 255))
                   for r, g, b, _ in (cmap(i) for i in range(len(long_tracks)))]

    for colour, (tid, obs) in zip(colours_255, long_tracks.items()):
        frame_ids = sorted(obs.keys())
        pts = [obs[f] for f in frame_ids]

        # Draw the trail as connected line segments
        for j in range(len(pts) - 1):
            p1 = (int(pts[j][0]),     int(pts[j][1]))
            p2 = (int(pts[j + 1][0]), int(pts[j + 1][1]))
            cv2.line(img_vis, p1, p2, colour, 1, cv2.LINE_AA)

        # Draw the current (last) position as a filled circle
        end = (int(pts[-1][0]), int(pts[-1][1]))
        cv2.circle(img_vis, end, 3, colour, -1, cv2.LINE_AA)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_vis)
    ax.set_title(
        f"LK tracks — {len(long_tracks)} active at frame {last_frame_idx} "
        f"(trails over {n_frames} frames)"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()
