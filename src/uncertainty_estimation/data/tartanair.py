"""TartanAir-v2 dataset loader.

TartanAir is a synthetic dataset with perfect ground truth poses and depth maps,
making it ideal for evaluating uncertainty estimation.

Expected directory structure (one environment/difficulty/sequence):
    ArchVizTinyHouseDay/
        Data_easy/
            P000/
                image_lcam_front/
                    000000_lcam_front.png
                    000001_lcam_front.png
                    ...
                depth_lcam_front/
                    000000_lcam_front_depth.png
                    ...
                pose_lcam_front.txt      ← one pose per line: x y z qx qy qz qw

Camera intrinsics are fixed across all TartanAir sequences:
    fx = fy = 320.0,  cx = 320.0,  cy = 240.0  (640x640 images)

Download: https://theairlab.org/tartanair-dataset/
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# TartanAir has fixed intrinsics for all sequences
TARTANAIR_K = np.array([
    [320.0,   0.0, 320.0],
    [  0.0, 320.0, 320.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


@dataclass
class TartanAirFrame:
    """A single frame from a TartanAir sequence.

    Attributes:
        image:    (H, W, 3) RGB image, uint8.
        depth:    (H, W) depth map in metres, float32. NaN where invalid.
        pose:     (4, 4) camera-to-world transform (SE3), float64.
        frame_id: integer index within the sequence.
    """
    image: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    frame_id: int


class TartanAirSequence:
    """Loader for a single TartanAir trajectory (e.g. P000).

    Usage:
        seq = TartanAirSequence("path/to/abandonedfactory/Easy/P000")
        K = seq.K
        frame = seq[0]
        frame.image    # (640, 640, 3) RGB
        frame.depth    # (640, 640) depth in metres
        frame.pose     # (4, 4) SE3 camera-to-world
        for frame in seq:
            ...
    """

    K: np.ndarray = TARTANAIR_K

    def __init__(self, sequence_path: str | Path):
        self.path = Path(sequence_path)
        if not self.path.exists():
            raise FileNotFoundError(f"TartanAir sequence path not found: {self.path}")
        
        self.left_img_dir = self.path / "image_lcam_front"
        self.left_depth_dir = self.path / "depth_lcam_front"
        self.left_pose_file = self.path / "pose_lcam_front.txt"
        
        if not self.left_img_dir.exists():
            raise FileNotFoundError(f"Left image directory not found: {self.left_img_dir}")
        if not self.left_depth_dir.exists():
            raise FileNotFoundError(f"Left depth directory not found: {self.left_depth_dir}")
        if not self.left_pose_file.exists():
            raise FileNotFoundError(f"Left pose file not found: {self.left_pose_file}")
        
        self._left_image_files = sorted(p for p in self.left_img_dir.glob("*.png") if not p.name.startswith("._"))
        self._left_depth_files = sorted(p for p in self.left_depth_dir.glob("*.png") if not p.name.startswith("._"))
        self._pose_lines = self.left_pose_file.read_text().splitlines()

    def __len__(self) -> int:
        return len(self._left_image_files)

    def __getitem__(self, idx: int) -> TartanAirFrame:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)})")
        
        img_filename = self._left_image_files[idx].name
        depth_filename = self._left_depth_files[idx].name
        left_img = cv2.imread(str(self.left_img_dir / img_filename), cv2.IMREAD_COLOR)
        if left_img is None:
            raise IOError(f"Failed to read left image: {self.left_img_dir / img_filename}")
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

        left_depth = read_decode_depth(self.left_depth_dir / depth_filename)
        if left_depth is None:
            raise IOError(f"Failed to read left depth: {self.left_depth_dir / depth_filename}")
        
        if idx >= len(self._pose_lines):
            raise IndexError(f"Pose index {idx} out of range [0, {len(self._pose_lines)})")
        pose_vec = np.fromstring(self._pose_lines[idx], sep=" ", dtype=np.float64)
        pose_mat = pose_vec_to_matrix(pose_vec)

        return TartanAirFrame(
            image=left_img,
            depth=left_depth,
            pose=pose_mat,
            frame_id=idx,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_poses(self) -> np.ndarray:
        """Return all ground truth poses as (N, 4, 4) SE3 matrices."""
        poses = []
        for line in self._pose_lines:
            pose_vec = np.fromstring(line, sep=" ", dtype=np.float64)
            pose_mat = pose_vec_to_matrix(pose_vec)
            poses.append(pose_mat)
        return np.stack(poses, axis=0)

    def get_window(self, start: int, n_frames: int) -> list[TartanAirFrame]:
        """Return a contiguous window of frames.

        Args:
            start:    index of the first frame.
            n_frames: number of frames to return.

        Returns:
            list of TartanAirFrame, length min(n_frames, len(seq) - start).
        """
        frames = []
        for i in range(start, min(start + n_frames, len(self))):
            frames.append(self[i])
        return frames


def pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """Convert TartanAir pose vector to 4x4 camera-to-world SE3 matrix.

    TartanAir pose format: [x, y, z, qx, qy, qz, qw]

    The quaternion encodes the body/NED orientation relative to the world frame,
    where the body frame follows NED convention:
        x = camera forward,  y = camera right,  z = camera down

    OpenCV's pinhole model expects the camera frame:
        x = right,  y = down,  z = forward (out of lens)

    So the camera-to-world rotation is R_body_to_world @ R_cam_to_ned, where:
        R_cam_to_ned maps OpenCV cam axes → NED body axes.

    Reference: TartanAir data_type.md; tartanair_tools cam2ned utility.

    Args:
        pose_vec: (7,) array [x, y, z, qx, qy, qz, qw].

    Returns:
        (4, 4) SE3 matrix: camera-to-world (OpenCV camera convention).
    """
    if pose_vec.shape != (7,):
        raise ValueError(f"Expected pose_vec shape (7,), got {pose_vec.shape}")

    # Rotation that maps OpenCV camera axes to NED body axes:
    #   cam x (right)   → NED y (right)
    #   cam y (down)    → NED z (down)
    #   cam z (forward) → NED x (forward)
    R_cam_to_ned = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)

    x, y, z, qx, qy, qz, qw = pose_vec
    t = np.array([x, y, z], dtype=np.float64)
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    R_body_to_world = quat_to_rotmat(q)
    R_cam_to_world = R_body_to_world @ R_cam_to_ned

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam_to_world
    T[:3, 3] = t
    return T

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to (3, 3) rotation matrix."""
    return Rotation.from_quat(q).as_matrix()


def read_decode_depth(depthpath):
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print("Usage: uv run python -m uncertainty_estimation.data.tartanair <path/to/P000>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading sequence from: {path}\n")

    seq = TartanAirSequence(path)
    print(f"[1] Sequence length: {len(seq)} frames")
    assert len(seq) > 0, "Sequence is empty"

    frame = seq[0]
    assert frame.image.shape == (640, 640, 3), f"Unexpected image shape: {frame.image.shape}"
    assert frame.image.dtype == np.uint8, f"Unexpected image dtype: {frame.image.dtype}"
    assert frame.depth.shape == (640, 640), f"Unexpected depth shape: {frame.depth.shape}"
    assert frame.depth.dtype == np.float32, f"Unexpected depth dtype: {frame.depth.dtype}"
    assert frame.pose.shape == (4, 4), f"Unexpected pose shape: {frame.pose.shape}"
    print(f"[2] Frame 0 shapes/dtypes OK")

    # Pose is a valid SE3 (R orthogonal, last row [0,0,0,1]) 
    R = frame.pose[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "Rotation is not orthogonal"
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), "Rotation determinant != 1"
    assert np.allclose(frame.pose[3], [0, 0, 0, 1]), "Last row of pose is not [0,0,0,1]"
    print(f"[3] Pose SE3 validity OK  (det={np.linalg.det(R):.6f})")

    # Depth values are finite and in a sensible range
    assert np.all(np.isfinite(frame.depth)), "Depth contains NaN or Inf"
    assert frame.depth.min() > 0, f"Depth has non-positive values: min={frame.depth.min()}"
    # TartanAir sky/background pixels have large depth values (10000m+) — this is expected

    print(f"[4] Depth range OK  (min={frame.depth.min():.2f}m, mean={frame.depth.mean():.2f}m, max={frame.depth.max():.2f}m)")

    # get_poses() is consistent with per-frame poses 
    all_poses = seq.get_poses()
    assert all_poses.shape == (len(seq), 4, 4), f"get_poses() shape: {all_poses.shape}"
    assert np.allclose(all_poses[0], frame.pose, atol=1e-9), "get_poses()[0] != seq[0].pose"
    print(f"[5] get_poses() consistent with __getitem__ OK")

    # get_window returns the right count 
    window = seq.get_window(0, 5)
    assert len(window) == min(5, len(seq)), f"get_window returned {len(window)} frames"
    print(f"[6] get_window(0, 5) returned {len(window)} frames OK")


    print("\nAll checks passed.\n")

    # Visual inspection: image + depth side by side 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(frame.image)
    axes[0].set_title("Frame 0 — RGB (verify colours look natural)")
    axes[0].axis("off")
    axes[1].imshow(frame.depth, cmap="plasma")
    axes[1].set_title("Frame 0 — Depth (verify foreground is closer)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    # Trajectory sanity check: raw file vs stored SE3 matrices 
    # Ground truth: parse translations directly from the pose file
    raw_lines = seq.left_pose_file.read_text().splitlines()
    gt_xyz = np.array([list(map(float, l.split()[:3])) for l in raw_lines if l.strip()])

    # Stored: extract translation column from all_poses
    stored_xyz = all_poses[:, :3, 3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, title in zip(axes, ["X-Z plane (top-down)", "X-Y plane (side)"]):
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

    # XZ
    axes[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], "b-", lw=1.5, label="ground truth (file)")
    axes[0].plot(stored_xyz[:, 0], stored_xyz[:, 2], "r--", lw=1, label="stored SE3")
    axes[0].scatter(gt_xyz[0, 0], gt_xyz[0, 2], c="green", zorder=5, label="start")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Z")

    # XY
    axes[1].plot(gt_xyz[:, 0], gt_xyz[:, 1], "b-", lw=1.5, label="ground truth (file)")
    axes[1].plot(stored_xyz[:, 0], stored_xyz[:, 1], "r--", lw=1, label="stored SE3")
    axes[1].scatter(gt_xyz[0, 0], gt_xyz[0, 1], c="green", zorder=5, label="start")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.suptitle("Trajectory sanity check  (blue=file, red=stored — should overlap perfectly)")
    plt.tight_layout()
    plt.show()
