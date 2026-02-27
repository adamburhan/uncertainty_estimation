"""3D point cloud visualization with uncertainty ellipsoids using Open3D."""

import numpy as np
import open3d as o3d


def _depth_colormap(z: np.ndarray) -> np.ndarray:
    """Map depth values to a blue→green→yellow colormap. Returns (N, 3) RGB in [0, 1]."""
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    # Simple perceptual ramp: blue → cyan → green → yellow
    r = np.clip(2 * z_norm - 0.5, 0, 1)
    g = np.clip(2 * z_norm, 0, 1) * np.clip(2 - 2 * z_norm, 0, 1) + np.clip(2 * z_norm - 1, 0, 1)
    b = np.clip(1 - 2 * z_norm, 0, 1)
    return np.stack([r, g, b], axis=1)


def _uncertainty_colormap(traces: np.ndarray) -> np.ndarray:
    """Map uncertainty (trace) to white→orange→red. Returns (N, 3) RGB in [0, 1]."""
    t_norm = (traces - traces.min()) / (traces.max() - traces.min() + 1e-8)
    r = np.ones_like(t_norm)
    g = np.clip(1 - t_norm, 0, 1) * 0.6
    b = np.clip(1 - 2 * t_norm, 0, 1) * 0.2
    return np.stack([r, g, b], axis=1)


def _ellipsoid_rings(
    center: np.ndarray,
    cov: np.ndarray,
    scale: float,
    color: np.ndarray,
    n_pts: int = 48,
) -> o3d.geometry.LineSet:
    """Three wireframe ellipse rings (one per principal plane) from a 3x3 covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    radii = np.sqrt(eigenvalues) * scale  # (3,) — semi-axes lengths

    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    all_pts, all_segs = [], []
    offset = 0
    for i in range(3):
        j = (i + 1) % 3
        # Ellipse in the plane spanned by eigenvectors i and j
        pts = (cos_t[:, None] * eigenvectors[:, i] * radii[i] +
               sin_t[:, None] * eigenvectors[:, j] * radii[j] +
               center)
        idx = np.arange(n_pts) + offset
        segs = np.column_stack([idx, np.roll(idx, -1)])
        all_pts.append(pts)
        all_segs.append(segs)
        offset += n_pts

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    ls.lines = o3d.utility.Vector2iVector(np.vstack(all_segs))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(np.vstack(all_segs)), 1))
    )
    return ls


def visualize_point_cloud_with_uncertainty(
    points: np.ndarray,
    covariances: np.ndarray,
    colors: np.ndarray | None = None,
    scale: float = 1.0,
    ellipsoid_resolution: int = 10,
    title: str = "Point Cloud with Uncertainty",
) -> None:
    """Display a 3D point cloud with wireframe uncertainty ellipsoids.

    Points are colored by depth (blue=close, yellow=far).
    Ellipsoid rings are colored by uncertainty magnitude (white=low, red=high).

    Args:
        points: (N, 3) array of 3D points.
        covariances: (N, 3, 3) array of 3x3 covariance matrices per point.
        colors: unused, kept for API compatibility.
        scale: scale factor for ellipsoid size (1.0 = 1-sigma).
        ellipsoid_resolution: unused, kept for API compatibility.
        title: window title.
    """
    geometries = []

    # Point cloud colored by depth
    pt_colors = _depth_colormap(points[:, 2])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pt_colors)
    geometries.append(pcd)

    # Ellipsoid rings colored by uncertainty magnitude
    traces = np.trace(covariances, axis1=1, axis2=2)
    ring_colors = _uncertainty_colormap(traces)

    for i in range(len(points)):
        ls = _ellipsoid_rings(points[i], covariances[i], scale, ring_colors[i])
        geometries.append(ls)

    # Coordinate frame at origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    print(f"  Rendering {len(points)} points with uncertainty ellipsoids")
    print(f"  Point colors: depth (blue=near, yellow=far, z=[{points[:,2].min():.1f}, {points[:,2].max():.1f}] m)")
    print(f"  Ellipsoid colors: uncertainty (white=low, red=high, σ=[{np.sqrt(traces.min()):.2f}, {np.sqrt(traces.max()):.2f}])")
    print("  Controls: left-drag=rotate, right-drag=pan, scroll=zoom, Q=quit")

    o3d.visualization.draw_geometries(geometries, window_name=title)


def visualize_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    title: str = "Point Cloud",
) -> None:
    """Display a 3D point cloud interactively."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([pcd], window_name=title)


def _camera_frustum(
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    img_size: tuple[int, int],
    depth: float = 3.0,
    color: list[float] | None = None,
) -> o3d.geometry.LineSet:
    """Wireframe camera frustum for one frame.

    Convention: R @ X_world + t = X_cam  (same as the pipeline's pose convention).

    Args:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        K: (3, 3) intrinsic matrix.
        img_size: (H, W) image dimensions.
        depth: frustum depth in meters.
        color: RGB color in [0, 1].
    """
    if color is None:
        color = [0.2, 0.9, 0.3]

    h, w = img_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 4 image corners as rays in camera coordinates at the given depth
    corners_cam = np.array([
        [(0 - cx) * depth / fx, (0 - cy) * depth / fy, depth],   # top-left
        [(w - cx) * depth / fx, (0 - cy) * depth / fy, depth],   # top-right
        [(w - cx) * depth / fx, (h - cy) * depth / fy, depth],   # bottom-right
        [(0 - cx) * depth / fx, (h - cy) * depth / fy, depth],   # bottom-left
    ])

    # Transform camera → world: X_world = R^T @ (X_cam - t)
    R_T = R.T
    center_world = R_T @ (np.zeros(3) - t)
    corners_world = (R_T @ (corners_cam - t).T).T  # (4, 3)

    pts = np.vstack([center_world, corners_world])   # 5 pts: 0=apex, 1-4=corners
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],        # apex → corners
              [1, 2], [2, 3], [3, 4], [4, 1]]        # base rectangle

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def visualize_reconstruction(
    points: np.ndarray,
    covariances: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    img_size: tuple[int, int],
    scale: float = 1.0,
    frustum_depth: float = 3.0,
    alongside_fig=None,
    title: str = "Reconstruction",
) -> None:
    """Point cloud with uncertainty ellipsoids and camera frustums in one window.

    Point colors: depth (blue=near, yellow=far).
    Ellipsoid colors: uncertainty magnitude (white=low, red=high).
    Frustums: wireframe pyramids showing each camera's position and FOV.

    The view is set so the scene matches the source image orientation:
    the viewer sits behind the frame-0 camera looking along +Z, with -Y up
    (since KITTI images have Y increasing downward).

    Args:
        points: (N, 3) triangulated 3D points.
        covariances: (N, 3, 3) per-point covariance matrices.
        poses: list of (R, t) pairs — one per frame, same convention as the
               pipeline (R @ X_world + t = X_cam).
        K: (3, 3) intrinsic matrix.
        img_size: (H, W) image dimensions for frustum corner rays.
        scale: scale factor applied to uncertainty ellipsoid semi-axes.
        frustum_depth: depth of the frustum pyramid in meters.
        alongside_fig: if a matplotlib Figure is provided, both windows are shown
                       simultaneously side by side instead of sequentially.
        title: Open3D window title.
    """
    geometries = []

    # Point cloud colored by depth
    pt_colors = _depth_colormap(points[:, 2])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pt_colors)
    geometries.append(pcd)

    # Uncertainty ellipsoids
    traces = np.trace(covariances, axis1=1, axis2=2)
    ring_colors = _uncertainty_colormap(traces)
    for i in range(len(points)):
        geometries.append(_ellipsoid_rings(points[i], covariances[i], scale, ring_colors[i]))

    # Camera frustums — cycle through a small palette so frames are distinguishable
    palette = [
        [0.2, 0.9, 0.3],   # green
        [0.9, 0.8, 0.1],   # yellow
        [0.2, 0.6, 1.0],   # blue
        [1.0, 0.4, 0.2],   # orange
        [0.8, 0.2, 0.9],   # purple
    ]
    for i, (R, t) in enumerate(poses):
        color = palette[i % len(palette)]
        geometries.append(_camera_frustum(R, t, K, img_size, depth=frustum_depth, color=color))

    # Coordinate frame at world origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Camera centers as small spheres for extra clarity
    for i, (R, t) in enumerate(poses):
        center = -R.T @ t
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.translate(center)
        sphere.paint_uniform_color(palette[i % len(palette)])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    print(f"  {len(points)} points  |  {len(poses)} camera frustums")
    print(f"  Point depth range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] m")
    print(f"  Uncertainty σ range: [{np.sqrt(traces.min()):.2f}, {np.sqrt(traces.max()):.2f}]")
    print("  Controls: left-drag=rotate, right-drag=pan, scroll=zoom, Q=quit")

    # View: viewer sits behind the frame-0 camera, looking forward along +Z.
    # front = direction from lookat toward viewer = -Z (viewer is at negative Z).
    # up = -Y because KITTI images have Y increasing downward.
    lookat = np.median(points, axis=0)
    front = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, -1.0, 0.0])
    zoom = 0.4

    if alongside_fig is not None:
        import matplotlib.pyplot as plt

        # Try to nudge the matplotlib window to the left half of the screen
        try:
            mgr = alongside_fig.canvas.manager
            if hasattr(mgr, 'window'):
                w = mgr.window
                if hasattr(w, 'wm_geometry'):       # TkAgg
                    w.wm_geometry("+0+50")
                elif hasattr(w, 'move'):             # Qt backends
                    w.move(0, 50)
        except Exception:
            pass

        plt.show(block=False)
        plt.pause(0.05)

        # Open3D on the right half
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=960, height=600, left=960, top=50)
        for g in geometries:
            vis.add_geometry(g)

        # Set view after first render so the view control takes effect
        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_lookat(lookat)
        ctr.set_zoom(zoom)

        while vis.poll_events():
            vis.update_renderer()
            try:
                plt.pause(0.02)
            except Exception:
                break  # matplotlib window was closed

        vis.destroy_window()
        plt.close(alongside_fig)
    else:
        o3d.visualization.draw_geometries(
            geometries, window_name=title,
            lookat=lookat, front=front, up=up, zoom=zoom,
        )


def visualize_cameras(
    camera_centers: np.ndarray,
    camera_rotations: np.ndarray | None = None,
    points: np.ndarray | None = None,
    title: str = "Cameras",
) -> None:
    """Visualize camera positions (and optionally orientations) with an optional point cloud."""
    geometries = []

    for i, center in enumerate(camera_centers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        if camera_rotations is not None:
            R = camera_rotations[i]
            z_cam = R.T @ np.array([0, 0, 1.0])
            endpoint = center + z_cam * 0.5
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([center, endpoint])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
            geometries.append(line)

    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(_depth_colormap(points[:, 2]))
        geometries.append(pcd)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    o3d.visualization.draw_geometries(geometries, window_name=title)
