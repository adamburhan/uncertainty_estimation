"""Depth-variance uncertainty estimator.

For each (track, frame) observation, looks at a patch of the depth map around
the keypoint and uses the local depth variability as a quality signal:

    σ_pix² = σ_base² · (1 + α · (σ_d_local / mean_d_local)²)

Intuition:
    - Smooth surface patch  → low depth variance → near-baseline σ
    - Depth edge / occlusion → high depth variance → inflated σ → down-weighted

Uses MAD (median absolute deviation) rather than raw std so a single occluded
pixel in an otherwise flat patch doesn't dominate. Sky / invalid pixels are
filtered with `max_depth` before computing statistics.

Returns scalar σ² · I — no anisotropy. To upgrade later, replace the scalar
with a 2×2 derived from local depth-gradient direction.
"""

import numpy as np
from uncertainty_estimation.uncertainty.base import UncertaintyEstimator


class DepthVarianceUncertainty(UncertaintyEstimator):
    """Inflate isotropic covariance by local depth variability around the keypoint.

    Args:
        depths:       list of (H, W) depth maps, indexed by frame_idx.
        sigma_base:   baseline pixel std (px). Recovered when depth patch is flat.
        alpha:        gain on the relative-depth-variance term. Larger → harsher
                      down-weighting of depth-edge points.
        patch_radius: half-size of the square patch (patch is (2r+1)×(2r+1)).
        max_depth:    pixels in the patch with depth ≥ this are ignored
                      (filters TartanAir sky at ~10000 m).
        min_valid:    if the patch has fewer valid pixels than this, fall back
                      to baseline σ (no inflation).
    """

    def __init__(
        self,
        depths: list[np.ndarray],
        sigma_base: float = 1.0,
        alpha: float = 50.0,
        patch_radius: int = 4,
        max_depth: float = 100.0,
        min_valid: int = 9,
    ):
        self.depths = depths
        self.sigma_base = sigma_base
        self.alpha = alpha
        self.patch_radius = patch_radius
        self.max_depth = max_depth
        self.min_valid = min_valid

    def _patch_relative_variance(self, depth_map: np.ndarray, u: float, v: float) -> float:
        H, W = depth_map.shape
        r = self.patch_radius
        ui, vi = int(round(u)), int(round(v))
        u0, u1 = max(0, ui - r), min(W, ui + r + 1)
        v0, v1 = max(0, vi - r), min(H, vi + r + 1)
        patch = depth_map[v0:v1, u0:u1].ravel()
        valid = patch[(patch > 0) & (patch < self.max_depth) & np.isfinite(patch)]
        if valid.size < self.min_valid:
            return 0.0
        med = float(np.median(valid))
        if med <= 0:
            return 0.0
        mad = float(np.median(np.abs(valid - med)))
        sigma_d = 1.4826 * mad  # MAD → std under Gaussian assumption
        return (sigma_d / med) ** 2

    def estimate(
        self,
        tracks: dict[int, dict[int, np.ndarray]],
        poses: list[tuple[np.ndarray, np.ndarray]],  # noqa: ARG002
        K: np.ndarray,                                # noqa: ARG002
        images: list[np.ndarray] | None = None,       # noqa: ARG002
    ) -> dict[tuple[int, int], np.ndarray]:
        result: dict[tuple[int, int], np.ndarray] = {}
        s2_base = self.sigma_base ** 2
        for tid, obs in tracks.items():
            for fid, (u, v) in obs.items():
                rel_var = self._patch_relative_variance(self.depths[fid], u, v)
                sigma2 = s2_base * (1.0 + self.alpha * rel_var)
                result[(tid, fid)] = sigma2 * np.eye(2)
        return result


class DepthVarianceInflated(UncertaintyEstimator):
    """Wrap any base estimator and inflate each cov by the depth-variance factor.

        cov_out = cov_base · (1 + α · rel_var_at_obs)

    Tests whether the depth-edge signal is complementary to a richer base
    estimator (e.g. temporal). Preserves the base estimator's anisotropy —
    only scales it.
    """

    def __init__(
        self,
        base: UncertaintyEstimator,
        depths: list[np.ndarray],
        alpha: float = 200.0,
        patch_radius: int = 4,
        max_depth: float = 100.0,
        min_valid: int = 9,
    ):
        self.base = base
        self._depth_var = DepthVarianceUncertainty(
            depths=depths,
            sigma_base=1.0,
            alpha=alpha,
            patch_radius=patch_radius,
            max_depth=max_depth,
            min_valid=min_valid,
        )
        self.alpha = alpha

    def estimate(
        self,
        tracks: dict[int, dict[int, np.ndarray]],
        poses: list[tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        images: list[np.ndarray] | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        base_covs = self.base.estimate(tracks, poses, K, images)
        result: dict[tuple[int, int], np.ndarray] = {}
        for (tid, fid), cov in base_covs.items():
            u, v = tracks[tid][fid]
            rel_var = self._depth_var._patch_relative_variance(
                self._depth_var.depths[fid], u, v
            )
            result[(tid, fid)] = cov * (1.0 + self.alpha * rel_var)
        return result
