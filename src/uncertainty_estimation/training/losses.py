"""Covariance prediction loss functions.

Primary training loss
---------------------
bearing_nll()  — Variant (c): bearing-space NLL via tangent-plane projection.
    This is the main loss used during training. It takes raw keypoint pairs
    and predicted pixel-space covariances; geometry (K, T, depth) is handled
    internally. The residual and covariance are projected onto the 2D tangent
    plane at the observed bearing vector, turning the rank-2 3x3 problem into
    a regular full-rank 2x2 NLL with no pseudoinverse needed.

Ablation variants (same network output, different propagation)
--------------------------------------------------------------
pixel_nll()  — Variant (a): NLL directly in pixel space. Ignores bearing geometry.
               Useful as a simplicity baseline.

The registry / get_loss() covers pixel_nll (and future simple losses) only.
Call bearing_nll() directly from the training loop — its signature is richer.
"""

import torch

from uncertainty_estimation.geometry.bearings import linear, to_homogeneous, to_3d_cov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tangent_basis(b: torch.Tensor) -> torch.Tensor:
    """Compute an orthonormal tangent basis for each unit bearing vector.

    Picks a reference vector (whichever standard basis vector is most
    orthogonal to b) to avoid numerical issues near coordinate axes.

    Args:
        b: (..., 3) unit bearing vectors on S^2.

    Returns:
        (..., 3, 2) — columns are two unit tangent vectors perpendicular to b.
    """
    abs_b = b.abs()
    # Use e_x when it is the least-aligned basis vector, otherwise e_y.
    use_ex = (abs_b[..., 0] <= abs_b[..., 1]) & (abs_b[..., 0] <= abs_b[..., 2])
    ref = torch.zeros_like(b)
    ref[..., 0] = use_ex.float()
    ref[..., 1] = (~use_ex).float()

    t1 = ref - (ref * b).sum(-1, keepdim=True) * b
    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = torch.cross(b, t1, dim=-1)  # already unit since b ⊥ t1

    return torch.stack([t1, t2], dim=-1)  # (..., 3, 2)


# ---------------------------------------------------------------------------
# Primary training loss — bearing-space NLL
# ---------------------------------------------------------------------------

def bearing_nll(
    kp_obs: torch.Tensor,
    kp_reproj: torch.Tensor,
    sigma_u: torch.Tensor,
    K_inv: torch.Tensor,
) -> torch.Tensor:
    """Bearing-space NLL with tangent-plane projection (Variant c).

    Pipeline:
        1. Convert pixel coords to homogeneous, pad pixel covs to 3x3.
        2. linear() → unit bearing vectors + 3x3 bearing-space covariance.
        3. Project residual and Σ_b to the 2D tangent plane at b_obs.
           This collapses the rank-2 3x3 problem to a full-rank 2x2 problem
           without any pseudoinverse.
        4. NLL = e_2d^T Σ_2d^{-1} e_2d + log det Σ_2d, averaged over P.

    Depth-independence: the Jacobian J = (||v||^2 I - v v^T) / ||v||^3
    depends only on K and the pixel location, not on scene depth. This is the
    key advantage of bearing space over 3D space (Variant b).

    Args:
        kp_obs:    (P, 2) observed keypoints in the target image.
        kp_reproj: (P, 2) reprojected keypoints into the target image.
        sigma_u:   (P, 2, 2) predicted pixel-space covariances at kp_obs.
        K_inv:     (3, 3) inverse camera intrinsics.

    Returns:
        Scalar mean NLL loss over P keypoints.
        Returns zero (with gradient) when P == 0.
    """
    if kp_obs.shape[0] == 0:
        return sigma_u.sum() * 0.0

    homo_obs    = to_homogeneous(kp_obs)     # (P, 3)
    homo_reproj = to_homogeneous(kp_reproj)  # (P, 3)
    cov_3x3     = to_3d_cov(sigma_u)         # (P, 3, 3)

    b_obs, Sigma_b = linear(homo_obs, K_inv, cov_3x3)  # (P, 3), (P, 3, 3)
    b_reproj       = linear(homo_reproj, K_inv)          # (P, 3)

    # Residual in 3D — approximately perpendicular to b_obs for small angles
    e = b_reproj - b_obs  # (P, 3)

    # Project to 2D tangent plane at b_obs
    T        = tangent_basis(b_obs)                             # (P, 3, 2)
    e_2d     = (T.mT @ e.unsqueeze(-1)).squeeze(-1)             # (P, 2)
    Sigma_2d = T.mT @ Sigma_b @ T                               # (P, 2, 2)

    # Standard 2x2 Gaussian NLL (no pseudoinverse needed)
    sign, log_det = torch.linalg.slogdet(Sigma_2d)              # (P,)
    Sigma_inv     = torch.linalg.inv(Sigma_2d)                  # (P, 2, 2)
    mahal = (e_2d.unsqueeze(-2) @ Sigma_inv @ e_2d.unsqueeze(-1)).squeeze(-1, -2)  # (P,)

    per_point = mahal + log_det

    # Guard against numerical blow-ups during early training
    valid = torch.isfinite(per_point) & (sign > 0)
    if not valid.any():
        return sigma_u.sum() * 0.0

    return per_point[valid].mean()


# ---------------------------------------------------------------------------
# Ablation: pixel-space NLL (Variant a)
# ---------------------------------------------------------------------------

def pixel_nll(pred_cov: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL directly in pixel space (Variant a ablation).

    No bearing propagation — uses the predicted 2x2 covariance as-is against
    the pixel reprojection residual. Simple but conflates angular and depth
    effects.

    Args:
        pred_cov:  (N, 2, 2) predicted pixel-space covariance matrices.
        residuals: (N, 2) pixel reprojection residuals (kp_reproj - kp_obs).

    Returns:
        Scalar mean NLL.
    """
    if pred_cov.shape[0] == 0:
        return pred_cov.sum() * 0.0

    sign, log_det = torch.linalg.slogdet(pred_cov)
    cov_inv       = torch.linalg.inv(pred_cov)
    mahal = (residuals.unsqueeze(-2) @ cov_inv @ residuals.unsqueeze(-1)).squeeze(-1, -2)

    per_point = mahal + log_det
    valid = torch.isfinite(per_point) & (sign > 0)
    if not valid.any():
        return pred_cov.sum() * 0.0

    return per_point[valid].mean()


# ---------------------------------------------------------------------------
# Beta-NLL and energy score (future work)
# ---------------------------------------------------------------------------

def beta_nll_loss(
    _pred_cov: torch.Tensor,
    _residuals: torch.Tensor,
    _beta: float = 0.5,
) -> torch.Tensor:
    """Beta-NLL loss (Seitzer et al., 2022). Not yet implemented."""
    raise NotImplementedError


def energy_score_loss(_pred_cov: torch.Tensor, _target_cov: torch.Tensor) -> torch.Tensor:
    """Energy score between predicted and target covariances. Not yet implemented."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry (pixel-space losses only — bearing_nll is called directly)
# ---------------------------------------------------------------------------

LOSSES = {
    "pixel_nll":    pixel_nll,
    "beta_nll":     beta_nll_loss,
    "energy_score": energy_score_loss,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name]
