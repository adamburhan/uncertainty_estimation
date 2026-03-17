"""Camera projection utilities.

Adapted from dnls_covs/scripts/covpred/math/projections.py (Muhle et al.).

Core function: linear() propagates a 2x2 pixel-space covariance to a 3x3
bearing-space covariance via the Jacobian of the pixel-to-unit-sphere map.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """(..., 2) pixel coords → (..., 3) homogeneous coords [u, v, 1]."""
    return F.pad(points, (0, 1), value=1.0)


def to_3d_cov(cov_2x2: torch.Tensor) -> torch.Tensor:
    """(..., 2, 2) pixel covariance → (..., 3, 3) zero-padded.

    The third row/col is zero, encoding that there is no uncertainty in the
    homogeneous coordinate. Required input format for linear().
    """
    return F.pad(cov_2x2, (0, 1, 0, 1))


def linear(
    points: torch.Tensor,
    K_inv: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Linear projection from image plane to unit-sphere bearing vectors.

    Computes:
        v     = K_inv @ point             (unnormalised 3D direction)
        bv    = v / ||v||                 (unit bearing vector)
        J     = (||v||^2 I - v v^T) / ||v||^3    (normalisation Jacobian)
        Σ_bv  = J K_inv Σ K_inv^T J^T    (propagated covariance)

    Args:
        points:      (..., N, 3) homogeneous pixel coordinates [u, v, 1].
        K_inv:       (..., 3, 3) inverse camera matrix.
        covariances: (..., N, 3, 3) zero-padded pixel covariances (see
                     to_3d_cov), or None to skip propagation.

    Returns:
        bearing vectors (..., N, 3), and optionally bearing covariances
        (..., N, 3, 3) when covariances is not None.
    """
    transformed = torch.einsum("...ij,...nj->...ni", K_inv, points)
    norms = torch.norm(transformed, dim=-1)  # (..., N)

    if covariances is not None:
        jacobians = (1.0 / torch.pow(norms, 3))[..., None, None] * (
            torch.pow(norms, 2)[..., None, None]
            * torch.eye(3, device=transformed.device, dtype=points.dtype)
            - torch.einsum("...i,...j->...ij", transformed, transformed)
        )
        covariances = torch.einsum(
            "...nij,...jk,...nkl,...ml,...npm->...nip",
            jacobians,
            K_inv,
            covariances,
            K_inv,
            jacobians,
        )
        bearing = transformed / norms[..., None]
        return bearing, covariances

    return transformed / norms[..., None]
