"""Bearing-space projection utilities.

Adapted from dnls_covs and used by both the stereo training losses and
downstream evaluation. The key primitive is ``linear()``, which propagates
pixel-space covariance to unit-bearing-space covariance via the Jacobian of
the pixel-to-unit-sphere map.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """(..., 2) pixel coords -> (..., 3) homogeneous coords [u, v, 1]."""
    return F.pad(points, (0, 1), value=1.0)


def to_3d_cov(cov_2x2: torch.Tensor) -> torch.Tensor:
    """(..., 2, 2) pixel covariance -> (..., 3, 3) zero-padded covariance."""
    return F.pad(cov_2x2, (0, 1, 0, 1))


def linear(
    points: torch.Tensor,
    K_inv: torch.Tensor,
    covariances: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Project homogeneous image points to unit bearings.

    When pixel-space covariances are provided, they are propagated through the
    normalization map to produce rank-2 bearing-space covariance matrices.
    """
    transformed = torch.einsum("...ij,...nj->...ni", K_inv, points)
    norms = torch.norm(transformed, dim=-1)

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
