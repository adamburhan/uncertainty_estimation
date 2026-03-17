from math import cos, sin
from typing import List, Optional, Tuple, Union

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    r"""Return skew of a single/multple vectors
    The skew of a vector is defined by

    :math:`\hat{v} =`
    :math:`\bar{x}_k = \sum_{i=1}^{n_k}x_{ik}`

    Args:
        vector (torch.Tensor): vector or vectors of size ..., 3

    Returns:
        torch.Tensor: skew symmetric matrix/matrices of size ..., 3
    """

    skew_diag = torch.zeros_like(vector[..., 0])
    first_row = torch.stack([skew_diag, -vector[..., 2], vector[..., 1]], dim=-1)
    second_row = torch.stack([vector[..., 2], skew_diag, -vector[..., 0]], dim=-1)
    third_row = torch.stack([-vector[..., 1], vector[..., 0], skew_diag], dim=-1)

    return torch.stack([first_row, second_row, third_row], dim=-2)


def gaussian_kl_divergence(sigma_0: torch.Tensor, sigma_1: torch.Tensor) -> torch.Tensor:
    """Computes the KL-Divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) for to zero mean mulitvariate gaussians
    KL = 1/2 * (log|simga_2| - log|sigma_1| - d + trace(sigma_2^-1 simga_1))
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

    Args:
        sigma_0 (torch.Tensor): (..., N, N) covariance matrices of the first distribution
        sigma_1 (torch.Tensor): (..., N, N)covariance matrices of the second distribution

    Returns:
        float: KL-Divergence between the two gaussians
    """
    assert sigma_0.shape == sigma_1.shape
    assert sigma_0.shape[-1] == sigma_0.shape[-2]

    return 0.5 * (
        torch.log(torch.linalg.det(sigma_1) / torch.linalg.det(sigma_0))
        - sigma_0.shape[-1]
        + torch.einsum("...ij,...jk->...ik", torch.linalg.inv(sigma_1), sigma_0)
        .diagonal(offset=0, dim1=-2, dim2=-1)
        .sum(dim=-1)
    )


def sphere_to_carthesian(vector: torch.Tensor, jacobian: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if vector.shape[-1] != 2:
        raise ValueError(f"Expected the last dimension of vector to be of size 2, got {vector.shape[-1]} instead.")
    carthesian = torch.stack(
        [
            torch.cos(vector[..., 1]) * torch.sin(vector[..., 0]),
            torch.sin(vector[..., 1]) * torch.sin(vector[..., 0]),
            torch.cos(vector[..., 0]),
        ],
        dim=-1,
    )
    if not jacobian:
        return carthesian, None

    jac = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(vector[..., 1]) * torch.cos(vector[..., 0]),
                    torch.sin(vector[..., 1]) * torch.cos(vector[..., 0]),
                    -torch.sin(vector[..., 0]),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    -torch.sin(vector[..., 1]) * torch.sin(vector[..., 0]),
                    torch.cos(vector[..., 1]) * torch.sin(vector[..., 0]),
                    torch.zeros_like(vector[..., 1]),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    return carthesian, jac


def carthesian_to_sphere(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected the last dimension of vector to be of size 3, got {vector.shape[-1]} instead.")
    norm_vector = vector / torch.linalg.norm(vector, dim=-1)[..., None]
    return torch.stack(
        [torch.arccos(norm_vector[..., 2]), torch.arctan2(norm_vector[..., 1], norm_vector[..., 0])], dim=-1
    )


def rotation_matrix_2d(angle: Union[torch.Tensor, float]) -> torch.Tensor:
    if isinstance(angle, float):
        return torch.tensor([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    return torch.tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]])


def make_positive_definite(matrices: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
    """Reconstruct 2x2 covariances, such that they satisfy the criterions to be a covariance matrix (be positive-definite and symmetric).
    The covariances passed to this function should already be symmetric. To fullfill the positive-definite criterion, Sylvester's criterions (https://en.wikipedia.org/wiki/Sylvester%27s_criterion) gives for a covariance matrix of the form:
          a b
    cov = b c,
    that a > 0, c > 0, b^2 < ac.

    Args:
        covariances (torch.Tensor): (..., 2, 2) covariance matrices that need to be corrected.
        epsilon (float): a, c must be at least this big

    Returns:
        torch.Tensor: (..., 2, 2) reconstructed covariance matrices that are positive-definite and symmetric
    """
    pd_matrices = torch.zeros_like(matrices)
    pd_matrices[..., 0, 0] = torch.clip(matrices[..., 0, 0], epsilon, None)
    pd_matrices[..., 1, 1] = torch.clip(matrices[..., 1, 1], epsilon, None)
    sqrt_ac = (
        torch.sqrt(torch.clip(matrices[..., 0, 0], epsilon, None) * torch.clip(matrices[..., 1, 1], epsilon, None))
        - epsilon
    )
    pd_matrices[..., 0, 1] = torch.clip((matrices[..., 0, 1] + matrices[..., 1, 0]) / 2.0, -sqrt_ac, sqrt_ac)
    pd_matrices[..., 1, 0] = torch.clip((matrices[..., 0, 1] + matrices[..., 1, 0]) / 2.0, -sqrt_ac, sqrt_ac)
    return pd_matrices
