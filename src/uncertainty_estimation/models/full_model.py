from typing import Tuple
import torch
import torch.nn as nn

from uncertainty_estimation.models.output_filter import OutputFilter
from uncertainty_estimation.models.parameterization import BaseParametrization

def extract_covs(
    img_covs: torch.Tensor,  # B*2, H, W, 2, 2  (interleaved: left=0, right=1)
    left_kps: torch.Tensor,  # B, P, 2  (pixel coords, xy)
    right_kps: torch.Tensor, # B, P, 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = left_kps.shape[0]
    left_covs = img_covs[0::2]   # B, H, W, 2, 2
    right_covs = img_covs[1::2]  # B, H, W, 2, 2

    H, W = img_covs.shape[1], img_covs.shape[2]

    def sample(covs, kps):
        # grid_sample requires coordinates in [-1, +1], not pixels -> normalize kps accordingly
        # covs: B, H, W, 2, 2  →  grid_sample expects B, C, H, W
        # kps: B, P, 2 (xy pixel coords)
        norm = kps.clone().float()
        norm[..., 0] = (kps[..., 0] * 2.0 / (W - 1)) - 1.0  # x
        norm[..., 1] = (kps[..., 1] * 2.0 / (H - 1)) - 1.0  # y
        flat = covs.flatten(-2, -1).permute(0, 3, 1, 2)  # B, 4, H, W
        out = torch.nn.functional.grid_sample(flat, norm[:, None, :, :], mode="nearest", align_corners=True)
        # out: B, 4, 1, P  →  B, P, 2, 2
        return out[:, :, 0, :].permute(0, 2, 1).reshape(B, -1, 2, 2)

    return sample(left_covs, left_kps), sample(right_covs, right_kps)

class UnetCovarianceModel(nn.Module):
    def __init__(
            self,
            unet: nn.Module,
            output_filter: OutputFilter,
            parameterization: BaseParametrization,
            model_cfg
    ) -> None:
        super().__init__()
        self.unet = unet
        self.output_filter = output_filter
        self.parameterization = parameterization
        self.cfg = model_cfg

    def forward(
        self,
        images: torch.Tensor # B, N, C, H, W
    ) -> torch.Tensor:
        images = images.view(-1, *images.shape[2:])  # (B*N), C, H, W
        unet_output = self.unet(images).permute(0, 2, 3, 1)  # (B*N), H, W, C_out
        return self.parameterization.back_transform()(
            self.output_filter.filter_output(unet_output), False, self.cfg.isotropic_covariances
        ) # (B*N), H, W, 2, 2