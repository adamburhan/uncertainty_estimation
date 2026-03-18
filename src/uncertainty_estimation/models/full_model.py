from typing import Tuple
import torch
import torch.nn as nn

from uncertainty_estimation.models.output_filter import OutputFilter
from uncertainty_estimation.models.parameterization import BaseParametrization


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