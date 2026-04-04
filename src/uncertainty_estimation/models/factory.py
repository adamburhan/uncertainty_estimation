"""Factory for building UnetCovarianceModel from config.

Output filters are tied to parameterization:
  sab         -> lukas (scale) / identity (rotation) / sigmoid(0,1) (anisotropy)
  entries     -> exp (σ_11) / identity (σ_12) / exp (σ_22)
  inv_entries -> exp (H_11) / identity (H_12) / exp (H_22)
"""

import torch

from uncertainty_estimation.models.full_model import UnetCovarianceModel
from uncertainty_estimation.models.output_filter import Filters, OutputFilter, get_filter
from uncertainty_estimation.models.parameterization import CovarianceParameterization, get_parametrization
from uncertainty_estimation.models.unet.unet_model import UNet, UNetM, UNetSmall, UNetXS


_UNET_REGISTRY = {
    "UNetXS":    UNetXS,
    "UNetSmall": UNetSmall,
    "UNetM":     UNetM,
    "UNet":      UNet,
}

_PARAM_REGISTRY = {
    "sab":         CovarianceParameterization.sab,
    "entries":     CovarianceParameterization.entries,
    "inv_entries": CovarianceParameterization.inv_entries,
}


def _make_output_filter(parameterization: str) -> OutputFilter:
    if parameterization == "sab":
        return OutputFilter(
            filter1=get_filter(Filters.lukas),
            filter2=get_filter(Filters.no),
            filter3=get_filter(Filters.sigmoid, min=0.0, max=1.0),
        )
    if parameterization in ("entries", "inv_entries"):
        # Diagonal entries must be positive; off-diagonal is unconstrained.
        return OutputFilter(
            filter1=get_filter(Filters.exp),
            filter2=get_filter(Filters.no),
            filter3=get_filter(Filters.exp),
        )
    raise ValueError(f"Unknown parameterization '{parameterization}'. Available: {list(_PARAM_REGISTRY)}")


def build_model(model_cfg) -> UnetCovarianceModel:
    """Build a UnetCovarianceModel from a model config node.

    Args:
        model_cfg: DictConfig or ModelConfig with fields:
                   architecture, parameterization, isotropic, checkpoint
        device:    torch device to move the model onto.
    """
    arch = model_cfg.architecture
    if arch not in _UNET_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(_UNET_REGISTRY)}")

    param_name = model_cfg.parameterization
    if param_name not in _PARAM_REGISTRY:
        raise ValueError(f"Unknown parameterization '{param_name}'. Available: {list(_PARAM_REGISTRY)}")

    unet = _UNET_REGISTRY[arch](n_channels=1, n_classes=3)
    output_filter = _make_output_filter(param_name)
    parameterization = get_parametrization(_PARAM_REGISTRY[param_name])

    model = UnetCovarianceModel(
        unet=unet,
        output_filter=output_filter,
        parameterization=parameterization,
        model_cfg=type("Cfg", (), {"isotropic_covariances": model_cfg.isotropic})(),
    )

    return model
