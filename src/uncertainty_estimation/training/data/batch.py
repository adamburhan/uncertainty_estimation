from typing import TypedDict

import torch


class StereoBatch(TypedDict):
    images:   torch.Tensor  # (2, C, H, W)  — [left, right]
    K:        torch.Tensor  # (3, 3)
    T_lr:     torch.Tensor  # (4, 4)  — left→right extrinsic
    baseline: torch.Tensor  # ()      — baseline in metres
