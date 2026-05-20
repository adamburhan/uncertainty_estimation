import torch.nn as nn


class ErrorRegressor(nn.Module):
    """Small CNN regressing log1p(reproj_error) from a 32x32 patch.

    The same architecture is reused for all 3 modalities; only `in_channels`
    differs (3 for image, 1 for depth, 4 for both).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
