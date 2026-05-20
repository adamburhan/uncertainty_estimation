import torch
import torch.nn as nn


class ErrorRegressor(nn.Module):

    def __init__(self, modality: str = "both"):
        super().__init__()
        assert modality in ("image", "depth", "both")
        self.modality = modality

        img_dim = depth_dim = 0
        if modality in ("image", "both"):
            self.img_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            img_dim = 64
        if modality in ("depth", "both"):
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            depth_dim = 32

        self.head = nn.Sequential(
            nn.Linear(img_dim + depth_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, img, depth):
        feats = []
        if self.modality in ("image", "both"):
            feats.append(self.img_encoder(img))
        if self.modality in ("depth", "both"):
            feats.append(self.depth_encoder(depth))
        h = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        return self.head(h).squeeze(-1)
