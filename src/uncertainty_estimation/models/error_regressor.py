import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _make_resnet_encoder(in_channels: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 patched for 32x32: 3x3 stride-1 stem, no maxpool, no fc.
    With `pretrained=True`, layers 1-4 keep ImageNet weights; the new stem is
    randomly initialized (its kernel/stride/in_channels all differ from default)."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    net.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


class ErrorRegressor(nn.Module):
    """Late-fusion regressor with ResNet-18 backbones per modality."""

    def __init__(self, modality: str = "both"):
        super().__init__()
        assert modality in ("image", "depth", "both")
        self.modality = modality

        img_dim = depth_dim = 0
        if modality in ("image", "both"):
            self.img_encoder = _make_resnet_encoder(in_channels=3)
            img_dim = 512
        if modality in ("depth", "both"):
            self.depth_encoder = _make_resnet_encoder(in_channels=1)
            depth_dim = 512

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
