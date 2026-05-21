import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _simple_encoder(in_channels: int, base: int) -> nn.Module:
    """3-layer CNN for 32x32: in -> base -> 2*base -> 2*base, GAP. Output dim = 2*base."""
    return nn.Sequential(
        nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(base, 2 * base, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16x16
        nn.Conv2d(2 * base, 2 * base, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def _resnet_encoder(in_channels: int, pretrained: bool = True) -> nn.Module:
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
    """Late-fusion regressor. `backbone` selects the per-branch encoder."""

    def __init__(self, modality: str = "both", backbone: str = "simple"):
        super().__init__()
        assert modality in ("image", "depth", "both")
        assert backbone in ("simple", "resnet")
        self.modality = modality

        img_dim = depth_dim = 0
        if modality in ("image", "both"):
            if backbone == "resnet":
                self.img_encoder, img_dim = _resnet_encoder(3), 512
            else:
                self.img_encoder, img_dim = _simple_encoder(3, base=32), 64
        if modality in ("depth", "both"):
            if backbone == "resnet":
                self.depth_encoder, depth_dim = _resnet_encoder(1), 512
            else:
                self.depth_encoder, depth_dim = _simple_encoder(1, base=32), 64

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
