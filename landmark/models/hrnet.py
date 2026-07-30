from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from .heatmap_baseline import decode_global_heatmaps


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class HRNetLandmarkBaseline(nn.Module):
    """Small two-resolution HRNet-style landmark heatmap baseline."""

    def __init__(
        self,
        input_channels: int = 1,
        num_landmarks: int = 129,
        width: int = 32,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.high_branch = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.to_low = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.low_branch = nn.Sequential(
            ResidualBlock(width * 2), ResidualBlock(width * 2)
        )
        self.low_to_high = nn.Sequential(
            nn.Conv2d(width * 2, width, 1, bias=False),
            nn.BatchNorm2d(width),
        )
        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            ResidualBlock(width),
            nn.Conv2d(width, num_landmarks, 1),
        )

    def forward(self, image, **_):
        high = self.high_branch(self.stem(image))
        low = self.low_branch(self.to_low(high))
        low = F.interpolate(
            self.low_to_high(low), high.shape[-2:], mode="bilinear", align_corners=False
        )
        return decode_global_heatmaps(self.fusion(high + low))
