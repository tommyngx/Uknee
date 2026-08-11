"""HRNet-W32/W48 full-frame heatmap baselines.

The stage layout follows the W32 pose backbone: one 2-branch stage, four
3-branch modules and three 4-branch modules with repeated multi-resolution
fusion. The prediction head stays at 1/4 input resolution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        output_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, output_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + identity)


def _residual_layer(block: type[nn.Module], in_channels: int, channels: int, blocks: int) -> nn.Sequential:
    output_channels = channels * block.expansion
    downsample = None
    if in_channels != output_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, 1, bias=False), nn.BatchNorm2d(output_channels)
        )
    layers = [block(in_channels, channels, downsample=downsample)]
    layers.extend(block(output_channels, channels) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    def __init__(self, channels: tuple[int, ...], blocks_per_branch: int = 4):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList(
            [_residual_layer(BasicBlock, channel, channel, blocks_per_branch) for channel in channels]
        )
        fuse_layers: list[nn.ModuleList] = []
        for output_index, output_channels in enumerate(channels):
            row = nn.ModuleList()
            for input_index, input_channels in enumerate(channels):
                if input_index == output_index:
                    row.append(nn.Identity())
                elif input_index > output_index:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(input_channels, output_channels, 1, bias=False),
                            nn.BatchNorm2d(output_channels),
                        )
                    )
                else:
                    operations: list[nn.Module] = []
                    current = input_channels
                    for step in range(output_index - input_index):
                        final = step == output_index - input_index - 1
                        operations.extend(
                            [
                                nn.Conv2d(current, output_channels if final else current, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(output_channels if final else current),
                            ]
                        )
                        if not final:
                            operations.append(nn.ReLU(inplace=True))
                        current = output_channels if final else current
                    row.append(nn.Sequential(*operations))
            fuse_layers.append(row)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        branches = [branch(value) for branch, value in zip(self.branches, inputs)]
        outputs: list[torch.Tensor] = []
        for output_index, row in enumerate(self.fuse_layers):
            target_size = branches[output_index].shape[-2:]
            fused = None
            for input_index, transform in enumerate(row):
                value = transform(branches[input_index])
                if input_index > output_index:
                    value = F.interpolate(value, target_size, mode="bilinear", align_corners=False)
                fused = value if fused is None else fused + value
            outputs.append(self.relu(fused))
        return outputs


def _transition(previous: tuple[int, ...], current: tuple[int, ...]) -> nn.ModuleList:
    layers = nn.ModuleList()
    for index, channels in enumerate(current):
        if index < len(previous):
            layers.append(
                nn.Identity()
                if previous[index] == channels
                else nn.Sequential(
                    nn.Conv2d(previous[index], channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        else:
            operations: list[nn.Module] = []
            source_channels = previous[-1]
            for step in range(index + 1 - len(previous)):
                output_channels = channels if step == index - len(previous) else source_channels
                operations.extend(
                    [
                        nn.Conv2d(source_channels, output_channels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                source_channels = output_channels
            layers.append(nn.Sequential(*operations))
    return layers


def _apply_transition(layers: nn.ModuleList, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [layer(inputs[index] if index < len(inputs) else inputs[-1]) for index, layer in enumerate(layers)]


class _HRNet(nn.Module):
    """Canonical high-resolution backbone and a 129-channel pose head."""

    def __init__(self, width: int, input_channels: int = 3, num_landmarks: int = 129):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _residual_layer(Bottleneck, 64, 64, 4)
        stage1 = (width, width * 2)
        stage2 = (width, width * 2, width * 4)
        stage3 = (width, width * 2, width * 4, width * 8)
        self.transition1 = _transition((256,), stage1)
        self.stage2 = nn.Sequential(HighResolutionModule(stage1))
        self.transition2 = _transition(stage1, stage2)
        self.stage3 = nn.Sequential(*(HighResolutionModule(stage2) for _ in range(4)))
        self.transition3 = _transition(stage2, stage3)
        self.stage4 = nn.Sequential(*(HighResolutionModule(stage3) for _ in range(3)))
        self.final_layer = nn.Conv2d(stage3[0], num_landmarks, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = [self.layer1(self.stem(image))]
        features = self.stage2(_apply_transition(self.transition1, features))
        features = self.stage3(_apply_transition(self.transition2, features))
        features = self.stage4(_apply_transition(self.transition3, features))
        return self.final_layer(features[0])


class HRNetW32(_HRNet):
    """HRNet-W32 with the standard 32/64/128/256 stage widths."""

    def __init__(self, input_channels: int = 3, num_landmarks: int = 129):
        super().__init__(32, input_channels=input_channels, num_landmarks=num_landmarks)


class HRNetW48(_HRNet):
    """HRNet-W48 with the standard 48/96/192/384 stage widths."""

    def __init__(self, input_channels: int = 3, num_landmarks: int = 129):
        super().__init__(48, input_channels=input_channels, num_landmarks=num_landmarks)


# Compatibility alias for published imports; this is now the full W32 model.
HRNetLandmarkBaseline = HRNetW32

__all__ = ["HRNetW32", "HRNetW48", "HRNetLandmarkBaseline"]
