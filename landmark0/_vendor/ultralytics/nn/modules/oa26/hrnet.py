# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Canonical HRNet-Pose backbone adapter for OA26 pose experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization and optional ReLU activation."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """The two 3x3 residual block used by HRNet multi-resolution stages."""

    def __init__(self, channels: int):
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, 3, 1)
        self.cv2 = ConvBNAct(channels, channels, 3, 1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.cv2(self.cv1(x)))


class Bottleneck(nn.Module):
    """The 1x1-3x3-1x1 residual block used by canonical HRNet stage 1."""

    expansion = 4

    def __init__(self, c1: int, planes: int):
        super().__init__()
        c2 = planes * self.expansion
        self.cv1 = ConvBNAct(c1, planes, 1, 1)
        self.cv2 = ConvBNAct(planes, planes, 3, 1)
        self.cv3 = ConvBNAct(planes, c2, 1, 1, act=False)
        self.shortcut = ConvBNAct(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shortcut(x) + self.cv3(self.cv2(self.cv1(x))))


class HRFusionStage(nn.Module):
    """One HRNet module: branch processing followed by all-resolution fusion."""

    def __init__(self, channels: tuple[int, ...], num_blocks: int = 4):
        super().__init__()
        self.channels = channels
        self.branches = nn.ModuleList(
            nn.Sequential(*(BasicBlock(c) for _ in range(num_blocks))) for c in channels
        )
        self.fuse = nn.ModuleList(
            nn.ModuleList(self._make_fuse_layer(source, target) for source in range(len(channels)))
            for target in range(len(channels))
        )
        self.act = nn.ReLU(inplace=True)

    def _make_fuse_layer(self, source: int, target: int) -> nn.Module:
        """Project a source branch to the target branch's channel count/resolution."""
        if source == target:
            return nn.Identity()
        if source > target:  # lower-resolution source, then bilinear upsampling in forward
            return ConvBNAct(self.channels[source], self.channels[target], 1, 1, act=False)

        layers = []  # higher-resolution source, progressively downsample with 3x3 convolutions
        in_channels = self.channels[source]
        for step in range(target - source):
            out_channels = self.channels[target] if step == target - source - 1 else in_channels
            layers.append(ConvBNAct(in_channels, out_channels, 3, 2, act=step != target - source - 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        fused = []
        for target, target_tensor in enumerate(x):
            y = self.fuse[target][target](target_tensor)
            for source, source_tensor in enumerate(x):
                if source == target:
                    continue
                z = self.fuse[target][source](source_tensor)
                if source > target:
                    z = F.interpolate(z, size=target_tensor.shape[-2:], mode="bilinear", align_corners=False)
                y = y + z
            fused.append(self.act(y))
        return fused


class HRNetLite(nn.Module):
    """Original lightweight four-branch HRNet-style baseline used by OA pose v4."""

    _VARIANTS = {
        "w18": (18, 36, 72, 144),
        "w32": (32, 64, 128, 256),
        "w48": (48, 96, 192, 384),
        "w64": (64, 128, 256, 512),
    }

    def __init__(
        self,
        variant: str = "w32",
        pretrained: bool = False,
        out_channels: tuple[int, int, int, int] | list[int] = (128, 256, 512, 512),
        return_p2: bool = True,
        num_blocks: int = 1,
        num_stages: int = 3,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._VARIANTS:
            raise ValueError(f"Unsupported HRNetLite variant '{variant}'. Supported variants: {self._VARIANTS}.")
        if pretrained:
            raise NotImplementedError("HRNet pretrained weights are not bundled for this clean-room adapter.")

        self.variant = variant
        self.return_p2 = bool(return_p2)
        widths = self._VARIANTS[variant]
        out_channels = tuple(int(c) for c in out_channels)
        if len(out_channels) != 4:
            raise ValueError("out_channels must contain four values for P2, P3, P4 and P5.")

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2),
            ConvBNAct(64, 64, 3, 2),
            BasicBlock(64),
            BasicBlock(64),
        )
        self.transition = nn.ModuleList(
            (
                ConvBNAct(64, widths[0], 3, 1),
                ConvBNAct(64, widths[1], 3, 2),
                nn.Sequential(ConvBNAct(64, widths[1], 3, 2), ConvBNAct(widths[1], widths[2], 3, 2)),
                nn.Sequential(
                    ConvBNAct(64, widths[1], 3, 2),
                    ConvBNAct(widths[1], widths[2], 3, 2),
                    ConvBNAct(widths[2], widths[3], 3, 2),
                ),
            )
        )
        self.stages = nn.ModuleList(HRFusionStage(widths, num_blocks=num_blocks) for _ in range(num_stages))
        self.adapters = nn.ModuleList(ConvBNAct(c1, c2, 1, 1) for c1, c2 in zip(widths, out_channels))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = [transition(x) for transition in self.transition]
        for stage in self.stages:
            features = stage(features)
        p2, p3, p4, p5 = [adapter(feature) for adapter, feature in zip(self.adapters, features)]
        return [p2, p3, p4, p5] if self.return_p2 else [p3, p4, p5]


class HRNet(nn.Module):
    """Canonical HRNet-Pose backbone returning adapted P2/P3/P4/P5 feature maps.

    Uses the standard pose layout: four bottlenecks in stage 1, then 1, 4 and 3
    multi-resolution fusion modules in stages 2, 3 and 4 respectively. No pretrained
    weights are bundled with this clean-room adapter.
    """

    _VARIANTS = {
        "w32": (32, 64, 128, 256),
        "w48": (48, 96, 192, 384),
    }

    def __init__(
        self,
        variant: str = "w32",
        pretrained: bool = False,
        out_channels: tuple[int, int, int, int] | list[int] = (128, 256, 512, 512),
        return_p2: bool = True,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._VARIANTS:
            raise ValueError(f"Unsupported HRNet variant '{variant}'. Supported variants: {self._VARIANTS}.")
        if pretrained:
            raise NotImplementedError("HRNet pretrained weights are not bundled for this clean-room adapter.")

        self.variant = variant
        self.return_p2 = bool(return_p2)
        widths = self._VARIANTS[variant]
        out_channels = tuple(int(c) for c in out_channels)
        if len(out_channels) != 4:
            raise ValueError("out_channels must contain four values for P2, P3, P4 and P5.")

        self.stem = nn.Sequential(ConvBNAct(3, 64, 3, 2), ConvBNAct(64, 64, 3, 2))
        layer1 = [Bottleneck(64, 64)]
        layer1.extend(Bottleneck(256, 64) for _ in range(3))
        self.layer1 = nn.Sequential(*layer1)  # stride 4, 256 channels

        self.transition1 = self._make_transition((256,), widths[:2])
        self.stage2 = self._make_stage(widths[:2], num_modules=1)
        self.transition2 = self._make_transition(widths[:2], widths[:3])
        self.stage3 = self._make_stage(widths[:3], num_modules=4)
        self.transition3 = self._make_transition(widths[:3], widths)
        self.stage4 = self._make_stage(widths, num_modules=3)
        self.adapters = nn.ModuleList(ConvBNAct(c1, c2, 1, 1) for c1, c2 in zip(widths, out_channels))

    @staticmethod
    def _make_stage(channels: tuple[int, ...], num_modules: int) -> nn.Sequential:
        return nn.Sequential(*(HRFusionStage(channels, num_blocks=4) for _ in range(num_modules)))

    @staticmethod
    def _make_transition(previous: tuple[int, ...], current: tuple[int, ...]) -> nn.ModuleList:
        layers = []
        for branch, out_channels in enumerate(current):
            if branch < len(previous):
                layers.append(nn.Identity() if previous[branch] == out_channels else ConvBNAct(previous[branch], out_channels, 3, 1))
                continue
            transition = []
            in_channels = previous[-1]
            for step in range(branch + 1 - len(previous)):
                final_step = step == branch - len(previous)
                next_channels = out_channels if final_step else in_channels
                transition.append(ConvBNAct(in_channels, next_channels, 3, 2))
                in_channels = next_channels
            layers.append(nn.Sequential(*transition))
        return nn.ModuleList(layers)

    @staticmethod
    def _apply_transition(x: list[torch.Tensor], transition: nn.ModuleList) -> list[torch.Tensor]:
        return [layer(x[i] if i < len(x) else x[-1]) for i, layer in enumerate(transition)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.layer1(self.stem(x))
        x = self.stage2(self._apply_transition([x], self.transition1))
        x = self.stage3(self._apply_transition(x, self.transition2))
        x = self.stage4(self._apply_transition(x, self.transition3))
        p2, p3, p4, p5 = [adapter(feature) for adapter, feature in zip(self.adapters, x)]
        return [p2, p3, p4, p5] if self.return_p2 else [p3, p4, p5]
