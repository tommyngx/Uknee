# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""ConvNeXtV2-Tiny backbone adapter for OA26 pose experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtV2T(nn.Module):
    """ConvNeXtV2-Tiny feature extractor that returns P2/P3/P4/P5 maps."""

    def __init__(
        self,
        pretrained: bool = False,
        out_channels: tuple[int, int, int, int] | list[int] = (128, 256, 512, 512),
    ):
        """Initialize a ConvNeXtV2-Tiny backbone with 1x1 adapters."""
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("ConvNeXtV2T requires timm. Please install it with pip install timm.") from e

        out_channels = tuple(int(c) for c in out_channels)
        if len(out_channels) != 4:
            raise ValueError("out_channels must contain four values for P2, P3, P4 and P5.")

        self.backbone = timm.create_model(
            "convnextv2_tiny.fcmae",
            pretrained=bool(pretrained),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        in_channels = tuple(int(c) for c in self.backbone.feature_info.channels())
        if len(in_channels) != 4:
            raise RuntimeError(f"Expected four ConvNeXtV2 feature maps, got channels={in_channels}.")

        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c1, c2, 1, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
            for c1, c2 in zip(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return `[P2, P3, P4, P5]` feature maps for the YOLO neck."""
        features = self.backbone(x)
        p2, p3, p4, p5 = [adapter(feature) for adapter, feature in zip(self.adapters, features)]
        # For 896x896 input: P2=224x224, P3=112x112, P4=56x56, P5=28x28.
        return [p2, p3, p4, p5]
