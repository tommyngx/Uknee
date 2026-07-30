from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from landmark.utils.checkpoint import load_checkpoint


def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = min(8, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class RWKVUNetBackboneAdapter(nn.Module):
    """Stable feature interface around the repository's RWKV U-Net variants.

    It deliberately executes named encoder/decoder stages instead of forward
    hooks, so export and compilation do not depend on hook side effects.
    """

    def __init__(
        self,
        backbone: nn.Module,
        query_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        required = ("encoder", "decoder1", "decoder2", "decoder3", "decoder4", "final_conv")
        missing = [name for name in required if not hasattr(backbone, name)]
        if missing:
            raise TypeError(f"Unsupported RWKV U-Net; missing modules: {missing}")
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        embed_dims = getattr(backbone, "embed_dims", [48, 72, 144, 240])
        self.high_projection = _projection(embed_dims[1], query_dim)
        self.mid_projection = _projection(embed_dims[2], query_dim)
        self.low_projection = _projection(embed_dims[3], query_dim)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, reference.shape[-2:], mode="bilinear", align_corners=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _backbone_forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image
        for block in self.backbone.encoder.stage0:
            x = block(x)
        for block in self.backbone.encoder.stage1:
            x = block(x)
        enc1 = x
        for block in self.backbone.encoder.stage2:
            x = block(x)
        enc2 = x
        for block in self.backbone.encoder.stage3:
            x = block(x)
        enc3 = x
        for block in self.backbone.encoder.stage4:
            x = block(x)
        feature_low = x

        enc3, enc2, enc1 = self.backbone.ccm([enc3, enc2, enc1])
        dec3 = self._match_size(self.backbone.decoder1(x), enc3)
        dec2 = self._match_size(
            self.backbone.decoder2(torch.cat([dec3, enc3], dim=1)), enc2
        )
        dec1 = self._match_size(
            self.backbone.decoder3(torch.cat([dec2, enc2], dim=1)), enc1
        )
        dec0 = self.backbone.decoder4(torch.cat([dec1, enc1], dim=1))
        logits = torch.nan_to_num(
            self.backbone.final_conv(dec0), nan=0.0, posinf=1e4, neginf=-1e4
        )
        return {
            "segmentation_logits": logits,
            "raw_high": dec2,
            "raw_mid": dec3,
            "raw_low": feature_low,
        }

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_channels = self.backbone.encoder.stage0[0].conv.conv.in_channels
        if image.shape[1] == 1 and expected_channels == 3:
            image = image.repeat(1, 3, 1, 1)
        if image.shape[1] != expected_channels:
            raise ValueError(
                f"Backbone expects {expected_channels} channels, got {image.shape[1]}"
            )
        if self.freeze_backbone:
            with torch.no_grad():
                raw = self._backbone_forward(image)
        else:
            raw = self._backbone_forward(image)
        return {
            "segmentation_logits": raw["segmentation_logits"],
            "feature_high": self.high_projection(raw["raw_high"]),
            "feature_mid": self.mid_projection(raw["raw_mid"]),
            "feature_low": self.low_projection(raw["raw_low"]),
        }


def build_rwkv_v3_backbone(
    input_channels: int,
    num_mask_classes: int,
    image_size: int,
    checkpoint: str | Path = "",
    strict: bool = True,
) -> nn.Module:
    # Lazy import keeps ViTPose/HRNet baselines independent from RWKV dependencies.
    from models.RWKV.RWKV_UNet.RWKV_UNetV3 import RWKV_UNetV3

    backbone = RWKV_UNetV3(
        input_channel=input_channels,
        num_classes=num_mask_classes,
        img_size=image_size,
        pretrained_path="",
    )
    if checkpoint:
        load_checkpoint(backbone, checkpoint, strict=strict)
    return backbone
