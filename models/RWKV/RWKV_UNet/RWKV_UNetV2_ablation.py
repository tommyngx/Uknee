import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .RWKV_UNet import RWKV_UNet_encoder_B, _sanitize_module_finite_
from .RWKV_UNetV2 import (
    BoundaryAwareRefinement,
    GatedSkipFusionBlock,
    HybridRWKVRefineBlock,
    PRETRAINED_ENCODER_URL,
)
from .ccm.ccm import CCMix
from .module.basic_modules import ConvNormAct


class PlainSkipFusionBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels, out_channels, use_rwkv=True, drop_path=0.0):
        super().__init__()
        self.dec_proj = ConvNormAct(dec_channels, out_channels, kernel_size=1, norm_layer="bn_2d", act_layer="silu")
        self.skip_proj = ConvNormAct(skip_channels, out_channels, kernel_size=1, norm_layer="bn_2d", act_layer="none")
        self.merge = ConvNormAct(out_channels * 2, out_channels, kernel_size=3, norm_layer="bn_2d", act_layer="silu")
        self.refine = HybridRWKVRefineBlock(out_channels, hidden_ratio=2.0, drop_path=drop_path, use_rwkv=use_rwkv)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec_proj(x)
        skip = self.skip_proj(skip)
        x = self.merge(torch.cat([x, skip], dim=1))
        return self.refine(x)


class RWKV_UNetV2Ablation(nn.Module):
    def __init__(
        self,
        input_channel=1,
        num_classes=1,
        img_size=256,
        pretrained_path=PRETRAINED_ENCODER_URL,
        deep_supervision=True,
        use_ccm=True,
        use_boundary_refine=True,
        use_gated_skip=True,
        use_bottleneck_rwkv=True,
        decoder_rwkv=(True, True, True, False),
    ):
        super().__init__()
        if len(decoder_rwkv) != 4:
            raise ValueError(f"decoder_rwkv must contain 4 booleans, received: {decoder_rwkv}")

        self.deep_supervision = bool(deep_supervision)
        self.use_ccm = bool(use_ccm)
        self.ablation_config = {
            "deep_supervision": self.deep_supervision,
            "use_ccm": bool(use_ccm),
            "use_boundary_refine": bool(use_boundary_refine),
            "use_gated_skip": bool(use_gated_skip),
            "use_bottleneck_rwkv": bool(use_bottleneck_rwkv),
            "decoder_rwkv": tuple(bool(flag) for flag in decoder_rwkv),
        }

        self.encoder = RWKV_UNet_encoder_B(dim_in=input_channel, img_size=img_size)
        self.embed_dims = [48, 72, 144, 240]
        self.stem_dim = 24

        if pretrained_path:
            if isinstance(pretrained_path, str) and pretrained_path.startswith(("http://", "https://")):
                checkpoint = load_state_dict_from_url(pretrained_path, progress=True, map_location="cpu")
            elif isinstance(pretrained_path, str) and os.path.isfile(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
            else:
                checkpoint = load_state_dict_from_url(PRETRAINED_ENCODER_URL, progress=True, map_location="cpu")
            encoder_state = self.encoder.state_dict()
            compatible_weights = {
                key: value
                for key, value in checkpoint.items()
                if key in encoder_state and value.shape == encoder_state[key].shape
            }
            encoder_state.update(compatible_weights)
            self.encoder.load_state_dict(encoder_state, strict=False)
            _sanitize_module_finite_(self.encoder)

        self.ccm = (
            CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size // 2)
            if self.use_ccm
            else None
        )

        decoder_block = GatedSkipFusionBlock if use_gated_skip else PlainSkipFusionBlock
        self.bottleneck = HybridRWKVRefineBlock(
            self.embed_dims[3],
            hidden_ratio=2.0,
            drop_path=0.05,
            use_rwkv=use_bottleneck_rwkv,
        )
        self.decoder1 = decoder_block(
            self.embed_dims[3],
            self.embed_dims[2],
            self.embed_dims[2],
            use_rwkv=decoder_rwkv[0],
            drop_path=0.05,
        )
        self.decoder2 = decoder_block(
            self.embed_dims[2],
            self.embed_dims[1],
            self.embed_dims[1],
            use_rwkv=decoder_rwkv[1],
            drop_path=0.05,
        )
        self.decoder3 = decoder_block(
            self.embed_dims[1],
            self.embed_dims[0],
            self.embed_dims[0],
            use_rwkv=decoder_rwkv[2],
            drop_path=0.03,
        )
        self.decoder4 = decoder_block(
            self.embed_dims[0],
            self.stem_dim,
            32,
            use_rwkv=decoder_rwkv[3],
            drop_path=0.0,
        )
        self.boundary_refine = BoundaryAwareRefinement(32) if use_boundary_refine else nn.Identity()
        self.head = nn.Sequential(
            ConvNormAct(32, 32, kernel_size=3, norm_layer="bn_2d", act_layer="silu"),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        if self.deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv2d(self.embed_dims[2], num_classes, kernel_size=1),
                    nn.Conv2d(self.embed_dims[1], num_classes, kernel_size=1),
                    nn.Conv2d(self.embed_dims[0], num_classes, kernel_size=1),
                ]
            )

        _sanitize_module_finite_(self)

    def forward(self, x):
        for blk in self.encoder.stage0:
            x = blk(x)
        enc0 = x

        for blk in self.encoder.stage1:
            x = blk(x)
        enc1 = x

        for blk in self.encoder.stage2:
            x = blk(x)
        enc2 = x

        for blk in self.encoder.stage3:
            x = blk(x)
        enc3 = x

        for blk in self.encoder.stage4:
            x = blk(x)

        if self.ccm is not None:
            enc3, enc2, enc1 = self.ccm([enc3, enc2, enc1])

        x = self.bottleneck(x)
        x = self.decoder1(x, enc3)
        aux3 = x
        x = self.decoder2(x, enc2)
        aux2 = x
        x = self.decoder3(x, enc1)
        aux1 = x
        x = self.decoder4(x, enc0)
        x = self.boundary_refine(x)
        x = self.head(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.deep_supervision:
            return [
                self.aux_heads[0](aux3),
                self.aux_heads[1](aux2),
                self.aux_heads[2](aux1),
                x,
            ]
        return x


def rwkv_unetv2_ablation(input_channel=3, num_classes=1, img_size=256, **kwargs):
    return RWKV_UNetV2Ablation(input_channel=input_channel, num_classes=num_classes, img_size=img_size, **kwargs)


def rwkv_unetv2_nods(input_channel=3, num_classes=1, img_size=256, **kwargs):
    kwargs.setdefault("deep_supervision", False)
    return RWKV_UNetV2Ablation(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )


def rwkv_unetv2_noboundary(input_channel=3, num_classes=1, img_size=256, **kwargs):
    kwargs.setdefault("use_boundary_refine", False)
    return RWKV_UNetV2Ablation(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )


def rwkv_unetv2_nogatedskip(input_channel=3, num_classes=1, img_size=256, **kwargs):
    kwargs.setdefault("use_gated_skip", False)
    return RWKV_UNetV2Ablation(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )


def rwkv_unetv2_nodecoderrwkv(input_channel=3, num_classes=1, img_size=256, **kwargs):
    kwargs.setdefault("decoder_rwkv", (False, False, False, False))
    return RWKV_UNetV2Ablation(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )


def rwkv_unetv2_noccm(input_channel=3, num_classes=1, img_size=256, **kwargs):
    kwargs.setdefault("use_ccm", False)
    return RWKV_UNetV2Ablation(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )
