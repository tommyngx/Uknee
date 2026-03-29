import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch.hub import load_state_dict_from_url

from .RWKV_UNet import RWKV_UNet_encoder_B, VRWKV_SpatialMix, _sanitize_module_finite_
from .ccm.ccm import CCMix
from .module.basic_modules import ConvNormAct, LayerScale2D, get_norm


PRETRAINED_ENCODER_URL = "https://huggingface.co/FengheTan9/U-Stone/resolve/main/net_B.pth"


class StripDirectionalRWKVMix2D(nn.Module):
    def __init__(self, dim, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.row_mix = VRWKV_SpatialMix(dim, channel_gamma=channel_gamma, shift_pixel=shift_pixel)
        self.row_reverse_mix = VRWKV_SpatialMix(dim, channel_gamma=channel_gamma, shift_pixel=shift_pixel)
        self.col_mix = VRWKV_SpatialMix(dim, channel_gamma=channel_gamma, shift_pixel=shift_pixel)
        self.col_reverse_mix = VRWKV_SpatialMix(dim, channel_gamma=channel_gamma, shift_pixel=shift_pixel)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(nn.Conv2d(dim * 4, dim, kernel_size=1, bias=True), nn.Sigmoid())

    def _mix_rows(self, x, mixer, reverse=False):
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> (b h) w c")
        if reverse:
            tokens = torch.flip(tokens, dims=[1])
        mixed = mixer(tokens, patch_resolution=(1, w))
        if reverse:
            mixed = torch.flip(mixed, dims=[1])
        return rearrange(mixed, "(b h) w c -> b c h w", b=b, h=h)

    def _mix_cols(self, x, mixer, reverse=False):
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> (b w) h c")
        if reverse:
            tokens = torch.flip(tokens, dims=[1])
        mixed = mixer(tokens, patch_resolution=(h, 1))
        if reverse:
            mixed = torch.flip(mixed, dims=[1])
        return rearrange(mixed, "(b w) h c -> b c h w", b=b, w=w)

    def forward(self, x):
        direction_outputs = [
            self._mix_rows(x, self.row_mix, reverse=False),
            self._mix_rows(x, self.row_reverse_mix, reverse=True),
            self._mix_cols(x, self.col_mix, reverse=False),
            self._mix_cols(x, self.col_reverse_mix, reverse=True),
        ]
        fused = torch.cat(direction_outputs, dim=1)
        return self.fuse(fused) * self.gate(fused)


class BoundaryAwareRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.refine = ConvNormAct(dim, dim, kernel_size=3, norm_layer="bn_2d", act_layer="silu")

    def forward(self, x):
        feature_mean = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(feature_mean, self.sobel_x, padding=1)
        grad_y = F.conv2d(feature_mean, self.sobel_y, padding=1)
        edge_strength = torch.sqrt(torch.clamp(grad_x.pow(2) + grad_y.pow(2), min=1e-6))
        edge_attention = self.edge_proj(edge_strength)
        refined = x + edge_attention * self.refine(x)
        return torch.nan_to_num(refined, nan=0.0, posinf=1e4, neginf=-1e4)


class HybridRWKVRefineBlock(nn.Module):
    def __init__(self, dim, hidden_ratio=2.0, norm_layer="bn_2d", drop_path=0.0, use_rwkv=True):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.use_rwkv = use_rwkv
        self.norm = get_norm(norm_layer)(dim)
        self.in_proj = ConvNormAct(dim, hidden_dim, kernel_size=1, norm_layer="none", act_layer="none")
        self.local_branch = ConvNormAct(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            groups=hidden_dim,
            norm_layer="bn_2d",
            act_layer="silu",
        )
        if use_rwkv:
            self.global_branch = StripDirectionalRWKVMix2D(hidden_dim)
            self.global_proj = ConvNormAct(hidden_dim, hidden_dim, kernel_size=1, norm_layer="bn_2d", act_layer="none")
            self.gate = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=True), nn.Sigmoid())
        else:
            self.global_branch = None
            self.global_proj = None
            self.gate = None

        self.out_proj = ConvNormAct(hidden_dim, dim, kernel_size=1, norm_layer="bn_2d", act_layer="none")
        self.layer_scale = LayerScale2D(dim, init_values=1e-5)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.in_proj(x)
        local_features = self.local_branch(x)

        if self.use_rwkv:
            global_features = self.global_proj(self.global_branch(x))
            fusion_gate = self.gate(torch.cat([local_features, global_features], dim=1))
            x = local_features + fusion_gate * global_features
        else:
            x = local_features

        x = self.out_proj(x)
        x = self.layer_scale(x)
        x = shortcut + self.drop_path(x)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class GatedSkipFusionBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels, out_channels, use_rwkv=True, drop_path=0.0):
        super().__init__()
        self.dec_proj = ConvNormAct(dec_channels, out_channels, kernel_size=1, norm_layer="bn_2d", act_layer="silu")
        self.skip_proj = ConvNormAct(skip_channels, out_channels, kernel_size=1, norm_layer="bn_2d", act_layer="none")
        self.skip_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.merge = ConvNormAct(out_channels * 2, out_channels, kernel_size=3, norm_layer="bn_2d", act_layer="silu")
        self.refine = HybridRWKVRefineBlock(out_channels, hidden_ratio=2.0, drop_path=drop_path, use_rwkv=use_rwkv)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec_proj(x)
        skip = self.skip_proj(skip)
        gated_skip = skip * self.skip_gate(torch.cat([x, skip], dim=1))
        x = self.merge(torch.cat([x, gated_skip], dim=1))
        return self.refine(x)


class RWKV_UNetV2(nn.Module):
    def __init__(
        self,
        input_channel=1,
        num_classes=1,
        img_size=256,
        pretrained_path=PRETRAINED_ENCODER_URL,
        deep_supervision=False,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
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

        self.ccm = CCMix([self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]], self.embed_dims[0], img_size // 2)
        self.bottleneck = HybridRWKVRefineBlock(self.embed_dims[3], hidden_ratio=2.0, drop_path=0.05, use_rwkv=True)
        self.decoder1 = GatedSkipFusionBlock(self.embed_dims[3], self.embed_dims[2], self.embed_dims[2], use_rwkv=True, drop_path=0.05)
        self.decoder2 = GatedSkipFusionBlock(self.embed_dims[2], self.embed_dims[1], self.embed_dims[1], use_rwkv=True, drop_path=0.05)
        self.decoder3 = GatedSkipFusionBlock(self.embed_dims[1], self.embed_dims[0], self.embed_dims[0], use_rwkv=True, drop_path=0.03)
        self.decoder4 = GatedSkipFusionBlock(self.embed_dims[0], self.stem_dim, 32, use_rwkv=False, drop_path=0.0)
        self.boundary_refine = BoundaryAwareRefinement(32)
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


def rwkv_unetv2(input_channel=3, num_classes=1, img_size=256, **kwargs):
    return RWKV_UNetV2(input_channel=input_channel, num_classes=num_classes, img_size=img_size, **kwargs)
