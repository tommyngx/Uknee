import math

import torch
import torch.nn as nn
from timm.layers import DropPath, create_act_layer, trunc_normal_

from .module.basic_modules import ConvNormAct, get_act, get_norm


inplace = True


def _sanitize_module_finite_(module):
    for name, parameter in module.named_parameters():
        if not parameter.is_floating_point():
            continue
        parameter.data = torch.nan_to_num(parameter.data, nan=0.0, posinf=1.0, neginf=-1.0)
        if "spatial_decay" in name or "spatial_first" in name:
            parameter.data.clamp_(-10.0, 10.0)

    for _, buffer in module.named_buffers():
        if buffer.is_floating_point():
            buffer.data = torch.nan_to_num(buffer.data, nan=0.0, posinf=1.0, neginf=-1.0)


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    assert gamma <= 1 / 4
    b, _, c = input.shape
    h, w = patch_resolution
    input = input.transpose(1, 2).reshape(b, c, h, w)
    output = torch.zeros_like(input)
    c1 = int(c * gamma)
    c2 = int(c * gamma * 2)
    c3 = int(c * gamma * 3)
    c4 = int(c * gamma * 4)
    output[:, 0:c1, :, shift_pixel:w] = input[:, 0:c1, :, 0 : w - shift_pixel]
    output[:, c1:c2, :, 0 : w - shift_pixel] = input[:, c1:c2, :, shift_pixel:w]
    output[:, c2:c3, shift_pixel:h, :] = input[:, c2:c3, 0 : h - shift_pixel, :]
    output[:, c3:c4, 0 : h - shift_pixel, :] = input[:, c3:c4, shift_pixel:h, :]
    output[:, c4:, ...] = input[:, c4:, ...]
    return output.flatten(2).transpose(1, 2)


class SE(nn.Module):
    def __init__(
        self,
        in_chs,
        rd_ratio=0.25,
        rd_channels=None,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        force_act_layer=None,
        rd_round_fn=None,
    ):
        super().__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class SafeVRWKVSpatialMix(nn.Module):
    """RWKV-style spatial mix without the CUDA WKV extension or token limit."""

    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma
        self._init_weights()

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        decay_speed = torch.linspace(-5.0, 3.0, steps=self.n_embd)
        zigzag = torch.tensor(
            [(i + 1) % 3 - 1 for i in range(self.n_embd)], dtype=torch.float32
        ) * 0.5
        self.spatial_decay = nn.Parameter(decay_speed)
        self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        k = torch.clamp(torch.nan_to_num(self.key(xk), nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
        v = torch.clamp(torch.nan_to_num(self.value(xv), nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
        r = torch.clamp(torch.nan_to_num(self.receptance(xr), nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)

        x = self.key_norm(k + v)
        x = torch.sigmoid(r) * x
        x = self.output(x)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class UpBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=False,
        has_skip=False,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        act_layer="relu",
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        drop_path=0.0,
        drop=0.0,
    ):
        super().__init__()
        self.has_skip = has_skip
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(
            dim_mid, dim_out, kernel_size=1, norm_layer="bn_2d", act_layer="relu", inplace=inplace
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.conv_local = ConvNormAct(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            dilation=dilation,
            groups=dim_mid,
            norm_layer="bn_2d",
            act_layer="silu",
            inplace=inplace,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj(x)
        x = self.proj_drop(x)
        return self.upsample(x)


class SafeIRRWKV(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=True,
        has_skip=True,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        act_layer="relu",
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        attn_s=True,
        drop_path=0.0,
        drop=0.0,
        channel_gamma=1 / 4,
        shift_pixel=1,
    ):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if attn_s:
            self.att = SafeVRWKVSpatialMix(dim_mid, channel_gamma, shift_pixel)
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="none", act_layer="none", inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.conv_local = ConvNormAct(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            dilation=dilation,
            groups=dim_mid,
            norm_layer="bn_2d",
            act_layer="silu",
            inplace=inplace,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.conv(x)
        if self.attn_s:
            b, hidden, h, w = x.size()
            tokens = x.view(b, hidden, -1).permute(0, 2, 1)
            tokens = tokens + self.drop_path(self.ln1(self.att(tokens, (h, w))))
            x = tokens.permute(0, 2, 1).contiguous().view(b, hidden, h, w)
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        x = self.proj_drop(x)
        x = self.proj(x)
        return shortcut + self.drop_path(x) if self.has_skip else x


class RWKVUNetEncoder(nn.Module):
    def __init__(
        self,
        dim_in=3,
        img_size=224,
        depths=None,
        stem_dim=24,
        embed_dims=None,
        exp_ratios=None,
        norm_layers=None,
        act_layers=None,
        dw_kss=None,
        se_ratios=None,
        attn_ss=None,
        drop=0.0,
        drop_path=0.05,
        channel_gamma=1 / 4,
        shift_pixel=1,
    ):
        super().__init__()
        depths = depths or [3, 3, 6, 3]
        embed_dims = embed_dims or [48, 72, 144, 240]
        exp_ratios = exp_ratios or [2.0, 2.5, 4.0, 4.0]
        norm_layers = norm_layers or ["bn_2d", "bn_2d", "ln_2d", "ln_2d"]
        act_layers = act_layers or ["silu", "silu", "gelu", "gelu"]
        dw_kss = dw_kss or [5, 5, 5, 5]
        se_ratios = se_ratios or [0.0, 0.0, 0.0, 0.0]
        attn_ss = attn_ss or [False, False, True, True]

        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList(
            [
                SafeIRRWKV(
                    dim_in,
                    stem_dim,
                    norm_in=False,
                    has_skip=False,
                    exp_ratio=1,
                    norm_layer=norm_layers[0],
                    act_layer=act_layers[0],
                    dw_ks=dw_kss[0],
                    stride=1,
                    dilation=1,
                    se_ratio=1,
                    attn_s=False,
                    drop_path=0.0,
                    drop=0.0,
                    shift_pixel=shift_pixel,
                )
            ]
        )

        img_size = img_size // 2
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]) : sum(depths[: i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = 2, False, False, exp_ratios[i] * 2
                    img_size = img_size // 2
                else:
                    stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
                layers.append(
                    SafeIRRWKV(
                        emb_dim_pre,
                        embed_dims[i],
                        norm_in=True,
                        has_skip=has_skip,
                        exp_ratio=exp_ratio,
                        norm_layer=norm_layers[i],
                        act_layer=act_layers[i],
                        dw_ks=dw_kss[i],
                        stride=stride,
                        dilation=1,
                        se_ratio=se_ratios[i],
                        attn_s=attn_s,
                        drop_path=dpr[j],
                        drop=drop,
                        channel_gamma=channel_gamma,
                        shift_pixel=shift_pixel,
                    )
                )
                emb_dim_pre = embed_dims[i]
            setattr(self, f"stage{i + 1}", nn.ModuleList(layers))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(
            m,
            (
                nn.LayerNorm,
                nn.GroupNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


class IdentitySkipFusion(nn.Module):
    def forward(self, features):
        return features


class RWKV_UNetV3(nn.Module):
    """RWKV_UNet variant for large inputs without compiling the legacy WKV op."""

    def __init__(self, input_channel=3, num_classes=1, img_size=256, pretrained_path=""):
        super().__init__()
        self.encoder = RWKVUNetEncoder(dim_in=input_channel, img_size=img_size)

        self.embed_dims = [48, 72, 144, 240]
        self.ccm = IdentitySkipFusion()
        self.decoder1 = UpBlock(
            self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder2 = UpBlock(
            self.embed_dims[2] * 2, self.embed_dims[1], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder3 = UpBlock(
            self.embed_dims[1] * 2, self.embed_dims[0], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder4 = UpBlock(
            self.embed_dims[0] * 2, 24, norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        _sanitize_module_finite_(self)

        if pretrained_path:
            raise ValueError(
                "RWKV_UNetV3 is standalone and does not load legacy RWKV_UNet pretrained weights."
            )

    @staticmethod
    def _match_size(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        for blk in self.encoder.stage0:
            x = blk(x)
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
        dec3 = self._match_size(self.decoder1(x), enc3)
        dec2 = self._match_size(self.decoder2(torch.cat([dec3, enc3], dim=1)), enc2)
        dec1 = self._match_size(self.decoder3(torch.cat([dec2, enc2], dim=1)), enc1)
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        out = self.final_conv(dec0)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)


def rwkv_unetv3(input_channel=3, num_classes=1, img_size=256, **kwargs):
    return RWKV_UNetV3(
        input_channel=input_channel,
        num_classes=num_classes,
        img_size=img_size,
        **kwargs,
    )
