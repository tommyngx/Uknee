import os
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from . import RWKV_UNet as rwkv_base


PRETRAINED_ENCODER_URL = "https://huggingface.co/FengheTan9/U-Stone/resolve/main/net_B.pth"


class SafeVRWKVSpatialMix(rwkv_base.VRWKV_SpatialMix):
    """RWKV spatial mix that keeps large feature maps from hitting WKV T_MAX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned_token_limit = False
        self._warned_cpu_fallback = False

    def _value_branch(self, sr, v):
        x = self.key_norm(v)
        x = sr * x
        x = self.output(x)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, x, patch_resolution=None):
        b, t, c = x.size()
        sr, k, v = self.jit_func(x, patch_resolution)

        if t > rwkv_base.T_MAX1:
            if not self._warned_token_limit:
                warnings.warn(
                    f"RWKV_UNetV3: token length T={t} exceeds WKV T_MAX={rwkv_base.T_MAX1}; "
                    "using value-branch fallback for this high-resolution feature map.",
                    RuntimeWarning,
                )
                self._warned_token_limit = True
            return self._value_branch(sr, v)

        if not x.is_cuda:
            if not self._warned_cpu_fallback:
                warnings.warn(
                    "RWKV_UNetV3: WKV CUDA branch is unavailable on CPU; using value-branch fallback.",
                    RuntimeWarning,
                )
                self._warned_cpu_fallback = True
            return self._value_branch(sr, v)

        safe_decay = torch.clamp(
            torch.nan_to_num(self.spatial_decay, nan=0.0, posinf=5.0, neginf=-5.0),
            min=-10.0,
            max=10.0,
        ) / max(t, 1)
        safe_first = torch.clamp(
            torch.nan_to_num(self.spatial_first, nan=0.0, posinf=5.0, neginf=-5.0),
            min=-10.0,
            max=10.0,
        ) / max(t, 1)
        x = rwkv_base.RUN_CUDA(b, t, c, safe_decay, safe_first, k, v)
        if not torch.isfinite(x).all():
            if not self._warned_non_finite:
                warnings.warn(
                    "RWKV_UNetV3 WKV branch produced non-finite values; using value-branch fallback.",
                    RuntimeWarning,
                )
                self._warned_non_finite = True
            x = v
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


@contextmanager
def _safe_spatial_mix_for_encoder():
    original_spatial_mix = rwkv_base.VRWKV_SpatialMix
    rwkv_base.VRWKV_SpatialMix = SafeVRWKVSpatialMix
    try:
        yield
    finally:
        rwkv_base.VRWKV_SpatialMix = original_spatial_mix


def _load_compatible_encoder_weights(encoder, pretrained_path):
    if not pretrained_path:
        return

    if isinstance(pretrained_path, str) and pretrained_path.startswith(("http://", "https://")):
        checkpoint = load_state_dict_from_url(pretrained_path, progress=True, map_location="cpu")
    elif isinstance(pretrained_path, str) and os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
    else:
        checkpoint = load_state_dict_from_url(PRETRAINED_ENCODER_URL, progress=True, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    encoder_state = encoder.state_dict()
    compatible_weights = {
        key: value
        for key, value in checkpoint.items()
        if torch.is_tensor(value) and key in encoder_state and value.shape == encoder_state[key].shape
    }
    encoder_state.update(compatible_weights)
    encoder.load_state_dict(encoder_state, strict=False)
    rwkv_base._sanitize_module_finite_(encoder)


class RWKV_UNetV3(nn.Module):
    """RWKV_UNet variant that can run large input sizes via safe WKV fallback."""

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        img_size=256,
        pretrained_path=PRETRAINED_ENCODER_URL,
    ):
        super().__init__()
        with _safe_spatial_mix_for_encoder():
            self.encoder = rwkv_base.RWKV_UNet_encoder_B(dim_in=input_channel, img_size=img_size)

        _load_compatible_encoder_weights(self.encoder, pretrained_path)

        self.embed_dims = [48, 72, 144, 240]
        self.ccm = rwkv_base.CCMix(
            [self.embed_dims[2], self.embed_dims[1], self.embed_dims[0]],
            self.embed_dims[0],
            img_size // 2,
        )
        self.decoder1 = rwkv_base.UpBlock(
            self.embed_dims[3], self.embed_dims[2], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder2 = rwkv_base.UpBlock(
            self.embed_dims[2] * 2, self.embed_dims[1], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder3 = rwkv_base.UpBlock(
            self.embed_dims[1] * 2, self.embed_dims[0], norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.decoder4 = rwkv_base.UpBlock(
            self.embed_dims[0] * 2, 24, norm_in=False, has_skip=False,
            exp_ratio=1.0, dw_ks=9, stride=1, dilation=1, se_ratio=0.0, drop_path=0.0, drop=0.0,
        )
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)
        rwkv_base._sanitize_module_finite_(self)

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
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
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
