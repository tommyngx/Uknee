
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class DynamicLerpV6(nn.Module):
    """
    Input-dependent interpolation for RWKV-6-inspired branches.

    Produces separately mixed representations for:
        r: receptance
        w: decay
        k: key
        v: value
        g: output gate
    """

    def __init__(
        self,
        dim: int,
        low_rank_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.low_rank_dim = low_rank_dim or max(32, dim // 8)

        self.mix_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_w = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_g = nn.Parameter(torch.full((1, 1, dim), 0.5))

        self.maa_down = nn.Linear(
            dim,
            self.low_rank_dim * 5,
            bias=False,
        )

        self.maa_r = nn.Linear(self.low_rank_dim, dim, bias=False)
        self.maa_w = nn.Linear(self.low_rank_dim, dim, bias=False)
        self.maa_k = nn.Linear(self.low_rank_dim, dim, bias=False)
        self.maa_v = nn.Linear(self.low_rank_dim, dim, bias=False)
        self.maa_g = nn.Linear(self.low_rank_dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.maa_down.weight, std=0.01)

        for layer in [
            self.maa_r,
            self.maa_w,
            self.maa_k,
            self.maa_v,
            self.maa_g,
        ]:
            nn.init.zeros_(layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        shifted_x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        delta = shifted_x - x

        dynamic = torch.tanh(self.maa_down(x))
        dr, dw, dk, dv, dg = dynamic.chunk(5, dim=-1)

        lerp_r = torch.sigmoid(self.mix_r + self.maa_r(dr))
        lerp_w = torch.sigmoid(self.mix_w + self.maa_w(dw))
        lerp_k = torch.sigmoid(self.mix_k + self.maa_k(dk))
        lerp_v = torch.sigmoid(self.mix_v + self.maa_v(dv))
        lerp_g = torch.sigmoid(self.mix_g + self.maa_g(dg))

        xr = x + delta * lerp_r
        xw = x + delta * lerp_w
        xk = x + delta * lerp_k
        xv = x + delta * lerp_v
        xg = x + delta * lerp_g

        return xr, xw, xk, xv, xg


class RWKV6MatrixStateScan(nn.Module):
    """
    Plain PyTorch reference implementation of a recurrent,
    multi-head, matrix-valued RWKV-6-inspired state scan.

    This implementation prioritizes clarity and correctness.
    It is slower than a fused CUDA or Triton kernel because it
    uses a Python loop over sequence length.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dtype = state_dtype

    def forward(
        self,
        r: torch.Tensor,
        decay: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        batch_size, length, channels = r.shape

        if channels != self.dim:
            raise ValueError(
                f"Expected {self.dim} channels, received {channels}"
            )

        heads = self.num_heads
        head_dim = self.head_dim

        r = r.view(batch_size, length, heads, head_dim)
        decay = decay.view(batch_size, length, heads, head_dim)
        k = k.view(batch_size, length, heads, head_dim)
        v = v.view(batch_size, length, heads, head_dim)

        if reverse:
            r = torch.flip(r, dims=[1])
            decay = torch.flip(decay, dims=[1])
            k = torch.flip(k, dims=[1])
            v = torch.flip(v, dims=[1])

        state = torch.zeros(
            batch_size,
            heads,
            head_dim,
            head_dim,
            device=r.device,
            dtype=self.state_dtype,
        )

        outputs = []

        for index in range(length):
            r_t = r[:, index].to(self.state_dtype)
            decay_t = decay[:, index].to(self.state_dtype)
            k_t = k[:, index].to(self.state_dtype)
            v_t = v[:, index].to(self.state_dtype)

            kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            state = state * decay_t.unsqueeze(-1) + kv_t

            y_t = torch.einsum(
                "bhd,bhde->bhe",
                r_t,
                state,
            )

            outputs.append(y_t.to(r.dtype))

        output = torch.stack(outputs, dim=1)

        if reverse:
            output = torch.flip(output, dims=[1])

        return output.reshape(batch_size, length, channels)


class RWKV6SequenceMix(nn.Module):
    """
    RWKV-6-inspired sequence mixer with:

    - dynamic LERP
    - data-dependent decay
    - multi-head matrix-valued recurrent state
    - forward and backward recurrence
    - receptance
    - output gate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        low_rank_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.num_heads = num_heads

        self.dynamic_lerp = DynamicLerpV6(
            dim=dim,
            low_rank_dim=low_rank_dim,
        )

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        self.forward_scan = RWKV6MatrixStateScan(
            dim=dim,
            num_heads=num_heads,
        )
        self.backward_scan = RWKV6MatrixStateScan(
            dim=dim,
            num_heads=num_heads,
        )

        self.output_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=dim,
            eps=1e-5,
        )

        self.output = nn.Linear(dim, dim, bias=False)

        self.direction_fusion = nn.Parameter(
            torch.tensor([0.5, 0.5], dtype=torch.float32)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in [
            self.receptance,
            self.key,
            self.value,
            self.gate,
        ]:
            nn.init.trunc_normal_(layer.weight, std=0.02)

        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, -2.0)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def shift_sequence(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted_x = self.shift_sequence(x)

        xr, xw, xk, xv, xg = self.dynamic_lerp(
            x=x,
            shifted_x=shifted_x,
        )

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        decay = torch.exp(
            -F.softplus(self.decay(xw))
        )

        forward_output = self.forward_scan(
            r=r,
            decay=decay,
            k=k,
            v=v,
            reverse=False,
        )

        backward_output = self.backward_scan(
            r=r,
            decay=decay,
            k=k,
            v=v,
            reverse=True,
        )

        weights = torch.softmax(
            self.direction_fusion,
            dim=0,
        )

        output = (
            weights[0] * forward_output
            + weights[1] * backward_output
        )

        output = self.output_norm(
            output.transpose(1, 2)
        ).transpose(1, 2)

        output = output * g
        output = self.output(output)

        return output


class AxialRWKV6SpatialMix(nn.Module):
    """
    2D RWKV-6-inspired spatial mixer.

    Horizontal path:
        each row is treated as a sequence.

    Vertical path:
        each column is treated as a sequence.

    Both paths are internally bidirectional.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        low_rank_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dim = dim

        self.horizontal_mix = RWKV6SequenceMix(
            dim=dim,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
        )

        self.vertical_mix = RWKV6SequenceMix(
            dim=dim,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
        )

        hidden_gate_dim = max(16, dim // 4)

        self.axis_gate = nn.Sequential(
            nn.Linear(dim, hidden_gate_dim),
            nn.SiLU(),
            nn.Linear(hidden_gate_dim, 2),
        )

        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        patch_resolution: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, token_count, channels = tokens.shape
        height, width = patch_resolution

        if token_count != height * width:
            raise ValueError(
                f"Token count {token_count} does not match "
                f"resolution {height}x{width}"
            )

        feature = tokens.view(
            batch_size,
            height,
            width,
            channels,
        )

        horizontal = feature.reshape(
            batch_size * height,
            width,
            channels,
        )
        horizontal = self.horizontal_mix(horizontal)
        horizontal = horizontal.view(
            batch_size,
            height,
            width,
            channels,
        )

        vertical = feature.permute(
            0, 2, 1, 3
        ).contiguous()
        vertical = vertical.view(
            batch_size * width,
            height,
            channels,
        )
        vertical = self.vertical_mix(vertical)
        vertical = vertical.view(
            batch_size,
            width,
            height,
            channels,
        ).permute(
            0, 2, 1, 3
        ).contiguous()

        pooled = feature.mean(dim=(1, 2))
        axis_weights = torch.softmax(
            self.axis_gate(pooled),
            dim=-1,
        )

        horizontal_weight = axis_weights[:, 0].view(
            batch_size, 1, 1, 1
        )
        vertical_weight = axis_weights[:, 1].view(
            batch_size, 1, 1, 1
        )

        output = (
            horizontal_weight * horizontal
            + vertical_weight * vertical
        )

        output = output.reshape(
            batch_size,
            token_count,
            channels,
        )

        return self.output(output)


class RWKV6ChannelMix(nn.Module):
    """
    RWKV-style gated channel mixer.
    """

    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 3.5,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)

        self.mix_k = nn.Parameter(
            torch.full((1, 1, dim), 0.5)
        )
        self.mix_r = nn.Parameter(
            torch.full((1, 1, dim), 0.5)
        )

        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

        nn.init.trunc_normal_(self.key.weight, std=0.02)
        nn.init.zeros_(self.value.weight)
        nn.init.trunc_normal_(self.receptance.weight, std=0.02)

    @staticmethod
    def shift_sequence(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted_x = self.shift_sequence(x)

        xk = x + (shifted_x - x) * torch.sigmoid(self.mix_k)
        xr = x + (shifted_x - x) * torch.sigmoid(self.mix_r)

        k = torch.relu(self.key(xk)).square()
        v = self.value(k)

        r = torch.sigmoid(self.receptance(xr))

        return r * v


class IRRWKV6Block(nn.Module):
    """
    Inverted-residual RWKV-6-inspired block for 2D medical images.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        exp_ratio: float = 2.0,
        stride: int = 1,
        num_heads: int = 4,
        drop_path: float = 0.0,
        use_spatial_mix: bool = True,
        local_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        dim_mid = int(dim_in * exp_ratio)
        dim_mid = math.ceil(dim_mid / num_heads) * num_heads

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_mid = dim_mid
        self.stride = stride
        self.use_spatial_mix = use_spatial_mix

        self.input_norm = nn.BatchNorm2d(dim_in)

        self.expand = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_mid,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(dim_mid),
            nn.SiLU(inplace=True),
        )

        self.local_conv = nn.Sequential(
            nn.Conv2d(
                dim_mid,
                dim_mid,
                kernel_size=local_kernel_size,
                stride=stride,
                padding=local_kernel_size // 2,
                groups=dim_mid,
                bias=False,
            ),
            nn.BatchNorm2d(dim_mid),
            nn.SiLU(inplace=True),
        )

        self.spatial_norm = nn.LayerNorm(dim_mid)
        self.channel_norm = nn.LayerNorm(dim_mid)

        if use_spatial_mix:
            self.spatial_mix = AxialRWKV6SpatialMix(
                dim=dim_mid,
                num_heads=num_heads,
            )

        self.channel_mix = RWKV6ChannelMix(
            dim=dim_mid,
            hidden_ratio=3.5,
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                dim_mid,
                dim_out,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(dim_out),
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

        self.use_residual = (
            stride == 1 and dim_in == dim_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.input_norm(x)
        x = self.expand(x)
        x = self.local_conv(x)

        batch_size, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        if self.use_spatial_mix:
            tokens = tokens + self.drop_path(
                self.spatial_mix(
                    self.spatial_norm(tokens),
                    patch_resolution=(height, width),
                )
            )

        tokens = tokens + self.drop_path(
            self.channel_mix(
                self.channel_norm(tokens)
            )
        )

        x = tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        x = self.project(x)

        if self.use_residual:
            x = shortcut + self.drop_path(x)

        return x


class RWKV6UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        stem_dim: int = 24,
        depths: Tuple[int, ...] = (2, 2, 4, 2),
        embed_dims: Tuple[int, ...] = (48, 72, 144, 240),
        exp_ratios: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.0),
        num_heads: Tuple[int, ...] = (1, 1, 4, 6),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if not (
            len(depths)
            == len(embed_dims)
            == len(exp_ratios)
            == len(num_heads)
            == 4
        ):
            raise ValueError(
                "depths, embed_dims, exp_ratios, and num_heads "
                "must all contain exactly four values"
            )

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(depths)
        drop_rates = torch.linspace(
            0,
            drop_path_rate,
            total_blocks,
        ).tolist()

        stages = []
        input_dim = stem_dim
        block_index = 0

        for stage_index, (
            depth,
            output_dim,
            exp_ratio,
            heads,
        ) in enumerate(
            zip(
                depths,
                embed_dims,
                exp_ratios,
                num_heads,
            )
        ):
            blocks = []

            for depth_index in range(depth):
                stride = 2 if depth_index == 0 else 1

                use_spatial_mix = (
                    stage_index >= 2 and depth_index > 0
                )

                blocks.append(
                    IRRWKV6Block(
                        dim_in=input_dim,
                        dim_out=output_dim,
                        exp_ratio=(
                            exp_ratio * 2
                            if depth_index == 0
                            else exp_ratio
                        ),
                        stride=stride,
                        num_heads=heads,
                        drop_path=drop_rates[block_index],
                        use_spatial_mix=use_spatial_mix,
                        local_kernel_size=5,
                    )
                )

                input_dim = output_dim
                block_index += 1

            stages.append(nn.Sequential(*blocks))

        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.stem(x)

        enc1 = self.stage1(x)
        enc2 = self.stage2(enc1)
        enc3 = self.stage3(enc2)
        enc4 = self.stage4(enc3)

        return enc1, enc2, enc3, enc4


class ConvDecoderBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        skip_dim: int,
        dim_out: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                dim_in + skip_dim,
                dim_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(dim_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(dim_out),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RWKV_UNetV6(nn.Module):
    """
    Axial RWKV-6-inspired U-Net for medical image segmentation.

    Notes:
    - No custom CUDA extension is required.
    - Matrix-valued recurrence is implemented in plain PyTorch.
    - The recurrent scan is only enabled in deeper encoder stages.
    - Practical input resolution is limited by memory and runtime.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        stem_dim: int = 24,
        depths: Tuple[int, ...] = (2, 2, 4, 2),
        embed_dims: Tuple[int, ...] = (48, 72, 144, 240),
        exp_ratios: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.0),
        num_heads: Tuple[int, ...] = (1, 1, 4, 6),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = RWKV6UNetEncoder(
            input_channels=input_channels,
            stem_dim=stem_dim,
            depths=depths,
            embed_dims=embed_dims,
            exp_ratios=exp_ratios,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )

        self.decoder3 = ConvDecoderBlock(
            dim_in=embed_dims[3],
            skip_dim=embed_dims[2],
            dim_out=embed_dims[2],
        )

        self.decoder2 = ConvDecoderBlock(
            dim_in=embed_dims[2],
            skip_dim=embed_dims[1],
            dim_out=embed_dims[1],
        )

        self.decoder1 = ConvDecoderBlock(
            dim_in=embed_dims[1],
            skip_dim=embed_dims[0],
            dim_out=embed_dims[0],
        )

        self.final_decoder = nn.Sequential(
            nn.Conv2d(
                embed_dims[0],
                stem_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(inplace=True),
        )

        self.segmentation_head = nn.Conv2d(
            stem_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        enc1, enc2, enc3, enc4 = self.encoder(x)

        x = self.decoder3(enc4, enc3)
        x = self.decoder2(x, enc2)
        x = self.decoder1(x, enc1)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.final_decoder(x)
        logits = self.segmentation_head(x)

        return logits


def rwkv_unet_v6(
    input_channel: int = 3,
    num_classes: int = 1,
    **kwargs,
) -> RWKV_UNetV6:
    return RWKV_UNetV6(
        input_channels=input_channel,
        num_classes=num_classes,
        **kwargs,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def self_test() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = rwkv_unet_v6(
        input_channel=1,
        num_classes=4,
    ).to(device)

    image = torch.randn(
        1,
        1,
        128,
        128,
        device=device,
    )

    target = torch.randint(
        low=0,
        high=4,
        size=(1, 128, 128),
        device=device,
    )

    logits = model(image)

    expected_shape = (1, 4, 128, 128)
    if logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {logits.shape}, "
            f"expected {expected_shape}"
        )

    loss = F.cross_entropy(logits, target)

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss detected: {loss.item()}"
        )

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"Non-finite gradient detected in {name}"
            )

    print(f"Device: {device}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Loss: {loss.item():.6f}")
    print("Forward and backward test passed.")


if __name__ == "__main__":
    self_test()
