
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility layers
# ============================================================


class DropPath(nn.Module):
    """Stochastic depth applied per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )

        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor



def _valid_group_count(channels: int, preferred_groups: int = 8) -> int:
    groups = min(preferred_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ConvGNAct(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.norm = nn.GroupNorm(
            num_groups=_valid_group_count(dim_out),
            num_channels=dim_out,
        )

        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_value: float = 1e-4,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((dim,), init_value)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, 1, -1)


# ============================================================
# Lightweight local encoder blocks
# ============================================================

def q_shift_2d(
    tokens: torch.Tensor,
    patch_resolution: Tuple[int, int],
    shift_pixel: int = 1,
    gamma: float = 0.25,
) -> torch.Tensor:
    if gamma > 0.25:
        raise ValueError("gamma must be <= 0.25")

    batch_size, token_count, channels = tokens.shape
    height, width = patch_resolution

    if token_count != height * width:
        raise ValueError(
            f"Token count {token_count} does not match "
            f"resolution {height}x{width}"
        )

    feature = tokens.transpose(1, 2).reshape(
        batch_size,
        channels,
        height,
        width,
    )

    output = torch.zeros_like(feature)

    c1 = int(channels * gamma)
    c2 = int(channels * gamma * 2)
    c3 = int(channels * gamma * 3)
    c4 = int(channels * gamma * 4)

    if shift_pixel >= width or shift_pixel >= height:
        return tokens

    output[:, 0:c1, :, shift_pixel:] = feature[:, 0:c1, :, :-shift_pixel]
    output[:, c1:c2, :, :-shift_pixel] = feature[:, c1:c2, :, shift_pixel:]
    output[:, c2:c3, shift_pixel:, :] = feature[:, c2:c3, :-shift_pixel, :]
    output[:, c3:c4, :-shift_pixel, :] = feature[:, c3:c4, shift_pixel:, :]
    output[:, c4:, :, :] = feature[:, c4:, :, :]

    return output.flatten(2).transpose(1, 2)


class LightweightDynamicShift(nn.Module):
    """
    Lightweight local RWKV-inspired shift module for Stage 3.

    It is intentionally non-recurrent:
    - bounded dynamic value mixing
    - static key/receptance/gate mixing
    - local directional Q-shift
    """

    def __init__(
        self,
        dim: int,
        low_rank_dim: int = 16,
        shift_pixel: int = 1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.shift_pixel = shift_pixel

        self.mix_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_g = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, dim), 0.5))

        self.value_down = nn.Linear(dim, low_rank_dim, bias=False)
        self.value_up = nn.Linear(low_rank_dim, dim, bias=False)

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_down.weight, std=0.01)
        nn.init.zeros_(self.value_up.weight)

        for layer in [
            self.key,
            self.value,
            self.receptance,
            self.gate,
        ]:
            nn.init.trunc_normal_(layer.weight, std=0.02)

        nn.init.trunc_normal_(self.output.weight, std=0.01)

    def forward(
        self,
        tokens: torch.Tensor,
        patch_resolution: Tuple[int, int],
    ) -> torch.Tensor:
        shifted = q_shift_2d(
            tokens,
            patch_resolution=patch_resolution,
            shift_pixel=self.shift_pixel,
        )
        delta = shifted - tokens

        dynamic_v = torch.tanh(self.value_down(tokens))
        dynamic_v = self.value_up(dynamic_v)

        mix_k = torch.sigmoid(self.mix_k)
        mix_r = torch.sigmoid(self.mix_r)
        mix_g = torch.sigmoid(self.mix_g)
        mix_v = torch.sigmoid(self.mix_v + dynamic_v)

        xk = tokens + delta * mix_k
        xr = tokens + delta * mix_r
        xg = tokens + delta * mix_g
        xv = tokens + delta * mix_v

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        g = F.silu(self.gate(xg))

        return self.output(r * (k + v) * g)


class LocalIRBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        exp_ratio: float = 2.0,
        stride: int = 1,
        kernel_size: int = 5,
        drop_path: float = 0.0,
        use_dynamic_shift: bool = False,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()

        dim_mid = int(dim_in * exp_ratio)

        self.use_residual = stride == 1 and dim_in == dim_out
        self.use_dynamic_shift = use_dynamic_shift

        self.expand = ConvGNAct(
            dim_in,
            dim_mid,
            kernel_size=1,
        )

        self.local = ConvGNAct(
            dim_mid,
            dim_mid,
            kernel_size=kernel_size,
            stride=stride,
            groups=dim_mid,
        )

        if use_dynamic_shift:
            self.token_norm = nn.LayerNorm(dim_mid)
            self.dynamic_shift = LightweightDynamicShift(
                dim=dim_mid,
                low_rank_dim=min(16, max(8, dim_mid // 32)),
            )
            self.shift_scale = LayerScale(
                dim=dim_mid,
                init_value=layer_scale_init,
            )

        self.project = ConvGNAct(
            dim_mid,
            dim_out,
            kernel_size=1,
            activation=False,
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.expand(x)
        x = self.local(x)

        if self.use_dynamic_shift:
            batch_size, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2)

            shifted = self.dynamic_shift(
                self.token_norm(tokens),
                patch_resolution=(height, width),
            )

            tokens = tokens + self.drop_path(
                self.shift_scale(shifted)
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


# ============================================================
# Optimized RWKV-6-inspired matrix-state bottleneck
# ============================================================

class PartialDynamicLerpV6(nn.Module):
    """
    Reduced-complexity dynamic mixing.

    Dynamic branches:
        w: recurrence decay
        v: value/content

    Static learned branches:
        r, k, g
    """

    def __init__(
        self,
        dim: int,
        low_rank_dim: int = 32,
    ) -> None:
        super().__init__()

        self.mix_r = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_w = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_k = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.mix_g = nn.Parameter(torch.full((1, 1, dim), 0.5))

        self.dynamic_down = nn.Linear(
            dim,
            low_rank_dim * 2,
            bias=False,
        )

        self.dynamic_w = nn.Linear(
            low_rank_dim,
            dim,
            bias=False,
        )

        self.dynamic_v = nn.Linear(
            low_rank_dim,
            dim,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.dynamic_down.weight, std=0.01)
        nn.init.zeros_(self.dynamic_w.weight)
        nn.init.zeros_(self.dynamic_v.weight)

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

        dynamic = torch.tanh(self.dynamic_down(x))
        dynamic_w, dynamic_v = dynamic.chunk(2, dim=-1)

        mix_r = torch.sigmoid(self.mix_r)
        mix_k = torch.sigmoid(self.mix_k)
        mix_g = torch.sigmoid(self.mix_g)

        mix_w = torch.sigmoid(
            self.mix_w + self.dynamic_w(dynamic_w)
        )

        mix_v = torch.sigmoid(
            self.mix_v + self.dynamic_v(dynamic_v)
        )

        xr = x + delta * mix_r
        xw = x + delta * mix_w
        xk = x + delta * mix_k
        xv = x + delta * mix_v
        xg = x + delta * mix_g

        return xr, xw, xk, xv, xg


class MatrixStateScan(nn.Module):
    """
    Plain PyTorch multi-head matrix-state recurrent scan.

    The recurrent state is kept in float32 for stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        state_dtype: torch.dtype = torch.float32,
        state_scale: Optional[float] = None,
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

        self.state_scale = (
            state_scale
            if state_scale is not None
            else self.head_dim ** -0.5
        )

    def forward(
        self,
        r: torch.Tensor,
        decay: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, channels = r.shape

        heads = self.num_heads
        head_dim = self.head_dim

        r = r.view(batch_size, length, heads, head_dim)
        decay = decay.view(batch_size, length, heads, head_dim)
        k = k.view(batch_size, length, heads, head_dim)
        v = v.view(batch_size, length, heads, head_dim)

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

            kv_t = (
                k_t.unsqueeze(-1)
                * v_t.unsqueeze(-2)
                * self.state_scale
            )

            state = (
                state * decay_t.unsqueeze(-1)
                + kv_t
            )

            y_t = torch.einsum(
                "bhd,bhde->bhe",
                r_t,
                state,
            )

            outputs.append(y_t.to(r.dtype))

        return torch.stack(outputs, dim=1).reshape(
            batch_size,
            length,
            channels,
        )


class UniRWKV6SequenceMix(nn.Module):
    """
    Unidirectional RWKV-6-inspired matrix-state sequence mixer.

    Reverse direction is handled outside this module by flipping the sequence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        low_rank_dim: int = 32,
    ) -> None:
        super().__init__()

        self.dynamic_lerp = PartialDynamicLerpV6(
            dim=dim,
            low_rank_dim=low_rank_dim,
        )

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Linear(dim, dim, bias=True)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)

        self.scan = MatrixStateScan(
            dim=dim,
            num_heads=num_heads,
        )

        self.norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=dim,
            eps=1e-5,
        )

        self.output = nn.Linear(dim, dim, bias=False)

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

        # Important: inner projection must be non-zero.
        nn.init.trunc_normal_(self.output.weight, std=0.01)

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

        output = self.scan(
            r=r,
            decay=decay,
            k=k,
            v=v,
        )

        output = self.norm(
            output.transpose(1, 2)
        ).transpose(1, 2)

        output = output * g
        return self.output(output)


class QuadAxialMatrixRWKV(nn.Module):
    """
    Lightweight four-direction axial matrix-state mixer.

    Directions:
        left  -> right
        right -> left
        top   -> bottom
        bottom-> top

    Reverse directions share weights with the matching forward axis.
    Direction fusion uses only four learned scalar logits.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        low_rank_dim: int = 32,
    ) -> None:
        super().__init__()

        self.horizontal = UniRWKV6SequenceMix(
            dim=dim,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
        )

        self.vertical = UniRWKV6SequenceMix(
            dim=dim,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
        )

        self.direction_logits = nn.Parameter(
            torch.zeros(4)
        )

        self.output = nn.Linear(dim, dim, bias=False)

        # Outer residual projection starts at zero.
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _reverse_scan(
        mixer: nn.Module,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        reversed_sequence = torch.flip(
            sequence,
            dims=[1],
        )
        output = mixer(reversed_sequence)
        return torch.flip(output, dims=[1])

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

        # Horizontal sequences: [B*H, W, C]
        horizontal_sequence = feature.reshape(
            batch_size * height,
            width,
            channels,
        )

        left_to_right = self.horizontal(
            horizontal_sequence
        )

        right_to_left = self._reverse_scan(
            self.horizontal,
            horizontal_sequence,
        )

        left_to_right = left_to_right.view(
            batch_size,
            height,
            width,
            channels,
        )

        right_to_left = right_to_left.view(
            batch_size,
            height,
            width,
            channels,
        )

        # Vertical sequences: [B*W, H, C]
        vertical_sequence = feature.permute(
            0, 2, 1, 3
        ).contiguous().view(
            batch_size * width,
            height,
            channels,
        )

        top_to_bottom = self.vertical(
            vertical_sequence
        )

        bottom_to_top = self._reverse_scan(
            self.vertical,
            vertical_sequence,
        )

        top_to_bottom = top_to_bottom.view(
            batch_size,
            width,
            height,
            channels,
        ).permute(
            0, 2, 1, 3
        ).contiguous()

        bottom_to_top = bottom_to_top.view(
            batch_size,
            width,
            height,
            channels,
        ).permute(
            0, 2, 1, 3
        ).contiguous()

        weights = torch.softmax(
            self.direction_logits,
            dim=0,
        )

        output = (
            weights[0] * left_to_right
            + weights[1] * right_to_left
            + weights[2] * top_to_bottom
            + weights[3] * bottom_to_top
        )

        output = output.reshape(
            batch_size,
            token_count,
            channels,
        )

        return self.output(output)


class LightweightChannelMix(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)

        self.fc1 = nn.Linear(
            dim,
            hidden_dim * 2,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout(x)
        return self.fc2(x)


class MatrixRWKVBottleneckBlock(nn.Module):
    """
    Paper-oriented bottleneck block.

    - Quad-axial shared-weight matrix-state recurrence
    - partial dynamic LERP (w and v only)
    - LayerScale
    - lightweight channel mix
    - GroupNorm-compatible small-batch design
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        low_rank_dim: int = 32,
        channel_ratio: float = 2.0,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.spatial_norm = nn.LayerNorm(dim)
        self.channel_norm = nn.LayerNorm(dim)

        self.spatial_mix = QuadAxialMatrixRWKV(
            dim=dim,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
        )

        self.channel_mix = LightweightChannelMix(
            dim=dim,
            hidden_ratio=channel_ratio,
            dropout=0.05,
        )

        self.spatial_scale = LayerScale(
            dim=dim,
            init_value=layer_scale_init,
        )

        self.channel_scale = LayerScale(
            dim=dim,
            init_value=layer_scale_init,
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        tokens = tokens + self.drop_path(
            self.spatial_scale(
                self.spatial_mix(
                    self.spatial_norm(tokens),
                    patch_resolution=(height, width),
                )
            )
        )

        tokens = tokens + self.drop_path(
            self.channel_scale(
                self.channel_mix(
                    self.channel_norm(tokens)
                )
            )
        )

        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )


# ============================================================
# Encoder and decoder
# ============================================================

class MedRWKV6Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        stem_dim: int = 24,
        depths: Tuple[int, int, int, int] = (2, 2, 3, 2),
        embed_dims: Tuple[int, int, int, int] = (48, 72, 128, 192),
        exp_ratios: Tuple[float, float, float, float] = (2.0, 2.0, 2.5, 2.0),
        drop_path_rate: float = 0.05,
        bottleneck_heads: int = 8,
        bottleneck_low_rank: int = 32,
    ) -> None:
        super().__init__()

        self.stem = ConvGNAct(
            input_channels,
            stem_dim,
            kernel_size=3,
            stride=1,
        )

        total_local_blocks = sum(depths)
        drop_rates = torch.linspace(
            0,
            drop_path_rate,
            total_local_blocks + 1,
        ).tolist()

        stages = []
        input_dim = stem_dim
        drop_index = 0

        for stage_index in range(4):
            blocks = []

            for block_index in range(depths[stage_index]):
                stride = 2 if block_index == 0 else 1

                use_dynamic_shift = (
                    stage_index == 2
                    and block_index > 0
                )

                blocks.append(
                    LocalIRBlock(
                        dim_in=input_dim,
                        dim_out=embed_dims[stage_index],
                        exp_ratio=(
                            exp_ratios[stage_index] * 2
                            if block_index == 0
                            else exp_ratios[stage_index]
                        ),
                        stride=stride,
                        kernel_size=5,
                        drop_path=drop_rates[drop_index],
                        use_dynamic_shift=use_dynamic_shift,
                        layer_scale_init=1e-4,
                    )
                )

                input_dim = embed_dims[stage_index]
                drop_index += 1

            stages.append(nn.Sequential(*blocks))

        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]

        self.matrix_bottleneck = MatrixRWKVBottleneckBlock(
            dim=embed_dims[3],
            num_heads=bottleneck_heads,
            low_rank_dim=bottleneck_low_rank,
            channel_ratio=2.0,
            drop_path=drop_path_rate,
            layer_scale_init=1e-4,
        )

        self.bottleneck_refine = LocalIRBlock(
            dim_in=embed_dims[3],
            dim_out=embed_dims[3],
            exp_ratio=2.0,
            stride=1,
            kernel_size=5,
            drop_path=drop_path_rate,
            use_dynamic_shift=False,
        )

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

        enc4 = self.matrix_bottleneck(enc4)
        enc4 = self.bottleneck_refine(enc4)

        return enc1, enc2, enc3, enc4


class SkipGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(skip))
        return skip * (1.0 + gate)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        skip_dim: int,
        dim_out: int,
        use_skip_gate: bool = True,
    ) -> None:
        super().__init__()

        self.skip_gate = (
            SkipGate(skip_dim)
            if use_skip_gate
            else nn.Identity()
        )

        self.conv1 = ConvGNAct(
            dim_in + skip_dim,
            dim_out,
            kernel_size=3,
        )

        self.conv2 = ConvGNAct(
            dim_out,
            dim_out,
            kernel_size=3,
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

        skip = self.skip_gate(skip)
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class MedAxialRWKV6UNet(nn.Module):
    """
    MedAxial-RWKV6 U-Net

    Design goals:
    - stable training with small medical-image batches
    - high-resolution input support
    - bottleneck-only matrix-state recurrence
    - limited model complexity for small datasets
    - optional two-level deep supervision

    Output:
    - deep_supervision=False:
        logits
    - deep_supervision=True:
        {"out": logits, "aux": auxiliary_logits}
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        stem_dim: int = 24,
        depths: Tuple[int, int, int, int] = (2, 2, 3, 2),
        embed_dims: Tuple[int, int, int, int] = (48, 72, 128, 192),
        exp_ratios: Tuple[float, float, float, float] = (2.0, 2.0, 2.5, 2.0),
        drop_path_rate: float = 0.05,
        bottleneck_heads: int = 8,
        bottleneck_low_rank: int = 32,
        use_skip_gate: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.deep_supervision = deep_supervision

        self.encoder = MedRWKV6Encoder(
            input_channels=input_channels,
            stem_dim=stem_dim,
            depths=depths,
            embed_dims=embed_dims,
            exp_ratios=exp_ratios,
            drop_path_rate=drop_path_rate,
            bottleneck_heads=bottleneck_heads,
            bottleneck_low_rank=bottleneck_low_rank,
        )

        self.decoder3 = DecoderBlock(
            dim_in=embed_dims[3],
            skip_dim=embed_dims[2],
            dim_out=embed_dims[2],
            use_skip_gate=use_skip_gate,
        )

        self.decoder2 = DecoderBlock(
            dim_in=embed_dims[2],
            skip_dim=embed_dims[1],
            dim_out=embed_dims[1],
            use_skip_gate=use_skip_gate,
        )

        self.decoder1 = DecoderBlock(
            dim_in=embed_dims[1],
            skip_dim=embed_dims[0],
            dim_out=embed_dims[0],
            use_skip_gate=use_skip_gate,
        )

        self.final_refine = nn.Sequential(
            ConvGNAct(
                embed_dims[0],
                stem_dim,
                kernel_size=3,
            ),
            ConvGNAct(
                stem_dim,
                stem_dim,
                kernel_size=3,
            ),
        )

        self.segmentation_head = nn.Conv2d(
            stem_dim,
            num_classes,
            kernel_size=1,
        )

        if deep_supervision:
            self.auxiliary_head = nn.Conv2d(
                embed_dims[1],
                num_classes,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[
        torch.Tensor,
        dict,
    ]:
        input_size = x.shape[-2:]

        enc1, enc2, enc3, enc4 = self.encoder(x)

        dec3 = self.decoder3(enc4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        x = F.interpolate(
            dec1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.final_refine(x)
        logits = self.segmentation_head(x)

        if not self.deep_supervision:
            return logits

        auxiliary_logits = self.auxiliary_head(dec2)
        auxiliary_logits = F.interpolate(
            auxiliary_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "out": logits,
            "aux": auxiliary_logits,
        }


def med_axial_rwkv6_unet(
    input_channel: int = 1,
    num_classes: int = 1,
    **kwargs,
) -> MedAxialRWKV6UNet:
    return MedAxialRWKV6UNet(
        input_channels=input_channel,
        num_classes=num_classes,
        **kwargs,
    )


# ============================================================
# Diagnostics and self-test
# ============================================================

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def check_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                f"Non-finite gradient detected in {name}"
            )


def report_missing_gradients(
    model: nn.Module,
    keyword: str = "matrix_bottleneck",
) -> None:
    missing = []

    for name, parameter in model.named_parameters():
        if keyword in name and parameter.requires_grad:
            if parameter.grad is None:
                missing.append(name)

    if missing:
        print("Parameters without gradients:")
        for name in missing:
            print(f"  - {name}")
    else:
        print(
            f"All trainable parameters containing "
            f"'{keyword}' received gradients."
        )


def self_test() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = med_axial_rwkv6_unet(
        input_channel=1,
        num_classes=4,
        deep_supervision=True,
    ).to(device)

    # 128x128 is used for a quick self-test.
    # The architecture itself supports dynamic spatial sizes.
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

    output = model(image)

    logits = output["out"]
    auxiliary_logits = output["aux"]

    expected_shape = (1, 4, 128, 128)

    if logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected main output shape: {logits.shape}"
        )

    if auxiliary_logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected auxiliary output shape: "
            f"{auxiliary_logits.shape}"
        )

    main_loss = F.cross_entropy(
        logits,
        target,
    )

    auxiliary_loss = F.cross_entropy(
        auxiliary_logits,
        target,
    )

    loss = main_loss + 0.3 * auxiliary_loss

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss detected: {loss.item()}"
        )

    loss.backward()

    check_finite_gradients(model)
    report_missing_gradients(model)

    print(f"Device: {device}")
    print(f"Main output: {tuple(logits.shape)}")
    print(f"Aux output: {tuple(auxiliary_logits.shape)}")
    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print(f"Loss: {loss.item():.6f}")
    print("Forward/backward self-test passed.")


if __name__ == "__main__":
    self_test()
