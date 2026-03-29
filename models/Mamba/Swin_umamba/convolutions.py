# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from .utils import get_act_layer, get_dropout_layer, get_norm_layer


def same_padding(kernel_size: Sequence[int] | int, dilation: Sequence[int] | int = 1):
    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)
    padding_np = ((kernel_size_np - 1) * dilation_np) / 2
    if np.any(padding_np < 0):
        raise AssertionError("padding value should not be negative.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(kernel_size: Sequence[int] | int, stride: Sequence[int] | int):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    output_padding_np = stride_np - kernel_size_np
    if np.any(output_padding_np < 0):
        raise AssertionError("output_padding should not be negative.")
    output_padding = tuple(int(p) for p in output_padding_np)
    return output_padding if len(output_padding) > 1 else output_padding[0]


def _conv_type(spatial_dims: int, is_transposed: bool):
    if spatial_dims == 1:
        return nn.ConvTranspose1d if is_transposed else nn.Conv1d
    if spatial_dims == 2:
        return nn.ConvTranspose2d if is_transposed else nn.Conv2d
    if spatial_dims == 3:
        return nn.ConvTranspose3d if is_transposed else nn.Conv3d
    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")


class ADN(nn.Sequential):
    def __init__(
        self,
        ordering: str,
        in_channels: int,
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        norm_dim: int | None = 1,
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
    ) -> None:
        super().__init__()
        layers = {
            "A": get_act_layer(act),
            "D": get_dropout_layer(dropout, dropout_dim),
            "N": get_norm_layer(norm, spatial_dims=norm_dim, channels=in_channels),
        }
        for key in ordering:
            layer = layers.get(key)
            if layer is not None and not isinstance(layer, nn.Identity):
                self.add_module(key, layer)


class Convolution(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = same_padding(kernel_size, dilation)

        conv_type = _conv_type(spatial_dims, is_transposed)
        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": strides,
            "padding": padding,
            "groups": groups,
            "bias": bias,
        }
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv_kwargs["output_padding"] = output_padding
            conv_kwargs["dilation"] = dilation
        else:
            conv_kwargs["dilation"] = dilation

        self.add_module("conv", conv_type(**conv_kwargs))

        if conv_only or (act is None and norm is None and dropout is None):
            return

        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )


class ResidualUnit(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if padding is None:
            padding = same_padding(kernel_size, dilation)

        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)
        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                spatial_dims=spatial_dims,
                in_channels=schannels,
                out_channels=out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )
            self.conv.add_module(f"unit{su}", unit)
            schannels = out_channels
            sstrides = 1

        if np.prod(np.atleast_1d(strides)) != 1 or in_channels != out_channels:
            residual_kernel = kernel_size
            residual_padding = padding
            if np.prod(np.atleast_1d(strides)) == 1:
                residual_kernel = 1
                residual_padding = 0
            conv_type = _conv_type(spatial_dims, is_transposed=False)
            self.residual = conv_type(
                in_channels,
                out_channels,
                kernel_size=residual_kernel,
                stride=strides,
                padding=residual_padding,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)
