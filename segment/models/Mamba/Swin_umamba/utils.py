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

import inspect

import torch.nn as nn

__all__ = [
    "split_args",
    "get_norm_layer",
    "get_act_layer",
    "get_dropout_layer",
    "get_pool_layer",
]


def split_args(args):
    if isinstance(args, str) or callable(args):
        return args, {}

    name_obj, name_args = args
    if not (isinstance(name_obj, str) or callable(name_obj)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be a string/callable or a pair of the form (name/object-types, argument dict)"
        raise TypeError(msg)
    return name_obj, name_args


def has_option(obj, key: str) -> bool:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    return key in signature.parameters


def _resolve_layer_type(name_obj, mapping: dict[str, object]):
    if callable(name_obj):
        return name_obj
    key = str(name_obj).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported layer type: {name_obj}")
    return mapping[key]


def _default_group_count(channels: int | None) -> int:
    if not channels or channels < 1:
        return 1
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def get_norm_layer(name: tuple | str, spatial_dims: int | None = 1, channels: int | None = 1):
    if name in ("", None):
        return nn.Identity()

    norm_name, norm_args = split_args(name)
    norm_map = {
        "instance": (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d),
        "batch": (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
        "syncbatch": (nn.SyncBatchNorm, nn.SyncBatchNorm, nn.SyncBatchNorm),
        "group": nn.GroupNorm,
        "layer": nn.LayerNorm,
        "localresponse": nn.LocalResponseNorm,
    }
    norm_type = _resolve_layer_type(
        norm_name,
        {
            "instance": norm_map["instance"][max(1, spatial_dims or 1) - 1],
            "batch": norm_map["batch"][max(1, spatial_dims or 1) - 1],
            "syncbatch": norm_map["syncbatch"][max(1, spatial_dims or 1) - 1],
            "group": norm_map["group"],
            "layer": norm_map["layer"],
            "localresponse": norm_map["localresponse"],
        },
    )

    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    if has_option(norm_type, "normalized_shape") and "normalized_shape" not in kw_args:
        kw_args["normalized_shape"] = channels
    if has_option(norm_type, "num_groups") and "num_groups" not in kw_args:
        kw_args["num_groups"] = _default_group_count(channels)
    if norm_type in {nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d}:
        kw_args.setdefault("affine", True)
    return norm_type(**kw_args)


def get_act_layer(name: tuple | str):
    if name in ("", None):
        return nn.Identity()

    act_name, act_args = split_args(name)
    act_map = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "relu6": nn.ReLU6,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "logsoftmax": nn.LogSoftmax,
        "swish": nn.SiLU,
        "memswish": nn.SiLU,
        "mish": nn.Mish,
    }
    act_type = _resolve_layer_type(act_name, act_map)
    return act_type(**act_args)


def get_dropout_layer(name: tuple | str | float | int, dropout_dim: int | None = 1):
    if name in ("", None):
        return nn.Identity()

    if isinstance(name, (int, float)):
        drop_name = "dropout"
        drop_args = {"p": float(name)}
    else:
        drop_name, drop_args = split_args(name)

    dim = max(1, dropout_dim or 1)
    drop_map = {
        "dropout": (nn.Dropout, nn.Dropout2d, nn.Dropout3d),
        "alphadropout": (nn.AlphaDropout, nn.AlphaDropout, nn.AlphaDropout),
    }
    drop_type = _resolve_layer_type(drop_name, {k: v[dim - 1] for k, v in drop_map.items()})
    return drop_type(**drop_args)


def get_pool_layer(name: tuple | str, spatial_dims: int | None = 1):
    if name in ("", None):
        return nn.Identity()

    pool_name, pool_args = split_args(name)
    dim = max(1, spatial_dims or 1)
    pool_map = {
        "max": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
        "adaptivemax": (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d),
        "avg": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d),
        "adaptiveavg": (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
    }
    pool_type = _resolve_layer_type(pool_name, {k: v[dim - 1] for k, v in pool_map.items()})
    return pool_type(**pool_args)
