from __future__ import annotations

from collections.abc import Callable

from torch import nn

from landmark.config.loader import ExperimentConfig

from .adaptive_rwkv import RWKVUNetLandmarkModel
from .backbone_adapter import build_rwkv_v3_backbone
from .hrnet import HRNetLandmarkBaseline
from .kneepv1 import KneePV1ContourDETR
from .kneepv2 import KneePV2TopologyDETR
from .vitpose import ViTPoseLandmarkBaseline


ModelBuilder = Callable[[ExperimentConfig], nn.Module]
_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str):
    def decorator(builder: ModelBuilder):
        if name in _REGISTRY:
            raise KeyError(f"Duplicate landmark model: {name}")
        _REGISTRY[name] = builder
        return builder

    return decorator


@register_model("adaptive_rwkv")
def _build_adaptive_rwkv(config: ExperimentConfig) -> nn.Module:
    model_config = config.model
    backbone = build_rwkv_v3_backbone(
        input_channels=model_config.input_channels,
        num_mask_classes=model_config.num_mask_classes,
        image_size=config.data.image_height,
        checkpoint=model_config.checkpoint,
        strict=model_config.strict_checkpoint,
    )
    return RWKVUNetLandmarkModel(backbone, model_config)


register_model("adaptive_detr_rwkv")(_build_adaptive_rwkv)


@register_model("kneepv1")
def _build_kneepv1(config: ExperimentConfig) -> nn.Module:
    model_config = config.model
    backbone = build_rwkv_v3_backbone(
        input_channels=model_config.input_channels,
        num_mask_classes=model_config.num_mask_classes,
        image_size=config.data.image_height,
        checkpoint=model_config.checkpoint,
        strict=model_config.strict_checkpoint,
    )
    return KneePV1ContourDETR(backbone, model_config)


@register_model("kneepv2")
def _build_kneepv2(config: ExperimentConfig) -> nn.Module:
    model_config = config.model
    backbone = build_rwkv_v3_backbone(
        input_channels=model_config.input_channels,
        num_mask_classes=model_config.num_mask_classes,
        image_size=config.data.image_height,
        checkpoint=model_config.checkpoint,
        strict=model_config.strict_checkpoint,
    )
    return KneePV2TopologyDETR(backbone, model_config)


@register_model("vitpose")
def _build_vitpose(config: ExperimentConfig) -> nn.Module:
    model = config.model
    return ViTPoseLandmarkBaseline(
        input_channels=model.input_channels,
        num_landmarks=model.num_landmarks,
        embed_dim=model.query_dim,
        patch_size=model.vit_patch_size,
        depth=model.vit_depth,
        attention_heads=model.attention_heads,
    )


@register_model("hrnet")
def _build_hrnet(config: ExperimentConfig) -> nn.Module:
    model = config.model
    return HRNetLandmarkBaseline(
        input_channels=model.input_channels,
        num_landmarks=model.num_landmarks,
        width=model.hrnet_width,
    )


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_model(config: ExperimentConfig) -> nn.Module:
    try:
        builder = _REGISTRY[config.model.name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown landmark model '{config.model.name}'. Available: {available_models()}"
        ) from exc
    return builder(config)
