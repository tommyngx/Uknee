from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    yaml_path: str = "Ref/yolo_mesko4GF2/data.yaml"
    image_height: int = 640
    image_width: int = 640
    num_workers: int = 4
    val_fraction: float = 0.15
    seed: int = 2006
    normalize_mean: float = 0.0
    normalize_std: float = 1.0
    augment: bool = True
    aug_strategy: str = "xray"
    rotation_degrees: float = 7.0
    scale_min: float = 0.95
    scale_max: float = 1.05
    translate_fraction: float = 0.04


@dataclass
class ModelConfig:
    name: str = "adaptive_rwkv"
    num_landmarks: int = 129
    num_bones: int = 4
    input_channels: int = 3
    num_mask_classes: int = 11
    bone_class_indices: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    bone_class_groups: list[list[int]] = field(
        default_factory=lambda: [[1, 6, 7], [2, 4, 8, 9, 10], [3, 4], [5]]
    )
    query_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    transformer_ffn_dim: int = 256
    transformer_dropout: float = 0.1
    local_patch_size: int = 24
    token_patch_size: int = 3
    freeze_backbone: bool = True
    checkpoint: str = ""
    strict_checkpoint: bool = True
    contour_tokens_per_bone: int = 512
    contour_temperature: float = 0.05
    contour_kernel_size: int = 3
    topology_mixer_layers: int = 2
    topology_unique_inference: bool = True
    vit_patch_size: int = 16
    vit_depth: int = 6
    hrnet_width: int = 32


@dataclass
class LossConfig:
    coarse_weight: float = 0.5
    coarse_heatmap_weight: float = 0.0
    coordinate_weight: float = 1.0
    heatmap_weight: float = 1.0
    bone_constraint_weight: float = 0.05
    heatmap_sigma: float = 1.5
    heatmap_temperature: float = 1.0
    topology_edge_weight: float = 0.0
    topology_curvature_weight: float = 0.0
    topology_duplicate_weight: float = 0.0


@dataclass
class TrainingConfig:
    experiment_name: str = "adaptive_rwkv_landmark"
    output_dir: str = "landmark/runs"
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 5
    coarse_only_epochs: int = 10
    teacher_forcing_epochs: int = 20
    teacher_noise_pixels: float = 15.0
    topology_start_epoch: int = 1
    topology_ramp_epochs: int = 0
    gradient_clip: float = 1.0
    amp: bool = True
    seed: int = 2006
    device: str = "auto"
    resume: str = ""
    save_every: int = 10
    plot_samples: int = 4


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _merge_dataclass(instance: T, values: dict[str, Any], prefix: str = "") -> T:
    valid = {item.name for item in fields(instance)}
    unknown = sorted(set(values) - valid)
    if unknown:
        location = prefix or type(instance).__name__
        raise ValueError(f"Unknown config keys in {location}: {unknown}")
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            _merge_dataclass(current, value, f"{prefix}{key}.")
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return _merge_dataclass(ExperimentConfig(), values)
