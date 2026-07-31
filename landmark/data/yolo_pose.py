from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from landmark.config.loader import DataConfig


BONE_NAMES = ("femur", "tibia", "fibula", "patella")
POINT_COUNTS = (45, 51, 24, 9)
POINT_OFFSETS = (0, 45, 96, 120)
NUM_LANDMARKS = sum(POINT_COUNTS)
POINT_BONE_IDS = torch.tensor(
    [bone_id for bone_id, count in enumerate(POINT_COUNTS) for _ in range(count)],
    dtype=torch.long,
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _resolve_dataset_root(yaml_path: Path, metadata: dict) -> Path:
    configured = Path(str(metadata.get("path", ""))).expanduser()
    if configured.is_dir():
        return configured
    if configured and not configured.is_absolute():
        candidate = (yaml_path.parent / configured).resolve()
        if candidate.is_dir():
            return candidate
    warnings.warn(
        f"Dataset path '{configured}' from {yaml_path} does not exist; "
        f"using the YAML directory '{yaml_path.parent}'.",
        RuntimeWarning,
    )
    return yaml_path.parent


def _resolve_split(root: Path, value: str | Iterable[str]) -> list[Path]:
    values = [value] if isinstance(value, str) else list(value)
    paths: list[Path] = []
    for entry in values:
        candidate = Path(entry)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_dir():
            paths.extend(
                item for item in sorted(candidate.iterdir())
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
        elif candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text(encoding="utf-8").splitlines():
                image_path = Path(line.strip())
                paths.append(image_path if image_path.is_absolute() else root / image_path)
        elif candidate.is_file():
            paths.append(candidate)
    return paths


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path must contain an 'images' directory: {image_path}")
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_yolo_pose_label(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    landmarks = torch.zeros(NUM_LANDMARKS, 2, dtype=torch.float32)
    visibility = torch.zeros(NUM_LANDMARKS, dtype=torch.float32)
    if not path.exists():
        return landmarks, visibility

    seen_classes: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 8 or (len(values) - 5) % 3:
            raise ValueError(f"Malformed YOLO-Pose row at {path}:{line_number}")
        class_id = int(float(values[0]))
        if class_id not in range(len(BONE_NAMES)):
            raise ValueError(f"Unsupported class {class_id} at {path}:{line_number}")
        if class_id in seen_classes:
            raise ValueError(f"Duplicate class {class_id} at {path}:{line_number}")
        seen_classes.add(class_id)

        raw = torch.tensor([float(item) for item in values[5:]], dtype=torch.float32).view(-1, 3)
        required = POINT_COUNTS[class_id]
        if raw.shape[0] < required:
            raise ValueError(
                f"Class {class_id} at {path}:{line_number} has {raw.shape[0]} slots; "
                f"at least {required} are required."
            )
        start = POINT_OFFSETS[class_id]
        points = raw[:required, :2]
        visible = (raw[:required, 2] > 0).float()
        valid = torch.isfinite(points).all(dim=-1) & (points >= 0).all(dim=-1) & (points <= 1).all(dim=-1)
        visibility[start : start + required] = visible * valid.float()
        landmarks[start : start + required] = points.clamp(0, 1)
    return landmarks, visibility


class YoloPoseLandmarkDataset(Dataset):
    """Load class-wise YOLO-Pose contours as one ordered 129-point target."""

    def __init__(
        self,
        image_paths: list[Path],
        config: DataConfig,
        augment: bool = False,
    ):
        if not image_paths:
            raise ValueError("The landmark dataset contains no images")
        self.image_paths = image_paths
        self.config = config
        strategy = config.aug_strategy.lower().strip()
        if strategy not in {"xray", "basic", "none", "off"}:
            raise ValueError(
                f"Unknown landmark augmentation strategy '{config.aug_strategy}'. "
                "Available: xray, basic, none"
            )
        self.aug_strategy = strategy
        self.augment = augment and strategy not in {"none", "off"}

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment_intensity(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        return image

    def _augment_geometry(
        self,
        image: Image.Image,
        landmarks: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        try:
            import cv2
        except ImportError:
            return image, landmarks, visibility

        width, height = image.size
        angle = random.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)
        scale = random.uniform(self.config.scale_min, self.config.scale_max)
        tx = random.uniform(-self.config.translate_fraction, self.config.translate_fraction) * width
        ty = random.uniform(-self.config.translate_fraction, self.config.translate_fraction) * height
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        array = np.asarray(image)
        array = cv2.warpAffine(
            array,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        xy = landmarks.clone()
        pixels = xy * xy.new_tensor([width, height])
        homogeneous = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=-1)
        transformed = homogeneous @ torch.as_tensor(matrix, dtype=xy.dtype).T
        xy = transformed / xy.new_tensor([width, height])
        inside = (xy >= 0).all(dim=-1) & (xy <= 1).all(dim=-1)
        return Image.fromarray(array), xy.clamp(0, 1), visibility * inside.float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        original_width, original_height = image.size
        landmarks, visibility = parse_yolo_pose_label(_label_path(image_path))

        if self.augment:
            image, landmarks, visibility = self._augment_geometry(image, landmarks, visibility)
            if self.aug_strategy == "xray":
                image = self._augment_intensity(image)

        image = image.resize(
            (self.config.image_width, self.config.image_height),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).unsqueeze(0)
        tensor = (tensor - self.config.normalize_mean) / max(self.config.normalize_std, 1e-6)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite pixels in {image_path}")
        return {
            "image": tensor,
            "landmarks": landmarks,
            "landmark_visibility": visibility,
            "image_id": image_path.stem,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.long),
        }


def build_dataloaders(
    config: DataConfig,
    batch_size: int,
    train: bool = True,
) -> tuple[DataLoader | None, DataLoader]:
    yaml_path = Path(config.yaml_path).expanduser().resolve()
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    root = _resolve_dataset_root(yaml_path, metadata)
    train_paths = _resolve_split(root, metadata.get("train", "images/train"))
    val_paths = _resolve_split(root, metadata.get("val", "images/val"))

    same_split = {item.resolve() for item in train_paths} == {item.resolve() for item in val_paths}
    if not val_paths or same_split:
        generator = torch.Generator().manual_seed(config.seed)
        order = torch.randperm(len(train_paths), generator=generator).tolist()
        val_count = max(1, int(round(len(order) * config.val_fraction)))
        val_indices = set(order[:val_count])
        val_paths = [path for idx, path in enumerate(train_paths) if idx in val_indices]
        train_paths = [path for idx, path in enumerate(train_paths) if idx not in val_indices]

    kwargs = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = None
    if train:
        train_loader = DataLoader(
            YoloPoseLandmarkDataset(train_paths, config, augment=config.augment),
            shuffle=True,
            drop_last=len(train_paths) >= batch_size,
            **kwargs,
        )
    val_loader = DataLoader(
        YoloPoseLandmarkDataset(val_paths, config, augment=False),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader
