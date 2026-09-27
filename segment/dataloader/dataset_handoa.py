"""Dataset adapter for HandOA (Hand Osteoarthritis) segmentation layout.

Designed for hand X-ray bone and joint space segmentation datasets,
such as HandOA_segX1, following the 11-class anatomy standard:
  0: background
  1: distal_phalanges (DP1-5)
  2: middle_phalanges (MP2-5)
  3: proximal_phalanges (PP1-5)
  4: metacarpal_1 (MC1)
  5: metacarpals_other (MC2-5)
  6: trapezium
  7: trapezoid
  8: scaphoid
  9: interphalangeal_plateau (DIP/PIP/IP articular contact)
  10: metacarpal_overlap (2D projection overlap)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from segment.dataloader.image_io import read_rgb_image


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
HANDOA_DATASET_NAMES = {
    "handoa",
    "handoa_segx1",
    "handoa_seg",
    "hand_oa",
    "hand_seg",
    "hand",
    "ram-h1200",
}

# Standard 11 Hand OA anatomical bone & joint classes
HANDOA_DEFAULT_CLASSES = [
    {
        "class_id": 0,
        "name": "background",
        "color": [0, 0, 0],
        "description": "Background non-bone regions",
    },
    {
        "class_id": 1,
        "name": "distal_phalanges",
        "color": [0, 210, 240],
        "bones": ["DP1", "DP2", "DP3", "DP4", "DP5"],
        "description": "Distal Phalanges (DP1-5) - Upper articular border of DIP and IP joints",
    },
    {
        "class_id": 2,
        "name": "middle_phalanges",
        "color": [0, 185, 115],
        "bones": ["MP2", "MP3", "MP4", "MP5"],
        "description": "Middle Phalanges (MP2-5) - Articular bridge between DIP and PIP joints",
    },
    {
        "class_id": 3,
        "name": "proximal_phalanges",
        "color": [232, 77, 186],
        "bones": ["PP1", "PP2", "PP3", "PP4", "PP5"],
        "description": "Proximal Phalanges (PP1-5) - Lower articular border of PIP joints",
    },
    {
        "class_id": 4,
        "name": "metacarpal_1",
        "color": [244, 164, 96],
        "bones": ["MC1"],
        "description": "1st Metacarpal (MC1) - Focal bone for 1st CMC Rhizarthrosis",
    },
    {
        "class_id": 5,
        "name": "metacarpals_other",
        "color": [255, 135, 75],
        "bones": ["MC2", "MC3", "MC4", "MC5"],
        "description": "Metacarpals 2 to 5 (MC2-5) - Structural contextual metacarpals",
    },
    {
        "class_id": 6,
        "name": "trapezium",
        "color": [135, 35, 225],
        "bones": ["Trapezium"],
        "description": "Trapezium (Xương Thang) - Forms 1st CMC joint with MC1",
    },
    {
        "class_id": 7,
        "name": "trapezoid",
        "color": [78, 176, 155],
        "bones": ["Trapezoid"],
        "description": "Trapezoid (Xương Thê) - Central carpal articulation with Scaphoid and MC2",
    },
    {
        "class_id": 8,
        "name": "scaphoid",
        "color": [65, 65, 205],
        "bones": ["Scaphoid"],
        "description": "Scaphoid (Xương Thuyền) - Forms STT joint",
    },
    {
        "class_id": 9,
        "name": "interphalangeal_plateau",
        "color": [139, 92, 246],
        "description": "Interphalangeal Articular Plateau at DIP, PIP, and Thumb IP contact",
    },
    {
        "class_id": 10,
        "name": "metacarpal_overlap",
        "color": [195, 95, 45],
        "description": "Metacarpal and Carpal overlap interface in 2D projection",
    },
]


def _list_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _index_images(directory: Path | None) -> dict[str, Path]:
    return {path.stem: path for path in _list_images(directory)}


def _load_summary(base_dir: Path) -> dict[str, Any]:
    summary_path = base_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        return summary if isinstance(summary, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _normalize_classes(raw_list: list[Any]) -> list[dict[str, Any]]:
    result = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        cid = item.get("class_id") if "class_id" in item else item.get("id")
        if cid is None:
            continue
        normalized = dict(item)
        normalized["class_id"] = int(cid)
        if "name" not in normalized:
            normalized["name"] = f"class_{cid}"
        result.append(normalized)
    return sorted(result, key=lambda x: x["class_id"])


def handoa_class_info(base_dir: Path) -> list[dict[str, Any]]:
    """Load class taxonomy from classes.json, summary.json, or default HandOA taxonomy."""
    base_path = Path(base_dir).expanduser()
    classes_path = base_path / "classes.json"
    if classes_path.is_file():
        try:
            with classes_path.open("r", encoding="utf-8") as file:
                classes = json.load(file)
            if isinstance(classes, list):
                parsed = _normalize_classes(classes)
                if parsed:
                    return parsed
        except (OSError, ValueError):
            pass

    summary = _load_summary(base_path)
    if "classes" in summary and isinstance(summary["classes"], list):
        parsed = _normalize_classes(summary["classes"])
        if parsed:
            return parsed

    # Check for yaml definition
    for yaml_name in ("data_unet.yaml", "data_segment.yaml", "data.yaml"):
        yaml_path = base_path / yaml_name
        if yaml_path.is_file():
            try:
                import yaml

                with yaml_path.open("r", encoding="utf-8") as file:
                    data = yaml.safe_load(file) or {}
                names = data.get("names", {})
                if isinstance(names, dict):
                    return [
                        {"class_id": int(cid), "name": str(name)}
                        for cid, name in sorted(names.items(), key=lambda item: int(item[0]))
                    ]
                if isinstance(names, list):
                    return [{"class_id": idx, "name": str(name)} for idx, name in enumerate(names)]
            except Exception:
                pass

    return list(HANDOA_DEFAULT_CLASSES)


def is_handoa_dataset(base_dir: str | Path, dataset_name: str = "") -> bool:
    """Return whether a path/name follows the HandOA segmentation contract."""
    base_path = Path(base_dir).expanduser()
    name = (dataset_name or "").strip().lower()
    if name in HANDOA_DATASET_NAMES or name.startswith("handoa") or name.startswith("hand_oa"):
        return True

    summary = _load_summary(base_path)
    summary_name = str(summary.get("dataset_name", "")).strip().lower()
    if (
        summary_name.startswith("handoa")
        or summary_name.startswith("hand_oa")
        or "handoa" in summary_name
    ):
        return (base_path / "images" / "train").is_dir() and (base_path / "masks" / "train").is_dir()

    # Check directory name if it clearly indicates HandOA
    dir_name = base_path.name.lower()
    if "handoa" in dir_name or "hand_oa" in dir_name:
        return (base_path / "images" / "train").is_dir() and (base_path / "masks" / "train").is_dir()

    return False


def _mask_class_ids_from_summary(base_dir: Path) -> list[int]:
    summary = _load_summary(base_dir)
    class_pixels = summary.get("class_pixels", {})
    if isinstance(class_pixels, dict) and class_pixels:
        try:
            return sorted({int(class_id) for class_id in class_pixels})
        except (TypeError, ValueError):
            pass
    num_classes = summary.get("num_classes")
    if isinstance(num_classes, int) and num_classes > 0:
        return list(range(num_classes))
    return []


def _scan_mask_class_ids(base_dir: Path) -> list[int]:
    class_ids = set()
    for split in ("train", "val", "test"):
        for mask_path in _list_images(base_dir / "masks" / split):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            class_ids.update(int(value) for value in np.unique(mask))
    return sorted(class_ids)


def handoa_class_ids(base_dir: str | Path) -> list[int]:
    """Read HandOA segmentation class IDs from metadata or mask files."""
    base_path = Path(base_dir).expanduser()
    from_summary = _mask_class_ids_from_summary(base_path)
    if from_summary:
        return from_summary
    from_classes = [item["class_id"] for item in handoa_class_info(base_path)]
    if from_classes:
        return from_classes
    return _scan_mask_class_ids(base_path)


def infer_handoa_num_classes(base_dir: str | Path) -> int | None:
    """Infer total number of segmentation classes for HandOA."""
    base_path = Path(base_dir).expanduser()
    summary = _load_summary(base_path)
    if "num_classes" in summary and isinstance(summary["num_classes"], int):
        return summary["num_classes"]
    class_ids = handoa_class_ids(base_path)
    return max(class_ids) + 1 if class_ids else None


def _resolve_split_dirs(base_dir: Path, split: str) -> tuple[Path, Path]:
    image_dir = base_dir / "images" / split
    mask_dir = base_dir / "masks" / split
    if _list_images(image_dir) and _list_images(mask_dir):
        return image_dir, mask_dir

    if split != "train":
        train_image_dir = base_dir / "images" / "train"
        train_mask_dir = base_dir / "masks" / "train"
        if _list_images(train_image_dir) and _list_images(train_mask_dir):
            return train_image_dir, train_mask_dir
    return image_dir, mask_dir


class HandOASegDataset(Dataset):
    """Paired RGB images and discrete multiclass masks for HandOA segmentation."""

    def __init__(
        self,
        base_dir: str | Path,
        mode: str = "train",
        transform: Any = None,
        num_classes: int | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.mode = "val" if mode == "validation" else mode
        self.transform = transform

        inferred_num_classes = infer_handoa_num_classes(self.base_dir) or len(HANDOA_DEFAULT_CLASSES)
        self.num_classes = int(num_classes or inferred_num_classes)
        if inferred_num_classes > self.num_classes:
            print(
                f"Auto-updating HandOASegDataset num_classes from {self.num_classes} "
                f"to {inferred_num_classes} based on HandOA metadata."
            )
            self.num_classes = inferred_num_classes

        if self.num_classes <= 1:
            raise ValueError(
                "HandOASegDataset requires multiclass masks (typically 11 classes). "
                "Set --num_classes explicitly."
            )

        self.class_info = handoa_class_info(self.base_dir)
        # Ensure class_info covers up to self.num_classes
        present_ids = {item["class_id"] for item in self.class_info}
        for cid in range(self.num_classes):
            if cid not in present_ids:
                self.class_info.append({"class_id": cid, "name": f"class_{cid}"})
        self.class_info = sorted(self.class_info, key=lambda x: x["class_id"])

        self.image_dir, self.mask_dir = _resolve_split_dirs(self.base_dir, self.mode)
        image_map = _index_images(self.image_dir)
        mask_map = _index_images(self.mask_dir)
        paired_stems = sorted(set(image_map) & set(mask_map))
        if not paired_stems:
            raise FileNotFoundError(
                f"No HandOA image/mask pairs found for split '{self.mode}'. "
                f"image_dir='{self.image_dir}', mask_dir='{self.mask_dir}'"
            )

        missing_masks = sorted(set(image_map) - set(mask_map))
        orphan_masks = sorted(set(mask_map) - set(image_map))
        if missing_masks or orphan_masks:
            print(
                f"HandOA split '{self.mode}' pairing warning: "
                f"missing_masks={len(missing_masks)}, orphan_masks={len(orphan_masks)}"
            )

        self.samples = [(stem, image_map[stem], mask_map[stem]) for stem in paired_stems]
        print(
            f"total {len(self.samples)} {self.mode} samples "
            f"(images={self.image_dir}, masks={self.mask_dir}, classes={self.num_classes})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case, image_path, mask_path = self.samples[idx]
        image = read_rgb_image(image_path)
        label = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise FileNotFoundError(
                f"Failed to read HandOA sample '{case}'. "
                f"image='{image_path}', mask='{mask_path}'"
            )
        if label.ndim == 3:
            label = label[..., 0]

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        label = np.asarray(label, dtype=np.int64)
        if label.ndim == 3:
            label = label[..., 0]

        max_label = int(label.max()) if label.size else 0
        if max_label >= self.num_classes:
            raise ValueError(
                f"HandOA sample '{case}' contains label value {max_label}, "
                f"but num_classes={self.num_classes}."
            )
        return {"image": image, "label": label, "case": case}


__all__ = [
    "HANDOA_DATASET_NAMES",
    "HANDOA_DEFAULT_CLASSES",
    "HandOASegDataset",
    "handoa_class_ids",
    "handoa_class_info",
    "infer_handoa_num_classes",
    "is_handoa_dataset",
]
