"""Dataset adapter for HipOA (Hip Osteoarthritis) segmentation layout.

Designed for pelvic and hip X-ray bone and joint space segmentation datasets,
such as HipOA_segX1 and HipOA_segX2, following the 7-class anatomy standard:
  0: background
  1: right_hip_joint_space
  2: left_hip_joint_space
  3: right_pelvis
  4: left_pelvis
  5: right_femur
  6: left_femur
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
HIPOA_DATASET_NAMES = {
    "hipoa",
    "hipoa_segx1",
    "hipoa_segx2",
    "hipoa_seg",
    "hip_oa",
    "hip_seg",
    "hip",
}

# Standard 7 Hip OA anatomical bone & joint classes
HIPOA_DEFAULT_CLASSES = [
    {
        "class_id": 0,
        "name": "background",
        "name_vi": "Nền không chứa xương",
        "color": [0, 0, 0],
        "hex": "#000000",
        "description": "Background non-bone regions",
    },
    {
        "class_id": 1,
        "name": "right_hip_joint_space",
        "name_vi": "Khe khớp háng phải (Diện tiếp xúc chỏm - ổ cối)",
        "color": [139, 92, 246],
        "hex": "#8B5CF6",
        "description": "Right Hip Articular Joint Space (matching tibial_plateau in PhenoX01) - Primary site for minimum Joint Space Width (mJSW) and narrowing (JSN)",
    },
    {
        "class_id": 2,
        "name": "left_hip_joint_space",
        "name_vi": "Khe khớp háng trái (Diện tiếp xúc chỏm - ổ cối)",
        "color": [139, 92, 246],
        "hex": "#8B5CF6",
        "description": "Left Hip Articular Joint Space (matching tibial_plateau in PhenoX01) - Primary site for minimum Joint Space Width (mJSW) and narrowing (JSN)",
    },
    {
        "class_id": 3,
        "name": "right_pelvis",
        "name_vi": "Xương chậu & Ổ cối phải",
        "color": [255, 141, 23],
        "hex": "#FF8D17",
        "description": "Right Hemipelvis & Acetabulum (matching tibia in PhenoX01 - Warm Orange)",
    },
    {
        "class_id": 4,
        "name": "left_pelvis",
        "name_vi": "Xương chậu & Ổ cối trái",
        "color": [20, 83, 45],
        "hex": "#14532D",
        "description": "Left Hemipelvis & Acetabulum (matching patella in PhenoX01 - Dark Pine Green)",
    },
    {
        "class_id": 5,
        "name": "right_femur",
        "name_vi": "Xương đùi & Chỏm xương đùi phải",
        "color": [0, 180, 255],
        "hex": "#00B4FF",
        "description": "Right Proximal Femur - Includes femoral head, femoral neck, greater trochanter, and proximal shaft (matching femur in PhenoX01 - Sky Blue)",
    },
    {
        "class_id": 6,
        "name": "left_femur",
        "name_vi": "Xương đùi & Chỏm xương đùi trái",
        "color": [255, 120, 220],
        "hex": "#FF78DC",
        "description": "Left Proximal Femur - Includes femoral head, femoral neck, greater trochanter, and proximal shaft (matching fibula in PhenoX01 - Pink)",
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


def hipoa_class_info(base_dir: Path) -> list[dict[str, Any]]:
    """Load class taxonomy from classes.json, summary.json, or default HipOA taxonomy."""
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
                colors = data.get("colors", {})
                if isinstance(names, dict):
                    result = []
                    for cid_str, name in sorted(names.items(), key=lambda item: int(item[0])):
                        cid = int(cid_str)
                        c_dict: dict[str, Any] = {"class_id": cid, "name": str(name)}
                        if isinstance(colors, dict) and cid in colors:
                            c_dict["color"] = colors[cid]
                        elif isinstance(colors, dict) and cid_str in colors:
                            c_dict["color"] = colors[cid_str]
                        result.append(c_dict)
                    return result
                if isinstance(names, list):
                    result = []
                    for idx, name in enumerate(names):
                        c_dict = {"class_id": idx, "name": str(name)}
                        if isinstance(colors, list) and idx < len(colors):
                            c_dict["color"] = colors[idx]
                        result.append(c_dict)
                    return result
            except Exception:
                pass

    return list(HIPOA_DEFAULT_CLASSES)


def is_hipoa_dataset(base_dir: str | Path, dataset_name: str = "") -> bool:
    """Return whether a path/name follows the HipOA segmentation contract."""
    base_path = Path(base_dir).expanduser()
    name = (dataset_name or "").strip().lower()
    if name in HIPOA_DATASET_NAMES or name.startswith("hipoa") or name.startswith("hip_oa"):
        return True

    summary = _load_summary(base_path)
    summary_name = str(summary.get("dataset_name", "")).strip().lower()
    if (
        summary_name.startswith("hipoa")
        or summary_name.startswith("hip_oa")
        or "hipoa" in summary_name
    ):
        return (base_path / "images" / "train").is_dir() and (base_path / "masks" / "train").is_dir()

    dir_name = base_path.name.lower()
    if "hipoa" in dir_name or "hip_oa" in dir_name:
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

    class_stats = summary.get("class_statistics", {})
    if isinstance(class_stats, dict) and class_stats:
        ids = set()
        for stat in class_stats.values():
            if isinstance(stat, dict) and "id" in stat:
                ids.add(int(stat["id"]))
        if ids:
            return sorted(ids)

    classes = summary.get("classes", [])
    if isinstance(classes, list) and classes:
        ids = set()
        for item in classes:
            if isinstance(item, dict):
                cid = item.get("class_id") if "class_id" in item else item.get("id")
                if cid is not None:
                    ids.add(int(cid))
        if ids:
            return sorted(ids)

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


def hipoa_class_ids(base_dir: str | Path) -> list[int]:
    """Read HipOA segmentation class IDs from metadata or mask files."""
    base_path = Path(base_dir).expanduser()
    from_summary = _mask_class_ids_from_summary(base_path)
    if from_summary:
        return from_summary
    from_classes = [item["class_id"] for item in hipoa_class_info(base_path)]
    if from_classes:
        return from_classes
    return _scan_mask_class_ids(base_path)


def infer_hipoa_num_classes(base_dir: str | Path) -> int | None:
    """Infer total number of segmentation classes for HipOA."""
    base_path = Path(base_dir).expanduser()
    summary = _load_summary(base_path)
    if "num_classes" in summary and isinstance(summary["num_classes"], int):
        return summary["num_classes"]
    class_ids = hipoa_class_ids(base_path)
    if class_ids:
        return max(class_ids) + 1

    # Check yaml definition
    for yaml_name in ("data_unet.yaml", "data_segment.yaml", "data.yaml"):
        yaml_path = base_path / yaml_name
        if yaml_path.is_file():
            try:
                import yaml

                with yaml_path.open("r", encoding="utf-8") as file:
                    data = yaml.safe_load(file) or {}
                if "num_classes" in data and isinstance(data["num_classes"], int):
                    return int(data["num_classes"])
                if "names" in data:
                    return len(data["names"])
            except Exception:
                pass
    return None


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


class HipOASegDataset(Dataset):
    """Paired RGB images and discrete multiclass masks for HipOA segmentation."""

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

        inferred_num_classes = infer_hipoa_num_classes(self.base_dir) or len(HIPOA_DEFAULT_CLASSES)
        self.num_classes = int(num_classes or inferred_num_classes)
        if inferred_num_classes > self.num_classes:
            print(
                f"Auto-updating HipOASegDataset num_classes from {self.num_classes} "
                f"to {inferred_num_classes} based on HipOA metadata."
            )
            self.num_classes = inferred_num_classes

        if self.num_classes <= 1:
            raise ValueError(
                "HipOASegDataset requires multiclass masks (typically 7 classes). "
                "Set --num_classes explicitly."
            )

        self.class_info = hipoa_class_info(self.base_dir)
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
                f"No HipOA image/mask pairs found for split '{self.mode}'. "
                f"image_dir='{self.image_dir}', mask_dir='{self.mask_dir}'"
            )

        missing_masks = sorted(set(image_map) - set(mask_map))
        orphan_masks = sorted(set(mask_map) - set(image_map))
        if missing_masks or orphan_masks:
            print(
                f"HipOA split '{self.mode}' pairing warning: "
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
                f"Failed to read HipOA sample '{case}'. "
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
                f"HipOA sample '{case}' contains label value {max_label}, "
                f"but num_classes={self.num_classes}."
            )
        return {"image": image, "label": label, "case": case}


__all__ = [
    "HIPOA_DATASET_NAMES",
    "HIPOA_DEFAULT_CLASSES",
    "HipOASegDataset",
    "hipoa_class_ids",
    "hipoa_class_info",
    "infer_hipoa_num_classes",
    "is_hipoa_dataset",
]
