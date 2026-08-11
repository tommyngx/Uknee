import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MESKO_DATASET_NAMES = {"mesko", "mesko5seg", "mesko_5seg", "unet_mesko5seg"}


def _load_yaml(path):
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _resolve_dataset_path(base_dir, value):
    if not value:
        return None

    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path

    return base_dir / value


def _list_image_files(directory):
    if directory is None or not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _index_files(directory):
    return {path.stem: path for path in _list_image_files(directory)}


def _load_class_info(base_dir):
    classes_path = base_dir / "classes.json"
    summary_path = base_dir / "summary.json"
    data_yaml_path = base_dir / "data.yaml"

    if classes_path.is_file():
        with classes_path.open("r", encoding="utf-8") as file:
            classes = json.load(file)
        if isinstance(classes, list):
            return sorted(classes, key=lambda item: int(item["class_id"]))

    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        classes = summary.get("classes", [])
        if isinstance(classes, list):
            return sorted(classes, key=lambda item: int(item["class_id"]))

    data_yaml = _load_yaml(data_yaml_path)
    names = data_yaml.get("names", {})
    if isinstance(names, dict):
        return [
            {"class_id": int(class_id), "name": str(name)}
            for class_id, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    if isinstance(names, list):
        return [{"class_id": index, "name": str(name)} for index, name in enumerate(names)]

    return []


def _split_dirs_from_yaml(base_dir, split):
    data_yaml = _load_yaml(base_dir / "data.yaml")
    image_value = data_yaml.get(split)
    masks = data_yaml.get("masks", {})
    mask_value = masks.get(split) if isinstance(masks, dict) else None

    image_dir = _resolve_dataset_path(base_dir, image_value)
    mask_dir = _resolve_dataset_path(base_dir, mask_value)

    if mask_dir is None and image_value:
        mask_dir = _resolve_dataset_path(base_dir, str(image_value).replace("images", "masks", 1))

    return image_dir, mask_dir


def _resolve_split_dirs(base_dir, split):
    image_dir = base_dir / "images" / split
    mask_dir = base_dir / "masks" / split
    if _list_image_files(image_dir) and _list_image_files(mask_dir):
        return image_dir, mask_dir

    yaml_image_dir, yaml_mask_dir = _split_dirs_from_yaml(base_dir, split)
    if _list_image_files(yaml_image_dir) and _list_image_files(yaml_mask_dir):
        return yaml_image_dir, yaml_mask_dir

    if split != "train":
        return _resolve_split_dirs(base_dir, "train")

    return image_dir, mask_dir


def is_mesko_dataset(base_dir, dataset_name=""):
    base_path = Path(base_dir)
    name = (dataset_name or "").lower()
    if name in MESKO_DATASET_NAMES:
        return True

    return (
        (base_path / "images" / "train").is_dir()
        and (base_path / "masks" / "train").is_dir()
        and ((base_path / "classes.json").is_file() or (base_path / "data.yaml").is_file())
    )


def infer_mesko_num_classes(base_dir):
    base_path = Path(base_dir).expanduser()
    class_info = _load_class_info(base_path)
    if not class_info:
        return None
    return max(int(item["class_id"]) for item in class_info) + 1


class Mesko5SegDataset(Dataset):
    def __init__(self, base_dir, mode="train", transform=None, num_classes=None):
        self.base_dir = Path(base_dir).expanduser()
        self.mode = "val" if mode == "validation" else mode
        self.transform = transform
        self.class_info = _load_class_info(self.base_dir)
        inferred_num_classes = (
            max((int(item["class_id"]) for item in self.class_info), default=-1) + 1
        )
        self.num_classes = int(num_classes or inferred_num_classes)

        if self.num_classes <= 1:
            raise ValueError(
                "Mesko5SegDataset is multiclass. Set --num_classes to the number of classes "
                "matching your dataset metadata (e.g. 7, 11)."
            )
        if inferred_num_classes > 1 and self.num_classes < inferred_num_classes:
            print(
                f"Auto-updating Mesko5SegDataset num_classes from {self.num_classes} to {inferred_num_classes} "
                f"based on dataset metadata."
            )
            self.num_classes = inferred_num_classes

        self.image_dir, self.mask_dir = _resolve_split_dirs(self.base_dir, self.mode)
        image_map = _index_files(self.image_dir)
        mask_map = _index_files(self.mask_dir)

        paired_stems = sorted(set(image_map) & set(mask_map))
        if not paired_stems:
            raise FileNotFoundError(
                f"No MESKO image/mask pairs found for split '{self.mode}'. "
                f"image_dir='{self.image_dir}', mask_dir='{self.mask_dir}'"
            )

        missing_masks = sorted(set(image_map) - set(mask_map))
        orphan_masks = sorted(set(mask_map) - set(image_map))
        if missing_masks or orphan_masks:
            print(
                f"MESKO split '{self.mode}' pairing warning: "
                f"missing_masks={len(missing_masks)}, orphan_masks={len(orphan_masks)}"
            )

        self.samples = [(stem, image_map[stem], mask_map[stem]) for stem in paired_stems]
        print(
            f"total {len(self.samples)} {self.mode} samples "
            f"(images={self.image_dir}, masks={self.mask_dir}, classes={self.num_classes})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case, image_path, mask_path = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise FileNotFoundError(
                f"Failed to read MESKO sample '{case}'. image='{image_path}', mask='{mask_path}'"
            )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        image = image.astype("float32")
        image = image.transpose(2, 0, 1) / 255.0

        label = np.asarray(label)
        if label.ndim == 3:
            label = label[..., 0]
        label = label.astype("int64")

        max_label = int(label.max()) if label.size else 0
        if max_label >= self.num_classes:
            raise ValueError(
                f"MESKO sample '{case}' contains label value {max_label}, "
                f"but num_classes={self.num_classes}."
            )

        return {"image": image, "label": label, "case": case}
