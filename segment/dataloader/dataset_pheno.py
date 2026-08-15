"""Dataset adapter for the PhenoX segmentation layout.

PhenoX exports also contain ``data_points.yaml`` for landmark training.  That
file describes four landmark groups, not the segmentation mask classes, so it
must not be used to infer ``num_classes`` for segmentation.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from segment.dataloader.image_io import read_rgb_image


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PHENO_DATASET_NAMES = {"pheno", "phenox", "phenox01", "pheno_x01"}


def _list_images(directory):
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _index_images(directory):
    return {path.stem: path for path in _list_images(directory)}


def _load_summary(base_dir):
    summary_path = base_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return summary if isinstance(summary, dict) else {}


def is_pheno_dataset(base_dir, dataset_name=""):
    """Return whether a path/name follows the PhenoX segmentation contract."""
    base_path = Path(base_dir).expanduser()
    name = (dataset_name or "").strip().lower()
    if name in PHENO_DATASET_NAMES or name.startswith("phenox"):
        return True

    try:
        summary_name = str(_load_summary(base_path).get("dataset_name", "")).lower()
    except (OSError, ValueError, TypeError):
        summary_name = ""
    return (
        summary_name.startswith("phenox")
        and (base_path / "images" / "train").is_dir()
        and (base_path / "masks" / "train").is_dir()
    )


def _mask_class_ids_from_summary(base_dir):
    try:
        class_pixels = _load_summary(base_dir).get("class_pixels", {})
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(class_pixels, dict):
        return []
    try:
        return sorted({int(class_id) for class_id in class_pixels})
    except (TypeError, ValueError):
        return []


def _scan_mask_class_ids(base_dir):
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


def pheno_class_ids(base_dir):
    """Read segmentation class IDs without consulting landmark metadata."""
    base_path = Path(base_dir).expanduser()
    return _mask_class_ids_from_summary(base_path) or _scan_mask_class_ids(base_path)


def infer_pheno_num_classes(base_dir):
    class_ids = pheno_class_ids(base_dir)
    return max(class_ids) + 1 if class_ids else None


def _resolve_split_dirs(base_dir, split):
    image_dir = base_dir / "images" / split
    mask_dir = base_dir / "masks" / split
    if _list_images(image_dir) and _list_images(mask_dir):
        return image_dir, mask_dir

    # PhenoX01 may be exported with a 100/0/0 ratio. Preserve the existing
    # runtime behaviour by using its train split for validation/zero-shot use.
    if split != "train":
        train_image_dir = base_dir / "images" / "train"
        train_mask_dir = base_dir / "masks" / "train"
        if _list_images(train_image_dir) and _list_images(train_mask_dir):
            return train_image_dir, train_mask_dir
    return image_dir, mask_dir


class PhenoSegDataset(Dataset):
    """Paired RGB images and discrete multiclass masks from a PhenoX export."""

    def __init__(self, base_dir, mode="train", transform=None, num_classes=None):
        self.base_dir = Path(base_dir).expanduser()
        self.mode = "val" if mode == "validation" else mode
        self.transform = transform

        class_ids = pheno_class_ids(self.base_dir)
        inferred_num_classes = max(class_ids, default=-1) + 1
        self.num_classes = int(num_classes or inferred_num_classes)
        if inferred_num_classes > self.num_classes:
            print(
                f"Auto-updating PhenoSegDataset num_classes from {self.num_classes} "
                f"to {inferred_num_classes} based on segmentation masks."
            )
            self.num_classes = inferred_num_classes
        if self.num_classes <= 1:
            raise ValueError(
                "PhenoSegDataset requires multiclass masks. Provide summary.json "
                "with class_pixels or set --num_classes explicitly."
            )

        self.class_info = [
            {"class_id": class_id, "name": f"class_{class_id}"}
            for class_id in range(self.num_classes)
        ]
        self.image_dir, self.mask_dir = _resolve_split_dirs(self.base_dir, self.mode)
        image_map = _index_images(self.image_dir)
        mask_map = _index_images(self.mask_dir)
        paired_stems = sorted(set(image_map) & set(mask_map))
        if not paired_stems:
            raise FileNotFoundError(
                f"No Pheno image/mask pairs found for split '{self.mode}'. "
                f"image_dir='{self.image_dir}', mask_dir='{self.mask_dir}'"
            )

        missing_masks = sorted(set(image_map) - set(mask_map))
        orphan_masks = sorted(set(mask_map) - set(image_map))
        if missing_masks or orphan_masks:
            print(
                f"Pheno split '{self.mode}' pairing warning: "
                f"missing_masks={len(missing_masks)}, orphan_masks={len(orphan_masks)}"
            )
        self.samples = [(stem, image_map[stem], mask_map[stem]) for stem in paired_stems]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case, image_path, mask_path = self.samples[idx]
        image = read_rgb_image(image_path)
        label = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise FileNotFoundError(
                f"Failed to read Pheno sample '{case}'. "
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
                f"Pheno sample '{case}' contains label value {max_label}, "
                f"but num_classes={self.num_classes}."
            )
        return {"image": image, "label": label, "case": case}
