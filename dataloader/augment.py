import inspect
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

AVAILABLE_AUG_STRATEGIES = ("none", "basic", "standard", "strong", "xray", "auto")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MaskRetainRandomResizedCrop(A.DualTransform):
    """
    Crop on the original image first, then resize to (height, width).

    When the transform is applied, it samples crop candidates and only accepts
    crops that keep at least `min_mask_retention` of the foreground mask.
    Set `p=0.3` later if you want this to affect roughly 30% of training samples.
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_mask_retention: float = 0.5,
        min_crop_scale: float = 0.6,
        max_crop_scale: float = 0.9,
        min_mask_occupancy: float = 0.0,
        num_attempts: int = 15,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        if not (0.0 <= min_mask_retention <= 1.0):
            raise ValueError("min_mask_retention must be between 0 and 1.")
        if not (0.0 <= min_mask_occupancy <= 1.0):
            raise ValueError("min_mask_occupancy must be between 0 and 1.")
        if not (0.0 < min_crop_scale <= max_crop_scale <= 1.0):
            raise ValueError("Crop scale must satisfy 0 < min_crop_scale <= max_crop_scale <= 1.")
        if num_attempts < 1:
            raise ValueError("num_attempts must be >= 1.")

        self.height = int(height)
        self.width = int(width)
        self.min_mask_retention = float(min_mask_retention)
        self.min_crop_scale = float(min_crop_scale)
        self.max_crop_scale = float(max_crop_scale)
        self.min_mask_occupancy = float(min_mask_occupancy)
        self.num_attempts = int(num_attempts)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        return (
            "height",
            "width",
            "min_mask_retention",
            "min_crop_scale",
            "max_crop_scale",
            "min_mask_occupancy",
            "num_attempts",
            "interpolation",
            "mask_interpolation",
        )

    def _sample_crop_size(self, image_height: int, image_width: int):
        target_ratio = self.width / self.height
        max_allowed_height = min(
            image_height,
            max(1, int(round(image_width / max(target_ratio, 1e-8)))),
        )
        min_crop_height = max(1, int(round(image_height * self.min_crop_scale)))
        max_crop_height = max(1, int(round(image_height * self.max_crop_scale)))
        max_crop_height = min(max_crop_height, max_allowed_height)
        min_crop_height = min(min_crop_height, max_crop_height)

        crop_height = np.random.randint(min_crop_height, max_crop_height + 1)
        crop_width = max(1, int(round(crop_height * target_ratio)))

        if crop_width > image_width:
            crop_width = image_width
            crop_height = max(1, int(round(crop_width / max(target_ratio, 1e-8))))

        crop_height = min(crop_height, image_height)
        crop_width = min(crop_width, image_width)
        return crop_height, crop_width

    def _sample_crop_origin(
        self,
        image_height: int,
        image_width: int,
        crop_height: int,
        crop_width: int,
        foreground_points,
    ):
        max_y = max(image_height - crop_height, 0)
        max_x = max(image_width - crop_width, 0)

        if foreground_points is None or len(foreground_points) == 0:
            y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            return x_min, y_min

        center_y, center_x = foreground_points[np.random.randint(0, len(foreground_points))]

        min_y = max(0, int(center_y - crop_height + 1))
        max_y_from_point = min(int(center_y), max_y)
        min_x = max(0, int(center_x - crop_width + 1))
        max_x_from_point = min(int(center_x), max_x)

        if min_y > max_y_from_point:
            y_min = max(0, min(max_y, int(center_y - crop_height // 2)))
        else:
            y_min = np.random.randint(min_y, max_y_from_point + 1)

        if min_x > max_x_from_point:
            x_min = max(0, min(max_x, int(center_x - crop_width // 2)))
        else:
            x_min = np.random.randint(min_x, max_x_from_point + 1)

        return x_min, y_min

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        mask = params["mask"]
        image_height, image_width = image.shape[:2]

        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            foreground = np.any(mask_array > 0, axis=2)
        else:
            foreground = mask_array > 0

        total_foreground = int(foreground.sum())
        foreground_points = np.argwhere(foreground) if total_foreground > 0 else None

        for _ in range(self.num_attempts):
            crop_height, crop_width = self._sample_crop_size(image_height, image_width)
            x_min, y_min = self._sample_crop_origin(
                image_height=image_height,
                image_width=image_width,
                crop_height=crop_height,
                crop_width=crop_width,
                foreground_points=foreground_points,
            )
            x_max = x_min + crop_width
            y_max = y_min + crop_height

            if total_foreground == 0:
                return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

            kept_foreground = int(foreground[y_min:y_max, x_min:x_max].sum())
            retain_ratio = kept_foreground / max(total_foreground, 1)
            occupancy_ratio = kept_foreground / max(crop_height * crop_width, 1)

            if (
                retain_ratio >= self.min_mask_retention
                and occupancy_ratio >= self.min_mask_occupancy
            ):
                return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

        return {"x_min": 0, "y_min": 0, "x_max": image_width, "y_max": image_height}

    def _crop_and_resize(self, img, x_min, y_min, x_max, y_max, interpolation):
        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            cropped = img

        resized = cv2.resize(cropped, (self.width, self.height), interpolation=interpolation)
        if img.ndim == 3 and img.shape[2] == 1 and resized.ndim == 2:
            resized = resized[..., None]
        return resized

    def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return self._crop_and_resize(img, x_min, y_min, x_max, y_max, self.interpolation)

    def apply_to_mask(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return self._crop_and_resize(img, x_min, y_min, x_max, y_max, self.mask_interpolation)


def resolve_aug_strategy(strategy: str = "auto", dataset_name: str = "", base_dir: str = "") -> str:
    strategy = (strategy or "auto").lower()
    if strategy not in AVAILABLE_AUG_STRATEGIES:
        raise ValueError(
            f"Unsupported aug_strategy '{strategy}'. "
            f"Available options: {', '.join(AVAILABLE_AUG_STRATEGIES)}"
        )

    if strategy != "auto":
        return strategy

    hint = f"{dataset_name} {base_dir}".lower()
    if any(tag in hint for tag in ("covid", "xray", "montgomery", "nih")):
        return "xray"
    if any(tag in hint for tag in ("kvasir", "isic", "ph2", "skin", "derm")):
        return "strong"
    return "standard"


def _resize(img_size: int):
    return A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR)


def _random_flip(p: float = 0.5):
    # `A.Flip` was removed in newer albumentations releases.
    if hasattr(A, "Flip"):
        return A.Flip(p=p)
    return A.OneOf(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        ],
        p=p,
    )


def _make_transform(transform_cls, **kwargs):
    signature = inspect.signature(transform_cls)
    valid_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return transform_cls(**valid_kwargs)


def _affine_like(shift_limit: float, scale_limit: float, rotate_limit: float, border_mode: int, p: float):
    if hasattr(A, "Affine"):
        return _make_transform(
            A.Affine,
            scale=(1.0 - scale_limit, 1.0 + scale_limit),
            translate_percent={
                "x": (-shift_limit, shift_limit),
                "y": (-shift_limit, shift_limit),
            },
            rotate=(-rotate_limit, rotate_limit),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=border_mode,
            fill=0,
            fill_mask=0,
            p=p,
        )
    return _make_transform(
        A.ShiftScaleRotate,
        shift_limit=shift_limit,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        border_mode=border_mode,
        p=p,
    )


def _elastic_transform():
    return _make_transform(
        A.ElasticTransform,
        alpha=40,
        sigma=6,
        alpha_affine=6,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    )


def _optical_distortion():
    return _make_transform(
        A.OpticalDistortion,
        distort_limit=0.05,
        shift_limit=0.05,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    )


def _gauss_noise(var_limit, p=1.0):
    signature = inspect.signature(A.GaussNoise)
    kwargs = {"p": p}
    if "var_limit" in signature.parameters:
        kwargs["var_limit"] = var_limit
    elif "std_range" in signature.parameters:
        low, high = var_limit
        kwargs["std_range"] = (
            max(0.0, low / 255.0),
            max(0.0, high / 255.0),
        )
        if "mean_range" in signature.parameters:
            kwargs["mean_range"] = (0.0, 0.0)
    return A.GaussNoise(**kwargs)


def _normalize_and_tensor() -> List:
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]


def _standard_spatial_ops() -> List:
    return [
        A.RandomRotate90(p=0.5),
        _random_flip(p=0.5),
        _affine_like(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf(
            [
                _elastic_transform(),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.2,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                _optical_distortion(),
            ],
            p=0.25,
        ),
    ]


def _standard_intensity_ops() -> List:
    return [
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                _gauss_noise(var_limit=(5.0, 25.0)),
            ],
            p=0.35,
        )
    ]


def _strong_intensity_ops(img_size: int) -> List:
    hole_size = max(8, img_size // 12)
    min_hole_size = max(4, img_size // 32)
    return [
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                A.RandomGamma(gamma_limit=(80, 120)),
            ],
            p=0.50,
        ),
        A.OneOf(
            [
                _gauss_noise(var_limit=(10.0, 40.0)),
                A.MotionBlur(blur_limit=5),
                A.Blur(blur_limit=3),
            ],
            p=0.25,
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=hole_size,
            max_width=hole_size,
            min_holes=1,
            min_height=min_hole_size,
            min_width=min_hole_size,
            fill_value=0,
            p=0.15,
        ),
    ]


def _xray_ops() -> List:
    return [
        A.HorizontalFlip(p=0.5),
        _affine_like(shift_limit=0.03, scale_limit=0.08, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf(
            [
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12),
                A.RandomGamma(gamma_limit=(85, 115)),
            ],
            p=0.45,
        ),
        A.OneOf(
            [
                _gauss_noise(var_limit=(4.0, 18.0)),
                A.Blur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ],
            p=0.20,
        ),
    ]


def _build_policy(strategy: str, img_size: int) -> List:
    if strategy == "none":
        return [_resize(img_size)]
    if strategy == "basic":
        return [_resize(img_size), A.RandomRotate90(p=0.5), _random_flip(p=0.5)]
    if strategy == "standard":
        return [_resize(img_size), *_standard_spatial_ops(), *_standard_intensity_ops()]
    if strategy == "strong":
        return [_resize(img_size), *_standard_spatial_ops(), *_strong_intensity_ops(img_size)]
    if strategy == "xray":
        return [_resize(img_size), *_xray_ops()]
    raise ValueError(f"Unknown augmentation strategy: {strategy}")


def build_train_transform(
    img_size: int,
    strategy: str = "auto",
    dataset_name: str = "",
    base_dir: str = "",
) -> A.Compose:
    resolved_strategy = resolve_aug_strategy(strategy, dataset_name, base_dir)
    return A.Compose(_build_policy(resolved_strategy, img_size))


def build_val_transform(img_size: int) -> A.Compose:
    return A.Compose([_resize(img_size)])


def build_tensor_train_transform(
    img_size: int,
    strategy: str = "auto",
    dataset_name: str = "",
    base_dir: str = "",
) -> A.Compose:
    resolved_strategy = resolve_aug_strategy(strategy, dataset_name, base_dir)
    return A.Compose([*_build_policy(resolved_strategy, img_size), *_normalize_and_tensor()])


def build_tensor_val_transform(img_size: int) -> A.Compose:
    return A.Compose([_resize(img_size), *_normalize_and_tensor()])
