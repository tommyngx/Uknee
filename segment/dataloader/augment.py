import inspect
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

AVAILABLE_AUG_STRATEGIES = ("none", "basic", "standard", "strong", "xray", "auto")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MaskRetainRandomCrop(A.DualTransform):
    def __init__(
        self,
        min_crop_fraction: float = 0.85,
        max_crop_fraction: float = 1.0,
        min_mask_retention: float = 0.80,
        num_attempts: int = 25,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        if not 0.0 < min_crop_fraction <= max_crop_fraction <= 1.0:
            raise ValueError(
                "Crop fractions must satisfy 0 < min_crop_fraction <= max_crop_fraction <= 1."
            )
        if not 0.0 <= min_mask_retention <= 1.0:
            raise ValueError(
                "min_mask_retention must be between 0 and 1."
            )
        if num_attempts < 1:
            raise ValueError("num_attempts must be at least 1.")

        self.min_crop_fraction = float(min_crop_fraction)
        self.max_crop_fraction = float(max_crop_fraction)
        self.min_mask_retention = float(min_mask_retention)
        self.num_attempts = int(num_attempts)

    @property
    def targets_as_params(self):
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        return (
            "min_crop_fraction",
            "max_crop_fraction",
            "min_mask_retention",
            "num_attempts",
        )

    def _calculate_crop(self, image, mask):
        h, w = image.shape[:2]
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            foreground = np.any(mask_array > 0, axis=2)
        else:
            foreground = mask_array > 0
        total_foreground = int(foreground.sum())

        for _ in range(self.num_attempts):
            crop_fraction = np.random.uniform(
                self.min_crop_fraction,
                self.max_crop_fraction,
            )
            crop_h = max(1, int(round(h * crop_fraction)))
            crop_w = max(1, int(round(w * crop_fraction)))
            max_y = h - crop_h
            max_x = w - crop_w
            y_min = (
                np.random.randint(0, max_y + 1)
                if max_y > 0
                else 0
            )
            x_min = (
                np.random.randint(0, max_x + 1)
                if max_x > 0
                else 0
            )
            y_max = y_min + crop_h
            x_max = x_min + crop_w
            if total_foreground == 0:
                return {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            kept_foreground = int(
                foreground[y_min:y_max, x_min:x_max].sum()
            )
            retention = (
                kept_foreground / max(total_foreground, 1)
            )
            if retention >= self.min_mask_retention:
                return {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
        return {
            "x_min": 0,
            "y_min": 0,
            "x_max": w,
            "y_max": h,
        }

    # Albumentations 1.x
    def get_params_dependent_on_targets(self, params):
        return self._calculate_crop(
            params["image"],
            params["mask"],
        )

    # Albumentations 2.x
    def get_params_dependent_on_data(self, params, data):
        return self._calculate_crop(
            data["image"],
            data["mask"],
        )

    def apply(
        self,
        img,
        x_min=0,
        y_min=0,
        x_max=0,
        y_max=0,
        **params,
    ):
        return img[y_min:y_max, x_min:x_max]

    def apply_to_mask(
        self,
        mask,
        x_min=0,
        y_min=0,
        x_max=0,
        y_max=0,
        **params,
    ):
        return mask[y_min:y_max, x_min:x_max]


def resolve_aug_strategy(
    strategy: str = "auto",
    dataset_name: str = "",
    base_dir: str = "",
) -> str:
    strategy = (strategy or "auto").lower()
    if strategy not in AVAILABLE_AUG_STRATEGIES:
        raise ValueError(
            f"Unsupported aug_strategy '{strategy}'. "
            f"Available options: {', '.join(AVAILABLE_AUG_STRATEGIES)}"
        )
    if strategy != "auto":
        return strategy
    hint = f"{dataset_name} {base_dir}".lower()
    if any(
        tag in hint
        for tag in (
            "covid",
            "xray",
            "radiograph",
            "montgomery",
            "nih",
            "knee",
            "uknee",
            "mesko",
            "osteoarthritis",
        )
    ):
        return "xray"
    if any(
        tag in hint
        for tag in (
            "kvasir",
            "isic",
            "ph2",
            "skin",
            "derm",
        )
    ):
        return "strong"
    return "standard"


def _resize(img_size: int):
    return _make_transform(
        A.Resize,
        height=img_size,
        width=img_size,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )


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
        low_var, high_var = var_limit
        kwargs["std_range"] = (
            np.sqrt(max(0.0, low_var)) / 255.0,
            np.sqrt(max(0.0, high_var)) / 255.0,
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


def _xray_ops(img_size: int) -> List:
    return [
        # Randomly trim the left, right, top, and bottom.
        # The crop keeps the original aspect ratio.
        MaskRetainRandomCrop(
            min_crop_fraction=0.88,
            max_crop_fraction=1.00,
            min_mask_retention=0.85,
            num_attempts=25,
            p=0.40,
        ),
        # Resize once after cropping.
        _resize(img_size),
        _affine_like(
            shift_limit=0.04,
            scale_limit=0.07,
            rotate_limit=8,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.55,
        ),
        _make_transform(
            A.GridDistortion,
            num_steps=5,
            distort_limit=0.04,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.07,
        ),
        A.OneOf(
            [
                A.CLAHE(
                    clip_limit=(1.5, 3.0),
                    tile_grid_size=(8, 8),
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.15,
                    p=1.0,
                ),
                A.RandomGamma(
                    gamma_limit=(88, 112),
                    p=1.0,
                ),
                A.RandomToneCurve(
                    scale=0.06,
                    p=1.0,
                ),
            ],
            p=0.55,
        ),
        A.OneOf(
            [
                _gauss_noise(
                    var_limit=(4.0, 14.0),
                    p=1.0,
                ),
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=(0.1, 0.7),
                    p=1.0,
                ),
                A.MotionBlur(
                    blur_limit=3,
                    p=1.0,
                ),
                A.Sharpen(
                    alpha=(0.05, 0.18),
                    lightness=(0.95, 1.05),
                    p=1.0,
                ),
            ],
            p=0.18,
        ),
        _make_transform(
            A.Downscale,
            scale_range=(0.85, 0.97),
            interpolation_pair={
                "downscale": cv2.INTER_AREA,
                "upscale": cv2.INTER_LINEAR,
            },
            p=0.12,
        ),
    ]


def _build_policy(strategy: str, img_size: int) -> List:
    if strategy == "none":
        return [_resize(img_size)]
    if strategy == "basic":
        return [
            _resize(img_size),
            _affine_like(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=7,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4,
            ),
        ]
    if strategy == "standard":
        return [_resize(img_size), *_standard_spatial_ops(), *_standard_intensity_ops()]
    if strategy == "strong":
        return [_resize(img_size), *_standard_spatial_ops(), *_strong_intensity_ops(img_size)]
    if strategy == "xray":
        return _xray_ops(img_size)
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
