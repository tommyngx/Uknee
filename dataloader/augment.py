import inspect
from typing import List

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

AVAILABLE_AUG_STRATEGIES = ("none", "basic", "standard", "strong", "xray", "auto")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
