import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check a binary segmentation dataset and generate train.txt / val.txt."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for split generation")
    parser.add_argument("--train_file", type=str, default="train.txt", help="Train split filename")
    parser.add_argument("--val_file", type=str, default="val.txt", help="Validation split filename")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train.txt / val.txt if they already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only check and print summary without writing split files",
    )
    return parser.parse_args()


def discover_mask_dir(data_dir: Path):
    masks_zero_dir = data_dir / "masks" / "0"
    masks_dir = data_dir / "masks"

    if masks_zero_dir.is_dir():
        return masks_zero_dir, True
    if masks_dir.is_dir():
        return masks_dir, False
    return None, False


def collect_files(directory: Path):
    files = []
    duplicates = []
    stem_to_path = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        stem = path.stem
        if stem in stem_to_path:
            duplicates.append((stem, stem_to_path[stem], path))
            continue
        stem_to_path[stem] = path
        files.append(path)

    return stem_to_path, files, duplicates


def read_image(path: Path):
    try:
        with Image.open(path) as image:
            return np.array(image)
    except Exception as exc:
        raise ValueError(f"Failed to read file: {path}") from exc


def describe_shape(array: np.ndarray):
    if array.ndim == 2:
        return array.shape[0], array.shape[1], 1
    return array.shape[0], array.shape[1], array.shape[2]


def format_counter(counter: Counter, limit=10):
    if not counter:
        return "none"
    items = counter.most_common(limit)
    text = ", ".join([f"{key}:{value}" for key, value in items])
    if len(counter) > limit:
        text += f", ... (+{len(counter) - limit} more)"
    return text


def write_split(path: Path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(f"{sample}\n")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    mask_dir, uses_loader_layout = discover_mask_dir(data_dir)
    if mask_dir is None:
        raise SystemExit(
            f"Mask directory not found. Expected either {data_dir / 'masks' / '0'} or {data_dir / 'masks'}"
        )

    image_map, image_files, image_duplicates = collect_files(images_dir)
    mask_map, mask_files, mask_duplicates = collect_files(mask_dir)

    image_stems = set(image_map.keys())
    mask_stems = set(mask_map.keys())
    paired_stems = sorted(image_stems & mask_stems)
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)

    warnings = []
    if not uses_loader_layout:
        warnings.append(
            "Detected masks in 'masks/' instead of 'masks/0/'. "
            "The current generic loader in this repo expects 'masks/0/'."
        )
    if image_duplicates:
        warnings.append(f"Found {len(image_duplicates)} duplicate image stems.")
    if mask_duplicates:
        warnings.append(f"Found {len(mask_duplicates)} duplicate mask stems.")
    if missing_masks:
        warnings.append(f"Found {len(missing_masks)} images without matching masks.")
    if orphan_masks:
        warnings.append(f"Found {len(orphan_masks)} masks without matching images.")

    image_ext_counter = Counter(path.suffix.lower() for path in image_files)
    mask_ext_counter = Counter(path.suffix.lower() for path in mask_files)

    if set(image_ext_counter.keys()) != {".png"}:
        warnings.append(
            "Generic loader currently reads images as '<name>.png'. Non-png image files will need renaming or loader changes."
        )
    if set(mask_ext_counter.keys()) != {".png"}:
        warnings.append(
            "Generic loader currently reads masks as '<name>.png'. Non-png mask files will need renaming or loader changes."
        )

    image_size_counter = Counter()
    mask_size_counter = Counter()
    image_channel_counter = Counter()
    mask_channel_counter = Counter()
    matched_size_count = 0
    unreadable_pairs = []
    unique_mask_values = set()
    non_single_channel_masks = 0
    foreground_ratios = []

    for stem in paired_stems:
        try:
            image = read_image(image_map[stem])
            mask = read_image(mask_map[stem])
        except Exception as exc:
            unreadable_pairs.append((stem, str(exc)))
            continue

        img_h, img_w, img_c = describe_shape(image)
        mask_h, mask_w, mask_c = describe_shape(mask)

        image_size_counter[(img_h, img_w)] += 1
        mask_size_counter[(mask_h, mask_w)] += 1
        image_channel_counter[img_c] += 1
        mask_channel_counter[mask_c] += 1

        if img_h == mask_h and img_w == mask_w:
            matched_size_count += 1

        if mask_c != 1:
            non_single_channel_masks += 1
            mask_values_view = mask[..., 0]
        else:
            mask_values_view = mask

        unique_values = np.unique(mask_values_view)
        unique_mask_values.update(unique_values.tolist())
        foreground_ratios.append(float(np.mean(mask_values_view > 0)))

    if unreadable_pairs:
        warnings.append(f"Found {len(unreadable_pairs)} unreadable image/mask pairs.")

    unique_mask_values_sorted = sorted(unique_mask_values)
    if len(unique_mask_values_sorted) > 32:
        unique_mask_values_preview = unique_mask_values_sorted[:32]
        warnings.append(
            f"Mask values contain more than 32 unique values. Preview: {unique_mask_values_preview}"
        )
    else:
        unique_mask_values_preview = unique_mask_values_sorted

    binary_like_values = set(unique_mask_values_sorted).issubset({0, 1, 255})
    if non_single_channel_masks > 0:
        warnings.append(f"Found {non_single_channel_masks} masks that are not single-channel.")

    valid_pairs = [
        stem
        for stem in paired_stems
        if stem not in {item[0] for item in unreadable_pairs}
    ]

    if not valid_pairs:
        raise SystemExit("No valid image/mask pairs found. Split files were not generated.")

    random.Random(args.seed).shuffle(valid_pairs)

    if len(valid_pairs) == 1:
        val_count = 0
    else:
        requested_val = int(round(len(valid_pairs) * args.val_ratio))
        val_count = min(max(requested_val, 1), len(valid_pairs) - 1)

    train_count = len(valid_pairs) - val_count
    train_samples = sorted(valid_pairs[:train_count])
    val_samples = sorted(valid_pairs[train_count:])

    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    if not args.dry_run:
        if (train_path.exists() or val_path.exists()) and not args.overwrite:
            raise SystemExit(
                f"Split file already exists. Use --overwrite to replace them: {train_path}, {val_path}"
            )
        write_split(train_path, train_samples)
        write_split(val_path, val_samples)

    print("=" * 72)
    print("Dataset Summary")
    print("=" * 72)
    print(f"Dataset root           : {data_dir}")
    print(f"Images directory       : {images_dir}")
    print(f"Masks directory        : {mask_dir}")
    print(f"Loader layout ready    : {'yes' if uses_loader_layout else 'no'}")
    print(f"Total image files      : {len(image_files)}")
    print(f"Total mask files       : {len(mask_files)}")
    print(f"Paired samples         : {len(paired_stems)}")
    print(f"Valid readable pairs   : {len(valid_pairs)}")
    print(f"Images without masks   : {len(missing_masks)}")
    print(f"Masks without images   : {len(orphan_masks)}")
    print(f"Image extensions       : {format_counter(image_ext_counter)}")
    print(f"Mask extensions        : {format_counter(mask_ext_counter)}")
    print(f"Image size distribution: {format_counter(image_size_counter)}")
    print(f"Mask size distribution : {format_counter(mask_size_counter)}")
    print(f"Image channels         : {format_counter(image_channel_counter)}")
    print(f"Mask channels          : {format_counter(mask_channel_counter)}")
    print(f"Matched image/mask size: {matched_size_count}/{len(valid_pairs)}")
    print(f"Mask unique values     : {unique_mask_values_preview}")
    print(f"Binary-like masks      : {'yes' if binary_like_values else 'no'}")

    if foreground_ratios:
        print(
            "Foreground ratio       : "
            f"min={min(foreground_ratios):.6f}, "
            f"mean={sum(foreground_ratios) / len(foreground_ratios):.6f}, "
            f"max={max(foreground_ratios):.6f}"
        )
    else:
        print("Foreground ratio       : unavailable")

    print("-" * 72)
    print("Split Summary")
    print("-" * 72)
    print(f"Seed                   : {args.seed}")
    print(f"Validation ratio       : {args.val_ratio}")
    print(f"Train samples          : {len(train_samples)}")
    print(f"Validation samples     : {len(val_samples)}")
    print(f"train.txt path         : {train_path}")
    print(f"val.txt path           : {val_path}")
    print(f"Files written          : {'no (dry-run)' if args.dry_run else 'yes'}")

    if warnings:
        print("-" * 72)
        print("Warnings")
        print("-" * 72)
        for warning in warnings:
            print(f"- {warning}")

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        print(f"- Missing mask examples: {preview}")
    if orphan_masks:
        preview = ", ".join(orphan_masks[:10])
        print(f"- Orphan mask examples : {preview}")
    if unreadable_pairs:
        preview = ", ".join([item[0] for item in unreadable_pairs[:10]])
        print(f"- Unreadable pairs     : {preview}")


if __name__ == "__main__":
    main()
