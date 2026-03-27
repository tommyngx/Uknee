import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize segmentation masks using a configurable thresholding rule."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Optional explicit mask directory. Defaults to masks/0 or masks under data_dir.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fixed",
        choices=["fixed", "otsu"],
        help="Thresholding method used to binarize masks.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Threshold used when --method fixed. Pixels >= threshold become foreground.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Treat darker pixels as foreground instead of brighter pixels.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "source"],
        help="Write normalized masks as .png or keep the source extension.",
    )
    parser.add_argument(
        "--keep_originals",
        action="store_true",
        help="Keep original mask files when writing to a new extension such as .png.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only inspect masks and report what would change without writing files.",
    )
    return parser.parse_args()


def discover_mask_dir(data_dir: Path):
    masks_zero_dir = data_dir / "masks" / "0"
    masks_dir = data_dir / "masks"

    if masks_zero_dir.is_dir():
        return masks_zero_dir
    if masks_dir.is_dir():
        return masks_dir
    return None


def collect_mask_files(mask_dir: Path):
    files = []
    duplicates = []
    stem_to_path = {}

    for path in sorted(mask_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if path.stem in stem_to_path:
            duplicates.append((path.stem, stem_to_path[path.stem], path))
            continue
        stem_to_path[path.stem] = path
        files.append(path)

    return files, duplicates


def read_mask(path: Path):
    try:
        with Image.open(path) as image:
            original_mask = np.array(image)
            grayscale_mask = np.array(image.convert("L"))
            return original_mask, grayscale_mask
    except Exception as exc:
        raise ValueError(f"Failed to read mask: {path}") from exc


def compute_otsu_threshold(grayscale_mask: np.ndarray):
    histogram = np.bincount(grayscale_mask.reshape(-1), minlength=256).astype(np.float64)
    total_pixels = float(grayscale_mask.size)
    if total_pixels == 0:
        return 0

    intensity_values = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(histogram, intensity_values))
    background_weight = 0.0
    background_sum = 0.0
    best_threshold = 0
    best_variance = -1.0

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue

        foreground_weight = total_pixels - background_weight
        if foreground_weight == 0:
            break

        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (total_sum - background_sum) / foreground_weight
        between_class_variance = (
            background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        )

        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = threshold

    return int(best_threshold)


def get_channel_count(mask_array: np.ndarray):
    if mask_array.ndim == 2:
        return 1
    return mask_array.shape[2]


def normalize_mask(mask_array: np.ndarray, grayscale_mask: np.ndarray, method: str, threshold: int, invert: bool):
    grayscale_mask = grayscale_mask.astype(np.uint8, copy=False)
    channel_count = get_channel_count(mask_array)

    if method == "otsu":
        applied_threshold = compute_otsu_threshold(grayscale_mask)
    else:
        applied_threshold = int(np.clip(threshold, 0, 255))

    if invert:
        foreground = grayscale_mask <= applied_threshold
    else:
        foreground = grayscale_mask >= applied_threshold

    normalized = foreground.astype(np.uint8) * 255
    unique_values = np.unique(grayscale_mask).tolist()
    already_binary = channel_count == 1 and np.array_equal(grayscale_mask, normalized)
    return normalized, channel_count, unique_values, already_binary, applied_threshold


def build_output_path(mask_path: Path, output_format: str):
    if output_format == "source":
        return mask_path
    return mask_path.with_suffix(".png")


def format_counter(counter: Counter, limit=10):
    if not counter:
        return "none"
    items = counter.most_common(limit)
    text = ", ".join([f"{key}:{value}" for key, value in items])
    if len(counter) > limit:
        text += f", ... (+{len(counter) - limit} more)"
    return text


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")

    if args.mask_dir:
        mask_dir = Path(args.mask_dir).expanduser().resolve()
    else:
        mask_dir = discover_mask_dir(data_dir)

    if mask_dir is None or not mask_dir.is_dir():
        raise SystemExit(
            f"Mask directory not found. Expected either {data_dir / 'masks' / '0'} or {data_dir / 'masks'}"
        )
    if not 0 <= args.threshold <= 255:
        raise SystemExit(f"--threshold must be in [0, 255], got {args.threshold}")

    mask_files, duplicates = collect_mask_files(mask_dir)
    if not mask_files:
        raise SystemExit(f"No mask files found in: {mask_dir}")

    warnings = []
    if duplicates:
        warnings.append(f"Found {len(duplicates)} duplicate mask stems.")
    if args.output_format == "source":
        warnings.append(
            "Writing back to JPEG masks can reintroduce non-binary values because JPEG is lossy. "
            "Use --output_format png for training-ready masks."
        )
    if args.output_format == "png" and args.keep_originals:
        warnings.append(
            "Keeping originals while writing .png files will leave duplicate stems in the mask directory."
        )

    input_ext_counter = Counter(path.suffix.lower() for path in mask_files)
    output_ext_counter = Counter()
    unique_values_counter = Counter()
    threshold_counter = Counter()

    total_files = len(mask_files)
    readable_files = 0
    written_files = 0
    changed_files = 0
    unchanged_files = 0
    removed_originals = 0
    single_channel_files = 0
    multi_channel_files = 0
    unreadable_files = []
    foreground_ratios = []

    progress_bar = tqdm(mask_files, total=total_files, desc="Normalizing masks", unit="mask")
    for mask_path in progress_bar:
        try:
            mask_array, grayscale_mask = read_mask(mask_path)
        except Exception as exc:
            unreadable_files.append((mask_path.name, str(exc)))
            continue

        readable_files += 1
        normalized_mask, channels, unique_values, already_binary, applied_threshold = normalize_mask(
            mask_array=mask_array,
            grayscale_mask=grayscale_mask,
            method=args.method,
            threshold=args.threshold,
            invert=args.invert,
        )
        output_path = build_output_path(mask_path, args.output_format)
        output_ext_counter[output_path.suffix.lower()] += 1
        threshold_counter[applied_threshold] += 1

        if channels == 1:
            single_channel_files += 1
        else:
            multi_channel_files += 1

        unique_values_counter[len(unique_values)] += 1
        foreground_ratios.append(float(np.mean(normalized_mask > 0)))

        output_needs_write = not already_binary or output_path != mask_path
        if output_needs_write:
            changed_files += 1
        else:
            unchanged_files += 1

        if args.dry_run or not output_needs_write:
            progress_bar.set_postfix(
                written=written_files,
                changed=changed_files,
                unreadable=len(unreadable_files),
            )
            continue

        Image.fromarray(normalized_mask, mode="L").save(output_path)
        written_files += 1

        if output_path != mask_path and not args.keep_originals:
            mask_path.unlink()
            removed_originals += 1

        progress_bar.set_postfix(
            written=written_files,
            changed=changed_files,
            unreadable=len(unreadable_files),
        )

    print("=" * 72)
    print("Mask Summary")
    print("=" * 72)
    print(f"Dataset root           : {data_dir}")
    print(f"Masks directory        : {mask_dir}")
    print(f"Total mask files       : {total_files}")
    print(f"Readable masks         : {readable_files}")
    print(f"Unreadable masks       : {len(unreadable_files)}")
    print(f"Single-channel masks   : {single_channel_files}")
    print(f"Multi-channel masks    : {multi_channel_files}")
    print(f"Changed masks          : {changed_files}")
    print(f"Unchanged masks        : {unchanged_files}")
    print(f"Files written          : {'no (dry-run)' if args.dry_run else written_files}")
    print(f"Originals removed      : {0 if args.dry_run else removed_originals}")
    print(f"Threshold method       : {args.method}")
    print(f"Fixed threshold        : {args.threshold if args.method == 'fixed' else 'n/a'}")
    print(f"Invert foreground      : {'yes' if args.invert else 'no'}")
    print(f"Input extensions       : {format_counter(input_ext_counter)}")
    print(f"Output extensions      : {format_counter(output_ext_counter)}")
    print(f"Unique-value counts    : {format_counter(unique_values_counter)}")
    print(f"Applied thresholds     : {format_counter(threshold_counter)}")
    if threshold_counter:
        threshold_values = list(threshold_counter.elements())
        print(
            "Threshold stats        : "
            f"min={min(threshold_values)}, "
            f"mean={sum(threshold_values) / len(threshold_values):.2f}, "
            f"max={max(threshold_values)}"
        )
    else:
        print("Threshold stats        : unavailable")
    if foreground_ratios:
        print(
            "Foreground ratio       : "
            f"min={min(foreground_ratios):.6f}, "
            f"mean={sum(foreground_ratios) / len(foreground_ratios):.6f}, "
            f"max={max(foreground_ratios):.6f}"
        )
    else:
        print("Foreground ratio       : unavailable")
    print(f"Output format          : {args.output_format}")
    print(f"Keep originals         : {'yes' if args.keep_originals else 'no'}")

    if warnings:
        print("-" * 72)
        print("Warnings")
        print("-" * 72)
        for warning in warnings:
            print(f"- {warning}")

    if duplicates:
        preview = ", ".join([item[0] for item in duplicates[:10]])
        print(f"- Duplicate stems      : {preview}")
    if unreadable_files:
        preview = ", ".join([item[0] for item in unreadable_files[:10]])
        print(f"- Unreadable masks     : {preview}")


if __name__ == "__main__":
    main()
