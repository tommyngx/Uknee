import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize segmentation masks so every non-zero pixel becomes 255."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Optional explicit mask directory. Defaults to masks/0 or masks under data_dir.",
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
            return np.array(image)
    except Exception as exc:
        raise ValueError(f"Failed to read mask: {path}") from exc


def normalize_mask(mask_array: np.ndarray):
    if mask_array.ndim == 2:
        normalized = (mask_array != 0).astype(np.uint8) * 255
        channels = 1
    else:
        normalized = np.any(mask_array != 0, axis=-1).astype(np.uint8) * 255
        channels = mask_array.shape[2]

    unique_values = np.unique(mask_array)
    already_binary = channels == 1 and set(unique_values.tolist()).issubset({0, 255})
    return normalized, channels, unique_values.tolist(), already_binary


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

    total_files = len(mask_files)
    readable_files = 0
    written_files = 0
    changed_files = 0
    unchanged_files = 0
    removed_originals = 0
    single_channel_files = 0
    multi_channel_files = 0
    unreadable_files = []

    progress_bar = tqdm(mask_files, total=total_files, desc="Normalizing masks", unit="mask")
    for mask_path in progress_bar:
        try:
            mask_array = read_mask(mask_path)
        except Exception as exc:
            unreadable_files.append((mask_path.name, str(exc)))
            continue

        readable_files += 1
        normalized_mask, channels, unique_values, already_binary = normalize_mask(mask_array)
        output_path = build_output_path(mask_path, args.output_format)
        output_ext_counter[output_path.suffix.lower()] += 1

        if channels == 1:
            single_channel_files += 1
        else:
            multi_channel_files += 1

        unique_values_counter[len(unique_values)] += 1

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
    print(f"Input extensions       : {format_counter(input_ext_counter)}")
    print(f"Output extensions      : {format_counter(output_ext_counter)}")
    print(f"Unique-value counts    : {format_counter(unique_values_counter)}")
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
